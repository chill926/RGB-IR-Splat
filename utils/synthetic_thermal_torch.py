"""Synthetic data adapter for the existing physical field and CUDA GS renderer."""
import math
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.graphics_utils import getProjectionMatrix
from utils.sh_utils import RGB2SH
from utils.thermal_physics import MaterialThermalField, UniformLWIRPlanckLUT
from utils.thermal_spatial import TemperatureSpatialPrior
from utils.thermal_structured import StructuredThermalField, build_temperature_basis


def tensor(value,device):
    return torch.as_tensor(value,dtype=torch.float32,device=device)


def setup_geometry(geometry,device):
    model=GaussianModel(0,device)
    for name,key,transform in [("_xyz","xyz",lambda x:x),("_scaling","scaling",torch.log),
            ("_rotation","rotation",lambda x:x),("_opacity","opacity",torch.logit)]:
        setattr(model,name,nn.Parameter(transform(tensor(geometry[key],device)),requires_grad=False))
    model._features_dc=nn.Parameter(RGB2SH(tensor(geometry["rgb"],device))[:,None,:],requires_grad=False)
    model._features_rest=nn.Parameter(torch.zeros(len(geometry["xyz"]),0,3,device=device),requires_grad=False)
    model._thermal_features_dc=nn.Parameter(model._features_dc.detach().clone(),requires_grad=False)
    model._thermal_features_rest=nn.Parameter(model._features_rest.detach().clone(),requires_grad=False)
    return model


def camera_from_record(record,device):
    rotation=np.asarray(record["camera_to_world_rotation"])
    center=np.asarray(record["center"])
    w2c=np.eye(4);w2c[:3,:3]=rotation.T;w2c[:3,3]=-rotation.T@center
    fovx=2*math.atan(record["width"]/(2*record["fx"]))
    fovy=2*math.atan(record["height"]/(2*record["fy"]))
    view=tensor(w2c,device).T.contiguous()
    projection=getProjectionMatrix(.01,100.,fovx,fovy).to(device).T.contiguous()
    return SimpleNamespace(image_width=record["width"],image_height=record["height"],FoVx=fovx,FoVy=fovy,
        world_view_transform=view,full_proj_transform=view@projection,camera_center=tensor(center,device),image_name=record["name"])


def render_values(model,camera,values):
    device=values.device
    pipe=SimpleNamespace(compute_cov3D_python=False,convert_SHs_python=True,debug=False)
    return render(camera,model,pipe,torch.zeros(3,device=device),0.,0.,0.,device,
                  override_color=values.reshape(-1,1).repeat(1,3),detach_geometry=True)["render"][:1]


def observed_normalized(observation,sensor,device):
    value=((observation["dn"].astype(np.float64)-sensor["offset"])/sensor["gain"]-sensor["radiance_min"])/sensor["radiance_span"]
    return tensor(value[None],device),tensor(observation["mask"][None],device)


def palette_tensor(value):
    f=((value-.06)/.24).clamp(0,1)
    return torch.cat(((2*f).clamp(0,1),1-(2*f-1).abs(),(2*(1-f)).clamp(0,1)),dim=0)


def build_field(document,geometry,frames,branch,device,controls=256,epsilon_bias=0.):
    count=len(geometry["xyz"])
    lower,upper=document["sensor"]["temperature_bounds_K"]
    initial=(document["initial_temperature_K"]-lower)/(upper-lower)
    samples=[]
    for _,observation in frames["fit"]:
        dn=observation["dn"][observation["mask"]][::8].astype(np.float64)
        samples.append(((dn-document["sensor"]["offset"])/document["sensor"]["gain"]-document["sensor"]["radiance_min"])/document["sensor"]["radiance_span"])
    env=float(np.clip(np.quantile(np.concatenate(samples),.1),.001,.999))
    epsilon=np.clip(np.array([m["epsilon0_prior"] for m in document["materials"]])+epsilon_bias,.02,.98)
    base=MaterialThermalField(torch.as_tensor(geometry["material_ids"],device=device),
        torch.full((count,1),initial,device=device),tensor(epsilon,device),
        material_names=[m["name"] for m in document["materials"]],branch=branch,
        initial_environment=env,temp_min=lower,temp_max=upper,temp_ref=document["temp_ref_K"]).to(device)
    base.set_branch_trainability()
    normals=tensor(geometry["normals"],device)
    prior=TemperatureSpatialPrior.build(tensor(geometry["xyz"],device),tensor(geometry["opacity"],device),
        tensor(geometry["rgb"],device),base.material_ids,torch.as_tensor(geometry["fit_support"],device=device),neighbors=6)
    basis,protocol=build_temperature_basis(tensor(geometry["xyz"],device),base.material_ids,
        torch.as_tensor(geometry["fit_support"],device=device),prior,normals,max_controls=controls)
    protocol["initial_temperature_source"]="public_constant_reference_temperature_not_surface_GT"
    field=StructuredThermalField(base,basis,normals,torch.ones(count,1,device=device),protocol,
        control_bound_K=75.,residual_bound_K=2.,residual_sigma_K=1.,environment_sh_bound=.05).to(device)
    planck=UniformLWIRPlanckLUT(lower,upper).to(device)
    scale=max(float(np.quantile(np.concatenate(samples),.99)-np.quantile(np.concatenate(samples),.01)),1e-4)
    return field,planck,prior,scale


def matched_renderer(device):
    cache={}
    @torch.no_grad()
    def callback(geometry,truth,camera,sensor):
        if not cache:
            cache["model"]=setup_geometry(geometry,device)
            cache["values"]=tensor((truth["radiance"]-sensor["radiance_min"])/sensor["radiance_span"],device)
        normalized=render_values(cache["model"],camera_from_record(camera,device),cache["values"])[0].cpu().numpy()
        return normalized*sensor["radiance_span"]+sensor["radiance_min"]
    return callback
