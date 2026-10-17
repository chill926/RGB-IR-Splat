"""Fit C/R/K on public synthetic views; surface truth requires explicit Oracle."""
import argparse
import copy
import json
import math
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from utils.synthetic_thermal import load_public,file_hash
from utils.synthetic_thermal_torch import (build_field,setup_geometry,camera_from_record,
    observed_normalized,render_values,palette_tensor)
from utils.thermal_training_utils import masked_mean
from utils.thermal_stage3 import atomic_save,lock_protocol


def install_oracle(field,dataset,document):
    """Explicit known T/E/epsilon0 implementation control, only k remains free."""
    truth=dict(np.load(dataset/"truth/gaussians.npz",allow_pickle=False))
    scene=json.loads((dataset/"truth/scene.json").read_text())
    device=field.initial_temperature.device;sensor=document["sensor"]
    E0=(scene["environment_mean"]-sensor["radiance_min"])/sensor["radiance_span"]
    coefficients=np.asarray(scene["illumination_SH_1_2"])*np.array([2/3]*3+[.25]*5)/sensor["radiance_span"]
    energy=8/(4*math.pi);allowed=min(field.environment_sh_bound,.95*min(E0,1-E0))
    remainder=allowed**2-energy*float(np.square(coefficients).sum())
    if not 0<E0<1 or remainder<=0:raise ValueError("Oracle environment cannot be represented within model bounds")
    with torch.no_grad():
        field.initial_temperature.copy_(torch.as_tensor(truth["temperature_K"],dtype=torch.float32,device=device)[:,None])
        field.epsilon0_by_material.copy_(torch.as_tensor(truth["epsilon0_by_material"],dtype=torch.float32,device=device))
        field.environment_raw.fill_(math.log(E0/(1-E0)))
        field.environment_sh_raw.copy_(torch.as_tensor(coefficients*math.sqrt(energy/remainder),dtype=torch.float32,device=device))
    for parameter in (field.temperature_control_raw,field.temperature_residual_raw,field.environment_raw,field.environment_sh_raw):parameter.requires_grad_(False)
    field.structure_protocol["initial_temperature_source"]="ORACLE_known_surface_temperature"
    field.frozen_structure_sha256=field.structure_digest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",required=True,type=Path);p.add_argument("--output",required=True,type=Path)
    p.add_argument("--branch",choices=["C","R","K"],required=True)
    p.add_argument("--steps",type=int,default=2000);p.add_argument("--eval_every",type=int,default=100)
    p.add_argument("--seed",type=int,default=0);p.add_argument("--controls",type=int,default=256)
    p.add_argument("--temperature_lr",type=float,default=.003);p.add_argument("--environment_lr",type=float,default=.0005)
    p.add_argument("--environment_sh_lr",type=float,default=.002);p.add_argument("--material_lr",type=float,default=.003)
    p.add_argument("--material_warmup",type=int,default=200)
    p.add_argument("--lambda_material",type=float,default=1e-5);p.add_argument("--lambda_spatial",type=float,default=1e-3)
    p.add_argument("--lambda_residual",type=float,default=1e-3);p.add_argument("--lambda_control",type=float,default=1e-6)
    p.add_argument("--lambda_environment",type=float,default=1e-3);p.add_argument("--lambda_environment_sh",type=float,default=1e-4)
    p.add_argument("--lambda_display",type=float,default=0.);p.add_argument("--epsilon_bias",type=float,default=0.)
    p.add_argument("--oracle",action="store_true");p.add_argument("--comparison_protocol",type=Path)
    a=p.parse_args();a.dataset=a.dataset.resolve();a.output=a.output.resolve()
    if a.steps<1 or a.eval_every<1 or a.controls<1 or a.material_warmup<0:p.error("Invalid budget")
    for name,value in vars(a).items():
        if isinstance(value,float) and (not math.isfinite(value) or ((name.startswith("lambda_") or name.endswith("_lr")) and value<0)):p.error("Invalid numeric option: "+name)
    if a.oracle and a.branch!="K":p.error("Oracle is an explicitly labeled known-T/E epsilon0, k-only K diagnostic")
    if a.output.exists():raise ValueError("Output exists; choose a new name")
    if a.output==a.dataset or a.dataset in a.output.parents or a.output in a.dataset.parents:raise ValueError("Output overlaps dataset")
    if not torch.cuda.is_available():raise RuntimeError("Synthetic GS fitting requires the CUDA rasterizer")
    device=torch.device("cuda:0");torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    document,geometry,frames=load_public(a.dataset,("fit","validation"))
    field,planck,prior,scale=build_field(document,geometry,frames,a.branch,device,a.controls,a.epsilon_bias)
    if a.oracle:install_oracle(field,a.dataset,document)
    model=setup_geometry(geometry,device)
    frozen={k:getattr(model,k).detach().clone() for k in ("_xyz","_scaling","_rotation","_opacity","_features_dc","_features_rest")}
    data={s:[(camera_from_record(c,device),*observed_normalized(o,document["sensor"],device)) for c,o in rows] for s,rows in frames.items()}
    anchor=field.environment.detach().clone()
    setting={k:v for k,v in vars(a).items() if k not in ("dataset","output","branch","comparison_protocol")}
    protocol={"kind":"synthetic_C_R_K_known_geometry_v1","dataset":str(a.dataset),"dataset_sha256":file_hash(a.dataset/"dataset.json"),
        "geometry_sha256":document["geometry_sha256"],"settings":setting,"loss_scale_fit_only":scale,
        "truth_used_for_optimization":a.oracle,"test_used_for_training_or_selection":False,
        "fit_names":[c[0].image_name for c in data["fit"]],"validation_names":[c[0].image_name for c in data["validation"]],
        "initialization":"constant_public_Tref_and_fit_observation_quantile_E" if not a.oracle else "ORACLE_known_T_E_epsilon0",
        "appearance_SH":"disabled_in_synthetic_physics_control; real_RoadBlock_pipeline_unchanged",
        "structure_sha256":field.frozen_structure_sha256,
        "source_sha256":{str(path.relative_to(ROOT)):file_hash(path) for path in [Path(__file__),ROOT/"utils/synthetic_thermal.py",ROOT/"utils/synthetic_thermal_torch.py",ROOT/"utils/thermal_physics.py",ROOT/"utils/thermal_structured.py"]}}
    lock=a.comparison_protocol or a.output.parent/("oracle_comparison.json" if a.oracle else "comparison.json")
    lock_protocol(lock,protocol)
    a.output.mkdir(parents=True);(a.output/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    groups=[]
    for name,param,lr in [("temperature_control",field.temperature_control_raw,a.temperature_lr),("temperature_residual",field.temperature_residual_raw,a.temperature_lr),
        ("environment",field.environment_raw,a.environment_lr),("environment_sh",field.environment_sh_raw,a.environment_sh_lr),
        ("material",field.k_raw if a.branch=="K" else field.delta_epsilon_raw,a.material_lr)]:
        if param.requires_grad:groups.append({"name":name,"params":[param],"lr":lr,"initial_lr":lr})
    optimizer=torch.optim.Adam(groups)
    def loss(prediction,target,mask):
        signal=masked_mean(torch.nn.functional.smooth_l1_loss((prediction-target)/scale,torch.zeros_like(prediction),beta=.02,reduction="none"),mask)
        display=masked_mean((palette_tensor(prediction)-palette_tensor(target)).abs(),mask)
        return signal+a.lambda_display*display
    @torch.no_grad()
    def validate():
        rows=[]
        for camera,target,mask in data["validation"]:
            prediction=render_values(model,camera,field.radiance(planck))
            mse=float(masked_mean((prediction-target).square(),mask))
            rows.append({"objective":float(loss(prediction,target,mask)),"signal_MSE":mse,"signal_PSNR":-10*math.log10(max(mse,1e-12))})
        return {key:sum(r[key] for r in rows)/len(rows) for key in rows[0]}
    def save(name,step,metrics):
        if field.structure_digest()!=field.frozen_structure_sha256:raise RuntimeError("Frozen structure changed")
        for key,value in frozen.items():
            if not torch.equal(value,getattr(model,key)):raise RuntimeError("Frozen geometry changed")
        payload=field.export_state();payload["state_dict"]={k:v.detach().cpu().clone() for k,v in payload["state_dict"].items()}
        payload.update(step=step,metadata={"kind":"synthetic_thermal_checkpoint_v1","protocol":protocol,"validation":metrics,"oracle":a.oracle})
        atomic_save(payload,a.output/name)
    def log(row):
        with (a.output/"training.jsonl").open("a") as f:f.write(json.dumps(row,allow_nan=False)+"\n")
        print(json.dumps(row,allow_nan=False),flush=True)
    metrics=validate();best=metrics["objective"];best_step=0;save("best.pt",0,metrics);log({"step":0,"validation":metrics})
    order=list(range(len(data["fit"])));rng=random.Random(a.seed)
    for step in range(1,a.steps+1):
        for group in groups:
            ratio=(step-1)/max(a.steps-1,1);group["lr"]=group["initial_lr"]*.01**ratio
            if group["name"]=="material" and not a.oracle and step<=a.material_warmup:group["lr"]=0.
        if (step-1)%len(order)==0:rng.shuffle(order)
        camera,target,mask=data["fit"][order[(step-1)%len(order)]]
        optimizer.zero_grad(set_to_none=True)
        data_loss=loss(render_values(model,camera,field.radiance(planck)),target,mask)
        residual,control,env_sh=field.structure_regularizers()
        spatial=prior(field.temperature)
        environment=torch.nn.functional.smooth_l1_loss(field.environment,anchor,beta=.02)
        regularization=(a.lambda_residual*residual+a.lambda_control*control+a.lambda_environment_sh*env_sh+
            a.lambda_spatial*spatial+a.lambda_environment*environment)
        if not a.oracle:regularization=regularization+a.lambda_material*field.branch_regularizer(delta=0.)
        total=data_loss+regularization
        if not bool(torch.isfinite(total)):raise FloatingPointError("Nonfinite synthetic loss")
        total.backward()
        for group in groups:
            for parameter in group["params"]:
                if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):raise FloatingPointError("Missing/nonfinite gradient: "+group["name"])
        optimizer.step()
        if step==1 or step%a.eval_every==0 or step==a.steps:
            metrics=validate();log({"step":step,"fit_data_loss":float(data_loss.detach()),"regularization":float(regularization.detach()),"validation":metrics,
                "environment":float(field.environment),"k":field.k_epsilon_by_material.detach().cpu().tolist(),"delta_epsilon":field.delta_epsilon_by_material.detach().cpu().tolist()})
            if metrics["objective"]<best:best=metrics["objective"];best_step=step;save("best.pt",step,metrics)
    save("final.pt",a.steps,metrics)
    summary={"completed":True,"steps":a.steps,"best_step":best_step,"best_validation_objective":best,"final_validation":metrics,
        "protocol":protocol,"frozen_geometry_verified":True,"hidden_truth_read":a.oracle}
    (a.output/"training_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({"completed":True,"output":str(a.output),"best_step":best_step}),flush=True)


if __name__=="__main__":main()
