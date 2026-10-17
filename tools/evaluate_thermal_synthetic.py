"""Evaluate held-out images and, explicitly, hidden synthetic physical truth."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
import torch
from utils.synthetic_thermal import load_public,file_hash
from utils.synthetic_thermal_torch import setup_geometry,camera_from_record,observed_normalized,render_values,palette_tensor
from utils.thermal_physics import MaterialThermalField,UniformLWIRPlanckLUT
from utils.thermal_joint import masked_ssim
from utils.thermal_training_utils import masked_mean


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True,type=Path);p.add_argument("--output",type=Path)
    p.add_argument("--split",choices=["all","fit","validation","test"],default="all")
    p.add_argument("--truth",action="store_true");p.add_argument("--save_images",action="store_true")
    a=p.parse_args();checkpoint=torch.load(a.checkpoint,map_location="cpu")
    metadata=checkpoint["metadata"]
    if metadata.get("kind")!="synthetic_thermal_checkpoint_v1":raise ValueError("Not a synthetic checkpoint")
    protocol=metadata["protocol"];dataset=Path(protocol["dataset"])
    if file_hash(dataset/"dataset.json")!=protocol["dataset_sha256"]:raise ValueError("Dataset changed")
    output=a.output or a.checkpoint.parent/("evaluation_"+a.checkpoint.stem)
    if (output/"metrics.json").exists():
        previous=json.loads((output/"metrics.json").read_text())
        if previous["checkpoint_sha256"]==file_hash(a.checkpoint) and previous["truth_evaluated"]==a.truth and previous["requested_split"]==a.split:
            print(json.dumps(previous,indent=2));return
        raise ValueError("Evaluation output exists for a different request")
    splits=("fit","validation","test") if a.split=="all" else (a.split,)
    document,geometry,frames=load_public(dataset,splits)
    device=torch.device("cuda:0");model=setup_geometry(geometry,device)
    field=MaterialThermalField.from_checkpoint(checkpoint,checkpoint["branch"],device,reset_branch_parameters=False).eval()
    planck=UniformLWIRPlanckLUT(*document["sensor"]["temperature_bounds_K"]).to(device)
    values=field.radiance(planck)
    truth=None;truth_values=None
    if a.truth:
        truth=dict(np.load(dataset/"truth/gaussians.npz",allow_pickle=False))
        true_T=torch.as_tensor(truth["temperature_K"],dtype=torch.float32,device=device)[:,None]
        true_epsilon=torch.as_tensor(truth["epsilon"],dtype=torch.float32,device=device)[:,None]
        true_E=torch.as_tensor((truth["environment"]-document["sensor"]["radiance_min"])/document["sensor"]["radiance_span"],dtype=torch.float32,device=device)[:,None]
        truth_values=true_epsilon*planck(true_T)+(1-true_epsilon)*true_E
    output.mkdir(parents=True,exist_ok=True)
    report={"checkpoint":str(a.checkpoint.resolve()),"checkpoint_sha256":file_hash(a.checkpoint),"step":checkpoint["step"],
        "branch":checkpoint["branch"],"oracle":metadata["oracle"],"renderer":document["renderer"],"truth_evaluated":a.truth,"requested_split":a.split,
        "metric_region":"foreground_eroded_per_object; full_signal_PSNR_also_reported",
        "signal_data_range":"fixed_250_450K_calibrated_normalized_radiance_span",
        "pseudocolor_mapping":"fixed_triangular_palette","splits":{}}
    for split,rows in frames.items():
        results=[]
        for record,observation in rows:
            target,mask=observed_normalized(observation,document["sensor"],device)
            prediction=render_values(model,camera_from_record(record,device),values)
            colored,gt=palette_tensor(prediction),palette_tensor(target)
            mse=float(masked_mean((prediction-target).square(),mask));color_mse=float(masked_mean((colored-gt).square(),mask))
            results.append({"view":record["name"],"signal_MSE":mse,"signal_PSNR":-10*math.log10(max(mse,1e-12)),
                "full_signal_PSNR":-10*math.log10(max(float((prediction-target).square().mean()),1e-12)),
                "DN_RMSE":math.sqrt(mse)*document["sensor"]["radiance_span"]*document["sensor"]["gain"],
                "IR_PSNR":-10*math.log10(max(color_mse,1e-12)),"IR_SSIM":float(masked_ssim(colored,gt,mask))})
            if truth_values is not None:
                reference=render_values(model,camera_from_record(record,device),truth_values)
                reference_mse=float(masked_mean((reference-target).square(),mask))
                results[-1]["truth_parameter_render_signal_PSNR"]=-10*math.log10(max(reference_mse,1e-12))
            if a.save_images:
                foreground=torch.as_tensor(observation["foreground"],device=device)
                parts=[gt,colored,(gt-colored).abs()*3]
                arrays=[(v*foreground).clamp(0,1).permute(1,2,0).cpu().numpy() for v in parts]
                image=np.concatenate(arrays,axis=1)
                (output/split).mkdir(exist_ok=True)
                Image.fromarray(np.rint(image*255).astype(np.uint8)).save(output/split/(record["name"]+".png"))
        report["splits"][split]={key:float(np.mean([r[key] for r in results])) for key in results[0] if key!="view"}
        report["splits"][split]["views"]=len(results)
        (output/(split+"_per_view.json")).write_text(json.dumps(results,indent=2)+"\n")
    estimates={"temperature_K":field.temperature[:,0].cpu().numpy(),"epsilon":field.emissivity()[:,0].cpu().numpy(),
        "environment_normalized":field.environment_by_gaussian[:,0].cpu().numpy(),"k_by_material":field.k_epsilon_by_material.cpu().numpy(),
        "epsilon0_by_material":field.epsilon0_by_material.cpu().numpy()}
    np.savez_compressed(output/"estimated_physical_fields.npz",**estimates)
    if a.truth:
        def errors(prediction,target,visible):
            weights=geometry["area"]*visible
            if weights.sum()==0:return None
            difference=prediction-target
            return {"MAE":float((np.abs(difference)*weights).sum()/weights.sum()),
                    "RMSE":float(np.sqrt((difference**2*weights).sum()/weights.sum())),
                    "Bias":float((difference*weights).sum()/weights.sum()),"gaussians":int(visible.sum())}
        physical={}
        E=estimates["environment_normalized"]*document["sensor"]["radiance_span"]+document["sensor"]["radiance_min"]
        for region,visible in [("all_gaussians",np.ones(len(geometry["xyz"]),dtype=bool)),("fit_visible",truth["fit_visible"]),("test_visible",truth["test_visible"])]:
            physical[region]={"temperature_K":errors(estimates["temperature_K"],truth["temperature_K"],visible),
                "epsilon":errors(estimates["epsilon"],truth["epsilon"],visible),"environment_radiance":errors(E,truth["environment"],visible)}
        physical["materials"]=[{"name":material["name"],"k_estimate":float(estimates["k_by_material"][i]),"k_truth":float(truth["k_by_material"][i]),
            "k_abs_error":float(abs(estimates["k_by_material"][i]-truth["k_by_material"][i])),
            "epsilon0_estimate":float(estimates["epsilon0_by_material"][i]),"epsilon0_truth":float(truth["epsilon0_by_material"][i])} for i,material in enumerate(document["materials"])]
        report["physical_truth_errors"]=physical
        report["truth_interpretation"]="Main training never uses these truths; Oracle is separately labeled. Unseen surface errors are reported separately."
    (output/"metrics.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=="__main__":main()
