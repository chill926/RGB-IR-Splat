"""Generate calibrated multi-view thermal images from known physical quantities."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils.synthetic_thermal import generate_dataset


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--renderer",choices=["analytic","gaussian_matched"],default="analytic")
    p.add_argument("--width",type=int,default=320);p.add_argument("--height",type=int,default=240)
    p.add_argument("--views",type=int,default=40);p.add_argument("--grid",type=int,default=24)
    p.add_argument("--seed",type=int,default=2027);p.add_argument("--k_scale",type=float,default=1.)
    p.add_argument("--temperature_span",type=float,default=60.);p.add_argument("--epsilon_bias",type=float,default=0.)
    p.add_argument("--noise_std_dn",type=float,default=0.);p.add_argument("--quantize",action="store_true")
    a=p.parse_args();callback=None
    if a.renderer=="gaussian_matched":
        from utils.synthetic_thermal_torch import matched_renderer
        callback=matched_renderer("cuda:0")
    result=generate_dataset(a.output,width=a.width,height=a.height,views=a.views,grid=a.grid,seed=a.seed,
        k_scale=a.k_scale,temperature_span=a.temperature_span,epsilon_bias=a.epsilon_bias,
        noise_std_dn=a.noise_std_dn,quantize=a.quantize,renderer=a.renderer,matched_renderer=callback)
    print(json.dumps({"dataset":str(a.output.resolve()),"renderer":a.renderer,
        "splits":{s:sum(f["split"]==s for f in result["frames"]) for s in ["fit","validation","test"]}},indent=2))


if __name__=="__main__":main()
