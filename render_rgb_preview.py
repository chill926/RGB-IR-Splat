import os, torch, torchvision
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.general_utils import safe_state

parser = ArgumentParser()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ModelParams(parser, device, sentinel=True)
PipelineParams(parser)
parser.add_argument("--n", type=int, default=0)
args = get_combined_args(parser, device)
safe_state(False, device)

gaussians = GaussianModel(args.sh_degree, device)
scene = Scene(args, gaussians, load_iteration=-1, shuffle=False)

bg = torch.zeros((3,), device=device)
outdir = os.path.join(args.model_path, "rgb_preview")
os.makedirs(outdir, exist_ok=True)
cams = scene.getTestCameras()
if args.n > 0:
    cams = cams[: args.n]

for i, cam in enumerate(cams):
    with torch.no_grad():
        pkg = render(cam, gaussians, args, bg, 0.0, 0.0, 0.0, device,
                     getattr(args, "is_6dof", False), feature_set="rgb")
    torchvision.utils.save_image(pkg["render"].clamp(0, 1), os.path.join(outdir, "%02d_%s_render.png" % (i, cam.image_name)))
    torchvision.utils.save_image(cam.original_rgb_image.clamp(0, 1), os.path.join(outdir, "%02d_%s_gt.png" % (i, cam.image_name)))
    print("%02d %s" % (i, cam.image_name))
print("DONE ->", outdir)
