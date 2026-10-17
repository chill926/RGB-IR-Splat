"""Three-channel non-DC SH appearance residual and joint-geometry checkpoints."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from utils.thermal_sh_residual import sh2_basis
from utils.thermal_joint import GEOMETRY_NAMES, geometry_digest

KIND = 'stage2_joint_rgb_sh24_v1'

class ThermalRGBSH(nn.Module):
    def __init__(self, xyz, support, bound=0.5, trainable=True):
        super().__init__()
        support = torch.as_tensor(support, dtype=torch.bool, device=xyz.device).reshape(-1)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) != len(support) or not len(xyz):
            raise ValueError('RGB SH support must match nonempty Gaussian positions')
        if not math.isfinite(bound) or bound <= 0:
            raise ValueError('RGB SH bound must be positive and finite')
        if not bool(support.any()):
            raise ValueError('No fit-view support for RGB SH')
        self.bound = float(bound)
        self.register_buffer('fit_support', support.clone())
        self.raw = nn.Parameter(xyz.new_zeros(len(xyz), 3, 8), requires_grad=trainable)

    @property
    def coefficients(self):
        # Each channel remains a linear SH expansion without DC. Smoothly
        # constrain coefficient norm; do not tanh the directional function.
        energy = 8.0 / (4.0 * math.pi)
        denominator = torch.sqrt(energy * (1 + self.raw.square().sum(-1, keepdim=True)))
        return self.bound * self.raw / denominator * self.fit_support[:, None, None].to(self.raw)

    def forward(self, xyz, center, detach_geometry=False):
        if xyz.shape != (len(self.raw), 3):
            raise ValueError('RGB SH and geometry counts differ')
        positions = xyz.detach() if detach_geometry else xyz
        basis = sh2_basis(positions - center.detach().reshape(1, 3))
        return (self.coefficients * basis[:, None, :]).sum(-1)

    def regularizer(self):
        squared = self.raw[self.fit_support].square().sum(-1)
        return (squared / (1 + squared)).mean()

    def export_state(self):
        return {'kind': 'rgb_sh_l1_l2_no_dc', 'format_version': 1, 'channels': 3,
                'coefficients_per_gaussian': 24, 'bound': self.bound,
                'state_dict': {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}}

    @classmethod
    def from_checkpoint(cls, checkpoint, xyz, trainable=False):
        saved = checkpoint.get('rgb_sh_residual')
        if not saved or saved.get('kind') != 'rgb_sh_l1_l2_no_dc' or saved.get('format_version') != 1:
            raise ValueError('Missing/unsupported 24-coefficient RGB SH checkpoint')
        if saved.get('channels') != 3 or saved.get('coefficients_per_gaussian') != 24:
            raise ValueError('RGB SH channel/coefficient mismatch')
        state = saved['state_dict']
        model = cls(xyz, state['fit_support'], saved['bound'], trainable).to(xyz.device)
        model.load_state_dict({k: v.to(xyz.device) for k, v in state.items()}, strict=True)
        if not bool(torch.isfinite(model.raw).all()):
            raise ValueError('Non-finite RGB SH parameters')
        return model

def appearance(field, planck, gaussians, camera, sh, detach_geometry=False):
    core = field.radiance(planck).expand(-1, 3)
    residual = sh(gaussians.get_xyz, camera.camera_center, detach_geometry)
    combined = core + residual
    clipped = ((combined < 0) | (combined > 1)).float().mean()
    return combined.clamp(0, 1), residual, clipped

def install_geometry(gaussians, checkpoint):
    state = checkpoint.get('joint_geometry')
    protocol = checkpoint.get('metadata', {}).get('joint_protocol', {})
    if checkpoint.get('kind') != KIND or not state or protocol.get('kind') != KIND:
        raise ValueError('Not a Stage2 joint RGB SH24 checkpoint')
    if geometry_digest(state) != protocol.get('trained_geometry_sha256'):
        raise ValueError('Joint geometry checksum mismatch')
    if len(gaussians.get_xyz) != protocol.get('gaussian_count'):
        raise ValueError('Joint geometry/Stage1 Gaussian count mismatch')
    for name in GEOMETRY_NAMES:
        value = state[name]
        original = getattr(gaussians, name)
        if value.shape != original.shape or value.dtype != original.dtype or not bool(torch.isfinite(value).all()):
            raise ValueError('Invalid joint geometry: ' + name)
    degree = int(state['active_sh_degree'])
    if not 0 <= degree <= gaussians.max_sh_degree:
        raise ValueError('Invalid RGB SH degree in geometry checkpoint')
    for name in GEOMETRY_NAMES:
        setattr(gaussians, name, nn.Parameter(state[name].to(gaussians.get_xyz.device).clone(), requires_grad=False))
    gaussians.active_sh_degree = degree
