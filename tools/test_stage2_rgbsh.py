"""Checks the SH convention, geometry gradient, support and checkpoint safety."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from utils.sh_utils import eval_sh
from utils.thermal_rgb_sh import KIND, ThermalRGBSH, install_geometry
from utils.thermal_joint import GEOMETRY_NAMES, geometry_digest

class RGBSHTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(27)
        self.xyz = torch.randn(64, 3, requires_grad=True)
        support = torch.ones(64, dtype=torch.bool); support[-1] = False
        self.sh = ThermalRGBSH(self.xyz, support)
        self.center = torch.randn(3)

    def test_zero_start_and_support(self):
        self.assertEqual(self.sh.raw.shape, (64, 3, 8))
        self.assertEqual(self.sh.raw.numel(), 64 * 24)
        self.assertTrue(torch.equal(self.sh(self.xyz, self.center), torch.zeros(64, 3)))
        with torch.no_grad(): self.sh.raw.normal_()
        self.assertTrue(torch.equal(self.sh(self.xyz, self.center)[-1], torch.zeros(3)))

    def test_canonical_SH_and_bound(self):
        with torch.no_grad(): self.sh.raw.normal_(std=100)
        coeff = torch.cat((torch.zeros(64, 3, 1), self.sh.coefficients), -1)
        expected = eval_sh(2, coeff, torch.nn.functional.normalize(self.xyz - self.center, dim=-1))
        actual = self.sh(self.xyz, self.center)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
        self.assertLess(float(actual.abs().max()), self.sh.bound)
        self.assertFalse(torch.allclose(actual[:, 0], actual[:, 1]))
        self.assertFalse(torch.allclose(actual, self.sh(self.xyz, self.center + 1)))

    def test_direction_gradient_and_detachment(self):
        with torch.no_grad(): self.sh.raw.normal_()
        self.sh(self.xyz, self.center).square().mean().backward()
        self.assertGreater(float(self.xyz.grad.norm()), 0)
        self.assertGreater(float(self.sh.raw.grad.norm()), 0)
        self.xyz.grad = None; self.sh.raw.grad = None
        self.sh(self.xyz, self.center, detach_geometry=True).square().mean().backward()
        self.assertIsNone(self.xyz.grad)
        self.assertGreater(float(self.sh.raw.grad.norm()), 0)

    def test_SH_roundtrip_and_wrong_format(self):
        with torch.no_grad(): self.sh.raw.normal_()
        checkpoint = {'rgb_sh_residual': self.sh.export_state()}
        restored = ThermalRGBSH.from_checkpoint(checkpoint, self.xyz)
        self.assertFalse(restored.raw.requires_grad)
        self.assertTrue(torch.equal(self.sh(self.xyz, self.center), restored(self.xyz, self.center)))
        checkpoint['rgb_sh_residual']['channels'] = 1
        with self.assertRaises(ValueError): ThermalRGBSH.from_checkpoint(checkpoint, self.xyz)

    def test_geometry_restoration_and_tamper(self):
        model = SimpleNamespace(active_sh_degree=0, max_sh_degree=3)
        for name in GEOMETRY_NAMES: setattr(model, name, torch.nn.Parameter(torch.zeros(5, 3)))
        # The helper uses the GaussianModel position property.
        model.get_xyz = model._xyz
        state = {name: torch.randn(5, 3) for name in GEOMETRY_NAMES}
        state['active_sh_degree'] = 2
        checkpoint = {'kind': KIND, 'joint_geometry': state,
            'metadata': {'joint_protocol': {'kind': KIND, 'gaussian_count': 5,
                'trained_geometry_sha256': geometry_digest(state)}}}
        install_geometry(model, checkpoint)
        for name in GEOMETRY_NAMES:
            self.assertTrue(torch.equal(getattr(model, name), state[name]))
            self.assertFalse(getattr(model, name).requires_grad)
        self.assertEqual(model.active_sh_degree, 2)
        bad = copy.deepcopy(checkpoint); bad['joint_geometry']['_xyz'][0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'checksum'): install_geometry(model, bad)

if __name__ == '__main__': unittest.main()
