import copy
import unittest

import torch

from utils.thermal_physics import MaterialThermalField
from utils.thermal_structured import StructuredThermalField, build_temperature_basis
from utils.thermal_spatial import TemperatureSpatialPrior


class StructuredFieldTests(unittest.TestCase):
    def make(self, branch="K"):
        base = MaterialThermalField(torch.tensor([0] * 4 + [1] * 4),
            torch.linspace(.2, .3, 8)[:, None], torch.tensor([.9, .3]),
            material_names=["a", "b"], branch=branch, initial_environment=.12)
        base.set_branch_trainability()
        points = torch.tensor([[0., 0, 0], [.01, 0, 0], [.02, 0, 0], [.03, 0, 0],
                               [1., 0, 0], [1.01, 0, 0], [1.02, 0, 0], [1.03, 0, 0]])
        normals = torch.tensor([[0., 0, 1.]] * 8)
        prior = TemperatureSpatialPrior.build(points, torch.ones(8), torch.zeros(8, 3),
            base.material_ids, torch.ones(8, dtype=torch.bool), neighbors=2)
        basis, protocol = build_temperature_basis(points, base.material_ids,
            torch.ones(8, dtype=torch.bool), prior, normals, max_controls=2)
        field = StructuredThermalField(base, basis, normals, torch.ones(8, 1), protocol)
        return base, field

    def test_zero_updates_preserve_D_temperature_and_signal_exactly(self):
        for branch in ("C", "R", "K"):
            base, field = self.make(branch)
            self.assertTrue(torch.equal(base.temperature, field.temperature))
            self.assertTrue(torch.equal(base.radiance(lambda t: t / 500), field.radiance(lambda t: t / 500)))

    def test_shared_controls_respect_disconnected_material_surfaces(self):
        _, field = self.make()
        self.assertEqual(len(field.temperature_control_raw), 2)
        with torch.no_grad():
            field.temperature_control_raw[0] = 1
        delta = field.temperature - field.initial_temperature
        self.assertTrue(bool((delta[:4] > 0).all()))
        self.assertEqual(float(delta[4:].abs().max()), 0.)
        self.assertTrue(torch.allclose(field.control_weights.sum(1), torch.ones(8)))

    def test_local_temperature_residual_is_bounded_and_penalized(self):
        _, field = self.make()
        with torch.no_grad():
            field.temperature_residual_raw[0] = 100
        self.assertLessEqual(float(field.temperature_local_delta.max()), .5)
        residual, control, environment = field.structure_regularizers()
        self.assertGreater(float(residual), 0)
        self.assertEqual(float(control), 0.)
        self.assertEqual(float(environment), 0.)

    def test_global_environment_is_bounded_even_for_extreme_SH_coefficients(self):
        _, field = self.make()
        for dc in (-15., 0., 15.):
            with torch.no_grad():
                field.environment_raw.fill_(dc)
                field.environment_sh_raw.fill_(10000)
            value = field.environment_by_gaussian
            self.assertTrue(bool(((value > 0) & (value < 1)).all()))
        self.assertEqual(field.environment_sh_raw.numel() + field.environment_raw.numel(), 9)

    def test_environment_effect_is_weighted_by_reflectivity(self):
        _, field = self.make("C")
        before = field.radiance(lambda t: t / 500).detach()
        with torch.no_grad():
            field.environment_sh_raw[2] = 1.
        signal_delta = field.radiance(lambda t: t / 500) - before
        env_delta = field.environment_by_gaussian - field.environment
        self.assertTrue(torch.allclose(signal_delta, (1 - field.emissivity()) * env_delta, atol=5e-8))

    def test_uncertain_normals_use_only_environment_DC(self):
        _, field = self.make()
        with torch.no_grad():
            field.environment_normal_confidence.zero_()
            field.environment_sh_raw.fill_(1)
        self.assertTrue(torch.equal(field.environment_by_gaussian, field.environment.expand(8, 1)))

    def test_new_parameters_receive_data_gradients(self):
        _, field = self.make()
        field.radiance(lambda t: t / 500).sum().backward()
        for value in (field.temperature_control_raw, field.temperature_residual_raw,
                      field.environment_raw, field.environment_sh_raw, field.k_raw):
            self.assertIsNotNone(value.grad)
            self.assertGreater(float(value.grad.abs().sum()), 0.)
            self.assertTrue(bool(torch.isfinite(value.grad).all()))
        self.assertIsNone(field.temperature_raw.grad)
        self.assertFalse(field.temperature_raw.requires_grad)

    def test_checkpoint_roundtrip_preserves_nonzero_parameters_and_predictions(self):
        _, field = self.make()
        with torch.no_grad():
            field.temperature_control_raw.fill_(.03)
            field.temperature_residual_raw.fill_(.05)
            field.environment_sh_raw.fill_(.1)
            field.k_raw.fill_(.02)
        checkpoint = copy.deepcopy(field.export_state())
        restored = MaterialThermalField.from_checkpoint(checkpoint, "K", "cpu", reset_branch_parameters=False)
        self.assertIsInstance(restored, StructuredThermalField)
        self.assertTrue(torch.equal(field.temperature, restored.temperature))
        self.assertTrue(torch.equal(field.radiance(lambda t: t / 500), restored.radiance(lambda t: t / 500)))

    def test_checkpoint_rejects_changed_frozen_structure(self):
        _, field = self.make()
        checkpoint = copy.deepcopy(field.export_state())
        checkpoint["state_dict"]["environment_normals"][0, 0] += .1
        with self.assertRaises(ValueError):
            MaterialThermalField.from_checkpoint(checkpoint, "K", "cpu", reset_branch_parameters=False)

    def test_legacy_checkpoint_still_uses_legacy_field(self):
        base, _ = self.make()
        field = MaterialThermalField.from_checkpoint(base.export_state(), "K", "cpu", reset_branch_parameters=False)
        self.assertIs(type(field), MaterialThermalField)
        self.assertTrue(torch.equal(base.temperature, field.temperature))

    def test_material_branch_trainability_is_preserved(self):
        for branch in ("C", "R", "K"):
            _, field = self.make(branch)
            self.assertEqual(field.k_raw.requires_grad, branch == "K")
            self.assertEqual(field.delta_epsilon_raw.requires_grad, branch == "R")

    def test_sparse_cache_drops_zero_padding_and_is_not_saved(self):
        _, field = self.make()
        matrix = field._temperature_interpolation
        self.assertTrue(matrix.is_coalesced())
        self.assertEqual(matrix._nnz(), int((field.control_weights != 0).sum()))
        self.assertNotIn("_temperature_interpolation", field.state_dict())

    def test_sparse_interpolation_matches_gather_value_and_gradient(self):
        _, field = self.make()
        with torch.no_grad():
            field.temperature_control_raw.copy_(torch.tensor([[.03], [-.07]]))
        control = field.temperature_control_delta
        expected = (control[field.control_indices, 0] * field.control_weights).sum(1, keepdim=True)
        actual = torch.sparse.mm(field._temperature_interpolation, control)
        grad_output = torch.arange(len(actual), dtype=actual.dtype)[:, None]
        grad_a = torch.autograd.grad((expected * grad_output).sum(), field.temperature_control_raw, retain_graph=True)[0]
        grad_b = torch.autograd.grad((actual * grad_output).sum(), field.temperature_control_raw)[0]
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(grad_a, grad_b, atol=1e-5, rtol=1e-5))

    def test_cache_is_rebuilt_after_loading_basis_buffers(self):
        _, field = self.make()
        state = copy.deepcopy(field.state_dict())
        state["control_indices"] = 1 - state["control_indices"]
        field.load_state_dict(state)
        expected = torch.sparse_coo_tensor(torch.stack((torch.arange(8), state["control_indices"][:, 0])),
                                           state["control_weights"][:, 0], (8, 2)).coalesce()
        self.assertTrue(torch.equal(field._temperature_interpolation.indices(), expected.indices()))


if __name__ == "__main__":
    unittest.main()
