import unittest

import torch

from utils.thermal_physics import MaterialThermalField, UniformLWIRPlanckLUT


class MaterialThermalFieldTest(unittest.TestCase):
    def make_field(self, branch="stage2"):
        return MaterialThermalField(
            torch.tensor([0, 1, -1]),
            torch.tensor([[0.25], [0.50], [0.75]]),
            epsilon0_by_material=[0.90, 0.95],
            material_names=["paint", "rubber"],
            material_confidence=[0.9, 0.8, 0.0],
            unknown_epsilon0=0.92,
            initial_environment=0.2,
            branch=branch,
        )

    def test_environment_is_global_scalar(self):
        self.assertEqual(tuple(self.make_field().environment.shape), (1,))

    def test_unknown_uses_fixed_fallback(self):
        field = self.make_field("K")
        field.k_raw.data.fill_(1.0)
        self.assertAlmostEqual(float(field.emissivity()[-1]), 0.92, places=6)

    def test_zero_k_matches_constant_branch(self):
        constant = self.make_field("C")
        temperature_model = self.make_field("K")
        lut = UniformLWIRPlanckLUT(num_temp=128, num_lambda=32)
        self.assertTrue(torch.allclose(constant.radiance(lut), temperature_model.radiance(lut)))

    def test_checkpoint_render_load_preserves_material_parameter(self):
        field = self.make_field("K")
        field.k_raw.data.fill_(0.25)
        checkpoint = field.export_state()
        loaded = MaterialThermalField.from_checkpoint(
            checkpoint, "K", torch.device("cpu"), reset_branch_parameters=False)
        self.assertTrue(torch.equal(field.k_raw, loaded.k_raw))


if __name__ == "__main__":
    unittest.main()
