import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

from train_thermal_stage3 import effective_training_protocol
from tools.tune_stage3 import make_command, rank_trial


class HyperparameterTest(unittest.TestCase):
    def setUp(self):
        self.recorded = {"temperature_lr": .001, "temperature_lr_final": .00001,
                         "environment_lr": .0001, "environment_lr_final": .000001,
                         "lr_hold_fraction": .5, "radiometric_protocol": {"manifest_sha256": "unchanged"},
                         "temp_min": 250, "temp_max": 450}

    def test_override_is_recorded_without_mutating_common_or_calibration(self):
        before = copy.deepcopy(self.recorded)
        args = SimpleNamespace(temperature_lr=1e-4, temperature_lr_final=1e-6,
                               environment_lr=1e-5, environment_lr_final=1e-7, lr_hold_fraction=.1)
        result = effective_training_protocol(self.recorded, args)
        for name, value in vars(args).items():
            self.assertEqual(result[name], value)
        self.assertEqual(self.recorded, before)
        self.assertEqual(result["radiometric_protocol"], before["radiometric_protocol"])
        self.assertEqual(result["temp_min"], 250)

    def test_no_override_preserves_inherited_settings(self):
        self.assertEqual(effective_training_protocol(self.recorded, SimpleNamespace()), self.recorded)

    def test_invalid_LR_and_hold_are_rejected(self):
        for overrides in ({"temperature_lr": float("nan")}, {"temperature_lr_final": .01},
                          {"environment_lr": 0}, {"environment_lr_final": float("inf")},
                          {"lr_hold_fraction": 1}, {"lr_hold_fraction": -.1}):
            with self.assertRaises(ValueError):
                effective_training_protocol(self.recorded, SimpleNamespace(**overrides))

    def test_search_prioritizes_best_validation_and_uses_final_only_for_ties(self):
        def row(identifier, best, final):
            return {"id": identifier, "summary": {"best_validation_objective": best,
                     "final_validation_metrics": {"selection_loss": final}}}
        rows = [row("a", .06, .08), row("b", .06, .07), row("c", .061, .065)]
        self.assertEqual(min(rows, key=rank_trial)["id"], "b")

    def test_trial_command_passes_actual_settings_explicitly(self):
        args = SimpleNamespace(common=Path('/tmp/common.pt'), output=Path('/tmp/search'),
                               steps=2000, seed=0, eval_every=100, save_every=1000,
                               material_lr=1e-4, material_lr_final=1e-6, lr_hold_fraction=.1)
        command = make_command(args, 'K_reg_01', 'K', 1e-4, 1e-5, 1e-4, Path('/tmp/lock.json'))
        def value(name):
            return command[command.index(name) + 1]
        self.assertAlmostEqual(float(value('--temperature_lr_final')), 1e-6)
        self.assertAlmostEqual(float(value('--environment_lr_final')), 1e-7)
        self.assertEqual(float(value('--lambda_k')), 1e-4)
        self.assertEqual(float(value('--lambda_delta_epsilon')), 1e-4)
        self.assertEqual(float(value('--lr_hold_fraction')), .1)
        self.assertEqual(value('--branch'), 'K')


if __name__ == '__main__':
    unittest.main()
