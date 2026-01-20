import unittest
from unittest import mock

import numpy as np

from dingo.MetabolicNetwork import MetabolicNetwork
import dingo.dfba as dfba_module


class TestDynamicFBA(unittest.TestCase):

    def setUp(self):
        lb = np.array([-10.0, 0.0], dtype=float)
        ub = np.array([0.0, 1000.0], dtype=float)
        S = np.array([
            [-1.0, 0.0],
            [1.0, -1.0],
        ])
        metabolites = ["A_ext", "A_int"]
        reactions = ["EX_A", "BIOMASS"]
        biomass_index = 1
        objective = np.array([0.0, 1.0], dtype=float)
        medium = {"EX_A": 10.0}
        medium_indices = {"EX_A": 0}
        exchanges = ["EX_A"]

        tuple_args = (
            lb,
            ub,
            S,
            metabolites,
            reactions,
            biomass_index,
            objective,
            medium,
            medium_indices,
            exchanges,
        )
        self.model = MetabolicNetwork(tuple_args)
        self.original_lb = np.copy(lb)

    def test_sampling_guided_dynamics(self):
        class DummySampler:
            def __init__(self, model):
                self.calls = 0

            def generate_steady_states_no_multiphase(
                self,
                method,
                n,
                burn_in,
                thinning,
                variance,
                bias_vector,
                ess,
            ):
                self.calls += 1
                base = np.array([[-5.0], [5.0]], dtype=float)
                samples = np.tile(base, (1, n))
                if self.calls > 1:
                    samples[0] -= 0.2
                    samples[1] += 0.2
                return samples

        simulator = dfba_module.DynamicFBA(
            self.model,
            time_step=0.1,
            total_time=0.3,
            initial_biomass=0.1,
            initial_concentrations={"EX_A": 5.0},
            sample_size=4,
        )

        with mock.patch.object(dfba_module, "PolytopeSampler", DummySampler):
            result = simulator.run()

        self.assertEqual(result.fluxes.shape[0], 3)
        self.assertEqual(result.fluxes.shape[1], len(self.model.reactions))
        self.assertGreater(result.biomass[-1], result.biomass[0])
        self.assertLess(result.concentrations["EX_A"][-1], result.concentrations["EX_A"][0])
        np.testing.assert_allclose(self.model.lb, self.original_lb)


if __name__ == "__main__":
    unittest.main()
