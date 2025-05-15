import unittest
import numpy as np
from environment.simulation_environment import SimulationEnvironment

class TestSimulationEnvironment(unittest.TestCase):
    def test_start_position_and_angle(self):
        route_data = {
            "route": {
                "start_position": { "latitude": 0, "longitude": -1 },
                "start_angle": 90,
                "checkpoints": [],
                "walls": []
            }
        }
        genes = {
            "speed": 2.0,
            "acceleration": 0.1,
            "braking_force": 0.2,
            "turn_angle": 30
        }
        env = SimulationEnvironment(route_data, genes)
        self.assertTrue(np.array_equal(env.vehicle_position, np.array([0, -1], dtype=float)))
        self.assertEqual(env.vehicle_angle, 90)

        env.reset()
        self.assertTrue(np.array_equal(env.vehicle_position, np.array([0, -1], dtype=float)))
        self.assertEqual(env.vehicle_angle, 90)

if __name__ == "__main__":
    unittest.main()