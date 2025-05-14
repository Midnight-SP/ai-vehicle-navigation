import unittest
from src.environment.simulation_environment import SimulationEnvironment

class TestSimulationEnvironment(unittest.TestCase):

    def setUp(self):
        self.env = SimulationEnvironment()

    def test_reset(self):
        initial_state = self.env.reset()
        self.assertIsNotNone(initial_state)
        self.assertEqual(self.env.current_step, 0)

    def test_step(self):
        self.env.reset()
        action = self.env.action_space.sample()  # Assuming action_space is defined
        next_state, reward, done, info = self.env.step(action)
        self.assertIsNotNone(next_state)
        self.assertIsInstance(reward, (int, float))
        self.assertIn(done, [True, False])

    def test_render(self):
        self.env.reset()
        output = self.env.render()
        self.assertIsNotNone(output)

if __name__ == '__main__':
    unittest.main()