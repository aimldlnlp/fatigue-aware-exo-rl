import unittest

from exo_rl.envs.fatigue import update_fatigue


class FatigueTest(unittest.TestCase):
    def test_fatigue_increases_with_effort(self) -> None:
        low = update_fatigue(0.2, effort=0.1, gain=0.03, recovery=0.01)
        high = update_fatigue(0.2, effort=0.9, gain=0.03, recovery=0.01)
        self.assertGreater(high, low)


if __name__ == "__main__":
    unittest.main()

