import unittest

from exo_rl.envs.rewards import compute_reward
from exo_rl.utils.io import load_json


class RewardTest(unittest.TestCase):
    def test_success_bonus(self) -> None:
        config = load_json("configs/smoke_test.json")["env"]
        success_reward, success = compute_reward(
            progress=0.08,
            distance=0.01,
            success_threshold=0.05,
            fatigue=0.3,
            human_effort=0.4,
            assist_ratio=0.2,
            action_delta=0.1,
            success_streak=4,
            terminated=True,
            config=config,
        )
        fail_reward, fail = compute_reward(
            progress=-0.01,
            distance=0.2,
            success_threshold=0.05,
            fatigue=0.3,
            human_effort=0.4,
            assist_ratio=0.2,
            action_delta=0.1,
            success_streak=0,
            terminated=False,
            config=config,
        )
        self.assertTrue(success)
        self.assertFalse(fail)
        self.assertGreater(success_reward, fail_reward)


if __name__ == "__main__":
    unittest.main()
