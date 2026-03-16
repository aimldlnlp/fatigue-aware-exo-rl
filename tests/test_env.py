import unittest

import numpy as np

from exo_rl.envs.arm2d_env import Arm2DEnv
from exo_rl.utils.io import load_json


class ArmEnvTest(unittest.TestCase):
    def test_step_shapes(self) -> None:
        config = load_json("configs/smoke_test.json")
        env = Arm2DEnv(config["env"], seed=1)
        obs, _ = env.reset()
        self.assertEqual(obs.shape[0], env.obs_size)
        result = env.step(np.zeros(env.action_size, dtype=np.float32))
        self.assertEqual(result.obs.shape[0], env.obs_size)
        self.assertIsInstance(result.reward, float)


if __name__ == "__main__":
    unittest.main()

