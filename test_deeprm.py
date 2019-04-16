import unittest
import torch
from pg_travel.deeprm.env_simu_sigle.environment import Env
from pg_travel.deeprm.hparams import HyperParams as hp
from pg_travel.deeprm.utils import utils


class TestUtils(unittest.TestCase):
    def test_get_action(self):
        action = utils.get_action(torch.Tensor([1, 2, 3, 4]))
        print(action.dtype, torch.LongTensor([0, 1, 2, 3]).dtype)
        action in torch.LongTensor([0, 1, 2, 3])
        # self.assertTrue(action in torch.Tensor([0, 1, 2, 3]))


class TestEnv(unittest.TestCase):
    def test_backlog(self):
        pa = hp()
        pa.num_nw = 5
        pa.simu_len = 50
        pa.num_ex = 10
        pa.new_job_rate = 1
        pa.compute_dependent_parameters()

        env = Env(pa)

        env.step(5)
        env.step(5)
        env.step(5)
        env.step(5)
        env.step(5)
        env.step(5)

        self.assertTrue(env.job_backlog.backlog[0] is not None)
        self.assertTrue(env.job_backlog.backlog[1] is None)

        print("New job is backlogged")

        env.step(5)
        env.step(5)
        env.step(5)
        env.step(5)

        job = env.job_backlog.backlog[0]
        env.step(0)
        self.assertEqual(env.job_slot.slot[0], job)

        job = env.job_backlog.backlog[0]
        env.step(0)
        self.assertEqual(env.job_slot.slot[0], job)

        job = env.job_backlog.backlog[0]
        env.step(1)
        self.assertEqual(env.job_slot.slot[1], job)

        job = env.job_backlog.backlog[0]
        env.step(1)
        self.assertEqual(env.job_slot.slot[1], job)

        env.step(5)

        job = env.job_backlog.backlog[0]
        env.step(3)
        self.assertEqual(env.job_slot.slot[3], job)

        print("- Backlog test passed -")


if __name__ == '__main__':
    unittest.main()
