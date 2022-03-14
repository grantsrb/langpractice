from langpractice.utils.utils import *
from langpractice.experience import *
import unittest
import numpy as np
import torch

class TestExperienceReplay(unittest.TestCase):
    def make_exp(self, hyps_override=dict()):
        hyps={
            "exp_len": 10,
            "env_type": "gordongames-v4",
            "batch_size": 3,
            "inpt_shape": (1,4,5),
            "seq_len": 2,
            "randomize_order": False,
            "roll_data": True,
            "use_count_words": 1,
            "skip_first_phase": False,
            "second_phase": 1,
            "n_frame_stack": 1,
            "randomize_order": False,
            "lang_on_drops_only": True,
            "targ_range": [1,10],
            "n_envs": 3,
        }
        hyps = {**hyps, **hyps_override}
        exp = ExperienceReplay(hyps)
        prev = exp.shared_exp
        exp = self.fill_exp_with_random_values(exp)
        assert prev == exp.shared_exp
        return exp

    def fill_exp_with_random_values(self, exp):
        shared = exp.shared_exp
        for k in shared.keys():
            prev = shared[k]
            dtype = shared[k].dtype
            shared[k][:] = torch.randint(0,10, shared[k].shape, dtype=dtype)
            assert np.array_equal(shared[k].numpy(), prev.numpy())
        return exp

    def fill_exp_with_arange(self, exp):
        shared = exp.shared_exp
        for k in shared.keys():
            prev = shared[k]
            dtype = shared[k].dtype
            arange = torch.arange(0,shared[k].numel(), dtype=dtype)
            shared[k][:] = arange.reshape(shared[k].shape)
            assert np.array_equal(
                shared[k].reshape(-1).numpy(),
                arange.numpy()
            )
        return exp

    def test_harvest(self):
        exp = self.make_exp()
        data = exp.harvest_exp()
        for k in data.keys():
            self.assertTrue(
                np.array_equal(data[k].numpy(), exp.shared_exp[k].numpy())
            )
        self.fill_exp_with_random_values(exp)
        for k in data.keys():
            self.assertFalse(
                np.array_equal(data[k].numpy(), exp.shared_exp[k].numpy())
            )

    def test_len_rolldatatrue(self):
        override = {
            "roll_data": True,
            "exp_len": 10,
            "seq_len": 2,
            }
        exp = self.make_exp(override)
        self.assertEqual(len(exp), override["exp_len"]-override["seq_len"])

    def test_len_rolldatafalse(self):
        override = {
            "roll_data": False,
            "exp_len": 10,
            "seq_len": 2,
            }
        exp = self.make_exp(override)
        self.assertEqual(
            len(exp),
            (override["exp_len"]-override["seq_len"])//override["seq_len"]
        )

    def test_getitem_rolldatatrue(self):
        override = {
            "roll_data": True,
            "exp_len": 10,
            "seq_len": 2,
            }
        exp = self.make_exp(override)
        exp = self.fill_exp_with_arange(exp)
        exp.harvest_exp()
        step0 = exp[0]
        step1 = exp[1]
        for k in step0.keys():
            self.assertFalse(np.array_equal(
                step0[k].numpy(),
                step1[k].numpy(),
            ))
        for k in step0.keys():
            self.assertTrue(np.array_equal(
                step0[k][:, 1:].numpy(),
                step1[k][:, :-1].numpy(),
            ))

    def test_getitem_rolldatafalse(self):
        override = {
            "roll_data": False,
            "exp_len": 10,
            "seq_len": 2,
            "inpt_shape": (1,4,5),
            }
        exp = self.make_exp(override)
        exp = self.fill_exp_with_arange(exp)
        exp.harvest_exp()
        step0 = exp[0]
        step1 = exp[1]
        for k in step0.keys():
            self.assertFalse(np.array_equal(
                step0[k].numpy(),
                step1[k].numpy(),
            ))
        for k in step0.keys():
            if k != "drops":
                self.assertFalse(np.array_equal(
                    step0[k][:, 1:].numpy(),
                    step1[k][:, :-1].numpy(),
                ))
        for k in step0.keys():
            if k != "drops":
                self.assertTrue(np.array_equal(
                    step0[k].numpy(),
                    exp.exp[k][:,:override["seq_len"]].numpy(),
                ))
                self.assertTrue(np.array_equal(
                    step1[k].numpy(),
                    exp.exp[k][:,override["seq_len"]:2*override["seq_len"]].numpy(),
                ))
        for k in step0.keys():
            if k == "obs":
                self.assertTrue(np.array_equal(
                    step0[k][0][0],
                    np.arange(
                        np.prod(override["inpt_shape"])
                    ).reshape(override["inpt_shape"])
                ))
                self.assertTrue(np.array_equal(
                    step0[k][0][1],
                    np.arange(
                        np.prod(override["inpt_shape"])
                    ).reshape(override["inpt_shape"])+np.prod(override["inpt_shape"])
                ))
                self.assertTrue(np.array_equal(
                    step1[k][0][0],
                    np.arange(
                        np.prod(override["inpt_shape"])
                    ).reshape(override["inpt_shape"])+\
                        np.prod(override["inpt_shape"])*override["seq_len"]
                ))
                self.assertTrue(np.array_equal(
                    step1[k][0][1],
                    np.arange(
                        np.prod(override["inpt_shape"])
                    ).reshape(override["inpt_shape"])+np.prod(override["inpt_shape"])+\
                        np.prod(override["inpt_shape"])*override["seq_len"]
                ))

    def test_next_rolldatatrue(self):
        override = {
            "roll_data": True,
            "exp_len": 10,
            "seq_len": 2,
            "inpt_shape": (1,4,5),
            }
        exp = self.make_exp(override)
        exp = self.fill_exp_with_arange(exp)
        exp.harvest_exp()
        prev = None
        k = "obs"
        for i,data in enumerate(exp):
            self.assertTrue(np.array_equal(
                data[k][0][0],
                np.arange(
                    np.prod(override["inpt_shape"])
                ).reshape(override["inpt_shape"])+np.prod(override["inpt_shape"])*i
            ))
            self.assertTrue(np.array_equal(
                data[k][0][1],
                np.arange(
                    np.prod(override["inpt_shape"])
                ).reshape(override["inpt_shape"])+np.prod(override["inpt_shape"])*(i+1)
            ))

    def test_next_rolldatafalse(self):
        override = {
            "roll_data": False,
            "exp_len": 10,
            "seq_len": 2,
            "inpt_shape": (1,4,5),
            }
        exp = self.make_exp(override)
        exp = self.fill_exp_with_arange(exp)
        exp.harvest_exp()
        prev = None
        k = "obs"
        for i,data in enumerate(exp):
            self.assertTrue(np.array_equal(
                data[k][0][0],
                np.arange(
                    np.prod(override["inpt_shape"])
                ).reshape(override["inpt_shape"])+\
                    i*np.prod(override["inpt_shape"])*override["seq_len"]
            ))
            self.assertTrue(np.array_equal(
                data[k][0][1],
                np.arange(
                    np.prod(override["inpt_shape"])
                ).reshape(override["inpt_shape"])+np.prod(override["inpt_shape"])+\
                    i*np.prod(override["inpt_shape"])*override["seq_len"]
            ))

    def test_get_drops_v1(self):
        hyps = {
            "lang_on_drops_only": True,
            "env_type": "gordongames-v1",
            "count_targs": False,
            "drops_perc_threshold": 0.05,
            }
        grabs = torch.LongTensor([
            [0,1,3,3,3,0,0],
            [3,3,3,0,0,3,3],
            [3,1,0,2,0,3,0],
        ])
        goal_drops = torch.LongTensor([
            [0,0,0,0,0,0,1],
            [0,0,0,0,1,0,0],
            [0,0,1,0,0,0,0],
        ])
        drops = ExperienceReplay.get_drops(hyps, grabs, torch.ones_like(grabs))
        self.assertTrue(np.array_equal(
            drops.numpy(),
            goal_drops.numpy()
        ))

    def test_get_drops_v1_lodoFalse(self):
        hyps = {
            "lang_on_drops_only": False,
            "env_type": "gordongames-v1",
            "count_targs": False,
            "drops_perc_threshold": 0.05,
            }
        grabs = torch.LongTensor([
            [0,1,3,3,3,0,0],
            [3,3,3,0,0,3,3],
            [3,1,0,2,0,3,0],
        ])
        goal_drops = torch.ones_like(grabs)
        drops = ExperienceReplay.get_drops(hyps, grabs, torch.ones_like(grabs))
        self.assertTrue(np.array_equal(
            drops.numpy(),
            goal_drops.numpy()
        ))

    def test_get_drops_v4_counttargsFalse(self):
        hyps = {
            "lang_on_drops_only": True,
            "env_type": "gordongames-v4",
            "count_targs": False,
            "drops_perc_threshold": 0.05,
            }
        grabs = torch.LongTensor([
            [0,1,1,1,1,0,0],
            [3,3,3,0,0,3,3],
            [3,1,0,2,0,3,0],
        ])
        goal_drops = torch.LongTensor([
            [0,1,1,1,1,0,0],
            [1,1,1,0,0,1,1],
            [1,1,0,1,0,1,0],
        ])
        drops = ExperienceReplay.get_drops(hyps, grabs, torch.ones_like(grabs))
        self.assertTrue(np.array_equal(
            drops.numpy(),
            goal_drops.numpy()
        ))

    def test_get_drops_v4_counttargsTrue(self):
        hyps = {
            "lang_on_drops_only": True,
            "env_type": "gordongames-v4",
            "count_targs": True,
            "drops_perc_threshold": 0.05,
            }
        grabs = torch.LongTensor([
            [0,1,1,1,1,0,0],
            [3,3,3,0,0,3,3],
            [3,1,0,2,0,3,0],
        ])
        goal_drops = torch.LongTensor([
            [0,1,1,1,1,0,0],
            [1,1,1,0,0,1,1],
            [1,1,0,1,0,1,0],
        ])
        is_animating = torch.zeros_like(grabs)
        is_animating[:,:2] = 1
        goal_drops[:,:2] = 1
        drops = ExperienceReplay.get_drops(hyps, grabs, is_animating)
        self.assertTrue(np.array_equal(
            drops.numpy(),
            goal_drops.numpy()
        ))

    def test_get_drops_v4_langloctype1(self):
        hyps = {
            "lang_on_drops_only": True,
            "env_type": "gordongames-v4",
            "count_targs": True,
            "lang_loc_type": 1,
            "drops_perc_threshold": 0.05,
            }
        grabs = torch.LongTensor([
            [0,1,1,1,1,0,0],
            [3,3,3,0,0,3,3],
            [3,1,0,2,0,3,0],
        ])
        goal_drops = torch.zeros_like(grabs)
        is_animating = torch.zeros_like(grabs)
        is_animating[:,:2] = 1
        goal_drops[:,:2] = 1
        drops = ExperienceReplay.get_drops(hyps, grabs, is_animating)
        self.assertTrue(np.array_equal(
            drops.numpy(),
            goal_drops.numpy()
        ))

if __name__=="__main__":
    unittest.main()
