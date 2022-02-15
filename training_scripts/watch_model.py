import torch
import numpy as np
import langpractice as lp
import sys
from langpractice.models import *
import matplotlib.pyplot as plt

"""
Must argue the path to a model folder for viewing. This script
automatically selects the best model from the training.

$ python3 watch_model.py exp_name/model_folder/
"""
if __name__ == "__main__":
    model_folder = sys.argv[1]
    checkpt = lp.utils.save_io.load_checkpoint(
        model_folder,
        use_best=False
    )
    hyps = checkpt["hyps"]
    hyps["n_eval_steps"] = 1000
    hyps["render"] = True
    model = globals()[hyps["model_type"]](**hyps).cuda()
    model.load_state_dict(checkpt["state_dict"])
    model.eval()
    model.reset()
    val_runner = lp.experience.ValidationRunner(hyps)
    val_runner.phase = 2
    eval_eps = 10
    state = val_runner.create_new_env(n_targs=None)
    model.reset(1)

    data = val_runner.collect_data(model, state, None)
