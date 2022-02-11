import langpractice as lp
from langpractice.models import *
from langpractice.utils.utils import try_key

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os
import sys

"""
Must argue the path to a model folder for validating. This script
automatically selects the last checkpoint from the training.

$ python3 validate_model.py exp_name/model_folder/
"""

def validate_model(model_folder,
                   n_eps=15,
                   targ_range=None,
                   val_max_actn=None):
    """
    Validates a model by running 

    Args:
        model_folder: str
            full path to the model folder that you want to validate
        n_eps: int
            the number of episodes to collect at each target value
        targ_range: tuple of ints or None
            the range of target values to sample from. if None,
            defaults to the val_targ_range in the hyperparameters
        val_max_actn: bool or None
            decides if actions should be sampled as the maximum over
            the output distribution or according to the output distr.
            if None, defaults to the choice in the hyperparameters.
    """
    checkpt = lp.utils.save_io.load_checkpoint(
        model_folder,
        use_best=False
    )
    hyps = checkpt["hyps"]
    if val_max_actn is not None:
        hyps["val_max_actn"] = val_max_actn
    model = globals()[hyps["model_type"]](**hyps).cuda()
    model.load_state_dict(checkpt["state_dict"])
    model.eval()
    model.cuda()
    model.reset()
    val_runner = lp.experience.ValidationRunner(hyps)

    df = None
    if targ_range is None:
        targ_range = try_key(hyps,"val_targ_range",hyps['targ_range'])
    for n_targs in tqdm(range(targ_range[0], targ_range[1]+1)):
        data = val_runner.rollout(
            phase=2,
            model=model,
            n_eps=n_eps,
            n_tsteps=None,
            n_targs=n_targs
        )

        keys = ["n_items", "n_targs", "n_aligned"]
        dones = data["dones"].reshape(-1)
        inpts = {key: data[key].reshape(-1) for key in keys}
        inpts = {key: val[dones==1] for key,val in inpts.items()}
        inpts = {k:v.cpu().data.numpy() for k,v in inpts.items()}
        inpts["epoch"] = [
            checkpt["epoch"] for i in range(len(inpts["n_items"]))
        ]
        inpts["phase"] = [
            checkpt["phase"] for i in range(len(inpts["n_items"]))
        ]
        if df is None:
            df = pd.DataFrame(inpts)
        else:
            df = df.append(pd.DataFrame(inpts))
    df["loaded_path"] = checkpt["loaded_path"]
    df["save_folder"] = checkpt["hyps"]["save_folder"]

    if not os.path.isdir(model_folder):
        splt = model_folder.split("/")
        if len(splt) > 1:
            model_folder = "/".join(splt[:-1])
        else: model_folder = "./"
    save_file = os.path.join(model_folder, "external_validation.csv")
    print("Saving to", save_file)
    df.to_csv(save_file, sep=",", mode="a")

if __name__ == "__main__":
    model_folder = sys.argv[1]
    validate_model(model_folder, val_max_actn=True)
