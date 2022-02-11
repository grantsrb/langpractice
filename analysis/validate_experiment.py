import torch
import numpy as np
import langpractice as lp
import sys
from langpractice.models import *
import matplotlib.pyplot as plt
import pandas as pd
from validate_model import validate_model
import time

"""
Must argue the path to an experiment folder for validating. This script
automatically selects the last checkpoint from each training within
the experiment folder.

$ python3 validate_model.py exp_name/model_folder/
"""

if __name__=="__main__":
    exp_folder = sys.argv[1]
    model_folders = lp.utils.save_io.get_model_folders(
            exp_folder,
            incl_full_path=True
    )
    for model_folder in model_folders:
        print("Beginning Validation for", model_folder)
        starttime = time.time()
        validate_model(model_folder)
        print("Finished", time.time()-starttime, "s")
