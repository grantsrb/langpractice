import torch
import langpractice as lp
from langpractice.utils.training import run_training
import torch.multiprocessing as mp
import sys

"""
Just argue the path to a model folder that you would like to resume
the training of

$ python3 resume.py /path/to/model/folder/

"""
torch.autograd.set_detect_anomaly(True)

if __name__ == "__main__":
    mp.set_start_method('forkserver')
    hyps = lp.utils.save_io.get_hyps(sys.argv[1])
    hyps["resume_folder"] = sys.argv[1]
    lp.training.train(0, hyps=hyps, verbose=True)

