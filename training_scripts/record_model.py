import torch
import numpy as np
import langpractice as lp
import sys
from langpractice.models import *
import matplotlib.pyplot as plt
import cv2

"""
Must argue the path to a model folder for viewing. This script
automatically selects the best model from the training.

$ python3 record_model.py exp_name/model_folder/
"""

n_episodes = 2 # Set this to longer to get more unique game xp
repeat = 6 # Set this to longer to linger on images longer
fps = 30
targ_range = (1,5)

if __name__ == "__main__":
    model_folder = sys.argv[1]
    checkpt = lp.utils.save_io.load_checkpoint(
        model_folder,
        use_best=False
    )
    hyps = checkpt["hyps"]
    hyps["n_eval_eps"] = n_episodes
    hyps["val_targ_range"] = targ_range
    model = globals()[hyps["model_type"]](**hyps).cuda()
    model.load_state_dict(checkpt["state_dict"])
    model.eval()
    model.reset()
    val_runner = lp.experience.ValidationRunner(hyps)
    val_runner.phase = 2
    state = val_runner.create_new_env(n_targs=None)
    model.reset(1)

    with torch.no_grad():
        data = val_runner.collect_data(model, state, None)
    torch.cuda.empty_cache()

    frames = np.asarray(data["states"])
    print("collected data:", frames.shape)
    frames = np.repeat(frames.transpose((0,2,3,1)), 3, axis=-1)

    frames[frames==0] = -2
    frames = (frames+2)
    frames = frames/np.max(frames)
    frames = np.uint8(frames*255).repeat(3,axis=1).repeat(3,axis=2)
    frames = np.repeat(frames, repeat, axis=0)
    output_name = hyps["exp_name"] + str(hyps["exp_num"]) + ".mp4"
    out = cv2.VideoWriter("vids/"+output_name,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        frames[0].shape[:2]
    )
    for frame in frames:
        out.write(frame)
    out.release()
