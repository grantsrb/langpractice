"""
This script takes a metaranges file and distributes the specified 
hyperranges (within the metaranges) to each of the gpus with its own
tmux session.

    $ python3 distr.py main.py metaranges.json

The meta ranges should have the following structure within a .json:

{
    "devices": [0,1,2],
    "split_key": "use_count_words",
    "hyperparams": "path/to/hyperparams.json",
    "hyperranges": "path/to/hyperranges.json"
}
"""
import sys
import os
from langpractice.utils.utils import load_json, save_json

# os.system("tmux new -s tes")
# tmux new-session -d -s \"myTempSession\" /opt/my_script.sh

def distr_ranges(script, meta, rng_paths):
    exp_name = load_json(meta["hyperparams"])["exp_name"]
    
    tmux_sesh = "tmux new -d -s"
    exe = "python3 {}".format(script)
    for rng_path, device in zip(rng_paths, meta["devices"]):
        cuda = "export CUDA_VISIBLE_DEVICES=" + str(device)
        command = "{} \"{}{}\" \'{}; {} {} {}\'".format(
            tmux_sesh,
            exp_name,
            device,
            cuda,
            exe,
            meta["hyperparams"],
            rng_path
        )
        print(command)
        os.system(command)

def split_ranges(meta):
    """
    Takes a hyperranges file and splits the ranges on the split_key into
    multiple different ranges files. One for each cuda device.

    Args:
        meta: dict
            "hyperparams": str
                path to a hyperparams file
            "hyperranges": str
                path to a hyperranges file
            "split_key": str
                the key that should be distributed among devices
            "devices": list of int
                the potential cuda device indices to train on
    Returns:
        rng_paths: list of str
            a list of paths to the new hyperranges files
    """
    ranges = load_json(meta["hyperranges"])

    # Save to folder that we found the ranges
    # Each ranges is saved as exp_name{cuda_device}.json
    save_path = os.path.abspath(meta["hyperranges"]).split("/")
    save_path[-1] = load_json(meta["hyperparams"])["exp_name"]
    save_path = "/".join(save_path)

    splt_key = meta["split_key"]
    keys = list(filter(lambda x: x!=splt_key, ranges.keys()))
    no_splt = {k:ranges[k] for k in keys}

    devices = meta["devices"]

    rng_paths = []
    for device,val in zip(devices, ranges[splt_key]):
        rngs = {splt_key: [val], **no_splt}
        path = save_path+"{}.json".format(device)
        rng_paths.append(path)
        save_json(rngs, path)
    if len(devices) < len(ranges[splt_key]):
        print("NOT ENOUGH DEVICES ARGUED FOR SPLIT KEY!!!")
        s = splt_key + " values missing: "
        for rng in ranges[splt_key][len(devices):]:
            s += str(rng) + ", "
        print(s[:-2])
    return rng_paths

if __name__ == "__main__":

    meta = load_json(sys.argv[2])
    rng_paths = split_ranges(meta)
    distr_ranges(sys.argv[1], meta, rng_paths)
