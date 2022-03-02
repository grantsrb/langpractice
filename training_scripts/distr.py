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

or

{
    "devices": [0,1,2],
    "split_keys": ["use_count_words"],
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
        sesh_name = "{}{}".format(exp_name[:4],device)
        command = "{} \"{}\" \'{}; {} {} {}\'".format(
            tmux_sesh,
            sesh_name,
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

    if "split_keys" in meta:
        splt_keys = set(meta["split_keys"])
    elif "split_key" in meta:
        splt_keys = [meta["split_key"]]
    else:
        assert False, "split_key or split_keys must be in metaparameters"
    keys = list(filter(lambda x: x not in splt_keys, ranges.keys()))
    no_splt = {k:ranges[k] for k in keys}
    devices = meta["devices"]

    # Divide up hyperranges equally amongst GPUs
    rng_paths = []
    range_dict = {i:{**no_splt} for i in devices}
    for splt_key in splt_keys:
        splt_len = len(ranges[splt_key])
        n_vals = splt_len//len(devices) # Number of split_key values per gpu
        if splt_len % len(devices) != 0:
            print("WARNING: SEARCH DIVIDED UNEQUALLY!!!")
        for i,device in enumerate(devices):
            vals = ranges[splt_key][i*n_vals:(i+1)*n_vals]
            range_dict[device] = {splt_key: vals, **range_dict[device]}

    # Save hyperranges to json files
    for device in devices:
        path = save_path+"{}.json".format(device)
        rng_paths.append(path)
        save_json(range_dict[device], path)
    return rng_paths

if __name__ == "__main__":

    meta = load_json(sys.argv[2])
    rng_paths = split_ranges(meta)
    distr_ranges(sys.argv[1], meta, rng_paths)
