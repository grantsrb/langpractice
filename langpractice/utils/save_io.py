import torch
import pickle
import supervised_gym.utils.utils as utils
import os

BEST_CHECKPT_NAME = "best_checkpt_0.pt.best"

def save_checkpt(save_dict, save_name, epoch, ext=".pt",
                                       del_prev_sd=True,
                                       best=False):
    """
    Saves a dictionary that contains a statedict

    save_dict: dict
        a dictionary containing all the things you want to save
    save_name: str
        the path to save the dict to.
    epoch: int
        an integer to be associated with this checkpoint
    ext: str
        the extension of the file
    del_prev_sd: bool
        if true, the state_dict of the previous checkpoint will be
        deleted
    best: bool
        if true, additionally saves this checkpoint as the best
        checkpoint under the filename set by BEST_CHECKPT_NAME
    """
    if del_prev_sd and epoch is not None:
        prev_path = "{}_{}{}".format(save_name,epoch-1,ext)
        prev_path = os.path.abspath(os.path.expanduser(prev_path))
        if os.path.exists(prev_path):
            device = torch.device("cpu")
            data = torch.load(prev_path, map_location=device)
            keys = list(data.keys())
            for key in keys:
                if "state_dict" in key or "optim_dic" in key:
                    del data[key]
            torch.save(data, prev_path)
        elif epoch != 0:
            print("Failed to find previous checkpoint", prev_path)
    if epoch is None: epoch = 0
    path = "{}_{}{}".format(save_name,epoch,ext)
    path = os.path.abspath(os.path.expanduser(path))
    torch.save(save_dict, path)
    if best:
        if "/" not in save_name:
            folder = os.path.join("./")
        else:
            folder = save_name.split("/")[:-1]
            folder = os.path.join(*folder)
            if save_name[0] == "/": folder = "/" + folder
        save_best_checkpt(save_dict, folder)

def save_best_checkpt(save_dict, folder):
    """
    Saves the checkpoint under the name set in BEST_CHECKPT_PATH to the
    argued folder

    save_dict: dict
        a dictionary containing all the things you want to save
    folder: str
        the path to the folder to save the dict to.
    """
    path = os.path.join(folder,BEST_CHECKPT_NAME)
    path = os.path.abspath(path)
    torch.save(save_dict,path)

def get_checkpoints(folder, checkpt_exts={'p', 'pt', 'pth'}):
    """
    Returns all .p, .pt, and .pth file names contained within the
    folder. They're sorted by their epoch. BEST_CHECKPT_PATH is not
    included in this list

    folder: str
        path to the folder of interest

    Returns:
    checkpts: list of str
        the full paths to the checkpoints contained in the folder
    """
    folder = os.path.expanduser(folder)
    assert os.path.isdir(folder)
    checkpts = []
    for f in os.listdir(folder):
        splt = f.split(".")
        if len(splt) > 1 and splt[-1] in checkpt_exts:
            path = os.path.join(folder,f)
            checkpts.append(path)
    sort_key = lambda x: int(x.split(".")[-2].split("_")[-1])
    checkpts = sorted(checkpts, key=sort_key)
    return checkpts

def foldersort(x):
    """
    A sorting key function to order folder names with the format:
    <path_to_folder>/<exp_name>_<exp_num>_<ending_folder_name>/

    x: str
    """
    splt = x.split("/")[-1].split("_")
    for i,s in enumerate(splt[1:]):
        try:
            return int(s)
        except:
            pass
    assert False

def is_model_folder(path, exp_name=None):
    """
    checks to see if the argued path is a model folder or otherwise.

    path: str
        path to check
    exp_name: str or None
    """
    check_folder = os.path.expanduser(path)
    if exp_name is not None:
        exp_splt = exp_name.split("_")
        if check_folder[-1]=="/": check_folder = check_folder[:-1]
        folder_splt = check_folder.split("/")
        folder_splt = folder_splt[-1].split("_")
        match = True
        for i in range(len(exp_splt)):
            if exp_splt[i] != folder_splt[i]: match = False
        if match: return True
    contents = os.listdir(check_folder)
    for content in contents:
        if ".pt" in content or "hyperparams.txt" == content:
            return True
    return False

def get_model_folders(main_folder, incl_ext=False):
    """
    Returns a list of paths to the model folders contained within the
    argued main_folder

    main_folder - str
        path to main folder
    incl_ext: bool
        include extension flag. If true, the expanded paths are
        returned. otherwise only the end folder (i.e.  <folder_name>
        instead of main_folder/<folder_name>)

    Returns:
        list of folder names (see incl_ext for full path vs end point)
    """
    folders = []
    main_folder = os.path.expanduser(main_folder)
    exp_name = main_folder.split("/")
    exp_name = exp_name[-1] if len(exp_name[-1])>0 else exp_name[-2]
    for d, sub_ds, files in os.walk(main_folder):
        for sub_d in sub_ds:
            check_folder = os.path.join(d,sub_d)
            if is_model_folder(check_folder,exp_name=exp_name):
                if incl_ext:
                    folders.append(check_folder)
                else:
                    folders.append(sub_d)
    return sorted(folders, key=foldersort)

def load_checkpoint(path,use_best=False):
    """
    Loads the save_dict into python. If the path is to a model_folder,
    the loaded checkpoint is the BEST checkpt if available, otherwise
    the checkpt of the last epoch

    Args:
        path: str
            path to checkpoint file or model_folder
        use_best: bool
            if true, will load the best checkpt based on validation metrics
    Returns:
        checkpt: dict
            a dict that contains all the valuable information for the
            training.
    """
    path = os.path.expanduser(path)
    hyps = None
    if os.path.isdir(path):
        hyps = utils.load_json(os.path.join(path, "hyperparams.json"))
        best_path = os.path.join(path,BEST_CHECKPT_NAME)
        if use_best and os.path.exists(best_path):
            path = best_path 
        else:
            checkpts = get_checkpoints(path)
            if len(checkpts)==0: return None
            path = checkpts[-1]
    data = torch.load(path, map_location=torch.device("cpu"))
    if "hyps" not in data: data["hyps"] = hyps
    return data

def load_model(path, models, load_sd=True, use_best=False,
                                           verbose=True):
    """
    Loads the model architecture and state dict from a .pt or .pth
    file. Or from a training save folder. Defaults to the last check
    point file saved in the save folder.

    path: str
        either .pt,.p, or .pth checkpoint file; or path to save folder
        that contains multiple checkpoints
    models: dict (just pass `globals()` as the arg)
        A dict of the potential model classes. This function is
        easiest if you import each of the model classes in the calling
        script and simply pass `globals()` as the argument for this
        parameter

        keys: str
            the class names of the potential models
        vals: Class
            the potential model classes
    load_sd: bool
        if true, the saved state dict is loaded. Otherwise only the
        model architecture is loaded with a random initialization.
    use_best: bool
        if true, will load the best model based on validation metrics
    """
    path = os.path.expanduser(path)
    hyps = None
    data = load_checkpoint(path,use_best=use_best)
    if 'hyps' in data:
        kwargs = data['hyps']
    else:
        kwargs = utils.load_json(os.path.join(path, "hyps.json"))
    model = models[kwargs['model_type']](**kwargs)
    if "state_dict" in data and load_sd:
        model.load_state_dict(data['state_dict'])
    else:
        print("state dict not loaded!")
    return model

def get_hyps(folder):
    """
    Returns a dict of the hyperparameters collected from the json
    save file in the model folder.

    folder: str
        path to the folder that contains checkpts and a hyps json file
    """
    folder = os.path.expanduser(folder)
    hyps_json = os.path.join(folder, "hyperparams.json")
    hyps = utils.load_json(hyps_json)
    return hyps

def get_next_exp_num(exp_name):
    """
    Finds the next open experiment id number.

    exp_name: str
        path to the main experiment folder that contains the model
        folders
    """
    folders = get_model_folders(exp_name)
    exp_nums = set()
    for folder in folders:
        exp_num = foldersort(folder)
        exp_nums.add(exp_num)
    for i in range(len(exp_nums)):
        if i not in exp_nums:
            return i
    return len(exp_nums)

def exp_num_exists(exp_num, exp_name):
    """
    Determines if the argued experiment number already exists for the
    argued experiment name.

    exp_num: int
        the number to be determined if preexisting
    exp_name: str
        path to the main experiment folder that contains the model
        folders
    """
    folders = get_model_folders(exp_name)
    for folder in folders:
        num = foldersort(folder)
        if exp_num == num:
            return True
    return False

def make_save_folder(hyps):
    """
    Creates the save name for the model.

    hyps: dict
        keys:
            exp_name: str
            exp_num: int
            search_keys: str
    """
    save_folder = "{}/{}_{}".format(hyps['exp_name'],
                                    hyps['exp_name'],
                                    hyps['exp_num'])
    save_folder += hyps['search_keys']
    return save_folder

