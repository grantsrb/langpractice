import torch.nn.functional as F
import numpy as np
import torch
import json
import os
import cv2

PIRAHA_WEIGHTS = {
        3:   torch.FloatTensor([.55, .45]),
        4:   torch.FloatTensor([.4, .6]),
        5:   torch.FloatTensor([.4, .6]),
        6:   torch.FloatTensor([.4, .6]),
        7:   torch.FloatTensor([.45, .55]),
        8:   torch.FloatTensor([.3, .7]),
        9:   torch.FloatTensor([.3, .7]),
        10:  torch.FloatTensor([.3, .7]),
    }

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")

def try_key(d, key, val):
    """
    d: dict
    key: str
    val: object
        the default value if the key does not exist in d
    """
    if key in d:
        return d[key]
    return val

def load_json(file_name):
    """
    Loads a json file as a python dict

    file_name: str
        the path of the json file
    """
    file_name = os.path.expanduser(file_name)
    with open(file_name,'r') as f:
        s = f.read()
        j = json.loads(s)
    return j

def save_json(data, file_name):
    """
    saves a dict to a json file

    data: dict
    file_name: str
        the path that you would like to save to
    """
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def resize2Square(img, size):
    """
    resizes image to a square with the argued size. Preserves the aspect
    ratio. fills the empty space with zeros.

    img: ndarray (H,W, optional C)
    size: int
    """
    h, w = img.shape[:2]
    c = None if len(img.shape) < 3 else img.shape[2]
    if h == w: 
        return cv2.resize(img, (size, size), cv2.INTER_AREA)
    if h > w: 
        dif = h
    else:
        dif = w
    interpolation = cv2.INTER_AREA if dif > size else\
                    cv2.INTER_CUBIC
    x_pos = int((dif - w)/2.)
    y_pos = int((dif - h)/2.)
    if c is None:
      mask = np.zeros((dif, dif), dtype=img.dtype)
      mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
      mask = np.zeros((dif, dif, c), dtype=img.dtype)
      mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]
    return cv2.resize(mask, (size, size), interpolation)

def rand_sample(arr, n_samples=1):
    """
    Uniformly samples a single element from the argued array.

    Args:
        arr: indexable sequence
    """
    if not isinstance(arr,list): arr = list(arr)
    if len(arr) == 0: print("len 0:", arr)
    samples = []
    perm = np.random.permutation(len(arr))
    for i in range(n_samples):
        samples.append(arr[perm[i]])
    if len(samples) == 1: return samples[0]
    return samples

def get_max_key(d):
    """
    Returns key corresponding to maxium value

    d: dict
        keys: object
        vals: int or float
    """
    max_v = -np.inf
    max_k = None
    for k,v in d.items():
        if v > max_v:
            max_v = v
            max_k = k
    return max_k

def update_shape(shape, depth, kernel=3, padding=0, stride=1, op="conv"):
    """
    Calculates the new shape of the tensor following a convolution or
    deconvolution. Does not operate in place on shape.

    shape: list-like (chan, height, width)
    depth: int
        the new number of channels
    kernel: int or list-like
        size of the kernel
    padding: list-like or int
    stride: list-like or int
    op: str
        'conv' or 'deconv'
    """
    heightwidth = np.asarray([*shape[-2:]])
    if type(kernel) == type(int()):
        kernel = np.asarray([kernel, kernel])
    else:
        kernel = np.asarray(kernel)
    if type(padding) == type(int()):
        padding = np.asarray([padding,padding])
    else:
        padding = np.asarray(padding)
    if type(stride) == type(int()):
        stride = np.asarray([stride,stride])
    else:
        stride = np.asarray(stride)

    if op == "conv":
        heightwidth = (heightwidth - kernel + 2*padding)/stride + 1
    elif op == "deconv" or op == "conv_transpose":
        heightwidth = (heightwidth - 1)*stride + kernel - 2*padding
    return (depth, *heightwidth)

def sample_action(pi):
    """
    Stochastically selects an action from the pi vectors.

    Args:
        pi: torch FloatTensor (..., N) (must sum to 1 across last dim)
            this is most likely going to be a model output vector that
            has passed through a softmax
    """
    pi = pi.cpu()
    rand_nums = torch.rand(*pi.shape[:-1])
    cumu_sum = torch.zeros(pi.shape[:-1])
    actions = -torch.ones(pi.shape[:-1])
    for i in range(pi.shape[-1]):
        cumu_sum += pi[...,i]
        actions[(cumu_sum >= rand_nums)&(actions < 0)] = i
    return actions

def sample_numpy(pi):
    """
    Stochastically selects an index from the pi vectors.

    Args:
        pi: ndarray (N,) (must sum to 1 across last dim)
    """
    rand_num = np.random.random()
    cumu_sum = 0
    action = 0
    for i in range(len(pi)):
        cumu_sum += pi[i]
        if cumu_sum > rand_num: return i
    return len(pi)-1

def softmax(arr):
    """
    arr: ndarray (N,)
        a single dimensional array
    """
    arr = np.exp(arr-np.max(arr))
    return arr / np.sum(arr)

def zipfian(low=1, high=9, order=1):
    """
    Draws a single integer from low (inclusive) to high (inclusive) in
    which the probability is proportional to 1/k^order.

    Args:
        low: int (inclusive)
            the lowest possible value
        high: int (inclusive)
            the highest possible value
        order: float
            the order of the exponent to weight the probability density
            for each possible value.
    Returns:
        sample: int
            returns a sample drawn from the zipfian distribution.
    """
    if low == high: return low
    assert low < high and low > 0

    probs = np.arange(low, high+1).astype("float")
    probs = 1/(probs**order)
    probs = probs/probs.sum()
    samp = sample_numpy(probs)
    return samp + low

def get_piraha_labels(labels, n_items):
    """
    Converts the number of items that exist in the game (not
    including the targets) to a count word in the piraha language.

    Uses the following probablities for each count word conversion.
    Probabilities taken from Frank 2008.
        number 0:
            labels: 0
            probabilities: [1]
        number 1:
            labels: 1
            probabilities: [1]
        number 2:
            labels: 2
            probabilities: [1]
        number 3:
            labels: 2,3
            probabilities: [.55, .45]
        numbers 4-6:
            labels: 2,3
            probabilities: [.4, .6]
        numbers 7:
            labels: 2,3
            probabilities: [.45, .55]
        numbers 8 and above:
            labels: 2,3
            probabilities: [.3, .7]

    Args:
        labels: torch Tensor (...,N)
            the count of the items on the board (a clone of n_items
            works just fine)
        n_items: torch Tensor (...,N)
            the count of the items on the board
    Returns:
        labels: torch LongTensor
            the updated labels. operates in place 
    """
    weights = {**PIRAHA_WEIGHTS}
    labels[n_items==1] = 1
    labels[n_items==2] = 2
    # Sample the Piraha count words with the appropriate length 
    # using weights found in Frank's 2008 "Number as a cog tech"
    min_key = np.min(list(weights.keys()))
    max_key = np.max(list(weights.keys()))
    for i in range(min_key,max_key):
        idx = (n_items==i)
        l = len(labels[idx])
        if l > 0:
            labs = torch.multinomial(weights[i], l, replacement=True)
            # samples are 0 indexed, so add 2 for the proper label
            labels[idx] = labs + 2
    # Repeat previous step for numbers greater than max specified
    idx = (n_items>=max_key)
    l = len(labels[idx])
    if l > 0:
        labs = torch.multinomial(weights[max_key], l, replacement=True)
        # samples are 0 indexed, so add 2 for the proper label
        labels[idx] = labs + 2
    return labels

def get_duplicate_labels(labels, n_items, max_label):
    """
    Converts the number of items that exist in the game (not
    including the targets) to a count word that is interchangeable
    with another count word meaning the same thing. For example, the
    label for the value "0" can either be 0 or 1 with equal probability.
    Similarly the label "1" can be either 2 or 3. This pattern continues
    up to the max label.

    Args:
        labels: torch Tensor (...,N)
            the count of the items on the board (a clone of n_items
            works just fine)
        n_items: torch Tensor (...,N)
            the count of the items on the board
        max_label: int
            the maximum available label. must be even so that we can
            stride 2 steps in labels for each step in n_items values.
            Then we equally distribute both labels amongst the n_items
            entries equal to the corresponding value
    Returns:
        labels: torch LongTensor
            the updated labels. operates in place 
    """
    rand_vals = torch.randint(0,2,labels.shape)
    for i in range(0,max_label,2):
        val = i//2
        if i == max_label-2:
            idx = n_items>=val
        else:
            idx = n_items==val
        labels[idx] = i+rand_vals[idx]
    return labels

def get_lang_labels(n_items, n_targs, max_label, use_count_words):
    """
    Determines the language labels based on the type of training.

    Args:
        n_items: torch Tensor (N,)
            the count of the items on the board
        n_targs: torch Tensor (N,)
            the count of the targs on the board
        max_label: int
            the maximum allowed language label. can usually use
            model.lang_size-1
    Returns:
        labels: torch Tensor (N,)
    """
    labels = n_items.clone()
    labels[labels>max_label] = max_label
    if int(use_count_words) == 0:
        labels[n_items<n_targs] = 0
        labels[n_items==n_targs] = 1
        labels[n_items>n_targs] = 2
    # Piraha labels
    elif int(use_count_words) == 2:
        labels = get_piraha_labels(labels, n_items)
    # Random labels
    elif int(use_count_words) == 3:
        labels = torch.randint(0, max_label+1, labels.shape)
        if n_items.is_cuda:
            labels = labels.to(DEVICE)
    # Duplicate labels
    elif int(use_count_words) == 4:
        labels = get_duplicate_labels(labels, n_items, max_label)
    return labels

def get_loss_and_accs(phase,
                      actn_preds,
                      lang_preds,
                      actn_targs,
                      lang_targs,
                      drops,
                      n_targs,
                      n_items,
                      prepender="",
                      loss_fxn=F.cross_entropy,
                      lang_p=0.5):
    """
    Calculates the loss and accuracies depending on the phase of
    the training.

        Phase 0: language loss when agent drops an item
        Phase 1: action loss at all steps in rollout
        Phase 2: lang and action loss at all steps in rollout

    Args:
        phase: int - 0,1,or 2
            the phase of the training
        actn_preds: torch FloatTensor (B,S,A)
            action predictions
        lang_preds: sequence of torch FloatTensors [(B,S,L),(B,S,L),...]
            a list of language predictions
        actn_targs: torch LongTensor (B*S,)
            action labels
        lang_targs: torch LongTensor (B*S,)
            language labels
        drops: torch LongTensor (B*S,)
            1s denote steps in which the agent dropped an item, 0s
            denote all other steps
        n_targs: torch LongTensor (B*S,)
            the number of target objects on the grid at this step
            of the episode
        n_items: torch Tensor (B*S,)
            the count of the items on the board
        loss_fxn: torch Module
            the loss function to calculate the loss. i.e.
            torch.nn.CrossEntropyLoss()
        prepender: str
            a string to prepend to all keys in the accs dict
        lang_p: float
            the language portion of the loss. only a factor for phase
            2 trainings
    Returns:
        loss: torch float tensor (1,)
            the appropriate loss for the phase
        accs: dict
            keys: str
            vals: float
                the appropriate label accuracies depending on the
                phase
    """
    # Phase 0: language labels when agent drops an item
    # Phase 1: action labels at all steps in rollout
    # Phase 2: lang and action labels at all steps in rollout
    loss = 0
    lang_accs = {}
    lang_losses = {}
    if phase == 0 or phase == 2:
        loss, lang_losses, lang_accs = calc_lang_loss_and_accs(
            lang_preds,
            lang_targs,
            drops,
            loss_fxn=loss_fxn,
            categories=n_items,
            prepender=prepender
        )
    actn_accs = {}
    if phase == 1 or phase == 2:
        actn_loss, actn_accs = calc_actn_loss_and_accs(
            actn_preds,
            actn_targs,
            n_targs,
            loss_fxn,
            prepender
        )
        p = lang_p if phase == 2 else 0
        loss = p*loss + (1-p)*actn_loss
    return loss, lang_losses, {**actn_accs, **lang_accs}

def calc_actn_loss_and_accs(logits,targs,n_targs,loss_fxn,prepender):
    """
    Args:
        logits: torch FloatTensor (B,S,A)
            action predictions
        targs: torch LongTensor (B*S,)
            action labels
        n_targs: torch LongTensor (B*S,)
            the number of target objects on the grid at this step
            of the episode
        loss_fxn: torch Module
            the loss function to calculate the loss. i.e.
            torch.nn.CrossEntropyLoss()
        prepender: str
            a string to prepend to all keys in the accs dict
    Returns:
        loss: torch float tensor (1,)
        accs: dict
            keys: str
                accuracy types
            vals: float
                accuracies
    """
    logits = logits.reshape(-1, logits.shape[-1])
    targs = targs.to(DEVICE)
    loss = loss_fxn(logits, targs)
    with torch.no_grad():
        actn_accs = calc_accs( # accs is a dict of floats
            logits=logits,
            targs=targs,
            categories=n_targs,
            prepender=prepender+"_actn"
        )
    return loss, actn_accs

def calc_lang_loss_and_accs(preds,
                            labels,
                            drops,
                            loss_fxn,
                            categories,
                            prepender=""):
    """
    Args:
        preds: sequence of torch FloatTensors [(B,S,L),(B,S,L),...]
            a list of language predictions
        labels: torch LongTensor (B*S,)
            language labels
        drops: torch LongTensor (B*S,)
            1s denote steps in which the agent dropped an item, 0s
            denote all other steps
        loss_fxn: torch Module
            the loss function to calculate the loss. i.e.
            torch.nn.CrossEntropyLoss()
        categories: torch long tensor (B, N) or None
            if None, this value is ignored. Otherwise it specifies
            categories for accuracy calculations.
        prepender: str
            a string to prepend to all keys in the accs dict
    Returns:
        loss: torch float tensor (1,)
        losses: dict
            keys: str
                loss types
            vals: float
                losses
        accs: dict
            keys: str
                accuracy types
            vals: float
                accuracies
    """
    accs_array = []
    losses_array = []
    idxs = drops==1
    categories = categories[idxs]
    labels = labels[idxs].to(DEVICE)
    loss = 0
    for lang in preds:
        lang = lang.reshape(-1, lang.shape[-1])
        lang = lang[idxs]
        loss += loss_fxn(lang, labels)
        with torch.no_grad():
            accs = calc_accs( # accs is a dict of floats
                logits=lang,
                targs=labels,
                categories=categories,
                prepender=prepender+"_lang"
            )
            accs_array.append(accs)

            losses = calc_losses( # accs is a dict of floats
                logits=lang,
                targs=labels,
                categories=categories,
                prepender=prepender+"_lang"
            )
            losses_array.append(losses)
    losses = avg_over_dicts(losses_array)
    accs = avg_over_dicts(accs_array)
    return loss, losses, accs

def avg_over_dicts(dicts_array):
    """
    This is a helper function to average over the keys in an array
    of dicts. The result is a dict with the same keys as every
    dict in the argued array, but the values are averaged over each
    dict within the argued array.

    Args:
        dicts_array: list of dicts
            this is a list of dicts. Each dict must consist of str
            keys and float or int vals. Each dict must have the
            same set of keys.
    Returns:
        avgs: dict
            keys: str
                same keys as all dicts in dicts_array
            vals: float
                the average over all dicts in the accs array for
                the corresponding key
    """
    if len(dicts_array) == 0: return dict()
    avgs = {k: 0 for k in dicts_array[0].keys()}
    for k in avgs.keys():
        avg = 0
        for i in range(len(dicts_array)):
            avg += dicts_array[i][k]
        avgs[k] = avg/len(dicts_array)
    return avgs

def calc_accs(logits, targs, categories=None, prepender=""):
    """
    Calculates the average accuracy over the batch for each possible
    category

    Args:
        logits: torch float tensor (B, N, K)
            the model predictions. the last dimension must be the
            same number of dimensions as possible target values.
        targs: torch long tensor (B, N)
            the targets for the predictions
        categories: torch long tensor (B, N) or None
            if None, this value is ignored. Otherwise it specifies
            categories for accuracy calculations.
        prepender: str
            a string to prepend to all keys in the accs dict
    Returns:
        accs: dict
            keys: str
                <prepender>_acc: float
                    the average accuracy over all categories
                <prepender>_acc_<category>: float
                    the average accuracy over this particular
                    category. for example, if one of the categories
                    is named 1, the key will be "1" and the value
                    will be the average accuracy over that
                    particular category.
    """
    if prepender!="" and prepender[-1]!="_": prepender = prepender+"_"
    prepender = prepender + "acc"
    logits = logits.reshape(-1, logits.shape[-1])
    try:
        argmaxes = torch.argmax(logits, dim=-1).reshape(-1)
    except:
        print("logits:", logits)
        return { prepender: 0 }
    targs = targs.reshape(-1)
    acc = (argmaxes.long()==targs.long()).float()
    accs = {
        prepender: acc.mean().item()
    }
    if len(argmaxes) == 0: return accs
    pre = prepender + "lbl_"
    targ_types = {*targs.cpu().data.numpy()}
    for t in targ_types:
        idxs = targs==t
        if idxs.float().sum() == 0: continue
        accs[pre+str(t)] = acc[idxs].mean().item()
    if type(categories) == torch.Tensor: # (B, N)
        categories = categories.reshape(-1).data.long()
        pre = prepender + "ctg_"
        cats = {*categories.numpy()}
        for cat in cats:
            idxs = categories==cat
            if idxs.float().sum() <= 0: continue
            accs[pre+str(cat)] = acc[idxs].mean().item()
    return accs

def calc_losses(logits,
                targs,
                categories=None,
                prepender="",
                loss_fxn=F.cross_entropy):
    """
    Calculates the average accuracy over the batch for each possible
    category

    Args:
        logits: torch float tensor (B, N, K)
            the model predictions. the last dimension must be the
            same number of dimensions as possible target values.
        targs: torch long tensor (B, N)
            the targets for the predictions
        categories: torch long tensor (B, N) or None
            if None, this value is ignored. Otherwise it specifies
            categories for accuracy calculations.
        prepender: str
            a string to prepend to all keys in the accs dict
    Returns:
        accs: dict
            keys: str
                <prepender>_acc: float
                    the average accuracy over all categories
                <prepender>_acc_<category>: float
                    the average accuracy over this particular
                    category. for example, if one of the categories
                    is named 1, the key will be "1" and the value
                    will be the average accuracy over that
                    particular category.
    """
    if prepender!="" and prepender[-1]!="_": prepender = prepender+"_" 
    prepender = prepender + "loss"
    logits = logits.reshape(-1, logits.shape[-1])
    targs = targs.reshape(-1)
    loss = loss_fxn(logits, targs, reduction="none")
    losses = {
        prepender: loss.mean().item()
    }
    pre = prepender + "lbl_"
    targ_types = {*targs.cpu().data.numpy()}
    for t in targ_types:
        idxs = targs==t
        if idxs.float().sum() == 0: continue
        losses[pre+str(t)] = loss[idxs].mean().item()
    if type(categories) == torch.Tensor: # (B, N)
        categories = categories.reshape(-1).cpu().data.long()
        pre = prepender + "ctg_"
        cats = {*categories.numpy()}
        for cat in cats:
            idxs = categories==cat
            if idxs.float().sum() <= 0: continue
            losses[pre+str(cat)] = loss[idxs].mean().item()
    return losses

def get_transformer_fwd_mask(s):
    """
    Generates a mask that looks like this:
        0, -inf, -inf
        0,   0,  -inf
        0,   0,  0

    Args:
        s: int
            the size of each sidelength of the mask
    """
    return torch.triu(torch.ones(s,s)*float("-inf"), diagonal=1)

