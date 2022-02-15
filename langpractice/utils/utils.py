import numpy as np
import torch
import json
import os
import cv2

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
    Randomly samples a single element from the argued array.

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
    probs = softmax(probs)
    samp = sample_numpy(probs)
    return samp + low

def get_piraha_labels(labels, n_items, n_targs):
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
        numbers 4-7:
            labels: 2,3
            probabilities: [.4, .6]
        numbers 8 and above:
            labels: 2,3
            probabilities: [.3, .7]

    Args:
        labels: torch Tensor (N,)
            the count of the items on the board (a clone of n_items
            works just fine)
        n_items: torch Tensor (N,)
            the count of the items on the board
        n_targs: torch Tensor (N,)
            the count of the targs on the board
    """
    weights = {
        "3":   torch.FloatTensor([.55, .45]),
        "4-7": torch.FloatTensor([.4, .6]),
        "8":   torch.FloatTensor([.3, .7])
    }
    labels[n_items==1] = 1
    labels[n_items==2] = 2
    # Sample the Piraha count words with the appropriate length 
    # using weights found in Frank's 2008 "Number as a cog tech"
    idx = (n_items==3)
    l = len(labels[idx])
    if l > 0:
        labs = torch.multinomial(weights["3"], l, replacement=True)
        # samples are 0 indexed, so add 2 for the proper label
        labels[idx] = labs + 2

    # Repeat previous step for numbers 4-7
    idx = (n_items>=4)&(n_items<=7)
    l = len(labels[idx])
    if l > 0:
        labs = torch.multinomial(weights["4-7"], l, replacement=True)
        # samples are 0 indexed, so add 2 for the proper label
        labels[idx] = labs + 2

    # Repeat previous step for numbers 8 and greater
    idx = (n_items>=8)
    l = len(labels[idx])
    if l > 0:
        labs = torch.multinomial(weights["8"], l, replacement=True)
        # samples are 0 indexed, so add 2 for the proper label
        labels[idx] = labs + 2
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
    """
    labels = n_items.clone()
    labels[labels>max_label] = max_label
    if int(use_count_words) == 0:
        labels[n_items<n_targs] = 0
        labels[n_items==n_targs] = 1
        labels[n_items>n_targs] = 2
    elif int(use_count_words) == 2:
        labels = get_piraha_labels(labels, n_items, n_targs)
    return labels

