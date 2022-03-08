import pandas as pd
import os
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
from langpractice.models import NullModel
from langpractice.envs import SequentialEnvironment
from langpractice.oracles import *
from langpractice.utils.utils import try_key, sample_action, zipfian, get_lang_labels
from collections import deque

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")

def next_state(env, obs_deque, obs, reset, n_targs=None):
    """
    Get the next state of the environment.

    env - environment of interest
    obs_deq - deque of the past n observations
    obs - ndarray returned from the most recent step of the environment
    reset - boolean denoting the reset signal from the most recent step 
            of the environment
    n_targs: int or None
        if int, decides the number of targets for the next episode
    """

    if reset or obs is None:
        obs = env.reset(n_targs=n_targs)
        for i in range(obs_deque.maxlen-1):
            obs_deque.append(np.zeros(obs.shape))
    obs_deque.append(obs)
    state = np.concatenate(obs_deque, axis=0)
    return state

class ExperienceReplay(torch.utils.data.Dataset):
    """
    This class holds the game experience. It holds a number of shared
    tensors. One for each of the rewards, actions, observations, done
    signals, etc. For each shared tensor the experience is held in a
    row corresponding to each parallelized environment. One
    environment exists for each row in the shared tensors. See the
    DataCollector class for more details on how the data collection is
    parallelized.
    """
    def __init__(self,
        hyps,
        share_tensors=True,
        *args,
        **kwargs
    ):
        """
        Args:
            exp_len: int
                the maximum length of the experience tensors
            batch_size: int
                the number of parallel environments
            inpt_shape: tuple (C, H, W)
                the shape of the observations. channels first,
                then height and width
            seq_len: int
                the length of returned sequences
            randomize_order: bool
                a bool to determine if the data should be
                randomized in the iter. if true, data returned
                from this class's iterable will be in a
                randomized order.
            share_tensors: bool
                if true, each tensor within shared_exp is moved to the
                shared memory for parallel processing
            env_type: str or None
                used to determine if n_targs, n_items, and n_aligned
                should be included in the shared_exp dict
        Members:
            shared_exp: dict
                keys: str
                    rews: torch float tensor (N, L)
                        the rewards collected by the environments
                    obs: torch float tensor (N, L, C, H, W)
                        the observations collected by the environments
                    dones: torch long tensor (N, L)
                        the done signals collected by the environments
                    actns: torch long tensor (N, L)
                        the actions taken during the data collection
                    n_targs: torch long tensor (N, L) or None
                        the number of goal targets if using gordongames
                        environment
                    n_items: torch long tensor (N, L) or None
                        the number of items in the env if using
                        gordongames environment
                    n_aligned: torch long tensor (N, L) or None
                        the number of aligned items in the env if using
                        gordongames environment
        """
        self.hyps = hyps
        self.exp_len = self.hyps["exp_len"]
        self.env_type = self.hyps["env_type"]
        self.batch_size = self.hyps["batch_size"]
        self.inpt_shape = self.hyps["inpt_shape"]
        self.seq_len = self.hyps["seq_len"]
        self.randomize_order = self.hyps["randomize_order"]
        self.roll_data = try_key(self.hyps, "roll_data", True)
        self.share_tensors = share_tensors
        assert self.exp_len > self.seq_len,\
            "sequence length must be less than total experience length"
        
        self.shared_exp = {
            "obs": torch.zeros((
                    self.batch_size,
                    self.exp_len,
                    *self.inpt_shape
                )).float(),
            "rews": torch.zeros((
                    self.batch_size,
                    self.exp_len
                )).float(),
            "dones": torch.zeros((
                    self.batch_size,
                    self.exp_len
                )).long(),
            "actns": torch.zeros((
                    self.batch_size,
                    self.exp_len
                )).long(),
        }
        self.info_keys = [
            "n_targs","n_items","n_aligned","grabs","disp_targs"
        ]
        for key in self.info_keys:
            self.shared_exp[key] = torch.zeros((
                self.batch_size,
                self.exp_len
            )).long()
        if self.share_tensors:
            for key in self.shared_exp.keys():
                self.shared_exp[key].share_memory_()
        self.harvest_exp()
    
    def harvest_exp(self):
        """
        Copys the shared tensors so that the runners can continue
        collecting without changing the data.

        Returns:
            exp: dict of tensors
                deep copies the shared experience
        """
        self.exp = {
            k: v.detach().clone() for k,v in self.shared_exp.items()
        }

    def __len__(self):
        raw_len = len(self.shared_exp["rews"][0]) - self.seq_len
        if self.roll_data:
            return raw_len
        return int(raw_len//self.seq_len)

    def __getitem__(self, idx):
        """
        Returns a chunk of data with the sequence length including all
        environments.

        Args:
            idx: int
                this value is multiplied by the sequence length if
                roll_data is false
        Returns:
            data: dict
                keys: str
                    obs: torch float tensor (N, S, C, H, W)
                    rews: torch float tensor (N, S)
                    dones: torch long tensor (N, S)
                    actns: torch long tensor (N, S)
        """
        if not self.roll_data:
            idx = idx*self.seq_len
        data = dict()
        for key in self.exp.keys():
            data[key] = self.exp[key][:, idx: idx+self.seq_len]
        data["drops"] = self.get_drops(
            self.hyps,
            data["grabs"],
            data["disp_targs"]
        )
        return data

    def __iter__(self):
        """
        Uses a permutation to track which index is next.

        Note that if __iter__ is called a second time, then any
        exiting iterable of this class will also be affected!
        """
        if self.randomize_order:
            self.idx_order = torch.randperm(self.__len__()).long()
        else:
            self.idx_order = torch.arange(self.__len__()).long()
        self.idx = 0
        return self

    def __next__(self):
        """
        Returns:
            data: dict
                keys: str
                    obs: torch float tensor (N, S, C, H, W)
                    rews: torch float tensor (N, S)
                    dones: torch long tensor (N, S)
                    actns: torch long tensor (N, S)
        """
        if not hasattr(self, "idx_order"):
            self.__iter__()
        if self.idx < self.__len__():
            idx = self.idx_order[self.idx]
            self.idx += 1
            return self.__getitem__(idx)
        else:
            raise StopIteration

    @staticmethod
    def get_drops(hyps, grabs, disp_targs):
        """
        Returns a tensor denoting steps in which the agent dropped an
        item. This always means that the player is still on the item
        when the prediction happens.
        
        WARNING: this tensor has been bastardized to simply denote
        when the agent should make a language prediction about n_items.

        The tensor returned is also actually a LongTensor, not a Bool
        Tensor. Assumes 1 means PILE, and 2 means BUTTON and 3 means
        ITEM within the grabs tensor.

        For variants 4 and 7, drops includes any frame in which the
        targets are displayed.

        Args:
            hyps: dict
                the hyperparameters
            grabs: Long Tensor (B,N)
                a tensor denoting the item grabbed by the agent at
                each timestep. Assumes 1 means PILE, and 2 means BUTTON
                and 3 means ITEM
            disp_targs: torch LongTensor (..., N)
                0s denote the environment was not displaying the targets
                anymore. 1s denote the targets were displayed
        Returns:
            drops: Long Tensor (B,N)
                a tensor denoting if the agent dropped an item with a 1,
                0 otherwise. See WARNING in description
        """
        env_types = {"gordongames-v4", "gordongames-v7", "gordongames-v8"}
        if type(grabs) == type(np.asarray([])):
            grabs = torch.from_numpy(grabs).long()
        if not try_key(hyps, "lang_on_drops_only", True):
            return torch.ones_like(grabs)
        drops = grabs.clone().long()

        if hyps["env_type"] in env_types:
            drops[drops>0] = 1
            drops = drops | disp_targs
            return drops
        drops[grabs!=3] = 0
        drops[grabs==3] = 1
        # Looks for situations where the sum drops off to 0
        if len(grabs.shape)==2:
            drops[:, 1:] = drops[:, :-1] - drops[:, 1:]
            drops[:,0] = 0
            drops = torch.roll(drops, 1, -1)
            drops[:,0] = 0
        elif len(grabs.shape)==1:
            drops[1:] = drops[:-1] - drops[1:]
            drops[0] = 0
            drops = torch.roll(drops, 1, -1)
            drops[0] = 0
        drops[drops!=1] = 0
        drops[drops>0] = 1
        # In case less than 5% of the batch are drops, we set the last
        # column to 1
        perc_threshold = try_key(hyps,"drops_perc_threshold",0)
        if drops.sum()<=(perc_threshold*drops.numel()):
            if perc_threshold == 0: perc_threshold = 0.1
            rand = torch.rand_like(drops.float())
            rand[rand<=perc_threshold] = 1
            rand[rand!=1] = 0
            drops = drops | rand.long()
        return drops

class DataCollector:
    """
    This class collects the training data by rolling out multiple
    environments in parallel. It places the data into the shared
    tensors within the experience replay.

    The data collector spawns multiple runners who collect data from
    their respective environments.
    """
    def __init__(self, hyps):
        """
        Creates the runners and initiates the initial data collection.
        Separate from the __init__ function because it's most efficient
        to get the observation size for the experience replay from
        the validation runner which is created internally

        Args:
            hyps: dict
                keys: str
                    batch_size: int
                        the number of parallel environments
                    n_envs: int
                        the number of runners to instantiate
        """
        self.hyps = hyps
        self.batch_size = self.hyps['batch_size']
        # Create gating mechanisms
        self.gate_q = mp.Queue(self.batch_size)
        self.stop_q = mp.Queue(self.batch_size)
        self.val_gate_q = mp.Queue(1)
        self.val_stop_q = mp.Queue(1)
        self.phase_q = mp.Queue(1)
        self.phase_q.put(0)
        # Get observation, actn, and lang shapes
        self.validator = ValidationRunner(
            self.hyps,
            gate_q=self.val_gate_q,
            stop_q=self.val_stop_q,
            phase_q=self.phase_q
        )
        self.validator.create_new_env()
        self.obs_shape = self.validator.env.shape
        self.hyps['inpt_shape'] = self.validator.state_bookmark.shape
        self.hyps["actn_size"] = self.validator.env.actn_size
        lang_range = try_key(
            self.hyps,
            "lang_range",
            self.hyps["targ_range"]
        )
        if lang_range is None: lang_range = self.hyps["targ_range"]
        self.hyps["lang_size"] = lang_range[1]+1
        if int(self.hyps["use_count_words"]) == 0:
            self.hyps["lang_size"] = 3
        elif int(self.hyps["use_count_words"]) == 2:
            self.hyps["lang_size"] = 4
        # Initialize Experience Replay
        self.exp_replay = ExperienceReplay(hyps)
        # Initialize runners
        self.runners = []
        offset = try_key(self.hyps, 'runner_seed_offset', 0)
        # We add one here because the validation environment defaults
        # to the argued seed without offset
        for i in range(1,self.hyps['n_envs']+1):
            seed = self.hyps["seed"] + offset + i
            temp_hyps = {**self.hyps, "seed": seed}
            runner = Runner(
                self.exp_replay.shared_exp,
                temp_hyps,
                self.gate_q,
                self.stop_q,
                self.phase_q
            )
            self.runners.append(runner)

        # Initiate Data Collection
    def init_runner_procs(self, model):
        """
        Initializes the runner processes

        Args:
            model: torch Module
                make sure the model's weights are shared so that they
                can be updated over the course of the training.
        """
        if not hasattr(self, "procs"):
            self.procs = []
        for i in range(len(self.runners)):
            if i > len(self.runners)//2: model = None
            proc = mp.Process(
                target=self.runners[i].run,
                args=(model,)
            )
            proc.start()
            self.procs.append(proc)

    def init_validator_proc(self, model):
        """
        Initializes the validation runner process

        Args:
            model: torch Module
                make sure the model's weights are shared so that they
                can be updated over the course of the training.
        """
        if not hasattr(self, "procs"):
            self.procs = []
        proc = mp.Process(
            target=self.validator.run,
            args=(model,)
        )
        proc.start()
        self.procs.append( proc )

    def await_runners(self):
        for i in range(self.hyps["batch_size"]):
            self.stop_q.get()

    def dispatch_runners(self):
        for i in range(self.hyps["batch_size"]):
            self.gate_q.put(i)

    def await_validator(self):
        self.val_stop_q.get()

    def dispatch_validator(self, epoch):
        self.val_gate_q.put(epoch)

    def terminate_runners(self):
        """
        Includes the validation runner process
        """
        for proc in self.procs:
            proc.terminate()

    def update_phase(self, phase):
        old_phase = self.phase_q.get()
        self.phase_q.put(phase)

def sample_zipfian(hyps):
    """
    A helper function to sample from the zipfian distribution according
    to the hyperparameters.

    Args:
        hyps: dict
            the hyperparameters
    Returns:
        sample: int or None
            the sampled value if zipf_order is not none
    """
    order = try_key(hyps, "zipf_order", None)
    if order is not None and order > 0:
        low, high = hyps["targ_range"]
        return zipfian(low, high, order)
    return None

class Runner:
    def __init__(self, shared_exp, hyps, gate_q, stop_q, phase_q):
        """
        Args:
            hyps: dict
                keys: str
                    "gamma": reward decay coeficient
                    "exp_len": number of steps to be taken in the
                                environment
                    "n_frame_stack": number of frames to stack for
                                     creation of the mdp state
                    "preprocessor": function to preprocess raw
                                    observations
                    "env_type": type of gym environment to be interacted
                                with. Follows OpenAI's gym api.
                    "oracle_type": str
                        the name of the Oracle Class to give the ideal
                        action from the environment
            shared_exp: dict
                keys: str
                vals: shared torch tensors
                    "obs": Collects the MDP states at each timestep t
                    "rews": Collects float rewards collected at each
                            timestep t
                    "dones": Collects the dones collected at each
                             timestep t
                    "actns": Collects actions performed at each
                             timestep t
                    "n_targs": Collects the number of targets for the
                               episode if using gordongames variant
                    "n_items": Collects the number of items over the
                               course of the episode if using
                               gordongames variant
                    "n_aligned": Collects the number of items aligned
                               with targets over the course of the
                               episode if using gordongames variant
                    "grabs": Collects information into whether the agent
                            is grabbing or not.
            gate_q: multiprocessing Queue.
                Allows main process to control when rollouts should be
                collected.
            stop_q: multiprocessing Queue.
                Used to indicate to main process that a rollout has
                been collected.
            phase_q: multiprocessing Queue.
                Used to indicate from the main process that the phase
                has changed.
        """

        self.hyps = hyps
        self.shared_exp = shared_exp
        self.gate_q = gate_q
        self.stop_q = stop_q
        self.phase_q = phase_q
        self.obs_deque = deque(maxlen=hyps['n_frame_stack'])
        env_type = self.hyps['env_type']
        self.oracle = globals()[self.hyps["oracle_type"]](env_type)

    def create_new_env(self, n_targs=None):
        """
        This function simplifies making a new environment and storing
        all the variables associated with it. It uses the language
        target range or the action target range depending on the phase
        of the experiment. Phase 0 means language, anything else means
        action.
        """
        if self.phase == 0:
            # Set defaults
            if try_key(self.hyps, "actn_range", None) is None:
                self.hyps["actn_range"] = self.hyps["targ_range"]
            if try_key(self.hyps, "lang_range", None) is None:
                self.hyps["lang_range"] = self.hyps["targ_range"]
            # Set environment targ range to language range
            self.hyps["targ_range"] = self.hyps["lang_range"]
        else:
            # Set environment targ range to action range
            self.hyps["targ_range"] = self.hyps["actn_range"]
        hyps = {**self.hyps}
        if n_targs is not None:
            hyps["targ_range"] =  (n_targs, n_targs)
        self.env = SequentialEnvironment(**hyps)
        state = next_state(
            self.env,
            self.obs_deque,
            obs=None,
            reset=True,
            n_targs=sample_zipfian(hyps)
        )
        self.state_bookmark = state
        self.h_bookmark = None
        return state

    def run(self, model=None):
        """
        run is the entry function to begin collecting rollouts from the
        environment. gate_q indicates when to begin collecting a
        rollout and is controlled from the main process. The stop_q is
        used to indicate to the main process that a new rollout has
        been collected.
        """
        self.phase = try_key(self.hyps, "first_phase", 0)
        self.model = model
        if model is None:
            self.model = NullModel(**self.hyps)
        state = self.create_new_env()
        self.ep_rew = 0
        while True:
            with torch.no_grad():
                # Await collection signal from main proc
                idx = self.gate_q.get()
                # Change phase if necessary
                phase = self.phase_q.get()
                self.phase_q.put(phase)
                if self.phase != phase:
                    self.phase = phase
                    state = self.create_new_env()
                # Collect rollout
                self.rollout(idx, self.model)
                # Signals to main process that data has been collected
                self.stop_q.put(idx)

    def rollout(self, idx, model):
        """
        rollout handles the actual rollout of the environment. It runs
        for n steps in the game. Collected data is placed into the
        shared_exp dict in the row corresponding to the argued idx.

        Args:
            idx: int
                identification number distinguishing the row of the
                shared_exp designated for this rollout
        """
        model.eval()
        if try_key(self.hyps, "reset_trn_env", False):
            state = next_state(
                self.env,
                self.obs_deque,
                obs=None,
                reset=True,
                n_targs=sample_zipfian(self.hyps)
            )
            model.reset(1)
        else:
            state = self.state_bookmark
            if self.h_bookmark is None:
                model.reset(1)
            else:
                model.h, model.c = self.h_bookmark
        exp_len = self.hyps['exp_len']
        for i in range(exp_len):
            # Collect the state of the environment
            t_state = torch.FloatTensor(state) # (C, H, W)
            self.shared_exp["obs"][idx,i] = t_state
            # Get actn
            actn_targ = self.oracle(self.env, state=t_state) # int
            if model.trn_whls == 1:
                actn = actn_targ
            else:
                inpt = t_state[None].to(DEVICE)
                actn_pred, _ = model.step(inpt)
                actn = sample_action(
                    F.softmax(actn_pred, dim=-1)
                ).item()
            # Step the environment
            obs, rew, done, info = self.env.step(actn)
            # Collect data
            self.shared_exp['rews'][idx,i] = rew
            self.shared_exp['dones'][idx,i] = float(done)
            self.shared_exp['actns'][idx,i] = actn_targ
            self.shared_exp["n_items"][idx,i] = info["n_items"]
            self.shared_exp["n_targs"][idx,i] = info["n_targs"]
            self.shared_exp["n_aligned"][idx,i] = info["n_aligned"]
            self.shared_exp["grabs"][idx,i] = int(info["grab"])
            self.shared_exp["disp_targs"][idx,i] = int(info["disp_targs"])

            state = next_state(
                self.env,
                self.obs_deque,
                obs=obs,
                reset=done,
                n_targs=sample_zipfian(self.hyps)
            )
            if done: model.reset(1)
        self.state_bookmark = state
        if hasattr(model, "h"):
            self.h_bookmark = (model.h, model.c)

class ValidationRunner(Runner):
    def __init__(self, hyps,
                       gate_q=None,
                       stop_q=None,
                       phase_q=None,
                       phase=0):
        """
        Args:
            hyps: dict
                keys: str
                    "gamma": reward decay coeficient
                    "exp_len": number of steps to be taken in the
                                environment
                    "n_frame_stack": number of frames to stack for
                                     creation of the mdp state
                    "preprocessor": function to preprocess raw
                                    observations
                    "env_type": type of gym environment to be interacted
                                with. Follows OpenAI's gym api.
                    "oracle_type": str
                        the name of the Oracle Class to give the ideal
                        action from the environment
            gate_q: multiprocessing Queue.
                Allows main process to control when rollouts should be
                collected.
            stop_q: multiprocessing Queue.
                Used to indicate to main process that a rollout has
                been collected.
            phase_q: multiprocessing Queue.
                Used to indicate from the main process that the phase
                has changed.
            phase: int
                the initial phase
        """
        self.hyps = {**hyps}
        self.gate_q = gate_q
        self.stop_q = stop_q
        self.phase_q = phase_q
        if try_key(self.hyps, "val_targ_range", None) is None:
            self.hyps["val_targ_range"] = self.hyps["targ_range"]
        self.hyps["targ_range"] = self.hyps["val_targ_range"]
        self.hyps["actn_range"] = self.hyps["val_targ_range"]
        self.hyps["lang_range"] = self.hyps["val_targ_range"]
        print("Validation runner target range:",self.hyps["targ_range"])
        self.phase = phase
        self.obs_deque = deque(maxlen=hyps['n_frame_stack'])
        self.oracle = globals()[self.hyps["oracle_type"]](**self.hyps)

    def rollout(self, epoch, model, *args, **kwargs):
        """
        rollout handles running the environment using the model's
        predictions to direct the game's MDP. This rollout function
        steps through each target quantity and runs the episodes to
        completion. It then saves the language predictions and the
        outcomes of the episodes.

        Args:
            epoch: int
                the current epoch
            model: torch Module
        """
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        model.eval()

        # Reset every on every validation run
        rng = range(
            self.hyps["targ_range"][0],
            self.hyps["targ_range"][1]+1
        )
        for n_targs in rng:
            state = self.create_new_env(n_targs=n_targs)
            model.reset(1)

            data = self.collect_data(model, state, n_targs)
            lang_labels = get_lang_labels(
                data["n_items"],
                data["n_targs"],
                max_label=model.lang_size-1,
                use_count_words=self.hyps["use_count_words"]
            )
            drops = ExperienceReplay.get_drops(
                self.hyps,
                data["grabs"],
                data["disp_targs"]
            )
            self.save_lang_data(
                data, lang_labels, drops, epoch, self.phase
            )
            self.save_ep_data(data, epoch, self.phase)

    def save_lang_data(self,
                       data,
                       labels,
                       drops,
                       epoch,
                       phase,
                       save_name="validation_lang.csv"):
        """
        Saves the stats at the end of each episode collected from the
        rollouts. Saves the data as a dataframe called `save_name`
        within the model's save_folder.

        Args:
            data: dict
                lang_preds: torch FloatTensor (K,N, L)
                    the language prediction logits. K is the number
                    of language outputs within the model
                n_targs: torch LongTensor (N,)
                    the number of target objects on the grid at this step
                    of the episode
                n_items: torch LongTensor (N,)
                    the number of item objects on the grid at this step
                    of the episode
                n_aligned: torch LongTensor (N,)
                    the number of aligned item objects on the grid at
                    this step of the episode
                dones: torch LongTensor (N,)
                    1 if episode ended on this step, 0 otherwise
            epoch: int
            phase: int
            save_name: str
        """
        inpts = {
            "n_items":None,
            "n_targs":None,
            "n_aligned":None,
            "pred":None,
            "label":None,
            "done":None
        }
        idxs = drops>=1
        if idxs.float().sum() <= 1: return
        lang = torch.argmax(data["lang_preds"].mean(0), dim=-1) # (N,)

        inpts["pred"] = lang[idxs]
        inpts["label"] = labels[idxs]
        inpts["n_targs"] = data["n_targs"][idxs]
        inpts["n_items"] = data["n_items"][idxs]
        inpts["n_aligned"] = data["n_aligned"][idxs]
        inpts["done"] = data["dones"][idxs]
        inpts = {k:v.cpu().data.numpy() for k,v in inpts.items()}

        df = pd.DataFrame(inpts)
        df["epoch"] = epoch
        df["phase"] = self.phase
        path = os.path.join(
            self.hyps["save_folder"],
            save_name
        )
        header = not os.path.exists(path)
        df.to_csv(
            path,
            sep=",",
            header=header,
            mode="a"
        )

    def save_ep_data(self,
                   data,
                   epoch,
                   phase,
                   save_name="validation_stats.csv"):
        """
        Saves the stats at the end of each episode collected from the
        rollouts. Saves the data as a dataframe called `save_name`
        within the model's save_folder.

        Args:
            data: dict
            epoch: int
            phase: int
            save_name: str
        """
        keys = ["n_items", "n_targs", "n_aligned"]
        dones = data["dones"].reshape(-1)
        inpts = {
            key: data[key].reshape(-1)[dones==1] for key in keys
        }
        inpts = {k:v.cpu().data.numpy() for k,v in inpts.items()}
        df = pd.DataFrame(inpts)
        df["epoch"] = epoch
        df["phase"] = self.phase
        path = os.path.join(
            self.hyps["save_folder"],
            save_name
        )
        header = not os.path.exists(path)
        df.to_csv(
            path,
            sep=",",
            header=header,
            mode="a"
        )

    def collect_data(self, model, state, n_targs=None):
        """
        Performs the actual rollouts using the model

        Args:
            model: Module
            state: ndarray ? I think
            n_targs: int
                the number of targets for the environment to display
        Returns:
            data: dict
                keys: str
                vals: shared torch tensors
                    actn_preds: float tensor (N, K)
                        Collects the predictions of the model for each
                        timestep t
                    lang_preds: float tensor (N, K)
                        Collects the predictions of the model for each
                        timestep t
                    actn_targs: long tensor (N,)
                        Collects the oracle actions at each timestep t
                    rews: float tensor (N,)
                        Collects the reward at each timestep t
                    dones: long tensor (N,)
                        Collects the done signals at each timestep t
                    n_targs: long tensor (N,)
                        Collects the number of targets in the episode
                        only relevant if using a gordongames
                        environment variant
                    n_items: long tensor (N,)
                        Collects the number of items over the course of
                        the episode. only relevant if using a
                        gordongames environment variant
                    n_aligned: long tensor (N,)
                        Collects the number of items that are aligned
                        with targets over the course of the episode.
                        only relevant if using a gordongames
                        environment variant
        """
        data = {
            "states":[],
            "actn_preds":[],
            "lang_preds":[],
            "actn_targs":[],
            "rews":[],
            "dones":[],
            "n_targs":[],
            "n_items":[],
            "n_aligned":[],
            "grabs":[],
            "disp_targs":[],
        }
        ep_count = 0
        n_eps = try_key(self.hyps,"n_eval_eps",10)
        while ep_count < n_eps:
            # Collect the state of the environment
            data["states"].append(state)
            t_state = torch.FloatTensor(state) # (C, H, W)
            # Get action prediction
            inpt = t_state[None].to(DEVICE)
            actn_pred, lang_pred = model.step(inpt)
            data["actn_preds"].append(actn_pred)
            # Batch Size is only ever 1
            # lang_pred: (1,1,L)
            if model.n_lang_denses == 1:
                lang = lang_pred[0].unsqueeze(0)
            else:
                lang = torch.stack(lang_pred, dim=0)
            # lang: (N,1,L) where N is number of lang models
            data["lang_preds"].append(lang)
            if try_key(self.hyps, "val_max_actn", False):
                actn = torch.argmax(actn_pred[0]).item()
            else:
                actn = sample_action(
                    F.softmax(actn_pred, dim=-1)
                ).item()
            # get target action
            targ = self.oracle(self.env)
            data["actn_targs"].append(targ)
            # Step the environment (use oracle if phase 0)
            if self.phase == 0: actn = targ
            obs, rew, done, info = self.env.step(actn)
            state = next_state(
                self.env,
                self.obs_deque,
                obs=obs,
                reset=done,
                n_targs=n_targs
            )
            if done: model.reset(1)
            data["dones"].append(int(done))
            data["rews"].append(rew)
            data["n_targs"].append(info["n_targs"])
            data["n_items"].append(info["n_items"])
            data["n_aligned"].append(info["n_aligned"])
            data["grabs"].append(info["grab"])
            data["disp_targs"].append(info["disp_targs"])
            if self.hyps["render"]: self.env.render()
            ep_count += int(done)
        self.state_bookmark = state
        if hasattr(model, "h"):
            self.h_bookmark = (model.h.data, model.c.data)
        # S stands for the collected sequence
        data["actn_preds"] = torch.cat(data["actn_preds"], dim=0) #(S,A)
        data["lang_preds"] = torch.cat(data["lang_preds"], dim=1) #(N,S,L)
        data["actn_targs"] = torch.LongTensor(data["actn_targs"])
        data["dones"] = torch.LongTensor(data["dones"])
        data["grabs"] = torch.LongTensor(data["grabs"])
        data["rews"] = torch.FloatTensor(data["rews"])
        data["n_targs"] = torch.LongTensor(data["n_targs"])
        data["n_items"] = torch.LongTensor(data["n_items"])
        data["n_aligned"] = torch.LongTensor(data["n_aligned"])
        data["disp_targs"] = torch.LongTensor(data["disp_targs"])
        return data

