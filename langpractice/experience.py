import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
from supervised_gym.envs import SequentialEnvironment
from supervised_gym.oracles import *
from supervised_gym.utils.utils import try_key, sample_action
from collections import deque

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")

def next_state(env, obs_deque, obs, reset):
    """
    Get the next state of the environment.

    env - environment of interest
    obs_deq - deque of the past n observations
    obs - ndarray returned from the most recent step of the environment
    reset - boolean denoting the reset signal from the most recent step 
            of the environment
    """

    if reset or obs is None:
        obs = env.reset()
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
        exp_len,
        batch_size,
        inpt_shape,
        seq_len,
        randomize_order,
        share_tensors=True,
        env_type=None,
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
        self.exp_len = exp_len
        self.batch_size = batch_size
        self.inpt_shape = inpt_shape
        self.seq_len = seq_len
        self.randomize_order = randomize_order
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
        if "gordongames" in env_type:
            keys = ["n_targs", "n_items", "n_aligned"]
            for key in keys:
                self.shared_exp[key] = torch.zeros((
                    self.batch_size,
                    self.exp_len
                )).long()
        if self.share_tensors:
            for key in self.shared_exp.keys():
                self.shared_exp[key].share_memory_()

    def __len__(self):
        return len(self.shared_exp["rews"][0]) - self.seq_len

    def __getitem__(self, idx):
        """
        Returns a chunk of data with the sequence length including all
        environments.

        Args:
            idx: int
        Returns:
            data: dict
                keys: str
                    obs: torch float tensor (N, S, C, H, W)
                    rews: torch float tensor (N, S)
                    dones: torch long tensor (N, S)
                    actns: torch long tensor (N, S)
        """
        data = dict()
        for key in self.shared_exp.keys():
            data[key] = self.shared_exp[key][:, idx: idx+self.seq_len]
        return data

    def __iter__(self):
        """
        Uses a permutation to track which index is next.

        Note that if __iter__ is called a second time, then any
        exiting iterable of this class will also be affected!
        """
        if self.randomize_order:
            self.idx_order = torch.arange(self.__len__()).long()
        else:
            self.idx_order = torch.randperm(self.__len__()).long()
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
            idx = self.idx
            self.idx += 1
            return self.__getitem__(idx)
        else:
            raise StopIteration

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
        # Check keys
        self.hyps = hyps
        self.batch_size = self.hyps['batch_size']
        # Get observation shape
        self.val_runner = ValidationRunner(self.hyps)
        self.obs_shape = self.val_runner.env.shape
        self.hyps['inpt_shape'] = self.val_runner.state_bookmark.shape
        self.hyps["actn_size"] = self.val_runner.env.actn_size
        # Create gating mechanisms
        self.gate_q = mp.Queue(self.batch_size)
        self.stop_q = mp.Queue(self.batch_size)
        # Initialize Experience Replay
        self.exp_replay = ExperienceReplay(**hyps)
        # Initialize runners
        self.runners = []
        offset = try_key(self.hyps, 'runner_seed_offset', 0)
        for i in range(self.hyps['n_envs']):
            seed = self.hyps["seed"] + offset + i
            temp_hyps = {**self.hyps, "seed": seed}
            runner = Runner(
                self.exp_replay.shared_exp, 
                self.hyps,
                self.gate_q,
                self.stop_q,
            )
            self.runners.append(runner)
        # Initiate Data Collection
        self.procs = []
        for i in range(len(self.runners)):
            proc = mp.Process(target=self.runners[i].run)
            self.procs.append(proc)
            proc.start()

    def await_runners(self):
        for i in range(self.hyps["batch_size"]):
            self.stop_q.get()

    def dispatch_runners(self):
        for i in range(self.hyps["batch_size"]):
            self.gate_q.put(i)

    def terminate_runners(self):
        for proc in self.procs:
            proc.terminate()

class Runner:
    def __init__(self, shared_exp, hyps, gate_q, stop_q):
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
            gate_q: multiprocessing Queue.
                Allows main process to control when rollouts should be
                collected.
            stop_q: multiprocessing Queue.
                Used to indicate to main process that a rollout has
                been collected.
        """

        self.hyps = hyps
        self.shared_exp = shared_exp
        self.gate_q = gate_q
        self.stop_q = stop_q
        self.obs_deque = deque(maxlen=hyps['n_frame_stack'])
        env_type = self.hyps['env_type']
        self.oracle = globals()[self.hyps["oracle_type"]](env_type)

    def run(self):
        """
        run is the entry function to begin collecting rollouts from the
        environment. gate_q indicates when to begin collecting a
        rollout and is controlled from the main process. The stop_q is
        used to indicate to the main process that a new rollout has
        been collected.
        """
        self.env = SequentialEnvironment(**self.hyps)
        state = next_state(
            self.env,
            self.obs_deque,
            obs=None,
            reset=True
        )
        self.state_bookmark = state
        self.ep_rew = 0
        while True:
            idx = self.gate_q.get() # Opened from main process
            self.rollout(idx)
            self.stop_q.put(idx) # Signals to main process that data has been collected

    def rollout(self, idx):
        """
        rollout handles the actual rollout of the environment. It runs
        for n steps in the game. Collected data is placed into the
        shared_exp dict in the row corresponding to the argued idx.

        Args:
            idx: int
                identification number distinguishing the row of the
                shared_exp designated for this rollout
        """
        state = self.state_bookmark
        exp_len = self.hyps['exp_len']
        for i in range(exp_len):
            # Collect the state of the environment
            t_state = torch.FloatTensor(state) # (C, H, W)
            self.shared_exp["obs"][idx,i] = t_state
            # Get oracle's actn
            actn = self.oracle(self.env, state=t_state) # int
            # Step the environment
            obs, rew, done, info = self.env.step(actn)
            # Collect data
            self.shared_exp['rews'][idx,i] = rew
            self.shared_exp['dones'][idx,i] = float(done)
            self.shared_exp['actns'][idx,i] = actn
            if "n_targs" in info:
                self.shared_exp["n_targs"][idx,i] = info["n_targs"]
                self.shared_exp["n_items"][idx,i] = info["n_items"]
                self.shared_exp["n_aligned"][idx,i] = info["n_aligned"]
            state = next_state(
                self.env,
                self.obs_deque,
                obs=obs,
                reset=done
            )
        self.state_bookmark = state

class ValidationRunner(Runner):
    def __init__(self, hyps):
        """
        Args:
            hyps: dict
                keys: str
                    "gamma": reward decay coeficient
                    "n_frame_stack": number of frames to stack for
                                     creation of the mdp state
                    "preprocessor": function to preprocess raw
                                    observations
                    "env_type": type of gym environment to be interacted
                                with. Follows OpenAI's gym api.
        """

        self.hyps = hyps
        self.obs_deque = deque(maxlen=hyps['n_frame_stack'])
        self.env = SequentialEnvironment(**self.hyps)
        state = next_state(
            self.env,
            self.obs_deque,
            obs=None,
            reset=True
        )
        self.state_bookmark = state
        self.h_bookmark = None
        self.ep_rew = 0
        self.oracle = globals()[self.hyps["oracle_type"]](**self.hyps)

    def rollout(self, model, n_tsteps, n_eps=None):
        """
        rollout handles the actual rollout of the environment. It runs
        for n steps in the game using the model to determine actions
        in the environment. Returns collected data from model and
        what the oracle would have done.

        Args:
            model: torch Module
            n_tsteps: int or None
                number of steps to rollout. must not be None if n_eps
                is None
            n_eps: int or None
                number of episodes to rollout. must not be None if
                n_tsteps is None
        Returns:
            data: dict
                keys: str
                vals: shared torch tensors
                    logits: float tensor (N, K)
                        Collects the predictions of the model for each
                        timestep t
                    targs: long tensor (N,)
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
            "states": [],
            "logits": [],
            "targs": [],
            "rews":  [],
            "dones": [],
            "n_targs": None,
            "n_items": None,
            "n_aligned": None,
        }
        if "gordongames" in self.hyps['env_type']:
            data["n_targs"] = []
            data["n_items"] = []
            data["n_aligned"] = []
        model.eval()
        state = self.state_bookmark
        if self.h_bookmark is None:
            model.reset(1)
        else:
            model.h, model.c = self.h_bookmark
        prev_h = self.h_bookmark
        with torch.no_grad():
            loop_count = 0
            max_count = n_tsteps if n_eps is None else n_eps
            while loop_count < max_count:
                # Collect the state of the environment
                t_state = torch.FloatTensor(state) # (C, H, W)
                data["states"].append(t_state)
                # Get action prediction
                logits = model.step(t_state[None].to(DEVICE))
                data["logits"].append(logits)
                actn = sample_action(F.softmax(logits, dim=-1)).item()
                # get target action
                targ = self.oracle(self.env)
                data["targs"].append(targ)
                # Step the environment
                obs, rew, done, info = self.env.step(actn)
                state = next_state(
                    self.env,
                    self.obs_deque,
                    obs=obs,
                    reset=done
                )
                if done: model.reset(1)
                data["dones"].append(int(done))
                data["rews"].append(rew)
                if "n_targs" in info and data["n_targs"] is not None:
                    data["n_targs"].append(info["n_targs"])
                    data["n_items"].append(info["n_items"])
                    data["n_aligned"].append(info["n_aligned"])
                if self.hyps["render"]: self.env.render()
                loop_count += int(n_tsteps is not None or done)
        self.state_bookmark = state
        self.h_bookmark = (model.h.data, model.c.data)
        data["logits"] = torch.cat(data["logits"], dim=0)
        data["targs"] = torch.LongTensor(data["targs"])
        data["dones"] = torch.LongTensor(data["dones"])
        data["rews"] = torch.FloatTensor(data["rews"])
        if data["n_targs"] is not None:
            data["n_targs"] = torch.LongTensor(data["n_targs"])
            data["n_items"] = torch.LongTensor(data["n_items"])
            data["n_aligned"] = torch.LongTensor(data["n_aligned"])
        return data

