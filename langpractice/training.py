from langpractice.experience import ExperienceReplay, DataCollector
from langpractice.models import * # SimpleCNN, SimpleLSTM
from langpractice.recorders import Recorder
from langpractice.utils.save_io import load_checkpoint
from langpractice.utils.utils import try_key
from langpractice.utils.training import get_resume_checkpt

from torch.optim import Adam, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import CrossEntropyLoss
from PIL import Image
import torch
import numpy as np
import time
from tqdm import tqdm
import copy

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")

def train(rank, hyps, verbose=True):
    """
    This is the main training function. Argue a set of hyperparameters
    and this function will train a model to solve an openai gym task
    given an AI oracle.

    Args:
        rank: int
            the index of the distributed training system.
        hyps: dict
            a dict of hyperparams
            keys: str
            vals: object
        verbose: bool
            determines if the function should print status updates
    """
    # If resuming, hyperparameters are updated appropriately
    _, hyps = get_resume_checkpt(hyps)
    # Set random seeds
    hyps['seed'] = try_key(hyps,'seed', int(time.time()))
    torch.manual_seed(hyps["seed"])
    np.random.seed(hyps["seed"])
    # Initialize Data Collector
    # DataCollector's Initializer does Important changes to hyps
    data_collector = DataCollector(hyps)
    # Initialize model
    model = make_model(hyps)
    model.cuda()
    shared_model = None
    if try_key(hyps, "trn_whls_epoch", None) is not None:
        shared_model = model
        shared_model.share_memory()
        print("Sharing model with runners")
    # Begin collecting data
    data_collector.init_runner_procs(shared_model)
    data_collector.dispatch_runners()
    # Record experiment settings
    recorder = Recorder(hyps, model)
    # initialize trainer
    trainer = Trainer(hyps, model, recorder, verbose=verbose)
    if not try_key(hyps,"skip_first_phase",False) and trainer.phase == 0:
        # Loop training
        exp_name = hyps["exp_name"]
        n_epochs = hyps["lang_epochs"] if exp_name != "test" else 2
        training_loop(
            n_epochs,
            data_collector,
            trainer,
            model,
            verbose=verbose
        )
    # Update phase accross training
    trainer.phase = hyps["second_phase"]
    data_collector.await_runners()
    data_collector.update_phase(trainer.phase)
    data_collector.dispatch_runners()
    # Fresh optimizer
    trainer.set_optimizer_and_scheduler(
        model,
        hyps["optim_type"],
        hyps["lr"],
        try_key(hyps, "resume_folder", None)
    )
    n_epochs = hyps["actn_epochs"] if hyps["exp_name"] != "test" else 2
    s = "\n\nBeginning Second Phase " + str(trainer.phase)
    recorder.write_to_log(s)
    print(s)
    training_loop(
        n_epochs,
        data_collector,
        trainer,
        model,
        verbose=verbose
    )
    data_collector.terminate_runners()
    trainer.end_training()

def make_model(hyps):
    """
    Makes the model. The model type specified in the hyperparams must
    be imported into the global scope.

    Args:
        hyps: dict
            dict of hyperparameters. See `README.md` for details
    """
    model = globals()[hyps["model_type"]](**hyps).to(DEVICE)
    folder = try_key(hyps, "resume_folder", None)
    if folder is not None and folder != "":
        checkpt, _ = get_resume_checkpt(hyps, in_place=False)
        model.load_state_dict(checkpt["state_dict"])
    return model

def resume_epoch(trainer):
    """
    If the training is resuming from a checkpoint and the phase of the
    checkpoint matches the phase of the trainer, then this function
    returns the epoch of the resumed checkpoint

    Args:
        trainer: Trainer
            a Trainer object. Must have valid hyps member
    Returns:
        epoch: int
            if the phase of the trainer and the checkpoint match, this
            will be the epoch of the checkpoint. In all other cases,
            defaults to 0.
    """
    folder = try_key(trainer.hyps, "resume_folder", "")
    if folder is not None and folder != "":
        checkpt = load_checkpoint(folder)
        if checkpt["phase"] == trainer.phase:
            return try_key(checkpt,"epoch",0)
    return 0

def training_loop(n_epochs,data_collector,trainer,model,verbose=True):
    """
    The epoch level training loop.

    Args:
        n_epochs: int
            the number of epochs to train for
        data_collector: DataCollector
        trainer: Trainer
        model: Model
    """
    if trainer.hyps["exp_name"] == "test":
        trainer.hyps["n_val_samples"] = 1
    # Potentially modify starting epoch for resumption of previous
    # training. Defaults to 0 if not resuming or not same phase
    start_epoch = resume_epoch(trainer)
    for epoch in range(start_epoch, n_epochs):
        if verbose:
            print()
            print("Phase:",
                trainer.phase,
                "-- Epoch",
                epoch,
                "--",
                trainer.hyps["save_folder"]
            )
        # Run environments, automatically fills experience replay's
        # shared_exp tensors
        data_collector.await_runners()
        trainer.train(model, data_collector.exp_replay)
        data_collector.dispatch_runners()
        if verbose: print("\nValidating")
        n_targs = None
        for val_sample in tqdm(range(trainer.hyps["n_val_samples"])):
            if try_key(trainer.hyps, "isolate_val_targs", True):
                n_targs = val_sample + 1
            trainer.validate(epoch,
                model,
                data_collector,
                n_targs=n_targs
            )
        trainer.end_epoch(epoch)
        trn_whls = try_key(trainer.hyps, "trn_whls_epoch", None)
        trn_whls_off = trn_whls is not None and epoch >= trn_whls
        if trainer.phase != 0 and trn_whls_off:
            model.trn_whls = 0

class Trainer:
    """
    This class handles the training of the model.
    """
    def __init__(self, hyps, model, recorder, verbose=True):
        """
        Args:
            hyps: dict
                keys: str
                vals: object
            model: torch.Module
            recorder: Recorder
                an object for recording the details of the experiment
            verbose: bool
                if true, some functions will print updates to the
                console
        """
        self.hyps = hyps
        self.model = model
        self.recorder = recorder
        self.verbose = verbose
        #phase: int [0,1,2]
        #    the phase of the training (is it (0) training the
        #    language network or (1) training the action network or
        #    (2) both together)
        self.phase = self.init_phase()
        self.set_optimizer_and_scheduler(
            self.model,
            self.hyps["optim_type"],
            self.hyps["lr"],
            try_key(self.hyps, "resume_folder", None)
        )
        self.loss_fxn = globals()[self.hyps["loss_fxn"]]()

    def init_phase(self):
        """
        Initializes the phase of the training depending on if the
        training is resuming from a checkpoint.

        Returns:
            phase: int
                either 0 or the phase that the resume model left off at
                during its previous training
        """
        folder = try_key(self.hyps, "resume_folder", "")
        if folder is not None and folder != "":
            checkpt = load_checkpoint(folder)
            return try_key(checkpt, "phase", 0)
        return 0

    def set_optimizer_and_scheduler(self,
                                    model,
                                    optim_type,
                                    lr,
                                    resume_folder=""):
        """
        Initializes an optimizer using the model parameters and the
        hyperparameters. Also sets a scheduler for the optimizer's
        learning rate.

        If a resume_folder is argued and the phase
        of the resume folder matches the current phase, then the
        optim_dict will be loaded from the resume folder.

        Args:
            model: Model or torch.Module
                any object that implements a `.parameters()` member
                function that returns a sequence of torch.Parameters
            optim_type: str (one of [Adam, RMSprop])
                the type of optimizer. 
            lr: float
                the learning rate
            resume_folder: str (optional)
                a model folder to resume training from. if a valid
                path is argued and the phase of the last checkpt is
                the same as the current phase, the optimizer loads
                thes saved optim_dict
        Returns:
            optim: torch optimizer
                the model optimizer
        """
        self.optim = globals()[optim_type](
            list(model.parameters()),
            lr=lr,
            weight_decay=self.hyps["l2"]
        )
        if resume_folder is not None and resume_folder != "":
            checkpt = load_checkpoint(resume_folder)
            # If same phase, then we want to load the state dict
            # otherwise it means that we originally resumed from phase
            # 0 and now we're past that so we won't want to load the sd
            if try_key(checkpt, "phase", None) == self.phase:
                self.optim.load_state_dict(checkpt["optim_dict"])
            elif try_key(checkpt["stats"],"phase",None) == self.phase:
                self.optim.load_state_dict(checkpt["optim_dict"])

        self.scheduler = ReduceLROnPlateau(
            self.optim,
            mode='min',
            factor=try_key(self.hyps,"factor", 0.9),
            patience=try_key(self.hyps, "patience", 10),
            threshold=try_key(self.hyps, "threshold", 0.0001),
            min_lr=try_key(self.hyps, "min_lr", 0),
            verbose=self.verbose
        )

    def reset_model(self, model, batch_size):
        """
        Determines what type of reset to do. If the data is provided
        in a random order, the model is simply reset. If, however,
        the data is provided in sequence, we must store the h value
        from the first forward loop in the last training loop.
        """
        if self.hyps["randomize_order"]:
            model.reset(batch_size=batch_size)
        else:
            model.reset_to_step(step=1)

    def train(self, model, data_iter):
        """
        This function handles the actual training. It loops through the
        available data from the experience replay to train the model.

        Args:
            model: torch.Module
                the model to be trained
            data_iter: iterable
                an iterable of the collected experience/data. each 
                iteration must return a dict of data with the keys:
                    obs: torch Float Tensor (N, S, C, H, W)
                    actns: torch Long Tensor (N,S)
                    dones: torch Long Tensor (N,S)
                    n_targs: None or torch LongTensor (N,S)
                The iter must also implement the __len__ member so that
                the data can be easily looped through.
        """
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        model.train()
        model.reset(self.hyps['batch_size'])
        for i,data in enumerate(data_iter):
            iter_start = time.time()
            self.optim.zero_grad()
            obs =   data["obs"]
            actns = data["actns"]
            dones = data["dones"]
            drops = data["drops"]
            if self.hyps["env_type"]=="gordongames-v4":
                drops = torch.ones_like(drops).long()
            n_items = data["n_items"]
            n_targs = data["n_targs"]
            labels = self.get_lang_labels(
                n_items,
                n_targs,
                max_label=model.lang_size-1
            )

            self.reset_model(model, len(obs))
            # model uses dones if it is recurrent
            logits, langs = model(
                obs.to(DEVICE),
                dones.to(DEVICE)
            )

            loss,accs = self.get_loss_and_accs(
                logits=logits,
                langs=langs,
                actns=actns.flatten(),
                labels=labels.flatten(),
                drops=drops.flatten(),
                n_targs=n_targs.flatten(),
                prepender="train"
            )
            # Backprop and update
            loss.backward()
            self.optim.step()
            # Calc acc
            # Record metrics
            metrics = {
                "train_loss": loss.item(),
                **accs}
            self.recorder.track_loop(metrics)
            key = "train_lang_acc" if self.phase==0 else "train_actn_acc"
            self.print_loop(
                i,
                len(data_iter),
                loss.item(),
                accs[key],
                iter_start
            )
            if self.hyps["exp_name"] == "test" and i >= 2: break
        key = "train_lang_loss" if self.phase==0 else "train_actn_loss"
        self.scheduler.step(
            np.mean(self.recorder.metrics[key])
        )

    def get_lang_labels(self, n_items, n_targs, max_label):
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
        if int(self.hyps["use_count_words"]) == 0:
            labels[n_items<n_targs] = 0
            labels[n_items==n_targs] = 1
            labels[n_items>n_targs] = 2
        elif int(self.hyps["use_count_words"]) == 2:
            labels = self.get_piraha_labels(labels, n_items, n_targs)
        return labels

    def get_piraha_labels(self, labels, n_items, n_targs):
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

    def get_loss_and_accs(self,
                          logits,
                          langs,
                          actns,
                          labels,
                          drops,
                          n_targs,
                          prepender):
        """
        Calculates the loss and accuracies depending on the phase of
        the training.

            Phase 0: language loss when agent drops an item
            Phase 1: action loss at all steps in rollout
            Phase 2: lang and action loss at all steps in rollout

        Args:
            logits: torch FloatTensor (..., A)
                action predictions
            langs: sequence of torch FloatTensors [(N, L),(N, L),...]
                a list of language predictions
            actns: torch LongTensor (N,)
                action labels
            labels: torch LongTensor (N,)
                language labels
            drops: torch LongTensor (N,)
                1s denote steps in which the agent dropped an item, 0s
                denote all other steps
            n_targs: torch LongTensor (N,)
                the number of target objects on the grid at this step
                of the episode
        Returns:
            loss: torch float tensor (1,)
                the appropriate loss for the phase
            accs: dict
                keys: str
                vals: float
                    the appropriate label accuracies depending on the
                    phase
        """
        logits = logits.reshape(-1, logits.shape[-1])
        # Phase 0: language labels when agent drops an item
        # Phase 1: action labels at all steps in rollout
        # Phase 2: lang and action labels at all steps in rollout
        loss = 0
        lang_accs = {}
        if self.phase == 0 or self.phase == 2:
            accs_array = []
            idxs = drops==1
            labels = labels[idxs]
            temp_targs = n_targs[idxs]
            for lang in langs:
                lang = lang.reshape(-1, lang.shape[-1])
                lang = lang[idxs]
                if len(lang)==0:
                    print("lang:", lang)
                    print("drops:", drops)
                    continue
                labels = labels.to(DEVICE)
                loss += self.loss_fxn(lang, labels)
                with torch.no_grad():
                    accs = self.calc_accs( # accs is a dict of floats
                        logits=lang,
                        targs=labels,
                        categories=temp_targs,
                        prepender=prepender+"_lang"
                    )
                    accs_array.append(accs)
            lang_accs = Trainer.avg_over_accs_array(accs_array)
        actn_accs = {}
        if self.phase == 1 or self.phase == 2:
            actns = actns.to(DEVICE)
            p = self.hyps["lang_p"] if self.phase == 2 else 0
            loss = p*loss + (1-p)*self.loss_fxn(logits, actns)
            with torch.no_grad():
                actn_accs = self.calc_accs( # accs is a dict of floats
                    logits=logits,
                    targs=actns,
                    categories=n_targs,
                    prepender=prepender+"_actn"
                )
        return loss, {**actn_accs, **lang_accs}

    @staticmethod
    def avg_over_accs_array(accs_array):
        """
        This is a helper function to average over the keys in an array
        of dicts. The result is a dict with the same keys as every
        dict in the argued array, but the values are averaged over each
        dict within the argued array.

        Args:
            accs_array: list of dicts
                this is a list of dicts. Each dict must consist of str
                keys and float or int vals. Each dict must have the
                same set of keys.
        Returns:
            avgs: dict
                keys: str
                    same keys as all dicts in accs_array
                vals: float
                    the average over all dicts in the accs array for
                    the corresponding key
        """
        if len(accs_array) == 0: return dict()
        avgs = {k: 0 for k in accs_array[0].keys()}
        for k in avgs.keys():
            avg = 0
            for i in range(len(accs_array)):
                avg += accs_array[i][k]
            avgs[k] = avg/len(accs_array)
        return avgs

    def calc_accs(self, logits, targs, categories=None, prepender=""):
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
                    total: float
                        the average accuracy over all categories
                    <categories_type_n>: float
                        the average accuracy over this particular
                        category. for example, if one of the categories
                        is named 1, the key will be "1" and the value
                        will be the average accuracy over that
                        particular category.
        """
        logits = logits.reshape(-1, logits.shape[-1])
        try:
            argmaxes = torch.argmax(logits, dim=-1).reshape(-1)
        except:
            print("logits:", logits)
            return { prepender + "_acc": 0 }
        targs = targs.reshape(-1)
        acc = (argmaxes.long()==targs.long()).float().mean()
        accs = {
            prepender + "_acc": acc.item()
        }
        if len(argmaxes) == 0: return accs
        if type(categories) == torch.Tensor: # (B, N)
            categories = categories.reshape(-1).data.long()
            cats = {*categories.numpy()}
            for cat in cats:
                idxs = categories==cat
                if idxs.float().sum() <= 0: continue
                try:
                    argmxs = argmaxes[idxs]
                except:
                    print("logits:", logits.shape)
                    print("argmaxes:", argmaxes.shape)
                    print("categs:", categories.shape)
                    print("idxs sum:", idxs.float().sum())
                    assert False
                trgs = targs[idxs]
                acc = (argmxs.long()==trgs.long()).float().mean()
                accs[prepender+"_acc_"+str(cat)] = acc.item()
        return accs

    def print_loop(self,
                   loop_count,
                   max_loops,
                   loss,
                   acc,
                   iter_start):
        """
        Printing statement for inner loop in the epoch.

        Args:
            loop_count: int
                the current loop
            max_loops: int
                the number of loops in the epoch
            loss: float
                the calculated loss
            acc: float
                the calculated accuracy
            iter_start: float
                a timestamp collected at the start of the loop
        """
        s = "Loss:{:.5f} | Acc:{:.5f} | {:.0f}% | t:{:.2f}"
        s = s.format(
            loss,
            acc,
            loop_count/max_loops*100,
            time.time()-iter_start
        )
        print(s, end=len(s)//4*" " + "\r")

    def validate(self, epoch, model, data_collector, n_targs=None):
        """
        Validates the performance of the model directly on an
        environment. Steps the learning rate scheduler based on the
        performance of the model.

        Args:
            runner: ValidationRunner
        """
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        # run model directly on an environment
        with torch.no_grad():
            # Returned tensors are mainly of shape (n_eval_steps,)
            model.reset(batch_size=1)
            eval_data = data_collector.val_runner.rollout(
                self.phase,
                model,
                n_tsteps=self.hyps["n_eval_steps"],
                n_eps=self.hyps["n_eval_eps"],
                n_targs=n_targs
            )
            lang_labels = self.get_lang_labels(
                eval_data["n_items"],
                eval_data["n_targs"],
                max_label=model.lang_size-1
            )
            drops = data_collector.exp_replay.get_drops(
                eval_data["grabs"]
            )
            if self.hyps["env_type"]=="gordongames-v4":
                drops = torch.ones_like(drops).long()
            loss, accs = self.get_loss_and_accs(
                logits=eval_data["actn_preds"],
                langs=eval_data["lang_preds"],
                actns=eval_data["actn_targs"],
                labels=lang_labels,
                drops=drops,
                n_targs=eval_data["n_targs"],
                prepender="val"
            )
        eval_eps = self.hyps["n_eval_eps"]
        eval_steps = self.hyps["n_eval_steps"]
        divisor = eval_eps if eval_steps is None else eval_steps
        avg_rew = eval_data["rews"].sum()/divisor
        metrics = {
            "val_loss": loss.item(),
            "val_rew": avg_rew.item(),
            **accs
        }
        # Extra metrics if using gordongames variant
        keys = ["n_items", "n_targs", "n_aligned"]
        dones = eval_data["dones"].reshape(-1)
        inpts = {key: eval_data[key].reshape(-1) for key in keys}
        inpts = {key: val[dones==1] for key,val in inpts.items()}
        targ_accs = self.calc_targ_accs(
            **inpts,
            prepender="val"
        )
        metrics = {**metrics, **targ_accs}
        inpts = {k:v.cpu().data.numpy() for k,v in inpts.items()}
        inpts["epoch"] = [
            epoch for i in range(len(inpts["n_items"]))
        ]
        inpts["phase"] = [
            self.phase for i in range(len(inpts["n_items"]))
        ]
        self.recorder.to_df(**inpts)
        self.recorder.track_loop(metrics)

    def calc_targ_accs(self,
        n_targs,
        n_items,
        n_aligned,
        prepender="val",
        **kwargs
    ):
        """
        Calculates the accuracy of the episodes with regards to matching
        the correct number of objects.

        Args:
            n_targs: ndarray or long tensor (N,)
                Collects the number of targets in the episode
                only relevant if using a gordongames
                environment variant
            n_items: ndarray or long tensor (N,)
                Collects the number of items over the course of
                the episode. only relevant if using a
                gordongames environment variant
            n_aligned: ndarray or long tensor (N,)
                Collects the number of items that are aligned
                with targets over the course of the episode.
                only relevant if using a gordongames
                environment variant
            prepender: str
                a simple string prepended to each key in the returned
                dict
        Returns:
            metrics: dict
                keys: str
                    "error": float
                        the difference between the number of target
                        objects and the number of item objects
                    "coef_of_var": float
                        the coefficient of variation. The avg error
                        divided by the goal size
                    "stddev": float
                        the standard deviation of the n_item responses.
                    "mean_resp": float
                        the mean response of the n_item responses.
        """
        fxns = {
            "error": calc_error,
            "coef_of_var": coef_of_var,
            "stddev": stddev,
            "mean_resp": mean_resp,
        }
        metrics = dict()
        if type(n_targs) == torch.Tensor:
            n_targs = n_targs.detach().cpu().numpy()
        if type(n_items) == torch.Tensor:
            n_items = n_items.detach().cpu().numpy()
        if type(n_aligned) == torch.Tensor:
            n_aligned = n_aligned.detach().cpu().numpy()
        inpts = {
            "n_items":  n_items,
            "n_targs":  n_targs,
            "n_aligned":n_aligned,
        }
        categories = set(n_targs.astype(np.int))
        for key,fxn in fxns.items():
            metrics[prepender+"_"+ key] = fxn(**inpts)
            # Calc for each specific target count
            for cat in categories:
                targs = n_targs[n_targs==cat]
                items = n_items[n_targs==cat]
                aligned = n_aligned[n_targs==cat]
                if len(targs)==0 or len(items)==0 or len(aligned)==0:
                    continue
                metrics[prepender+"_"+key+"_"+str(cat)] = fxn(
                    n_items=items,
                    n_targs=targs,
                    n_aligned=aligned,
                )
        return metrics

    def end_epoch(self, epoch):
        """
        Records, prints, cleans up the epoch statistics. Call this
        function at the end of the epoch.

        Args:
            epoch: int
                the epoch that has just finished.
        """
        self.recorder.save_epoch_stats(
            self.phase,
            epoch,
            self.model,
            self.optim,
            verbose=self.verbose
        )
        self.recorder.reset_stats()

    def end_training(self):
        """
        Perform all cleanup actions here. Mainly recording the best
        metrics.
        """
        pass

def mean_resp(n_items, **kwargs):
    """
    Args:
        n_items: ndarray (same dims as n_targs)
    Returns:
        mean: float
            the standard deviation of the responses
    """
    return n_items.mean()

def stddev(n_items, **kwargs):
    """
    Args:
        n_items: ndarray (same dims as n_targs)
    Returns:
        std: float
            the standard deviation of the responses
    """
    return n_items.std()

def calc_error(n_items, n_targs, **kwargs):
    """
    The square root of the mean squared distance between n_items and 
    n_targs.

    Args:
        n_items: ndarray (same dims as n_targs)
        n_targs: ndarray (same dims as n_items)
    Returns:
        error: float
            the square root of the average squared distance from the
            goal.
    """
    return np.sqrt(((n_items-n_targs)**2).mean())

def coef_of_var(n_items, n_targs, **kwargs):
    """
    Returns the coefficient of variation which is the error divided
    by the average n_targs

    Args:
        n_items: ndarray (same dims as n_targs)
        n_targs: ndarray (same dims as n_items)
    Returns:
        coef_var: float
            the error divided by the average n_targs
    """
    if len(n_items) == 0: return np.inf
    mean = n_items.mean()
    if mean == 0: return np.inf
    return n_items.std()/n_items.mean()

def perc_aligned(n_aligned, n_targs, **kwargs):
    """
    Calculates the percent of items that are aligned

    Args:
        n_aligned: ndarray (same dims as n_targs)
        n_targs: ndarray (same dims as n_aligned)
    Returns:
        perc: float
            the average percent aligned over all entries
    """
    perc = n_aligned/n_targs
    return perc.mean()*100

def perc_unaligned(n_items, n_aligned, n_targs, **kwargs):
    """
    Calculates the percent of items that are unaligned

    Args:
        n_items: ndarray (same dims as n_targs)
        n_aligned: ndarray (same dims as n_targs)
        n_targs: ndarray (same dims as n_items)
    Returns:
        perc: float
            the average percent unaligned over all entries
    """
    perc = (n_items-n_aligned)/n_targs
    return perc.mean()*100

def perc_over(n_items, n_targs, **kwargs):
    """
    Calculates the average proportion in which the number of items
    was greater than the number of targets. If the number of items
    was less than or equal to the number of targets, that entry is
    counted as 0%

    Args:
        n_items: ndarray (same dims as n_targs)
        n_targs: ndarray (same dims as n_items)
    Returns:
        perc: float
            the average amount of items over the number of targets
    """
    n_items = n_items.copy()
    n_items[n_items<n_targs] = n_targs[n_items<n_targs]
    perc = (n_items-n_targs)/n_targs
    return perc.mean()*100

def perc_under(n_items, n_targs, **kwargs):
    """
    Calculates the average proportion in which the number of items
    was less than the number of targets. If the number of items
    was greater than or equal to the number of targets, that entry is
    counted as 0%

    Args:
        n_items: ndarray (same dims as n_targs)
        n_targs: ndarray (same dims as n_items)
    Returns:
        perc: float
            the average amount of items less than the number of targets
    """
    n_items = n_items.copy()
    n_items[n_items>n_targs] = n_targs[n_items>n_targs]
    perc = (n_targs-n_items)/n_targs
    return perc.mean()*100

def perc_off(n_items, n_targs, **kwargs):
    """
    Calculates the average proportion in which the number of items
    was different than the number of targets.

    Args:
        n_items: ndarray (same dims as n_targs)
        n_targs: ndarray (same dims as n_items)
    Returns:
        perc: float
            the average amount of items different than the number of
            targets
    """
    perc = torch.abs(n_targs-n_items)/n_targs
    return perc.mean()*100

def perc_correct(n_aligned, n_targs, **kwargs):
    """
    Calculates the average proportion in which the number of aligned
    items is equal to the number of targets.

    Args:
        n_aligned: ndarray (same dims as n_targs)
        n_targs: ndarray (same dims as n_aligned)
    Returns:
        perc: float
            the average number of entries in which the number of
            aligned items is equal to the number of targets.
    """
    perc = (n_aligned == n_targs)
    return perc.mean()*100

