from supervised_gym.experience import ExperienceReplay, DataCollector
from supervised_gym.models import * # SimpleCNN, SimpleLSTM
from supervised_gym.recorders import Recorder
from supervised_gym.utils.utils import try_key

from torch.optim import Adam, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import CrossEntropyLoss
import torch
import numpy as np
import time
from tqdm import tqdm

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")


def train(rank, hyps, verbose=False):
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
    # Set random seeds
    hyps['seed'] = try_key(hyps,'seed', int(time.time()))
    torch.manual_seed(hyps["seed"])
    np.random.seed(hyps["seed"])
    # Initialize Data Collector and Begin Collecting Data
    # DataCollector's Initializer does Important changes to hyps
    data_collector = DataCollector(hyps)
    data_collector.dispatch_runners()
    # Initialize model
    model = globals()[hyps["model_type"]](**hyps)
    model.to(DEVICE)
    # Record experiment settings
    recorder = Recorder(hyps, model)
    # initialize trainer
    trainer = Trainer(hyps, model, recorder, verbose=verbose)
    # Loop training
    n_epochs = hyps["n_epochs"] if hyps["exp_name"] != "test" else 2
    for epoch in range(n_epochs):
        if verbose:
            print()
            print("Starting Epoch", epoch, "--", hyps["save_folder"])
        # Run environments, automatically fills experience replay's
        # shared_exp tensors
        data_collector.await_runners()
        trainer.train(model, data_collector.exp_replay)
        data_collector.dispatch_runners()
        if verbose:
            print("\nValidating")
        for val_sample in tqdm(range(hyps["n_val_samples"])):
            trainer.validate(epoch, model, data_collector)
        trainer.end_epoch(epoch)
    data_collector.terminate_runners()
    trainer.end_training()

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
        self.optim = self.get_optimizer(
            self.model,
            self.hyps["optim_type"],
            self.hyps["lr"]
        )
        self.scheduler = ReduceLROnPlateau(
            self.optim,
            'min',
            factor=0.5,
            patience=6,
            verbose=self.verbose
        )
        self.loss_fxn = globals()[self.hyps["loss_fxn"]]()

    def get_optimizer(self, model, optim_type, lr, *args, **kwargs):
        """
        Initializes an optimizer using the model parameters and the
        hyperparameters.
    
        Args:
            model: Model or torch.Module
                any object that implements a `.parameters()` member
                function that returns a sequence of torch.Parameters
            optim_type: str (one of [Adam, RMSprop])
                the type of optimizer. 
            lr: float
                the learning rate
        Returns:
            optim: torch optimizer
                the model optimizer
        """
        return globals()[optim_type](list(model.parameters()), lr=lr)

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
            obs =   data['obs']
            actns = data['actns'].to(DEVICE)
            dones = data["dones"]
            self.reset_model(model, len(obs))
            # model uses dones if it is recurrent
            logits = model(obs.to(DEVICE), dones.to(DEVICE)) 
            loss = self.loss_fxn(
                logits.reshape(-1, logits.shape[-1]),
                actns.flatten()
            )
            # Backprop and update
            loss.backward()
            self.optim.step()
            # Calc acc
            accs = self.calc_accs( # accs is a dict of floats
                logits=logits,
                targs=actns,
                categories=data["n_targs"],
                prepender="train"
            )
            # Record metrics
            metrics = {
                "train_loss": loss.item(),
                **accs}
            self.recorder.track_loop(metrics)
            self.print_loop(
                i,
                len(data_iter),
                loss.item(),
                accs["train_acc"],
                iter_start
            )
            if self.hyps["exp_name"] == "test" and i >= 2: break

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
        argmaxes = torch.argmax(logits, dim=-1).squeeze()
        targs = targs.reshape(-1)
        acc = (argmaxes.long()==targs.long()).float().mean()
        accs = {
            prepender + "_acc": acc.item()
        }
        if type(categories) == torch.Tensor: # (B, N)
            categories = categories.reshape(-1).data.long()
            cats = {*categories.numpy()}
            for cat in cats:
                argmxs = argmaxes[categories==cat]
                trgs = targs[categories==cat]
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

    def validate(self, epoch, model, data_collector):
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
                model,
                n_tsteps=self.hyps["n_eval_steps"],
                n_eps=self.hyps["n_eval_eps"]
            )
            # Calc Loss
            logits = eval_data["logits"] # already CUDA (N, K)
            targs = eval_data["targs"].to(DEVICE) # (N,)
            n_targs = eval_data["n_targs"] # (N,) or None
            loss = self.loss_fxn(logits, targs)
            # Calc Acc
            accs = self.calc_accs( # accs is a dict
                logits,
                targs,
                n_targs,
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
        if "gordongames" in self.hyps["env_type"]:
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
    return n_items.std()/n_targs.mean()

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

