# Language as a Tool or a Practice

## Description
This project seeks to disentangle the reason why language is correlated with some cognitive tasks. Specifically, this project seeks to determine if language is used as a tool for counting or it acts as a practice for learning counting concepts.

The experiment first creates two models. One is pretrained to use language to count objects. The other is pretrained on a task that requires counting but only uses a trinary signal of more, less, or equal to as its output. The models are then transferred to tasks that require manipulating an environment based on a visual count. If the two models perform equally, we can attribute the cognitive abilities associated with language to the practice that comes with learning a language. If the lingual model performs better, however, we can attribute the improvements to the model using language as a tool.

## How to Use this Repo
### Training
To train a model you will need to have a hyperparameters json and a hyperranges json. The hyperparameters json details the values of each of the training parameters that will be used for the training. See the [training scripts readme](training_scripts/readme.md) for parameter details. The hyperranges json contains a subset of the hyperparameter keys each coupled to a list of values that will be cycled through for training. Every combination of the hyperranges key value pairs will be scheduled for training. This allows for easy hyperparameter searches. For example, if `lr` is the only key in the hyperranges json, then trainings for each listed value of the learning rate will be queued and processed in order. If `lr` and `l2` each are in the hyperranges json, then every combination of the `lr` and `l2` values will be queued for training.

To run a training session, navigate to the `training_scripts` folder:

```
$ cd training_scripts
```

And then select the cuda device index you will want to use (in this case 0) and type the following command:

```
$ CUDA_VISIBLE_DEVICES=0 python3 main.py path_to_hyperparameters.json path_to_hyperranges.json
```
It is also possible to split the hyperranges into multiple tmux sessions on multiple gpus using the `./training_scripts/distr_main.py` script. To use this script, you must create a metaparams.json file which is detailed within the `distr_main.py` script. If it does not work, you may need to upgrade tmux to at least 3.0 or you may need to upgrade ubuntu to 20.04 or later.

## Setup
After cloning the repo, install all necessary packages locally:
```sh
python3.6 -m pip install --user -r requirements.txt
```
Next you will to install this pip package. Run the following:
```sh
python3.6 -m pip install --user -e .
```

## Repo Usage
### Watching Your Trained Policy
After training your policy, you can watch the policy run in the environment using the `watch_model.py` script. To use this file, pass the name of the saved model folder that you would like to watch. The viewing script will automatically create a version of the environment that the model was trained on, load the best version of the model based on the evaluated performance during training, and run the model on the environment.

Here's an example:

    $ python watch_model.py search_name/search_name_0_whateverwaslistedhere/

### Automated Hyper Parameter Search
Much of deep learning consists of tuning hyperparameters. It can be extremely addicting to change the hyperparameters by hand and then stare at the average reward as the algorithm trains. THIS IS A HOMERIAN SIREN! DO NOT SUCCUMB TO THE PLEASURE! It is too easy to change hyperparameters before their results are fully known. It is difficult to keep track of what you did, and the time spent toying with hyperparameters can be spent reading papers, studying something useful, or calling your Mom and telling her that you love her (you should do that right now. Your dad, too)

This repo can automate your hyperparameter searches using a `hyperranges json`. Simply specify the key you would like to search over and specify a list of values that you would like that key to take. If multiple keys are listed, then all combinations of the possible values will be searched. 

### Saved Outputs
Each combination of hyperparameters gets saved to a "model\_folder" under the path within a greater folder called the "exp\_folder". The experiment folder is specified within the `hyperparams.json`.

Within the `model_folder` a checkpoint file exists for each epoch in the training. The checkpoints contain details about the statistics for that epoch as well as holding the hyperparameters for that epoch. The "best" and "last" checkpoints also contain the model parameters. You can save a copy of the model parameters at every epoch using the `hyperparams.json`.

### `validation\_stats.csv`
For this project a csv is created for each training session. The csv consists of the ending metrics of each validation episode for each epoch. This should enable you to analyze how the model performance changed over the course of the training and allow you to calculate statistics such as the coefficient of variation.


#### List of Valid Keys for hyperparams json
Set values in a json and run `$ python3 main.py your_hyperparams_json.json` to use the specified parameters.

    "exp_name": str
        the name of the folder in which the hyperparameter search will
        be saved to. This is different than the path. If you would like
        to save the experiment to a different folder than the one in
        which you run `main.py`, use the hyperparemter called `save_root`
    "exp_num_offset": int
        a number by which to offset all experiment numbers. This is
        useful in cases where you want to run trainings on different
        machines but avoid overlapping experiment numbers
    "save_root": str
        this value is prepended to the exp_name when creating the save
        folder for the hyperparameter search.
    "resume_folder": str
        path to a training to resume from. The epochs argued in this
        hyperparameter set must be larger than where the resumed
        training left off.
    "lang_checkpt": str
        path to a language checkpoint to initialize the model from.
        If a model_folder is argued (instead of a specific checkpoint
        file), the loaded weights will be from the last epoch of phase
        0. This makes it possible to save time by using pretrained
        language models. This will make the training skip straight to
        the second phase but will leave the `skip_first_phase` parameter
        unchanged.
    "description": str
        this is an option key used to write notes or a description of
        the hyperparameter search
    "render": bool
        if true, the validation operations are rendered
    "del_prev_sd": bool
        option to only keep the most recent state dict in the
        checkpoints. This is used to conserve space on the disk. If
        true, previous state dicts are deleted from their checkpts
    "seed": int
        The random seed for the training
    "runner_seed_offset": int
        An offset for the random seed for each of the parallel
        environments
    "lang_epochs": int
        The number of training epochs for the language training phase
    "actn_epochs": int
        The number of training epochs for the action training phase
    "trn_whls_epoch": int
        the epoch in which the training wheels are removed from the
        model. This means that the training data is collected using
        the actions from the model. The action targets continue to be
        those produced by the oracle.

    "use_count_words": int [0, 1, or 2]
        this determines what type of count words should be used. If 0,
        the model learns to count by issuing a less-than, equal-to, or
        greater-than prediction of the number of items relative to
        the number of targets. If 1, the model learns to count out the
        items using english number words. Lastly if 2 is argued, the
        model learns the Piraha count words which are probabilistically
        derived at runtime to match the statistics found in Frank 2008.
        In this case, 4 possible labels exist. 0, 1, and 2 each are used
        100% of the time for numbers 0,1,and 2. But for numbers 3 and
        greater, either label 2 or 3 is used for the count word with
        a probability matching the published values.
    "skip_first_phase": bool
        if true, the training will skip phase 0 and go straight to
        phase 1 or 2 (depending on the value of `second_phase`).
        Defaults to false.
    "first_phase": int (0, 1 or 2)
        this determines if the model will be trained with language,
        actions, or both during the first phase of the training.
        0 means language only. 1 means the model is trained with actions
        only. 2 means the model is trained with both language and
        actions.
    "second_phase": int (0, 1 or 2)
        this determines if the model will be trained with language,
        actions, or both during the second phase of the training.
        0 means language only. 1 means actions only. 2 means both
        language and actions.
    "blind_lang": bool
        if true, the observations are zeroed out during language 
        training creating the effect of the model purely learning
        language. If false, this variable is effectively ignorned.
        Defaults to False.

    "model_type": str
        the name of the model class that you wish to use for the
        training. i.e. "SimpleCNN"
    "lstm_lang_first": bool
        only used in multi-lstm model types. If true, the h
        vector from the first LSTM will be used as the input
        to the language layers. The second h vector will then
        be used for the action layers. If False, the second h
        vector will be used for language and the first h for
        actions. Defaults to True

    "h_size": int
        the size of the hidden dimension for dense networks
    "bnorm": bool
        determines if the model should use batchnorm. true means it
        does use batchnorm
    "lnorm": bool
        determines if the model should use layernorm. true means it
        does use layernorm on both the h and c recurrent vectors just
        after the lstm cell. Not that using the layernorm after the
        cell still results in a normalized input for the next step
        in time while normalizing the input for the action and language
        networks.
    "n_frame_stack": int
        the number of frames to stack for an observation of the game
    "lr": float
        the learning rate
    "l2": float
        the weight decay or l2 regularization parameter
    "lang_p": float between 0 and 1
        the portion of the loss during the second phase attributed
        to language.
    "conv_noise": float
        the standard deviation of gaussian noise applied to the
        convolutional layers of the model. if 0, has no effect
    "dense_noise": float
        the standard deviation of gaussian noise applied to the
        dense layers of the model. if 0, has no effect
    "feat_drop_p": float
        the probability of zeroing a neuron within the features
        of the cnn output.
    "drop_p": float
        the probability of zeroing a neuron within the dense
        layers of the network.

    "langall": bool
        if false, language predictions only occur when the agent drops
        an object. Otherwise the language predictions occur at every
        step. defaults to false. this parameter overrides `count_targs`
        and `lang_targs_only`
    "drop_prec_threshold": float
        if 0 no effect. Otherwise the drops array will randomly add
        ones so that the language training signal is increased.
    "count_targs": bool
        Only applies to v4, v7, and v8 variants of gordongames. if true,
        the model will learn to count out the targets in addition to
        the items. If false, model will only count the items. Differs
        from langall=True in that it skips the steps that
        the agent takes to move about the grid. Overridden by
        `lang_targs_only`
    "lang_targs_only": int
        Only applies to v4, v7, and v8 variants of gordongames. if 0,
        effectively does nothing. If 1, the language labels will only
        be for the targets. No counting is performed on the items.
        This argument is overridden by langall being true.
        count_targs is overridden by this argument. drop_perc_threshold
        has no impact on this argument.

    "env_type": str
        the name of the gym environment to be used for the training
    "harsh": bool
        an optional parameter to determine the reward scheme for
        gordongames variants
    "pixel_density": int
        the side length (in pixels) of a unit square in the game. Only
        applies to gordongames variants
    "grid_size": list of ints (height, width)
        the number of units (height, width) of the game. Only applies
        to gordongames variants
    "targ_range": list of ints (low, high)
        the range of potential target counts. This acts as the default
        if lang_range or actn_range are not specified. both low and
        high are inclusive. only applies to gordongames variants.
    "lang_range": list of ints (low, high)
        the range of potential target counts for training the language
        model during phase 0. both low and high are
        inclusive. only applies to gordongames variants.
    "actn_range": list of ints (low, high)
        the range of potential target counts for training the action
        model during phases other than 0. both low and high are
        inclusive. only applies to gordongames variants.
    "zipf_order": float or None
        if greater than 0, the targets are drawn proportionally to the
        zipfian distribution using `zipfian_order` as the exponent.

    "n_envs": int
        the number of parallel environments to collect data in the
        training
    "batch_size": int
        the number of rollouts to collect when collecting data and
        the batch size for the training
    "seq_len": int
        the number of consecutive frames to feed into the model for
        a single batch of data
    "exp_len": int
        the "experience" length for a single rollout. This is the
        number of steps to take in the environment for a single row
        in the batch during data collection.
    "reset_trn_env": bool
        if true, the training environments are reset at the beginning
        of each collection. This ensures that the model never has to
        jump into the middle of an episode during training.
    "roll_data": bool
        if true, training data is fed in such that the next training
        loop is only one time step in the future from the first step
        in the last loop (so seq_len-1 steps overlap with the previous
        loop). Otherwise training data is fed in such that
        the first data point in the current training loop is 1 time
        step after the last time step in the last loop (so there is
        no data overlap).

    "dino": bool
        determines if the transformer should attempt to train a memory
        token using the DINO self-supervised training method.
    "dino_weight": float
        the weight of the dino loss relative to the simclr and total
        losses.
    "simclr": bool
        determines if the transformer should attempt to train a memory
        token using the SimCLR self-supervised training method.
    "simclr_weight": float
        the weight of the simclr loss relative to the dino and total
        losses.
    "augment": bool
        if true, images are augmented before being used by the network
    "smoothing": float
        the label smoothing when calculating the cross entropy between
        the student and teacher distributions. See pytorch documentation
        about the cross_entropy loss for more details.

    "val_targ_range": list of ints (low, high) or None
        the range of target counts during the validation phase. both
        low and high are inclusive. only applies to gordongames
        variants. if None, defaults to training range.
    "val_max_actn": bool
        if true, actions during the validation phase are selected as
        the maximum argument over the model's action probability output.
        If False, actions are sampled from the action output with
        probability equal to the corresponding output probability
    "n_eval_eps": int or null
        the number of episodes to collect for each target value during
        validation.

    "oracle_type": str
        the name of the class to use for the oracle. i.e. "GordonOracle"
    "optim_type": str
        the name of the class to use for the optimizer. i.e. "Adam"
    "factor": float
        the factor to decrease the learning rate by. new_lr = factor*lr
    "patience": int
        the number of epochs to wait for loss improvement before
        reducing the learing rate
    "threshold": float
        the threshold by which to determine if the training has
        plateaued. smaller threshold means smaller changes count as
        meaningful progress in the training. This is proportional to
        the size of the loss.
        loss_to_beat = best_actual_loss*(1-threshold)
        If loss_to_beat is further away from actual loss, then a
        learning rate change is more likely to occur.
    "min_lr": float
        the minimum learning rate allowed by the scheduler
    "loss_fxn": str
        the name of the class to use for the loss function. i.e.
        "CrossEntropyLoss"
    "preprocessor": str
        the name of the preprocessing function. this function operates
        directly on observations from the environment before they are
        concatenated to form the "state" of the environment
    "best_by_key": str
        the name of the metric to use for determining the best
        performing model. i.e. "val_perc_correct_avg"

