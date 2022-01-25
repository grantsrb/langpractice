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
    "save_root": str
        this value is prepended to the exp_name when creating the save
        folder for the hyperparameter search.
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
    "second_phase": int (1 or 2)
        this determines if the model will be trained with language
        during the second phase of the training. If 1, the model is
        trained with actions only, if 2, the model is trained with
        both language and actions during the second phase.

    "model_type": str
        the name of the model class that you wish to use for the
        training. i.e. "SimpleCNN"
    "h_size": int
        the size of the hidden dimension for dense networks
    "bnorm": bool
        determines if the model should use batchnorm. true means it
        does use batchnorm
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
        the range of potential target counts. both low and high are
        inclusive. only applies to gordongames variants.

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
    "n_val_samples": int
        the number of validation loops to perform per training epoch
    "n_eval_eps": int or null
        the number of episodes to collect for evaluation during one
        validation loop. if null, n_eval_steps must be not null
    "n_eval_steps": int
        the number of environment steps to collect for evaluation
        during one validation loop. if null, n_eval_eps must be not null
    "randomize_order": bool
        determines if the order of data during training should be
        randomized. the order of a sequence within the batch is
        preserved.

    "oracle_type": str
        the name of the class to use for the oracle. i.e. "GordonOracle"
    "optim_type": str
        the name of the class to use for the optimizer. i.e. "Adam"
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

