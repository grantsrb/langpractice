import numpy as np
import torch
import torch.nn as nn
from langpractice.utils.torch_modules import Flatten, Reshape, GaussianNoise
from langpractice.utils.utils import update_shape
# update_shape(shape, kernel=3, padding=0, stride=1, op="conv"):

class Model(torch.nn.Module):
    """
    This is the base class for all models within this project. It
    ensures the appropriate members are added to the model.

    All models that inherit from Model must implement a step function
    that takes a float tensor of dims (B, C, H, W)
    """
    def __init__(self,
        inpt_shape,
        actn_size,
        lang_size,
        n_lang_denses=1,
        h_size=128,
        bnorm=False,
        lnorm=False,
        conv_noise=0,
        dense_noise=0,
        *args, **kwargs
    ):
        """
        Args: 
            inpt_shape: tuple or listlike (..., C, H, W)
                the shape of the input
            actn_size: int
                the number of potential actions
            lang_size: int
                the number of potential words
            n_lang_denses: int
                the number of duplicate language model outputs
            h_size: int
                the size of the hidden dimension for the dense layers
            bnorm: bool
                if true, the model uses batch normalization
            lnorm: bool
                if true, the model uses layer normalization on the h
                and c recurrent vectors after the recurrent cell
            conv_noise: float
                the standard deviation of noise added after each
                convolutional operation
            dense_noise: float
                the standard deviation of noise added after each
                dense operation (except for the output)
        """
        super().__init__()
        self.inpt_shape = inpt_shape
        self.actn_size = actn_size
        self.lang_size = lang_size
        self.h_size = h_size
        self.bnorm = bnorm
        self.lnorm = lnorm
        self.conv_noise = conv_noise
        self.dense_noise = dense_noise
        self.n_lang_denses = n_lang_denses
        self.register_buffer("_trn_whls", torch.ones(1))

    @property
    def is_cuda(self):
        try:
            return self._trn_whls.is_cuda
        except:
            return False

    @property
    def trn_whls(self):
        """
        This is essentially a boolean used to communicate if the
        runners should use the model predictions or the oracle
        predictions for training.

        Returns:
            training_wheel_status: int
                if 1, then the training wheels are still on the bike.
                if 0, then that sucker is free to shred
        """
        if hasattr(self,"_trn_whls"):
            return self._trn_whls.item()
        return 1

    @trn_whls.setter
    def trn_whls(self, status):
        """
        Sets the status of the training wheels

        Args:
            status: int
                if set to 1, the training wheels are considered on and
                the data collection will use the oracle. if 0, then
                the training data collection will use the actions from
                the model but still use the oracle's actions as the
                labels.
        """
        self._trn_whls[0] = status

    def get_device(self):
        try:
            return next(self.parameters()).get_device()
        except:
            return False

    def reset(self, batch_size):
        """
        Only necessary to override if building a recurrent network.
        This function should reset any recurrent state in a model.

        Args:
            batch_size: int
                the size of the incoming batches
        """
        pass

    def reset_to_step(self, step=1):
        """
        Only necessary to override if building a recurrent network.
        This function resets all recurrent states in a model to the
        recurrent state that occurred after the first step in the last
        call to forward.

        Args:
            step: int
                the index + 1 of the step to revert the recurrence to
        """
        pass

    def step(self, x):
        """
        Performs a single step rather than a complete sequence of steps

        Args:
            x: torch FloatTensor (B, C, H, W)
        Returns:
            actn: torch Float Tensor (B, K)
            lang: torch Float Tensor (B, L)
        """
        pass

    def forward(self, x):
        """
        Performs multiple steps in time rather than a single step.

        Args:
            x: torch FloatTensor (B, S, C, H, W)
        Returns:
            actn: torch Float Tensor (B, S, K)
            lang: torch Float Tensor (B, S, L)
        """
        pass

class NullModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, *args, **kwargs):
        """
        Args:
            x: torch Float Tensor (B, S, C, H, W) or (B,C,H,W)
        Returns:
            actn: torch FloatTensor (B, S, A) or (B, A)
            lang: tuple of torch FloatTensors (B, S, L) or (B, L)
        """
        if len(x.shape) == 4:
            return self.step(x)
        else:
            # Action
            actn = torch.zeros(*x.shape[:2], self.actn_size).float()
            # Language
            lang = torch.zeros(*x.shape[:2], self.lang_size).float()
            if x.is_cuda:
                actn.cuda()
                lang.cuda()
            return actn, (lang,)

    def step(self, x):
        """
        Args:
            x: torch Float Tensor (B, C, H, W)
        """
        actn = torch.zeros((x[0], self.actn_size)).float()
        lang = torch.zeros((x[0], self.lang_size)).float()
        if self.is_cuda:
            actn = actn.cuda()
            lang = lang.cuda()
        return actn, (lang,)

class RandomModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(inpt_shape=None, **kwargs)

    def forward(self, x, dones=None):
        """
        Args:
            x: torch Float Tensor (B, S, C, H, W)
            dones: torch LongTensor (B, S)
        """
        if len(x.shape) == 4:
            return self.step(x)
        else:
            # Action
            actn = torch.zeros(*x.shape[:2], self.actn_size).float()
            rand = torch.randint(
                low=0,
                high=self.actn_size,
                size=(int(np.prod(x.shape[:2])),)
            )
            actn = actn.reshape(int(np.prod(x.shape[:2])), -1)
            actn[torch.arange(len(actn)).long(), rand] = 1

            # Language
            lang = torch.zeros(*x.shape[:2], self.lang_size).float()
            rand = torch.randint(
                low=0,
                high=self.lang_size,
                size=(int(np.prod(x.shape[:2])),)
            )
            lang = lang.reshape(int(np.prod(x.shape[:2])), -1)
            lang[torch.arange(len(lang)).long(), rand] = 1
            if x.is_cuda:
                actn.cuda()
                lang.cuda()
            return actn, (lang,)

    def step(self, x):
        """
        Args:
            x: torch Float Tensor (B, C, H, W)
        """
        rand = torch.randint(
            low=0,
            high=self.actn_size,
            size=(len(x),)
        )
        actn = torch.zeros(len(x), self.actn_size).float()
        actn[torch.arange(len(x)).long(), rand] = 1
        rand = torch.randint(
            low=0,
            high=self.lang_size,
            size=(len(x),)
        )
        lang = torch.zeros(len(x), self.lang_size).float()
        lang[torch.arange(len(x)).long(), rand] = 1
        if x.is_cuda:
            actn.cuda()
            lang.cuda()
        return actn, (lang,)

class SimpleCNN(Model):
    """
    A simple convolutional network with no recurrence.
        conv2d
        bnorm
        relu
        conv2d
        bnorm
        relu
        linear
        bnorm
        relu
        linear
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.depths = [self.inpt_shape[-3], 32, 48]
        self.kernels = [3, 3]
        self.strides = [1, 1]
        self.paddings = [0, 0]
        modules = []
        shape = [*self.inpt_shape[-3:]]
        self.shapes = [shape]
        for i in range(len(self.depths)-1):
            # CONV
            modules.append(
                nn.Conv2d(
                    self.depths[i],
                    self.depths[i+1],
                    kernel_size=self.kernels[i],
                    stride=self.strides[i],
                    padding=self.paddings[i]
                )
            )
            # RELU
            modules.append(GaussianNoise(self.conv_noise))
            modules.append(nn.ReLU())
            # Batch Norm
            if self.bnorm:
                modules.append(nn.BatchNorm2d(self.depths[i+1]))
            # Track Activation Shape Change
            shape = update_shape(
                shape, 
                depth=self.depths[i+1],
                kernel=self.kernels[i],
                stride=self.strides[i],
                padding=self.paddings[i]
            )
            self.shapes.append(shape)
        self.features = nn.Sequential(*modules)
        self.flat_size = int(np.prod(shape))

        # Make Action MLP
        modules = [
            Flatten(),
            nn.Linear(self.flat_size, self.h_size),
            GaussianNoise(self.dense_noise),
            nn.ReLU()
        ]
        if self.bnorm:
            modules.append(nn.BatchNorm1d(self.h_size))
        self.actn_dense = nn.Sequential(
            *modules,
            nn.Linear(self.h_size, self.actn_size)
        )

        # Make Language MLP
        self.lang_denses = nn.ModuleList([])
        for i in range(self.n_lang_denses):
            modules = [
                Flatten(),
                nn.Linear(self.flat_size, self.h_size),
                nn.ReLU()
            ]
            if self.bnorm:
                modules.append(nn.BatchNorm1d(self.h_size))
            self.lang_denses.append(nn.Sequential(
                *modules,
                nn.Linear(self.h_size, self.lang_size)
            ))

    def step(self, x, *args, **kwargs):
        """
        Performs a single step rather than a complete sequence of steps

        Args:
            x: torch FloatTensor (B, C, H, W)
        Returns:
            pred: torch Float Tensor (B, K)
        """
        fx = self.features(x)
        actn = self.actn_dense(fx)
        langs = []
        for dense in self.lang_denses:
            lang = dense(fx)
            langs.append(lang)
        return actn, langs

    def forward(self, x, *args, **kwargs):
        """
        Args:
            x: torch FloatTensor (B, S, C, H, W)
        Returns:
            actns: torch FloatTensor (B, S, N)
                N is equivalent to self.actn_size
        """
        b,s = x.shape[:2]
        actn, langs = self.step(x.reshape(-1, *x.shape[2:]))
        langs = torch.stack(langs, dim=0).reshape(len(langs), b, s, -1)
        return actn.reshape(b,s,-1), langs

class SimpleLSTM(Model):
    """
    A recurrent LSTM model.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.bnorm == False,\
            "bnorm must be False. it does not work with Recurrence!"

        # Convs
        cnn = SimpleCNN(*args, **kwargs)
        self.shapes = cnn.shapes
        self.features = cnn.features

        # LSTM
        self.flat_size = cnn.flat_size
        self.lstm = nn.LSTMCell(self.flat_size, self.h_size)

        # Action Dense
        self.actn_dense = nn.Sequential(
            GaussianNoise(self.dense_noise),
            nn.ReLU(),
            nn.Linear(self.h_size, self.actn_size),
        )

        # Lang Dense
        self.lang_denses = nn.ModuleList([])
        for i in range(self.n_lang_denses):
            self.lang_denses.append(nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.h_size, 2*self.h_size),
                nn.ReLU(),
                nn.Linear(2*self.h_size, self.lang_size),
            ))

        print("lang_denses:", self.n_lang_denses)
        # Memory
        if self.lnorm:
            self.layernorm_c = nn.LayerNorm(self.h_size)
            self.layernorm_h = nn.LayerNorm(self.h_size)
        self.h = None
        self.c = None
        self.reset(batch_size=1)

    def reset(self, batch_size=1):
        """
        Resets the memory vectors

        Args:
            batch_size: int
                the size of the incoming batches
        Returns:
            None
        """
        self.h = torch.zeros(batch_size, self.h_size).float()
        self.c = torch.zeros(batch_size, self.h_size).float()
        # Ensure memory is on appropriate device
        if self.features[0].weight.is_cuda:
            self.h.to(self.get_device())
            self.c.to(self.get_device())
        self.prev_hs = [self.h]
        self.prev_cs = [self.c]

    def partial_reset(self, dones):
        """
        Uses the done signals to reset appropriate parts of the h and
        c vectors.

        Args:
            dones: torch LongTensor (B,)
                h and c are zeroed along any row in which dones[row]==1
        Returns:
            h: torch FloatTensor (B, H)
            c: torch FloatTensor (B, H)
        """
        mask = (1-dones).unsqueeze(-1)
        h = self.h*mask
        c = self.c*mask
        return h,c

    def reset_to_step(self, step=1):
        """
        This function resets all recurrent states in a model to the
        recurrent state that occurred after the first step in the last
        call to forward.

        Args:
            step: int
                the index + 1 of the step to revert the recurrence to
        """
        assert (step-1) < len(self.prev_hs) and (step-1) >= 0, "invalid step"
        self.h = self.prev_hs[step-1].detach().data
        self.c = self.prev_cs[step-1].detach().data
        if self.is_cuda:
            self.h.to(self.get_device())
            self.c.to(self.get_device())

    def step(self, x, *args, **kwargs):
        """
        Performs a single step rather than a complete sequence of steps

        Args:
            x: torch FloatTensor (B, C, H, W)
        Returns:
            actn: torch Float Tensor (B, K)
            langs: list of torch Float Tensor (B, L)
        """
        if x.is_cuda:
            self.h = self.h.to(x.get_device())
            self.c = self.c.to(x.get_device())
        fx = self.features(x)
        fx = fx.reshape(len(x), -1) # (B, N)
        self.h, self.c = self.lstm( fx, (self.h, self.c) )
        if self.lnorm:
            self.c = self.layernorm_c(self.c)
            self.h = self.layernorm_h(self.h)
        langs = []
        for dense in self.lang_denses:
            langs.append(dense(self.h))
        return self.actn_dense(self.h), langs

    def forward(self, x, dones, *args, **kwargs):
        """
        Args:
            x: torch FloatTensor (B, S, C, H, W)
            dones: torch Long Tensor (B, S)
                the done signals for the environment. the h and c
                vectors are reset when encountering a done signal
        Returns:
            actns: torch FloatTensor (B, S, N)
                N is equivalent to self.actn_size
            langs: torch FloatTensor (N,B,S,L)
        """
        seq_len = x.shape[1]
        actns = []
        langs = []
        self.prev_hs = []
        self.prev_cs = []
        if x.is_cuda:
            dones = dones.to(x.get_device())
        for s in range(seq_len):
            actn, lang = self.step(x[:,s])
            actns.append(actn.unsqueeze(1))
            if self.n_lang_denses == 1:
                lang = lang[0].unsqueeze(0).unsqueeze(2) # (1, B, 1, L)
            else:
                lang = torch.stack(lang, dim=0).unsqueeze(2)# (N, B, 1, L)
            langs.append(lang)
            self.h, self.c = self.partial_reset(dones[:,s])
            self.prev_hs.append(self.h.detach().data)
            self.prev_cs.append(self.c.detach().data)
        return torch.cat(actns, dim=1), torch.cat(langs, dim=2)




