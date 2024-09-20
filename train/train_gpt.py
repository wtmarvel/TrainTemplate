import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from train.ema_pytorch import EMA
from train.trainer import DistributedTrainerBase
from helper import cprint, init_seeds
from net.resnet import resnet18 as net
from dataloader.dataset_mock import Dataset, CollateFn


class TrainProcess(DistributedTrainerBase):
    """
    A class to handle the distributed training process.

    Attributes:
        rank: Global rank of the process in distributed training.
        local_rank: Local GPU rank on the current node.
        world_size: Total number of processes participating in training.
        start_epoch: Epoch from which to resume training.
        start_step: Step from which to resume training.
        current_lr: The current learning rate.
        device: Device to use for training (e.g., 'cuda:0').
    """

    def __init__(self, rank: int, local_rank: int, opt: Namespace):
        super().__init__()
        if rank == 0:
            cprint(f'#### [TrainTemplate] Start main Process. pid={os.getpid()}', 'red')

        # Process and training parameters
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = opt.world_size
        self.start_epoch = 0
        self.start_step = 0
        self.opt = opt
        self.current_lr = opt.learning_rate
        self.dtype = self.set_dtype(opt.dtype)
        self.start_time = time.time()

    @staticmethod
    def set_dtype(dtype_str: str) -> torch.dtype:
        """
        Maps string-based dtype to PyTorch dtype.

        Args:
            dtype_str: String representing the data type ('float32', 'fp16', 'bf16').

        Returns:
            Corresponding PyTorch dtype.
        """
        dtype_map = {
            'float32': torch.float32,
            'fp16': torch.float16,
            'bf16': torch.bfloat16
        }
        return dtype_map.get(dtype_str.lower(), torch.float32)

    def run(self):
        """
        Main entry point for the distributed training process.
        Initializes the training environment and starts the training loop.
        """
        cprint(f'#### Start run rank {self.rank} (local_rank={self.local_rank}) Process. pid={os.getpid()}', 'red')

        # Set the GPU device for the current process
        torch.cuda.set_device(self.local_rank)

        # Initialize seeds for reproducibility
        init_seeds(self.opt.seed + self.rank)

        # Initialize the distributed environment
        self.init_distributed_env()

        # Initialize necessary components like dataloaders, model, and tensorboard
        self.set_meter(['total'])
        self.set_dataprovider(self.opt)
        self.set_model_and_loss()

        if self.rank == 0:
            logpath = os.path.join(self.opt.model_save_dir, 'logs')
            self.sw = SummaryWriter(logpath)

        # Optionally test the model before training begins
        if self.opt.test_before_train and (self.opt.test_distributed or self.rank == 0):
            self.test(self.start_step)

        # Begin the training loop
        self.train()

    def init_distributed_env(self):
        """
        Initializes the distributed training environment with the NCCL backend.
        """
        cprint(f'[rank-{self.rank}] Initializing distributed environment...', 'cyan')

        # Initializes the distributed process group for NCCL
        torch.distributed.init_process_group(backend='nccl', world_size=self.opt.world_size, rank=self.rank)
        self.device = torch.device(f'cuda:{self.local_rank}')

    def set_dataprovider(self, opt):
        """
        Sets up the data loaders for training and testing in a distributed environment.

        Args:
            opt: The options/arguments for training, including batch size and worker counts.
        """
        cprint(f'[rank-{self.rank}] Setting up dataloader...', 'cyan')

        distributed = self.opt.world_size > 1

        # Create the training dataset and sampler for distributed training
        self.trainset = Dataset(training=True, rank=self.rank, verbose=self.rank == 0, **opt)
        train_sampler = DistributedSampler(self.trainset, rank=self.rank, num_replicas=self.world_size, shuffle=True) \
            if distributed else None

        self.train_loader = DataLoader(self.trainset,
                                       collate_fn=None,
                                       num_workers=opt.num_dl_workers,
                                       shuffle=False if distributed else True,
                                       sampler=train_sampler,
                                       batch_size=opt.batch_size_per_gpu,
                                       pin_memory=False,
                                       persistent_workers=True,
                                       drop_last=True)

        # Set up the test dataloader if applicable
        if opt.test_distributed or self.rank == 0:
            self.testset = Dataset(training=False, rank=self.rank, verbose=self.rank == 0, **opt)
            test_sampler = DistributedSampler(self.testset, rank=self.rank, num_replicas=self.world_size,
                                              shuffle=False) if distributed else None
            self.test_loader = DataLoader(self.testset,
                                          collate_fn=None,
                                          num_workers=opt.num_dl_workers,
                                          shuffle=False,
                                          sampler=test_sampler,
                                          batch_size=opt.batch_size_per_gpu,
                                          pin_memory=False,
                                          persistent_workers=True,
                                          drop_last=True)

    def set_model_and_loss(self):
        """
        Sets up the model, optimizer, and loss function for training.
        Also loads the model checkpoint if specified.
        """
        cprint(f'[rank-{self.rank}] Setting up model...', 'cyan')

        # Initialize the model and move it to the correct device
        self.model = net().to(self.device)
        self.load_model()

        # Enable synchronized BatchNorm for distributed training
        if self.opt.world_size > 1:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank],
                                                             find_unused_parameters=self.opt.find_unused_parameters)

        # Initialize optimizer
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=self.opt.learning_rate,
                                       weight_decay=self.opt.weight_decay, betas=[self.opt.beta1, self.opt.beta2])

        # Define the loss function
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)

        if self.rank == 0:
            self.count_parameter(self.model)

        # Set up Exponential Moving Average (EMA) for model weights if enabled
        if self.rank == 0 and self.opt.use_ema:
            self.model_ema = EMA(
                self.model,
                beta=self.opt.ema_beta,
                update_after_step=0,
                update_every=self.opt.ema_update_every,
                start_step=1000000,
                ma_device=torch.device('cpu')
            )
