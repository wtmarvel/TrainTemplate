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
    def __init__(self, rank, local_rank, opt):
        super().__init__()
        if rank == 0:
            cprint('#### [TrainTempolate] Start main Process. pid=%d' % os.getpid(), 'red')
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = opt.world_size
        self.start_epoch = 0
        self.start_step = 0
        self.opt = opt
        self.current_lr = opt.learning_rate
        self.dtype = self.set_dtype(opt.dtype)
        self.start_time = time.time()

    def run(self):
        """
        Main entry point for the distributed training process.
        Initializes the training environment and starts the training loop.
        """
        cprint(f'#### Start run rank {self.rank} (local_rank={self.local_rank}) Process. pid={os.getpid()}', 'red')
        torch.cuda.set_device(self.local_rank)
        # Initialize seeds for reproducibility
        init_seeds(self.opt.seed + self.rank)
        # Initialize the distributed environment
        self.init_distributed_env()
        # Initialize necessary components like dataloaders, model, and tensorboard
        self.set_meter(['total', ])
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
        torch.distributed.init_process_group(backend='nccl', world_size=self.opt.world_size, rank=self.rank)
        self.device = torch.device(f'cuda:{self.local_rank}')

    def set_dataprovider(self, opt):
        cprint(f'[rank-{self.rank}] Setting up dataloader...', 'cyan')

        distributed = self.opt.world_size > 1
        self.trainset = Dataset(training=True, rank=self.rank, verbose=self.rank == 0, **opt)
        train_sampler = DistributedSampler(self.trainset, rank=self.rank, num_replicas=self.world_size,
                                           shuffle=True) if distributed else None
        self.train_loader = DataLoader(self.trainset,
                                       collate_fn=None,
                                       num_workers=opt.num_dl_workers,
                                       shuffle=False if distributed else True,
                                       sampler=train_sampler,
                                       batch_size=opt.batch_size_per_gpu,
                                       pin_memory=False,
                                       persistent_workers=True,
                                       drop_last=True)

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
        cprint(f'[rank-{self.rank}] Setting up model...', 'cyan')
        self.model = net().to(self.device)
        self.load_model()

        if self.opt.world_size > 1:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank],
                                                             find_unused_parameters=self.opt.find_unused_parameters)

        self.optim = torch.optim.AdamW(self.model.parameters(), lr=self.opt.learning_rate,
                                       weight_decay=self.opt.weight_decay, betas=[self.opt.beta1, self.opt.beta2])
        # self.optim = DeepSpeedCPUAdam(self.net.parameters(), lr=opt.learning_rate,
        #                        betas=(opt.beta1, opt.beta2), weight_decay=0.0001)

        # self.optim = torch.optim.SGD(self.model.parameters(), momentum=0.9, lr=opt.learning_rate)

        self.loss_fn = nn.CrossEntropyLoss().to(self.device)
        if self.rank == 0:
            self.count_parameter(self.model)

        if self.rank == 0 and self.opt.use_ema:
            self.model_ema = EMA(
                self.model,
                beta=self.opt.ema_beta,  # exponential moving average factor
                update_after_step=0,  # only after this number of .update() calls will it start updating. default=100
                update_every=self.opt.ema_update_every,
                # how often to actually update, to save on compute (updates every 10th .update() call)
                start_step=1000000,
                ma_device=torch.device('cpu')
            )

    def load_model(self):
        opt = self.opt
        filepath = None
        if hasattr(opt, 'load_ckpt') and isinstance(opt.load_ckpt, str):
            if os.path.isdir(opt.load_ckpt):
                filepath = self.get_lastest_ckpt(opt.load_ckpt)
            elif os.path.isfile(opt.load_ckpt):
                filepath = opt.load_ckpt
            else:
                cprint('checkpoint file not found! %s' % opt.load_ckpt, 'red', attrs=['blink'])

        if filepath is not None:
            cprint('load ckpt from %s' % filepath, on_color='on_red')
            checkpoint = torch.load(filepath, map_location='cpu')
            self.start_epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
            self.start_step = checkpoint['step'] if 'step' in checkpoint else 0
            if 'net' in checkpoint and checkpoint['net'] is not None:
                self.model.load_state_dict(self.get_match_ckpt(self.model, checkpoint['net']))
            if 'optim' in checkpoint:
                self.optim.load_state_dict(checkpoint['optim'])
        return

    def preprocess(self, mb):
        for name, x in mb.items():
            if name in ['image']:
                x = x.to(self.device)
                if x.shape[-1] == 3:
                    x = torch.permute(x, [0, 3, 1, 2])  # NHWC->NCHW
                if x.dtype == torch.uint8:
                    x = x.float() / 127.5 - 1
            elif name in ['label']:
                if isinstance(x, torch.Tensor):
                    x = x.to(self.device)
            mb[name] = x
        return mb

    def do_optimize(self, mb, step, scaler):
        self.model.train()
        opt = self.opt
        gradient_accu_steps = opt.gradient_accumulation_steps
        if opt.world_size > 1:
            self.model.require_backward_grad_sync = False if step % gradient_accu_steps != 0 else True

        with autocast(enabled=self.opt.enable_amp):
            output = self.model(mb['image'])
        total_loss = self.loss_fn(output.float(), mb['label'].long())

        # self.optim.zero_grad()
        scaler.scale(total_loss / gradient_accu_steps).backward()
        if step % gradient_accu_steps == 0:
            scaler.unscale_(self.optim)
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 1)
            scaler.step(self.optim)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            self.optim.zero_grad()

        if opt.world_size > 1:
            torch.distributed.all_reduce(total_loss, op=torch.distributed.ReduceOp.AVG)

        mb_losses = {}
        mb_losses['total'] = total_loss
        self.update_meter(mb_losses)
        return mb_losses

    def train(self):
        opt = self.opt
        rank = self.rank
        world_size = self.world_size
        grad_accu_steps = opt.gradient_accumulation_steps

        grad_scaler = GradScaler(enabled=opt.enable_amp)
        self.model.train()
        torch.cuda.empty_cache()

        mb_size = self.opt.batch_size_per_gpu
        t0 = time.time()
        step = self.start_step if hasattr(self, 'start_step') else 0
        grad_step = step * grad_accu_steps  # when grad_accu_steps==1, grad_step is step
        # start_epoch = step // (len(self.trainset) // opt.batch_size) + 1
        start_epoch = 0
        for epoch in range(start_epoch, opt.max_epochs):
            for mb in self.train_loader:
                # print('rank=%d step=%d' % (rank, step))
                # torch.distributed.barrier(async_op=False)  # sync for training
                # all nodes need update learning rate
                if (grad_step == self.start_step * grad_accu_steps or grad_step % (
                        opt.print_freq * grad_accu_steps) == 0):
                    self.adjust_learning_rate_sqrt(step, [self.optim, ],
                                                   min_lr=opt.min_lr,
                                                   max_lr=opt.learning_rate,
                                                   warm_steps=opt.warm_steps,
                                                   total_steps=opt.total_steps)

                # forward
                grad_step += 1
                step = grad_step // grad_accu_steps
                mb = self.preprocess(mb)
                mb_losses = self.do_optimize(mb, grad_step, grad_scaler)
                if hasattr(self, 'model_ema'):  # only rank-0 own model_ema
                    self.model_ema.update()

                if rank == 0 and grad_step % (opt.print_freq * grad_accu_steps) == 0:
                    # torch.cuda.synchronize()  # sync for logging
                    now = time.time()
                    ct = now - t0
                    speed = opt.print_freq * grad_accu_steps * world_size * mb_size / ct
                    speed_iter = float(opt.print_freq) / ct
                    str = 'epoch %d ([Step%d]x[Bs%dx%d]x[GPU%d]) took %0.1fs(%0.2fh) %0.1fimgs/s %0.2fiter/s lr=%0.6f' % \
                          (epoch, step, mb_size, grad_accu_steps, world_size, ct,
                           (now - self.start_time) / 3600, speed, speed_iter, self.current_lr)
                    str += self.get_meter(str=True)
                    cprint(str)

                    self.sw.add_scalar("train/lr", self.current_lr, step)
                    for k, v in self.get_meter().items():
                        self.sw.add_scalar("train/%s" % k, v, step)

                    self.reset_meter()
                    t0 = time.time()

                if (opt.test_distributed or rank == 0) and grad_step % (opt.test_freq * grad_accu_steps) == 0:
                    test_loss = self.test(step)
                    self.reset_meter()  # maybe used in test
                    self.model.train()
                    if rank == 0:
                        self.save_model(epoch, step, test_loss, opt.model_save_dir, max_time_not_save=0 * 60)

                if rank == 0 and torch.isnan(mb_losses['total'].data):
                    cprint('Nan in train.', 'red', attrs=['blink'])
                    self.save_model(epoch, step, 404, opt.model_save_dir, max_time_not_save=0)
                    exit()

    @torch.no_grad()
    def test(self, step):
        self.model.eval()

        t0 = time.time()
        cnt, loss_sum = 0, 0
        for mb in self.test_loader:
            mb = self.preprocess(mb)
            B = mb['image'].shape[0]
            with autocast(enabled=self.opt.enable_amp):
                output = self.model(mb['image'])
            total_loss = self.loss_fn(output.float(), mb['label'].long())

            loss_sum += total_loss * B  # (mean loss) * (img num in the mini-batch)
            cnt += B

        loss = loss_sum / cnt
        if self.opt.test_distributed:  # test on every gpu
            cnt *= self.world_size
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)

        if self.rank == 0:
            speed = cnt / (time.time() - t0)
            print(
                f'rank={self.rank} took {(time.time() - t0):0.2f}s num_images={cnt} speed={speed:.2f}imgs/s loss={loss:0.4f}')
            self.sw.add_scalar("test/loss", loss, step)
        return loss
