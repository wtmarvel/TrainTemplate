import os
import time
import numpy as np
import math
import torch
from termcolor import cprint
import glob


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, mom=0):
        self.mom = mom
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        if torch.is_tensor(val):
            if val.is_cuda:
                val = val.data.float().cpu().numpy()
            else:
                val = val.data.float().numpy()
        if isinstance(val, np.ndarray):
            val = val.item()

        if self.count == 0:
            self.val = val
        else:
            self.val = self.mom * self.val + (1 - self.mom) * val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count


# function to calculate the Exponential moving averages for the Generator weights
# This function updates the exponential average weights based on the current training
@torch.no_grad()
def update_average(model_tgt, model_src, beta=0.9):
    """
    update the model_target using exponential moving averages
    :param model_tgt: target model
    :param model_src: source model
    :param beta: value of decay beta
    :return: None (updates the target model)
    """

    # utility function for toggling the gradient requirements of the models
    def toggle_grad(model, requires_grad):
        for p in model.parameters():
            p.requires_grad_(requires_grad)

    # turn off gradient calculation
    # toggle_grad(model_tgt, False)
    # toggle_grad(model_src, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert (p_src is not p_tgt)
        p_tgt.copy_(beta * p_tgt + (1. - beta) * p_src)
        # p_tgt.data.copy_(beta * p_tgt.data + (1. - beta) * p_src.data)

    # turn back on the gradient calculation
    # toggle_grad(model_tgt, True)
    # toggle_grad(model_src, True)


class TrainerBase(object):
    """
    TrainerBase
    """

    def __init__(self, FLAGS=None):
        pass

    @staticmethod
    def set_dtype(dtype_str):
        dtype_map = {
            'float32': torch.float32,
            'fp16': torch.float16,
            'bf16': torch.bfloat16
        }
        return dtype_map.get(dtype_str.lower(), torch.float32)

    def set_meter(self, names):
        for name in names:
            name = 'meter_' + name
            if not hasattr(self, name):
                self.__setattr__(name, AverageMeter(0))
        return

    def update_meter(self, loss):
        for k, v in loss.items():
            name = 'meter_' + k
            if hasattr(self, name):
                self.__getattribute__(name).update(v)
        return

    def reset_meter(self):
        for name in self.__dict__.keys():
            if name.startswith('meter_'):
                self.__getattribute__(name).reset()
        return

    def get_meter(self, str=False):
        m = {}
        for name in self.__dict__.keys():
            if name.startswith('meter_'):
                v = self.__getattribute__(name).avg
                nm = name.split('meter_')[-1]
                m[nm] = v
        if str:
            s = ''
            for nm, v in m.items():
                if nm == 'newline':
                    s += '\n          '
                elif abs(v) > 1e-2:
                    s += ' %s=%0.4f' % (nm, v)
                elif abs(v) > 1e-3:
                    s += ' %s=%0.5f' % (nm, v)
                elif abs(v) > 1e-4:
                    s += ' %s=%0.6f' % (nm, v)
                elif v == 0:
                    s += ' %s=%0.0f' % (nm, v)
                else:
                    s += ' %s=%0.3e' % (nm, v)
            return s
        else:
            return m

    def get_lastest_ckpt(self, dir, print=False):
        if not os.path.isdir(dir):
            return None

        files = glob.glob(os.path.join(dir, '*.pth'))
        if len(files) == 0:
            return None

        timestamps = [(i, os.path.getmtime(file)) for i, file in enumerate(files)]
        timestamps.sort(reverse=True, key=lambda x: x[1])
        if print:
            for i, _ in timestamps:
                print(files[i])
        return files[timestamps[0][0]]

    def delete_older_ckpt_dir(self, dir, maxN=2, verbose=True):
        folders = glob.glob(dir)
        if len(folders) <= maxN:
            return None

        timestamps = [(f, os.path.getmtime(f)) for i, f in enumerate(folders) if os.path.isdir(f)]
        timestamps.sort(reverse=True, key=lambda x: x[1])

        for ts in timestamps[maxN:]:
            cmd = 'rm -r %s' % ts[0]
            if verbose:
                print(cmd)
            os.system(cmd)
        return None

    def delete_older_ckpt(self, dir, maxN=2, verbose=True):
        if not os.path.isdir(dir):
            return None

        files = glob.glob(os.path.join(dir, '*.pth'))
        if len(files) <= maxN:
            return None

        timestamps = [(file, os.path.getmtime(file)) for i, file in enumerate(files)]
        timestamps.sort(reverse=True, key=lambda x: x[1])

        for ts in timestamps[maxN:]:
            cmd = 'rm %s' % ts[0]
            if verbose:
                print(cmd)
            os.system(cmd)
        return None

    def train(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def adjust_learning_rate(self, type, step, optimizer, **config):
        raise NotImplementedError

    def set_param_lr(self, optims, lr):
        if isinstance(optims, list):
            for optim in optims:
                for param_group in optim.param_groups:
                    param_group['lr'] = lr
        else:
            for param_group in optims.param_groups:
                param_group['lr'] = lr
        return None

    def adjust_learning_rate_sqrt(self, step, optimizer, **config):
        max_lr = config.get('max_lr', None) or config.get('learning_rate', None)
        min_lr = config.get('min_lr', 1e-6)
        warm_steps = config.get('warm_steps', 0)
        init_steps = config.get('init_steps', 10000)

        if step < warm_steps:
            scale = step / warm_steps
        else:
            scale = (init_steps / (step - warm_steps + init_steps)) ** 0.5

        lr = max_lr * scale
        lr = max(lr, min_lr)
        self.set_param_lr(optimizer, lr)
        self.current_lr = lr
        return lr

    def adjust_learning_rate_cosine(self, step, optimizer, **config):
        start_lr = 0.000001
        max_lr = config.get('max_lr', None) or config.get('learning_rate', None)
        min_lr = config.get('min_lr', 1e-6)
        warm_steps = config.get('warm_steps', 0)
        total_steps = config['total_steps']

        if step >= total_steps and step < total_steps + 20:
            print('STOP TRAIN!!! step=%d lr=%0.6f' % (step, self.current_lr))

        if step < warm_steps:
            lr = ((max_lr - start_lr) * step) / warm_steps + start_lr
        else:
            step = min(step, total_steps - 1e-6)
            lr = max_lr * (math.cos(math.pi * (step - warm_steps) / (total_steps - warm_steps)) + 1) / 2

        lr = max(lr, min_lr)
        self.set_param_lr(optimizer, lr)
        self.current_lr = lr
        return lr

    def adjust_learning_rate_exponential(self, epoch, step, optimizer, **config):
        learning_rate = config['learning_rate']
        warm_steps = config.get('warm_steps', 0)
        lr_decay = config.get('lr_decay', 0)
        if step < warm_steps:
            scale = step / warm_steps
        else:
            scale = 1.0

        scale *= lr_decay ** epoch
        lr = learning_rate * scale

        self.set_param_lr(optimizer, lr)
        self.current_lr = lr
        return lr

    def save_model(self, epoch, step, loss,
                   save_dir,
                   higher_is_better=False,
                   max_time_not_save=60 * 60,
                   save_ema=True):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if not hasattr(self, 'best'):
            is_best = True
            self.best = loss
        else:
            is_best = loss > self.best if higher_is_better else loss < self.best

        long_time_save = True
        if hasattr(self, 'last_save_timestamp'):
            dt = time.time() - self.last_save_timestamp
            if max_time_not_save is None or dt < max_time_not_save:
                long_time_save = False

        if is_best or long_time_save:
            self.best = max(loss, self.best) if higher_is_better else min(loss, self.best)

            model_name = 'Model_E{}S{}_L{:.6f}.pth'.format(epoch, step, loss)
            model_path = os.path.join(save_dir, model_name)
            cprint('save model to >>> %s' % model_path, 'cyan')
            state = {'epoch': epoch,
                     'step': step,
                     'best': self.best,
                     'model': self.model.state_dict(),
                     'optimizer': self.optim.state_dict()
                     }
            if hasattr(self, 'model_ema') and save_ema:
                state['model_ema'] = self.model_ema.state_dict()
            torch.save(state, model_path)
            self.last_save_timestamp = time.time()
        return False

    def get_match_ckpt(self, model, ckpt_src):
        if model is None: return None
        ckpt = model.state_dict()
        for k, v in ckpt.items():
            k_without_module = k[7:] if 'module.' in k else k
            k_module = 'module.' + k_without_module
            if k_module in ckpt_src and ckpt[k].shape == ckpt_src[k_module].shape:
                ckpt[k] = ckpt_src[k_module].to(v.device)
            elif k_without_module in ckpt_src and ckpt[k].shape == ckpt_src[k_without_module].shape:
                ckpt[k] = ckpt_src[k_without_module].to(v.device)
            else:
                if self.rank == 0:
                    print('%s is not loaded.' % k)
        return ckpt


class DistributedTrainerBase(torch.multiprocessing.Process, TrainerBase):
    """
    DistributedTrainerBase
    """

    def __init__(self, ):
        super().__init__()
        pass

    def count_parameter(self, model, tag=''):
        num_total, num_learn = 0, 0
        for p in model.parameters():
            num_total += p.numel()
            num_learn += p.numel() if p.requires_grad else 0
        cprint(f'{tag} parameters: {(num_learn / 1e6):0.2f}M(learn)/{(num_total / 1e6):0.2f}M(total)', color='red')

    def test_dataloader_speed(self, dataloader):
        cnt = 0
        t0 = time.perf_counter()
        for i, mb in enumerate(dataloader):
            keyname = list(mb.keys())[0]
            cnt += len(mb[keyname])
            if time.perf_counter() - t0 > 2 or i >= 3:
                dt = time.perf_counter() - t0
                speed = cnt / dt
                print(f"[dataloader_speed] rank={self.rank} count={cnt}  dt={dt:0.2f}s speed={speed:0.2f}:samples/s ")
                cnt = 0
                t0 = time.perf_counter()
            if i >= 3:
                break
        return speed
