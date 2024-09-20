import time
import copy
import torch
from torch import nn


def exists(val):
    return val is not None


def clamp(value, min_value=None, max_value=None):
    assert exists(min_value) or exists(max_value)
    if exists(min_value):
        value = max(value, min_value)

    if exists(max_value):
        value = min(value, max_value)

    return value


class EMA(nn.Module):
    """
    Implements exponential moving average shadowing for your model.

    Utilizes an inverse decay schedule to manage longer term training runs.
    By adjusting the power, you can control how fast EMA will ramp up to your specified beta.

    @crowsonkb's notes on EMA Warmup:

    If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are
    good values for models you plan to train for a million or more steps (reaches decay
    factor 0.999 at 31.6K steps, 0.9999 at 1M steps), gamma=1, power=3/4 for models
    you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999 at
    215.4k steps).

    Args:
        inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
        power (float): Exponential factor of EMA warmup. Default: 1.
        min_value (float): The minimum EMA decay rate. Default: 0.
    """

    def __init__(
            self,
            model,
            ema_model=None,
            # if your model has lazylinears or other types of non-deepcopyable modules, you can pass in your own ema model
            beta=0.9999,
            update_after_step=100,
            update_every=10,
            inv_gamma=1.0,
            power=2 / 3,
            min_value=0.0,
            param_or_buffer_names_no_ema=set(),
            ignore_names=set(),
            ignore_startswith_names=set(),
            include_online_model=True,
            start_step=0,
            ma_device=None,
            # set this to False if you do not wish for the online model to be saved along with the ema model (managed externally)
    ):
        super().__init__()
        self.device = self.model.device if ma_device is None else ma_device
        self.beta = beta

        # whether to include the online model within the module tree, so that state_dict also saves it

        self.include_online_model = include_online_model

        if include_online_model:
            self.online_model = model
        else:
            self.online_model = [model]  # hack

        # ema model

        self.ema_model = ema_model

        if not exists(self.ema_model):
            try:
                # self.ema_model = copy.deepcopy(model.to(self.device))
                self.ema_model = copy.deepcopy(model).to(self.device)
            except:
                print('Your model was not copyable. Please make sure you are not using any LazyLinear')
                exit()

        self.ema_model.requires_grad_(False)

        self.parameter_names = {name for name, param in self.ema_model.named_parameters() if param.dtype == torch.float}
        self.buffer_names = {name for name, buffer in self.ema_model.named_buffers() if buffer.dtype == torch.float}

        self.update_every = update_every
        self.update_after_step = update_after_step

        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value

        assert isinstance(param_or_buffer_names_no_ema, (set, list))
        self.param_or_buffer_names_no_ema = param_or_buffer_names_no_ema  # parameter or buffer

        self.ignore_names = [name for name, p in self.model.named_parameters() if p.requires_grad == False]
        self.ignore_startswith_names = ignore_startswith_names

        self.register_buffer('initted', torch.Tensor([False]))
        self.register_buffer('step', torch.tensor([start_step]))

    @property
    def model(self):
        return self.online_model if self.include_online_model else self.online_model[0]

    def state_dict(self):
        return self.ema_model.state_dict()

    def restore_ema_model_device(self):
        device = self.initted.device
        self.ema_model.to(device)

    def get_params_iter(self, model):
        for name, param in model.named_parameters():
            if name not in self.parameter_names:
                continue
            yield name, param

    def get_buffers_iter(self, model):
        for name, buffer in model.named_buffers():
            if name not in self.buffer_names:
                continue
            yield name, buffer

    def copy_params_from_model_to_ema(self):
        for (_, ma_params), (_, current_params) in zip(self.get_params_iter(self.ema_model),
                                                       self.get_params_iter(self.model)):
            ma_params.data.copy_(current_params.data)

        for (_, ma_buffers), (_, current_buffers) in zip(self.get_buffers_iter(self.ema_model),
                                                         self.get_buffers_iter(self.model)):
            ma_buffers.data.copy_(current_buffers.data)

    def get_current_decay(self):
        step = clamp(self.step.item() - self.update_after_step - 1, min_value=0.)
        value = 1 - (1 + step / self.inv_gamma) ** - self.power
        # print(f'[get_current_decay] value={value}')
        if step <= 0:
            return 0.

        return clamp(value, min_value=self.min_value, max_value=self.beta)

    def update(self):
        step = self.step.item()
        self.step += 1

        if (step % self.update_every) != 0:
            return

        if step <= self.update_after_step:
            self.copy_params_from_model_to_ema()
            return

        if not self.initted.item():
            self.copy_params_from_model_to_ema()
            self.initted.data.copy_(torch.Tensor([True]))
        t0 = time.perf_counter()
        self.update_moving_average(self.ema_model, self.model)
        # print(f'ema cost {time.perf_counter() - t0:0.4f}sec')

    @torch.no_grad()
    def update_moving_average(self, ma_model, current_model):
        current_decay = self.get_current_decay()

        for (name, current_params), (_, ma_params) in zip(self.get_params_iter(current_model),
                                                          self.get_params_iter(ma_model)):
            if name in self.ignore_names:
                continue

            if any([name.startswith(prefix) for prefix in self.ignore_startswith_names]):
                continue

            if name in self.param_or_buffer_names_no_ema:
                ma_params.data.copy_(current_params.data)
                continue

            # ma_params.data.lerp_(current_params.data, 1. - current_decay)
            ma_params.data = self.update_tensor(ma_params.data, current_params.data, current_decay,
                                                self.device)

        for (name, current_buffer), (_, ma_buffer) in zip(self.get_buffers_iter(current_model),
                                                          self.get_buffers_iter(ma_model)):
            if name in self.ignore_names:
                continue

            if any([name.startswith(prefix) for prefix in self.ignore_startswith_names]):
                continue

            if name in self.param_or_buffer_names_no_ema:
                ma_buffer.data.copy_(current_buffer.data)
                continue

            # ma_buffer.data.lerp_(current_buffer.data, 1. - current_decay)
            ma_buffer.data = self.update_tensor(ma_buffer.data, current_buffer.data, current_decay,
                                                self.device)

    # def update_tensor(self, src, dst, decay, src_device, dst_device):
    #     src = src.to(dst_device)
    #     # y = src * decay + dst * (1 - decay)
    #     src.lerp_(dst, 1. - decay)
    #     # diff = (y - src).abs().mean()
    #     # if diff > 1e-4:
    #     #     print(f'diff={diff}')
    #     return src.to(src_device)

    def update_tensor(self, ema, dst, decay, src_device):
        # ema = ema * decay + dst.to(src_device) * (1 - decay)
        ema.lerp_(dst.to(src_device), 1. - decay)
        # diff = (y - src).abs().mean()
        # if diff > 1e-4:
        #     print(f'diff={diff}')
        return ema.to(src_device)

    def __call__(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)
