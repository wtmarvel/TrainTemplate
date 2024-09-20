import random
import numpy as np
import cv2
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        self.rank = kwargs.get('rank', None)
        self.training = kwargs.get('training', True)
        self.verbose = kwargs.get('verbose', False)
        self.image_size = kwargs.get('image_size', False)
        self.datalist = kwargs.get('train_data_list') if self.training else kwargs.get('test_data_list')
        self.N = 10000

    def __len__(self):
        if self.training:
            return max(self.N, 10000)
        else:
            return self.N

    def __getitem__(self, index):
        image = torch.randint(0, 255, (self.image_size, self.image_size, 3), dtype=torch.uint8)
        label = torch.zeros((), dtype=torch.float32)

        return {"image": image,
                "label": label,
                }


class CollateFn():
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, batch):
        batch_stack = {}
        for item in batch:
            for k, v in item.items():
                if k in batch_stack:
                    batch_stack[k].append(v)
                else:
                    batch_stack.append([v, ])

        for k, v in batch_stack.items():
            batch_stack[k] = torch.stack(v)

        return batch_stack
