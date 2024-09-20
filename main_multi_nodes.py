import os
from helper import set_cuda_visible_devices

set_cuda_visible_devices()

import json
import argparse
import torch
from pprint import pprint
from helper import AttrDict, cprint, get_port, get_opt_from_python_config
import importlib


def dynamic_import(train_script_name):
    module_name = f'train.{train_script_name}'
    class_name = 'TrainProcess'

    try:
        module = importlib.import_module(module_name)
        train_class = getattr(module, class_name)
        return train_class
    except (ModuleNotFoundError, AttributeError) as e:
        print(f"Error importing {class_name} from {module_name}: {e}")
        return None


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', type=str, default='tmp', help='tag')
    parser.add_argument('--config_file', type=str, default='configs/config_debug.py', help='')
    parser.add_argument('--total_batch_size', type=int, default=None)

    opt = parser.parse_args()
    opt = AttrDict(vars(opt))
    return opt


def main(rank, local_rank):
    if rank == 0:
        cprint('=> torch version : {}'.format(torch.__version__), 'blue')
        cprint('Initializing Training Process..', 'yellow')

    opt_default = parse_arguments()
    opt = get_opt_from_python_config(opt_default.config_file)
    for key, value in vars(opt_default).items():
        if value is not None:
            setattr(opt, key, value)

    opt.model_save_dir = os.path.join('ckpts/', opt.tag)

    opt.world_size = int(os.environ.get('WORLD_SIZE', '1'))
    opt.gradient_accumulation_steps = max(1, opt.total_batch_size // (opt.world_size * opt.batch_size_per_gpu))

    cprint(f"WORLD_SIZE: {opt.world_size}, RANK: {rank}, LOCAL_RANK: {local_rank}", 'red')

    if rank == 0:
        tensorboard_logpath = os.path.join(opt.model_save_dir, 'logs')
        os.system('rm -r %s/*.*' % tensorboard_logpath)
        pprint(vars(opt))

    train_script_name = opt.train_script_name if hasattr(opt, 'train_script_name') else 'train_bucket'
    TrainProcess = dynamic_import(train_script_name)
    p = TrainProcess(rank, local_rank, opt)

    # Save opt as a JSON file
    opt_json_path = os.path.join(opt.model_save_dir, 'opt.json')
    if rank == 0 and not os.path.exists(opt.model_save_dir):
        os.makedirs(opt.model_save_dir)
        with open(opt_json_path, 'w') as f:
            json.dump(opt, f, indent=4)

    p.run()


if __name__ == '__main__':
    # os.environ['MASTER_ADDR'] = "localhost"
    # os.environ['MASTER_PORT'] = "8000"
    rank = int(os.environ.get("RANK", '0'))
    local_rank = int(os.environ.get("LOCAL_RANK", '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))

    ## only use for python debug:
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = "localhost"
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = get_port()

    main(rank, local_rank)

###
# torchrun --nproc_per_node=2 main_multi_nodes.py --config_file='configs/config_debug.py' --total_batch_size=32
