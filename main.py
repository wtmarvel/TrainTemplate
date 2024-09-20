import os
import argparse
import torch
import torch.multiprocessing as mp
from pprint import pprint
from helper import AttrDict, get_gpu_id, cprint, get_port, get_opt_from_python_config
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


def main():
    cprint('=> torch version: {}'.format(torch.__version__), 'blue')
    cprint('Initializing Training Process..', 'yellow')

    opt_default = parse_arguments()
    opt = get_opt_from_python_config(opt_default.config_file)
    for key, value in vars(opt_default).items():
        if value is not None:
            setattr(opt, key, value)

    opt.model_save_dir = os.path.join('ckpts/', opt.tag)
    tensorboard_logpath = os.path.join(opt.model_save_dir, 'logs')
    os.system('rm -r %s/*.*' % tensorboard_logpath)

    opt.world_size = opt.num_gpus
    opt.gradient_accumulation_steps = max(1, opt.total_batch_size // (opt.world_size * opt.batch_size_per_gpu))
    os.environ['CUDA_VISIBLE_DEVICES'] = get_gpu_id(opt.world_size)
    os.environ['WORLD_SIZE'] = str(opt.world_size)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = get_port()
    # os.environ['NCCL_DEBUG'] = 'INFO'
    # os.environ['TORCH_CUDNN_V8_API_DISABLED'] = '1'
    pprint(vars(opt))

    train_script_name = opt.train_script_name if hasattr(opt, 'train_script_name') else 'train'
    TrainProcess = dynamic_import(train_script_name)

    if 1 or opt.world_size > 1:
        processes = []
        mp.set_start_method("spawn", force=True)
        for rank in range(opt.world_size):
            os.environ['RANK'] = str(rank)
            p = TrainProcess(rank, rank, opt)
            p.start()
            processes.append(p)
        for p in processes:
            p.join()  # the main process is blocked until the process p completes its execution
    else:
        rank = 0
        os.environ['RANK'] = str(rank)
        p = TrainProcess(rank, rank, opt)
        p.run()


if __name__ == '__main__':
    main()
