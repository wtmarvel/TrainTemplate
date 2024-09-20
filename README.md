# PyTorch Multi-Nodes Training Template

## Quick Start
This guide offers several methods to initiate model training. Select the option that best suits your needs and environment.

### Method 1 (Recommended)
Use torchrun to parallelly start multiple processes for training (recommended for multi-node environments).
```bash
torchrun --nproc_per_node=1 main_multi_nodes.py --config_file='configs/config_debug.py' --total_batch_size=32
```

### Method 2 (Supports Single Node Only)
Directly run the training using the python command.
```bash
python main.py --config_file='configs/config_debug.py'
```