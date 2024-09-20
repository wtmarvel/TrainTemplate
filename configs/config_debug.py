tag = 'tmp'
train_script_name = 'train'

###############Train#########################
enable_amp = True
dtype = "fp16"  # float32 fp16 bf16 fp8

num_gpus = 1
image_size = 224
batch_size_per_gpu = 8
total_batch_size = batch_size_per_gpu * num_gpus

# world_size = num_gpus
print_freq = 10  # don't print too often
test_freq = print_freq * 20  # also the model-saving frequency
seed = 3507
compile = False
num_max_save_models = 40
test_distributed = True
test_before_train = False

# set to true if model is differ than original
use_gradient_ckpt = True  # unet_requires_grad
find_unused_parameters = False
# load_ckpt = '/'

###############Model EMA#####################
use_ema = True
ema_beta = 0.97  # ema_update_every=20
ema_update_every = 2 * print_freq  # every update cost about 7sec

###############Optimizer#####################
gradient_accumulation_steps = 1
warm_steps = 1000
total_steps = 10000000
max_epochs = 9999999
learning_rate = 1e-4
min_lr = 5e-6  # learning_rate / 10 usually
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95  # make a bit bigger when batch size per iter is small

###############Net Structure#########################


###############DataLoader#####################
num_dl_workers = 1  # number of dataloader processes per GPU(process)
train_dataset = []
test_dataset = []
