import argparse
import os
from loguru import logger
import torch
import yaml
import sys
from pathlib import Path
import datetime
print(f'file: {Path(__file__).parents[1]}')
sys.path.append(str(Path(__file__).parents[1]))
print(sys.path)
from trackers.ettrack.ettrack2 import Predict
# Use Deterministic mode and set random seed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)

import sys
# Define a custom argument type for a list of strings
def list_of_strings(arg):
    return arg.split(',')

def get_parser():
    parser = argparse.ArgumentParser(
        description='ETTrack Trainer')
    parser.add_argument('--dataset', default='dancetrack',   nargs='+', choices=['dancetrack','MOT17','MOT20']) # MOT20 dancetrack sportsmot
    parser.add_argument('--dataset_root_path', default=Path('datasets/processed_output'), type=Path)
    parser.add_argument('--single_dataset_file', default=None, type=Path, help="only use the single dataset json file specified.")
    parser.add_argument('--save_dir', type=Path, required=True, help='Directory root to save checkpoints')
    parser.add_argument('--model_name', default='et_track', help='Your model name')

    parser.add_argument('--ettrack_model_path', type=Path, help='path to ettrack model, if set overrides '
                                                                'the default location when loading the model')
    parser.add_argument('--device', type=str, default='gpu', choices=['gpu', 'cpu'])
    parser.add_argument('--phase', default='train', help="Set this value to 'train' or 'test'", choices=['train', 'test'])

    parser.add_argument('--load_model_epoch', default=1, type=int, help="load pretrained model epoch number for test or training")

    parser.add_argument('--seq_length', default=20, type=int)

    parser.add_argument('--pred_length', default=12, type=int)

    parser.add_argument('--batch_size', default=16, type=int)  # 4
    parser.add_argument('--test_batch_size', default=16, type=int) #4

    parser.add_argument('--num_epochs', default=2, type=int)

    parser.add_argument('--learning_rate', default=0.0015, type=float)
    parser.add_argument('--ignore_direction_loss',action='store_true', help="ignore direction loss function in training ")

    parser.add_argument('--augmentation', action='store_true', help="Augment the tracks")
    parser.add_argument('--augmentation_list', default='all',  type=list_of_strings, help="Augmentation list, coma separated list: lr,ud,time")
    parser.add_argument('-lr_mul', type=float, default=2.0)
    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-dropout', type=float, default=0.1)
    logger.debug('created parser')
    return parser


def load_arg(p):
    # save arg
    logger.info(f'Attempting to load {p.config}')
    if os.path.exists(p.config):
        logger.info(f'Loading config: {p.config}')
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                try:
                    assert (k in key)
                except:
                    s = 1
        parser.set_defaults(**default_arg)
        return parser.parse_args()
    else:
        logger.debug('No config file to load')
        return False


def save_arg(args):
    # save arg
    arg_dict = vars(args)
    #if not os.path.exists(args.model_dir):
    #    os.makedirs(args.model_dir)
    logger.info(f"Saving config {args.config}")
    with open(args.config, 'w') as f:
        yaml.dump(arg_dict, f)


if __name__ == '__main__':

    parser = get_parser()
    p = parser.parse_args()
    ds = '_'.join(p.dataset)
    p.save_dir = p.save_dir / Path(p.model_name) / Path(ds)

    logger.add(p.save_dir/Path(f"Log_{datetime.datetime.today().strftime('%d_%m_%Y')}.txt"), level='DEBUG')
    logger.debug(f'Starting trainval with torch version {torch.__version__}')

    #check the CWD
    logger.info(f'Current working directory: {Path.cwd()}')
    if Path.cwd().stem != 'ETTrack2':
        logger.error(f'CWD should be project root, not {Path.cwd()}')
        sys.exit(1)


    #p.save_dir = p.save_base_dir + str(p.test_set) + '/'
    logger.info(f'Save directory is: {p.save_dir}')
    p.save_dir.mkdir(parents=True, exist_ok=True)
    p.config = p.save_dir/ Path(f'config_{p.phase}.yaml')

    #if not load_arg(p):
    save_arg(p)

    #args = load_arg(p)
    args = p
    torch.cuda.set_device(0)
    logger.info(f'Args: {args}')
    trainer = Predict(args,training=True)

    if args.phase == 'test':
        logger.info('Starting test phase')
        net = trainer.test()
        print("well,done")
    else:
        logger.info('Starting training phase')
        trainer.train()
