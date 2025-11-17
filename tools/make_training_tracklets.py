from loguru import logger
from tools.utils.TrackletDataset import make_data_set_json_files
import argparse
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_help('generates the processed json file from gt.txt files.')
parser.add_argument('--output_dir',type=Path,default=Path('data/datasets/processed_output'))
parser.add_argument('--datasets',type=Path, nargs='+' ,required=True)
parser.add_argument('--type',type=str, choices=['train', 'val','mix'], help="Which dataset to use. mix will split the train set into "
                                                                            "train and val sets.")
parser.add_argument('--filter',type=str,default=None, help='If set, the string must appear in the directory '
                                                           'name. eg --filter "FRCNN" will only process directories with FRCNN in name. '
                                                           'This is needed for MOT17.')
args = parser.parse_args()

logger.info(f'Processing {args.datasets}')
make_data_set_json_files(args.datasets,args.type, out_path=args.output_dir,filter=args.filter)
