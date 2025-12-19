import argparse
from loguru import logger
from pathlib import Path
from datetime import datetime

#from mpmath.ctx_mp_python import return_mpc

from trackers.ettrack.ettrack2 import Predict as Predict_Birch
from trackers.ettrack.datasets.MOTDatasetET import MOTDatasetET2
import torch
from tools.utils.MOTEvaluator import MOTEvaluator
from tools.utils.HOTA import run_hota_command

def make_parser():
    parser = argparse.ArgumentParser("ETTrack parameters")
    # ettrack
    parser.add_argument("--dataset_dir", type=Path, default='data/datasets', help="dataset directory")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name, use date if not set")
    parser.add_argument("--track_thresh", type=float, default=0.6, help="detection confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
    # parser.add_argument('-d_model', type=int, default=512)  # 521
    parser.add_argument('--ettrack_model_path', type=str, help='Your model name', required=True)
    parser.add_argument('--run_hota_command_only', action='store_true', help='run hota on pre-generated results')
    parser.add_argument("--dataset", type=str.lower, default="dancetrack", help="mot17, dancetrack, mot20",
                        choices=["dancetrack", "mot17", "mot20"])
    parser.add_argument('--exp_type', type=str, default='val', choices=['val', 'test', 'train'],
                        help="val or test dataset")
    parser.add_argument('--yolo_dump_dir', type=Path, default='data/yolo_outputs')
    parser.add_argument('--live_yolo', action='store_true', help='run live yolo on the dataset, rather than saved results')
    parser.add_argument('--yolox_weights', type=Path, default='pretrained/yolox/ocsort_x_mot17.pth.tar')

    #parser.add_argument('--restrict_file', type=str, help='restrict dataset to specific file')
    parser.add_argument('--output_dir', type=Path, default=Path('YOLOX_outputs'))
    parser.add_argument('--img_size', type=int, nargs=2, default=[800,1440])
    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument('--HOTA', action='store_true', help='run HOTA command only')
    parser.add_argument('--annotation_file',type=Path,default=None, help='name of annotation file. It will default to val.json, test.test, or train.json depending on --exp_type')
    return parser

def main(args,output_dir):


    # attempt to guess the annotation file name, depending on the dataset and experiment type.
    # Assuming --annotation_file is not set.
    if args.annotation_file:
        json_file = args.annotation_file
    else:
        if args.dataset in ['mot17','mot20']:
            if args.exp_type in ['val','train']:
                json_file = f'{args.exp_type}_half.json'
            else:
                json_file = f'{args.exp_type}.json'  #must be test
        else:
            json_file = f'{args.exp_type}.json'   #dancetrack
    # load dataset
    #if args.dataset == 'mot17':  # for some reason the MOT17 is put in datasets/mot
    #    args.dataset = 'mot'

    dataset_dir = args.dataset_dir / Path(args.dataset)
    logger.debug(f'Dataset directory: {dataset_dir}')
    dataloader_set = MOTDatasetET2(data_dir=dataset_dir, json_file=json_file, name=args.exp_type,
                                   dataset=args.dataset, img_size=args.img_size, return_image=args.live_yolo,
                                   run_tracking=False, yolo_detections_dir=args.yolo_dump_dir)

    sampler = torch.utils.data.SequentialSampler(dataloader_set)  # todo what does this do?
    dataloader_kwargs = {
        "num_workers": 0,
        # "pin_memory": True,
        "sampler": sampler,
        "batch_size": 1,  # todo
    }
    val_loader = torch.utils.data.DataLoader(dataloader_set, **dataloader_kwargs)

    # load tracker

    et = Predict_Birch(args, training=False)
    et_net = et.get_network()  # returns tnc_transformer
    # create evaluator
    evaluator = MOTEvaluator(et_net, val_loader, output_dir, args, args.img_size)

    # run evaluator
    *_, summary = evaluator.evaluate_ettrack()
    logger.info(f"Summary: {summary}")


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    if not args.exp_name:
        args.exp_name = datetime.now().strftime("%Y%m%d-%H%M%S")


    output_dir = args.output_dir / args.exp_name /Path(args.dataset)/Path(args.exp_type)
    logger.info(f"Output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.HOTA:
        main(args,output_dir)

    run_hota_command(args.dataset, str(output_dir), args.exp_type ,args.dataset_dir)


