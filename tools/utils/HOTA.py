from loguru import logger
import os
from pathlib import Path

def run_hota_command(dataset: str, results_folder: str, mode:str, data_set_dir:Path):
    assert mode in ["train", "val", "test"]
    if mode =='test':
        logger.info('Test set used. No ground truth to compare against.')
        return


    if dataset == "dancetrack":
        gt_folder = data_set_dir / Path(dataset)/Path(mode)
        seq_file = gt_folder/Path(f'{mode}_seqmap.txt')

        # f"--GT_FOLDER datasets/dancetrack/{mode} " \
        # f"--SEQMAP_FILE datasets/dancetrack/{mode}/{mode}_seqmap.txt " \

        hota_command = f"python3 scripts/TrackEval/scripts/run_mot_challenge.py " \
                       "--SPLIT_TO_EVAL val  " \
                       "--METRICS HOTA CLEAR Identity " \
                       f"--GT_FOLDER {str(gt_folder)} " \
                       f"--SEQMAP_FILE {str(seq_file)} " \
                       "--SKIP_SPLIT_FOL True " \
                       "--TRACKERS_TO_EVAL '' " \
                       "--TRACKER_SUB_FOLDER ''  " \
                       "--USE_PARALLEL True " \
                       "--NUM_PARALLEL_CORES 8 " \
                       "--PLOT_CURVES False " \
                       "--TRACKERS_FOLDER " + results_folder
    elif dataset == "mot17":
        hota_command = "python scripts/TrackEval/scripts/run_mot_challenge.py " \
                       "--BENCHMARK MOT17 " \
                       "--SPLIT_TO_EVAL train " \
                       "--TRACKERS_TO_EVAL '' " \
                       "--METRICS HOTA CLEAR Identity VACE " \
                       "--TIME_PROGRESS False " \
                       "--USE_PARALLEL False " \
                       "--NUM_PARALLEL_CORES 1  " \
                       "--GT_FOLDER datasets/mot/ " \
                       "--TRACKERS_FOLDER " + results_folder + " " \
                       "--GT_LOC_FORMAT {gt_folder}/{seq}/gt/gt.txt"
                       #"--GT_LOC_FORMAT {gt_folder}/{seq}/gt/gt_val_half.txt"
    elif dataset == "mot20":
        hota_command = "python scripts/TrackEval/scripts/run_mot_challenge.py " \
                       "--BENCHMARK MOT20 " \
                       "--SPLIT_TO_EVAL train " \
                       "--TRACKERS_TO_EVAL '' " \
                       "--METRICS HOTA CLEAR Identity VACE " \
                       "--TIME_PROGRESS False " \
                       "--USE_PARALLEL False " \
                       "--NUM_PARALLEL_CORES 1  " \
                       "--GT_FOLDER datasets/MOT20/ " \
                       "--TRACKERS_FOLDER " + results_folder + " " \
                       "--GT_LOC_FORMAT {gt_folder}/{seq}/gt/gt_val_half.txt"

    logger.debug(f'Hota command: {hota_command}')
    os.system(hota_command)


