from trackers.ettrack.byte_track import byte_tp
from tqdm import tqdm
from loguru import logger
from pathlib import Path
from collections import defaultdict
import time
import torch
from .utils import write_results
from yolox.utils.boxes import xyxy2xywh

import contextlib
import tempfile
import json
import io


def time_synchronized():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


class MOTEvaluator():
    def __init__(self, tracking_network, dataloader, results_dir: Path,args ,image_size:tuple):
        self.tracking_network = tracking_network
        self.dataloader = dataloader
        self.args = args
        self.results_dir = results_dir
        self.img_size = image_size


    def evaluate_ettrack(self):

        results = []
        data_list = []
        video_names = defaultdict()
        inference_time = 0
        track_time = 0
        tracker = None

        for cur_iter, (imgs, dets, info_imgs, ids) in enumerate(tqdm(self.dataloader)):
            frame_id = info_imgs[2].item()
            img_file_name = info_imgs[4]
            video_name = img_file_name[0].split('/')[0]
            video_id = info_imgs[3].item()
            if video_name not in video_names:   # store the list of video file names, so we can save the previous
                video_names[video_id] = video_name

            #set up and save last results
            if frame_id == 1:
                # first frame, create tracker
                tracker = byte_tp(self.tracking_network, self.args)


                if (len(results)) != 0:
                    results_filename = self.results_dir/Path(f'{video_names[video_id-1]}.txt')
                    write_results(results_filename, results)
                    results = []
                logger.info(f'Processing file {video_name}')
            # skip the last iters since batchsize might be not enough for batch inference
            is_time_record = cur_iter < len(self.dataloader) - 1
            if is_time_record:
                start = time.time()
            outputs = dets
            if is_time_record:
                infer_end = time_synchronized()
                inference_time += infer_end - start
            if outputs[0].dim() == 1:
                logger.debug('wrong dim')
                outputs[0]=outputs[0].unsqueeze(0)
            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            #run tracker
            online_targets = tracker.update(outputs[0], info_imgs, self.img_size)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                if tlwh[2] * tlwh[3] > self.args.min_box_area:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
            # save results
            results.append((frame_id, online_tlwhs, online_ids, online_scores))
            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            #if cur_iter == len(self.dataloader) - 1:  # save the final video
        logger.debug('save final result')
        results_filename = self.results_dir / Path(f'{video_names[video_id]}.txt')
        write_results(results_filename, results)

        n_samples = len(self.dataloader) - 1
        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        eval_results = self.evaluate_prediction(data_list, statistics)
        #synchronize()
        return eval_results

    def convert_to_coco_format(self, outputs, info_imgs, ids):
        data_list = []
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()
            if output.dim() == 1:   #PMB I think there is a bug where one object is not passed as 2D array
                output = output.unsqueeze(0)
                logger.debug('unsqueezed')

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            bboxes = xyxy2xywh(bboxes)

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)
        return data_list

    def evaluate_prediction(self, data_dict, statistics):
        #if not is_main_process():
        #    return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        track_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_track_time = 1000 * track_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "track", "inference"],
                    [a_infer_time, a_track_time, (a_infer_time + a_track_time)],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco
            # TODO: since pycocotools can't process dict in py36, write data to json file.
            _, tmp = tempfile.mkstemp()
            json.dump(data_dict, open(tmp, "w"))
            cocoDt = cocoGt.loadRes(tmp)
            from yolox.layers import COCOeval_opt as COCOeval
            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info

