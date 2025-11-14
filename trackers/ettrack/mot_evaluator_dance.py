from loguru import logger
from yolox.evaluators import MOTEvaluatorDance
import torch
from collections import defaultdict
from tqdm import tqdm
from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)
import itertools
import time
import os
from utils.utils import write_results
from trackers.ettrack.byte_track import byte_tp
import json
from json import JSONEncoder
from pathlib import Path

class Tensor2List(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()

class MOTEvaluatorET(MOTEvaluatorDance):
    def __init__(self, args, dataloader, img_size, confthre, nmsthre, num_classes,
                 use_saved_detections: bool = False, save_detections_dir : Path = None):
        super().__init__(args, dataloader, img_size, confthre, nmsthre, num_classes)
        self._use_saved_detections = use_saved_detections
        self._save_detections_dir = save_detections_dir


    def evaluate_ettrack(self, args, model, distributed=False, half=False, trt_file=None, decoder=None,
                         test_size=None, result_folder=None, net=None, store_yolo=False, use_pickle=True):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.
        NOTE: This function will change training mode to False, please save states if needed.
        Args:
            args : command line arguments
            model : model to evaluate.
            trt_file:
            decoder: used for trt
            test_size:
            net: ettrack network
            store_yolo: save yolo detections to file
        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1
        #if self._use_saved_detections and not self.dataloader.dataset.use_stored: #dataload failed to find stored.
        #    store_yolo = True
        #    self._use_saved_detections = False
        if store_yolo:
            self._save_detections_dir.mkdir(parents=True, exist_ok=True)

        tracker = byte_tp(net, args)
        yolo_dump={}
        # dataloader.__getitem__ returns img: ndarray, labels :[max_labels, 5] each label is [class, xc,yc,h],
        # info_img : (h, w, nh, nw, dx, dy), img_id: int

        for cur_iter, (imgs, dets, info_imgs, ids) in enumerate(progress_bar(self.dataloader)):
            with torch.no_grad():
                # init tracker

                #img = torch.ones((1,3,800,1440), dtype=torch.float16, device='cuda')
                #dets = model(img)
                #outputs = postprocess(dets, self.num_classes, self.confthre, self.nmsthre)
                #pass

                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]
                #logger.debug(f'{video_name} {frame_id}')
                if dets ==['empty']:
                    dets = [None]
                else:
                    dets = [torch.squeeze(dets).cuda()]
                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:  #it's the first frame
                    tracker = byte_tp(net, args)

                    # save the results of the previous video
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results(result_filename, results)
                        results = []
                        if store_yolo:
                            if use_pickle:
                                yolo_file = self._save_detections_dir/Path(f'{video_names[video_id - 1]}.pt')
                                torch.save(yolo_dump, yolo_file)
                            else: # json
                                yolo_file = self._save_detections_dir / Path(f'{video_names[video_id - 1]}.json')
                                with open(yolo_file, 'wt') as f:
                                    json.dump(yolo_dump, f, cls=Tensor2List)
                            logger.info(f'Saved {yolo_file}')

                            yolo_dump = {}
                if not self._use_saved_detections:
                    imgs = imgs.type(tensor_type)

                    # skip the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()
                    # get detector outputs   #pmb run yolox
                if not self._use_saved_detections:
                    outputs = model(imgs)
                    if decoder is not None:
                        outputs = decoder(outputs, dtype=outputs.type())
                    outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
                    if store_yolo:
                        yolo_dump[frame_id] = outputs
                else:
                    outputs = dets
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start
            if outputs[0].dim() == 1:
                logger.debug('wrong dim')
                outputs[0]=outputs[0].unsqueeze(0)
            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            # run tracking
            #logger.debug(outputs[0])
            if info_imgs[2]==torch.tensor(248) and 'dancetrack0013' in info_imgs[4][0]:
                logger.debug(f'good {outputs[0]}, {info_imgs}')
            try:
                online_targets = tracker.update(outputs[0], info_imgs, self.img_size)
            except IndexError:
                logger.debug(f'Index Error {outputs[0]}, {info_imgs}')
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

            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results(result_filename, results)
                if store_yolo:
                    yolo_file = self._save_detections_dir / Path(f'{video_names[video_id]}.json')
                    # yolo_file.mkdir(parents=True, exist_ok=True)
                    logger.info(f'Saving {yolo_file}')
                    with open(yolo_file, 'wt') as f:
                        json.dump(yolo_dump, f, cls=Tensor2List)
                    yolo_dump = {}

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results
