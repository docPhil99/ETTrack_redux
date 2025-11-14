from yolox.data.datasets import MOTDataset
from yolox.data.datasets.datasets_wrapper import Dataset
from pathlib import Path
import cv2
import os
import numpy as np
import json
from loguru import logger
import torch
import sys
from collections import defaultdict

class MOTDatasetET2(MOTDataset):
    def __init__(self, data_dir:str = None, json_file: str = "train_half.json", name:str = "train", img_size:tuple[int,int] = (800, 1440),
                 dataset:str = 'dancetrack', return_image=True,
                 yolo_detections_dir:Path=Path('yolo_outputs'), run_tracking = False,preproc = None, yolo_dets_filename: Path=Path('yolo_dets.pt')):
        #  use_stored = False, store_dir: Path = None, use_pickle = False)
        """
        :param data_dir:  dir of image dataset 
        :param json_file: 
        :param name:  train, test, val
        :param img_size: 
        :param preproc: 
        :param run_tracking: true for tracking mode, not for training
        """

        super().__init__(data_dir, json_file, name, img_size, preproc, run_tracking)
        self.data_dir = data_dir
        self.json_file = json_file
        self.name = self.name
        self.yolo_detections_dir = yolo_detections_dir
        self._return_image = return_image

        logger.debug(
            f'MOTDatasetET Data directory is {self.data_dir} name is {self.name}, json_file is {self.json_file}',
            f'yolo_detections_dir is {self.yolo_detections_dir}')
        filename = yolo_detections_dir / Path(dataset)/Path(name)/yolo_dets_filename
        logger.info(f"Processing yolo det file {filename}")
        all_outputs_list = torch.load(filename)

        self.all_outputs_dict = defaultdict(dict)  #conver to dict [filename][frame_num]   not zero based
        for dat in all_outputs_list:
            v_name, v_id, f_id, dets = dat
            self.all_outputs_dict[v_name][f_id.item()] = dets

    @Dataset.resize_getitem
    def __getitem__(self, index):
        return self.__getitem(index)

    def __getitem(self, index):
        img, target, img_info, img_id = self.pull_item(index)
        if self.preproc is not None:
            img, target, raw_image = self.preproc(img, target,
                                                  self.input_dim)  # array for [C, H, W], targets, raw_image [H, W, C]

        filename = self.annotations[index][2]
        key = filename.split('/')[0]  # file
        det = self.all_outputs_dict[key][img_info[2]]
        #logger.debug(f'key: {key}, img_info: {img_info}')
        return img, det, img_info, img_id,

    def pull_item(self, index):
        id_ = self.ids[index]

        res, img_info, file_name = self.annotations[index]
        # res, img_info, file_name = self.annotations[id_]

        # load image and preprocess
        img_file = os.path.join(
            self.data_dir, self.name, file_name
        )
        if self._return_image:
            img = cv2.imread(img_file)
            assert img is not None, f'failed to load {img_file}'
        else:
            img = 0
        return img, res.copy(), img_info, np.array([id_])


class MOTDatasetET(MOTDataset):

    def __init__( self, data_dir=None, json_file="train_half.json", name="train", img_size=(608, 1088),
            preproc=None, run_tracking=False,use_stored=False, store_dir: Path =None, use_pickle=False):
        super().__init__(data_dir,json_file,name,img_size,preproc,run_tracking)
        logger.debug(f'MOTDatasetET Data directory is {self.data_dir} name is {self.name}, json_file is {self.json_file}')
        self.store_dir = store_dir
        self.use_stored = use_stored
        self.stored_yolo = {}
        if use_stored:
            data_path = Path(self.store_dir)/Path(name)
            logger.debug(f'Loading save detections from {data_path}')
            if use_pickle:
                logger.info(f"Found {len([Path(data_path).glob('*.pt')])} files in {data_path}")
                for path in Path(data_path).glob('*.pt'):
                    data = torch.load(path)
                    self.stored_yolo[path.stem] = data
            else:
                logger.info(f"Found {len([Path(data_path).glob('*.json')])} files in {data_path}")
                for path in Path(data_path).glob('*.json'):
                    try:
                        with open(path, 'rt') as f:
                            data = json.load(f)
                            for key, value in data.items():
                                if value[0] is None:  # no detections
                                    data[key]= 'empty'
                                    logger.debug(f'{key} : {value}')
                                else:
                                    data[key] = torch.squeeze(torch.tensor(value))

                            self.stored_yolo[path.stem] = data
                    except Exception as e:
                        logger.exception(f'Failed to load or process {path}')
                        self.use_stored = False


    # def restrict_to_one_file(self, name:str):
    #     """update the ids so only one file is processed.
    #     name: filename part to search for eg 'MOT17-02-FRCNN'
    #     """
    #     logger.error("not implemented!")
    #     sys.exit(1)
    #     logger.info(f'Restrict dataset to one file {name}')
    #     self.ids = []
    #     for ind, anno in enumerate(self.annotations):
    #         if name in anno[2]:
    #             self.ids.append(ind)

    @Dataset.resize_getitem
    def __getitem__(self, index):

        if not self.use_stored:
            img, target, img_info, img_id = self.pull_item(index)

            if self.preproc is not None:
                img, target, raw_image = self.preproc(img, target,
                                                      self.input_dim)  # array for [C, H, W], targets, raw_image [H, W, C]

            if not self.run_tracking:  # TODO: [hgx 0427] dataloader related
                return img, target, img_info, img_id  # [hgx 0427] do not return 'raw_image' when training
            else:
                return img, target, img_info, img_id, raw_image  # [hgx 0427] return 'raw_image' when tracking

        else:

            return self.__getitem(index)


    def __getitem(self, index):
        img, target, img_info, img_id = self.pull_item(index)
        if self.preproc is not None:
            img, target, raw_image = self.preproc(img, target,
                                                  self.input_dim)  # array for [C, H, W], targets, raw_image [H, W, C]

        filename = self.annotations[index][2]
        #filename = self.annotations[self.ids[index]][2]
        key = filename.split('/')[0]  # file
        key_frame = f'{img_info[2]}'
        dets = self.stored_yolo[key]
        det = dets[key_frame]
        return img, det, img_info, img_id,

    def pull_item(self, index):
        id_ = self.ids[index]

        res, img_info, file_name = self.annotations[index]
        #res, img_info, file_name = self.annotations[id_]

        # load image and preprocess
        img_file = os.path.join(
            self.data_dir, self.name, file_name
        )
        img = cv2.imread(img_file)

        assert img is not None, f'failed to load {img_file}'
        return img, res.copy(), img_info, np.array([id_])

if __name__ == '__main__':
    import os
    import cv2
    from yolox.data import get_yolox_datadir
    import timeit
    dataset='dancetrack'
    DS = MOTDatasetET2(data_dir=os.path.join(get_yolox_datadir(), dataset),json_file='val.json',name="val",)

    def my_get():
        for indx in range(len(DS)):
            img, dets, img_info, img_id = DS.__getitem__(indx)

    logger.debug(timeit.timeit(my_get, number=1))

    DS._return_image = False
    logger.debug(timeit.timeit(my_get, number=1))


    DS._return_image = True
    for indx in range(len(DS)):
        img, dets, img_info, img_id = DS.__getitem__(indx)

    for indx in range(len(DS)):

        img, dets, img_info, img_id = DS.__getitem__(indx)
        img = cv2.resize(img,DS.img_size[::-1])

        tlwhs = dets.cpu().numpy()
        for i, tlwh in enumerate(tlwhs):
            x1, y1, w, h, c1, c2, c3 = tlwh
            intbox = tuple(map(int, (x1, y1, w, h)))

            cv2.rectangle(img, intbox[0:2], intbox[2:4], color=(255, 0, 0), thickness=2)

        cv2.imshow("img", img)
        q = cv2.waitKey(10)
        if q == ord('q'):
            break