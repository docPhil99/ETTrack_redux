from yolox.exp import Exp as MyExp
from loguru import logger
from trackers.ettrack.datasets.MOTDatasetET import MOTDatasetET2
#from yolox.data import MOTDataset
from pathlib import Path
import torch

class Exp(MyExp):
    def __init__(self,args):
        super(Exp, self).__init__()
        self.args = args
        self.depth = 1.33
        self.width = 1.25
        self.num_classes = 1

        if args.annotation_file:
            self.json_file = args.annotation_file
        else:
            self.json_file = f'{args.exp_type}.json'
        logger.info(f"Processing {self.json_file}")
        if not args.img_size:
            args.img_size = (800, 1440)
        self.input_size = args.img_size
        self.test_mode = args.img_size



    def get_data_loader(self):
        dataset = self.args.dataset
        if dataset == 'mot17':
            dataset='mot'
        data_dir = self.args.dataset_dir/Path(dataset)
        dataloader_set = MOTDatasetET2(data_dir=data_dir, json_file=self.json_file, name=self.args.type,
                                       dataset=self.args.dataset, img_size=self.args.img_size, return_image=True,
                                       run_tracking=False, yolo_detections_dir=None,resize=self.input_size)

        sampler = torch.utils.data.SequentialSampler(dataloader_set)  # always sample in same order.
        dataloader_kwargs = {
            "num_workers": 0,
            # "pin_memory": True,
            "sampler": sampler,
            "batch_size": self.args.batch_size,  # todo
        }
        val_loader = torch.utils.data.DataLoader(dataloader_set, **dataloader_kwargs)
        return val_loader
