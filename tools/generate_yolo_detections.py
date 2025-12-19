from loguru import logger
from pathlib import Path
import argparse
import torch
#import exp.yolox_dancetrack_val_ettrack as Exp
from tools.utils.Exp import Exp
from yolox.utils import fuse_model, get_model_info, setup_logger, postprocess
from tqdm import tqdm
import cv2
import numpy as np
import os


def playback(args, val_loader,yolo_dump_dir):
    #output_path = args.yolo_outputs / Path(args.dataset)
    current_pt_file = None
    current_video_file = None
    rgb_means = (0.485, 0.456, 0.406),
    std = (0.229, 0.224, 0.225)

    filename = yolo_dump_dir / Path(f"yolo_dets.pt")
    all_outputs_list = torch.load(filename)
    logger.info(f"Processing {filename}")

    for cur_iter, data in enumerate(tqdm(val_loader)):
        (imgs, dets, info_imgs, ids) = data
        frame_id = info_imgs[2]
        video_id = info_imgs[3]
        img_file_name = info_imgs[4]
        video_name = [a.split('/')[0] for a in img_file_name]



        for vid_name, vid_id, f_id, img in zip(video_name, video_id, frame_id, imgs):
            # output_dict[vid_name] = {'video_id':vid_id,[f_id] = outs
            if current_video_file != vid_name:
        #        filename = yolo_dump_dir / Path(f"{vid_name}.pt")
        #        outputs_list = torch.load(filename)
                logger.info(f"Processing {vid_name}")
                current_video_file = vid_name
                outputs_list = [a for a in all_outputs_list if a[0]==vid_name]
            #logger.debug(f"Processing {vid_name}")
            nimg = img.numpy()
            # really should be (img*std+mean)*255
            nimg -=np.min(nimg)
            nimg /= np.max(nimg)
            nimg *=255.0
            nimg = nimg.astype(np.uint8)
            nimg = np.transpose(nimg, (1, 2, 0))
            nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)

            tlwhs = outputs_list[f_id.item()-1][3]  #f_id starts at 1
            tlwhs = tlwhs.cpu().numpy()
            for i, tlwh in enumerate(tlwhs):
                x1, y1, w, h ,c1,c2 ,c3= tlwh
                intbox = tuple(map(int, (x1, y1,  w, h)))

                cv2.rectangle(nimg, intbox[0:2], intbox[2:4], color=(255,0,0), thickness=2)

            cv2.imshow("img", nimg)
            q=cv2.waitKey(10)
            if q==ord('q'):
                return

if __name__ == "__main__":
    parser= argparse.ArgumentParser()
    parser.add_argument('-c','--yolox_weights',type=Path, help='yolox weights file', default=Path('pretrained/bytetrack_dance_model.pth.tar'))
    parser.add_argument("-b", "--batch-size", type=int, default=8, help="batch size")
    parser.add_argument("-d", "--dataset", type=str.lower, default='dancetrack', help="dataset name, eg dancetrack, MOT17")
    parser.add_argument("--dataset_dir",type=Path, default='data/datasets', help="dataset directory")
    parser.add_argument('--yolo_outputs',type=Path,default='data/yolo_outputs', help="yolox detection output directory")
    parser.add_argument('-t','--exp_type',type=str,choices=['val','train','test'], default='val')
    parser.add_argument("--test",action="store_true",help="test, by showing replay")
    parser.add_argument('--annotation_file', type=Path, default=None,
                        help='name of annotation file. It will default to val.json, test.test, or train.json depending on --exp_type')
    parser.add_argument('--img_size', type=int, nargs=2, default=None)
    args = parser.parse_args()


    if args.dataset=='mot17':
        d_path=Path('mot')
    else:
        d_path=Path(args.dataset)
    yolo_dump_dir = args.yolo_outputs /d_path / Path(args.exp_type) # dir to store detections
    yolo_dump_dir.mkdir(parents=True, exist_ok=True)

    exp = Exp(args)
    val_loader = exp.get_data_loader()

    #exp = Exp.Exp()
    #os.environ["YOLOX_DATADIR"] = str(args.dataset_dir)
    #exp.val_ann = "val_half.json"
    #val_loader = exp.get_eval_loader(args.batch_size, False, testdev=False, run_tracking=False, yolo_dump_dir=None
    #                    ,use_stored_yolo=False, dataset ="mot", restrict_file=None ,use_pickle=False)

    if args.test:
        playback(args,val_loader, yolo_dump_dir)
        exit(0)


    yolo_model = exp.get_model()

    yolo_model.cuda()
    yolo_model.eval()

    #load weights
    weights = torch.load(args.yolox_weights,map_location='cuda:0')
    yolo_model.load_state_dict(weights["model"])
    logger.info("loaded checkpoint done.")
    logger.info("Model Summary: {}".format(get_model_info(yolo_model, exp.test_size)))

    logger.info('Fusing...')
    yolo_model = fuse_model(yolo_model)

    # eval

    tensor_type = torch.cuda.FloatTensor


    output_dict = {}
    outputs_list = []
    first_vid_id = None
    for cur_iter, data in enumerate(tqdm(val_loader)):
        (imgs, dets, info_imgs, ids) = data
        with torch.no_grad():
            imgs = imgs.type(tensor_type)
            frame_id = info_imgs[2]
            video_id = info_imgs[3]
            img_file_name = info_imgs[4]
            video_name = [a.split('/')[0] for a in img_file_name]

            if first_vid_id is None:
                first_vid_id = video_id[0]
                first_vid_name = video_name[0]

            #if torch.any(video_id != first_vid_id):
            #    print(torch.any(video_id != first_vid_id))

            #video_name = img_file_name[0].split('/')

            outputs = yolo_model(imgs)
            outputs = postprocess(outputs, exp.num_classes, exp.test_conf, exp.nmsthre)

            #output_path = args.yolo_outputs/Path(args.dataset)/Path(args.type)
            #output_path.mkdir(parents=True, exist_ok=True)

            #outputs = [None]
            for vid_name, vid_id, f_id, outs in zip(video_name, video_id, frame_id, outputs):
                #output_dict[vid_name] = {'video_id':vid_id,[f_id] = outs
                #print(f_id)

                # if vid_id != first_vid_id:
                #     filename = yolo_dump_dir/Path(f"{first_vid_name}.pt")
                #     torch.save(outputs_list,filename)
                #     logger.info(f"saved {filename}")
                #     first_vid_id = vid_id
                #     first_vid_id = vid_id
                #     first_vid_name = vid_name
                #
                outputs_list.append([vid_name, vid_id, f_id, outs])

    filename = yolo_dump_dir / Path(f"yolo_dets.pt")
    torch.save(outputs_list, filename)
    logger.info(f'Saved {filename}')