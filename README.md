# ETTrack

Refactored code from the paper  X. Han et al., “ETTrack: enhanced temporal motion predictor for multi-object tracking,” Appl. Intell., vol. 55, no. 1, Jan. 2025. https://doi.org/10.1007/s10489-024-05866-4
  
This is a cleaned up version of the code from the paper. The netwok is the same, but some changes to the dataloader have been updated to speed things up. 

## TODO

- [] Only DanceTrack has been tested in this version. Check the dataloader.
- [] Re-implement momementum loss to work on CUDA. The current version used numpy on the CPU and very slow.
- [] Test install


## Install
Install via uv

## Layout

Datasets MOT17,DanceTrack,MOT20 must be pre-procesed and set up the 
same way as an Byte-Track. The code is not included here yet.
Also, download the pretrained YOLOX weights from ByteTrack. YOLOX training is not included.

### Data Directory Layout
- data
  - datasets
    - dancetrack
    - mot
    - MOT20
- pretrained
  - ettrack
    - dancetrack
      - your weights here
    - yolox
      - yolox pretrained

You don't have to stick with this, since its configured 
via argparse.


## Run on pre-trained

YOLOX and loading images takes considerably more
time than the tracking. It is faster to 
preprocess the yolo detections.
eg.
```
python tools/generate_yolo_detections.py --yolox_weights pretrained/yolox/bytetrack_dance_model.pth.tar\
--dataset dancetrack --dataset_dir data\datasets --type val --yolo_outputs data\yolo_outputs
```
will create the detection file for the dancetrack validation set.

To evaluate this, using the star_1616.tar ettrack weights:

```
python tools/run_ettrack.py --dataset_dir data/datasets --dataset dancetrack \
--yolo_dump_dir data/yolo_outputs --output_dir Results \ 
--ettrack_model_path pretrained/ettrack/dancetrack/star_1616.tar
``` 

## Training

We preprocess the ground truth files to a set of json files. 
```
python tools/make_training_tracks.py --output_dir data/datasets/processed_outputs --type val \
--datasets data/datasets/dancetrack
```

To train
```
python tools\train_ettrack.py --dataset dancetrack --dataset_root_path data/datasets/processed_outputs \
--save_dir training_output --epoch 40
```
