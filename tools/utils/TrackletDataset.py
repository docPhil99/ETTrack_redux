from pathlib import Path
import math
import json
import configparser
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from loguru import logger

"""
Dataset loader and prefilter. Run the make_data_json_files first, these are read by the Dataset clas
"""





#set_type = 'train'
#path_list = [Path(f'datasets/dancetrack/{set_type}') , Path(f'datasets/MOT20/{set_type}'),Path(f'datasets/MOT17/{set_type}')]#,Path('datasets/SportsMOT/sportsmot_publish/dataset/{set_type}')]
#stub_list  = [Path('dancetrack'),Path('MOT20'),Path('MOT17')]#,Path('sportsMOT')]


# gt format
# <frame>,  <id>, <bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<x>,<y>,<z>
def make_data_set_json_files(path_list,set_type, out_path = Path('datasets/processed_output'), filter='FRCNN'):
    """
    makes the processed ground truth json files
    :param path_list: list of paths to dataset eg Path('datasets/dancetrack')
    :param set_type: 'train' or 'val', 'mix' to generate 50/50 split for MOT
    :param out_path: path to save the processed ground truth json files
    :param filter: 'FRCNN' string must be in the directory name, otherwise it will be ignored
    :return: nothing
    """
    if set_type == 'mix':  # if mix split the training set in two
        split_set = True
        set_type = 'train'
    else:
        split_set = False
    if isinstance(path_list, Path):
        path_list = [path_list]

    for base_path in path_list:
        set_path = base_path / Path(set_type)
        for paths in set_path.iterdir():
            if not paths.is_dir():
                continue
            if filter:
                if not filter in str(paths):
                    continue
            ground_truth_file = paths / Path('gt/gt.txt')
            logger.info(f"Processing {paths}")

            # get the frame rate
            seqinfo_file = paths / Path('seqinfo.ini')
            config = configparser.ConfigParser()
            config.read_file(open(seqinfo_file))
            fps = float(config['Sequence']['frameRate'])
            output = {}
            output['ground_truth_file'] = str(ground_truth_file)
            output['fps'] = fps

            try:
                with open(ground_truth_file, 'r') as f:
                    gt_s = f.readlines()
            except:
                logger.warning(f'Could open {ground_truth_file}')
                continue
            gt = [[int(float(val.rstrip())) for val in line.split(',')] for line in gt_s]

            data={}
            max_frame_num = 0
            for fr in gt:  #scan each line
                pos = fr[2:6]
                dat={'frame':fr[0],'pos':pos}
                if fr[0] > max_frame_num:  # find the number of frames in the video
                    max_frame_num = fr[0]
                if fr[1] in data:   # check if ID exists, if not make new key, else append
                    data[fr[1]].append(dat)
                else:
                    data[fr[1]]=[dat]
            if not split_set:
                output['data'] = data
                out_file = out_path / base_path.stem / Path(set_type) / paths.stem / Path('gt.json')
                out_file.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Saving {out_file}")
                with open(out_file, 'w') as f:
                    json.dump(output, f)
            else:
                # second half goes to val set
                frame_split =  max_frame_num//2
                train_data = {}
                val_data = {}
                for id in data:
                    trdat = [a for a in data[id] if a['frame'] <= frame_split]
                    if len(trdat) > 0:
                        train_data[id] = trdat
                    trdat = [a for a in data[id] if a['frame'] > frame_split]
                    if len(trdat) > 0:
                        val_data[id] = trdat

                output['data'] = train_data
                base_stem = base_path.stem
                if str(base_stem) == 'mot':
                    base_stem = base_path.parents[-1] / Path('MOT17')
                out_file = out_path / base_stem / Path('train') / paths.stem / Path('gt.json')
                out_file.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Saving {out_file}")
                with open(out_file, 'w') as f:
                    json.dump(output, f)
                output['data'] = val_data
                out_file = out_path / base_stem / Path('val') / paths.stem / Path('gt.json')
                out_file.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Saving {out_file}")
                with open(out_file, 'w') as f:
                    json.dump(output, f)

class TrackletDataset(Dataset):
    def __init__(self, data_set_path: Path | list[Path], set_type: str, tracklet_length=20, dtype=torch.float, transforms = None, single_dataset_file : Path=None):
        """

        :param data_set_path:  path to root of dataset eg dataset/drancetrack
        :param set_type: train or val
        :param tracklet_length: length of tracklets
        """
        super().__init__()
        if isinstance(data_set_path, Path):
            self.data_set_path = [data_set_path/Path(set_type)]
        else:
            self.data_set_path = [p/Path(set_type) for p in data_set_path]
        self.single_dataset_file = single_dataset_file
        self.set_type = set_type
        self.tracklet_length = tracklet_length
        self.transforms = transforms
        # data
        self.all_tracklets = []
        self.all_tracklets_xxyy = []
        self.get_all_data()
        self.all_tracklets_xxyy = torch.tensor(self.all_tracklets_xxyy, dtype=dtype)
        self._calc_deltas()

    def __len__(self):
        #return len(self.all_tracklets)
        return self.all_tracklets_xxyy.shape[0]

    def __getitem__(self, idx):
        track = self.all_tracklets_xxyy[idx,:,:]
        if self.transforms:
            for trans in self.transforms:
                track = trans(track)
        return track
        #return self.all_tracklets_xxyy[idx,:,:]

    def _get_tracklets_by_ID(self,data,ID):
        track = data['data'][f'{ID}']

        contiguous_tracks = []
        tracks = []
        counter = 1
        for p in range(len(track)):
            # print(f'{p} {counter} {track[p]}')
            if counter == track[p]['frame']:
                # print('ok')
                tracks.append(track[p])
                counter += 1
            else:
                #print('append')
                contiguous_tracks.append(tracks)
                tracks = []
                tracks.append(track[p])
                counter = track[p]['frame'] + 1
                #print(f'Set counter to {counter}')
        contiguous_tracks.append(tracks)
        num_tracklets = [len(track) // self.tracklet_length for track in contiguous_tracks]
        tracklets = []
        for num, track in zip(num_tracklets, contiguous_tracks):
            for n in range(num):
                start = n * self.tracklet_length
                end = (n + 1) * self.tracklet_length
                #print(start, end)
                tr = track[start:end]
                tracklets.append(tr)
        return tracklets, contiguous_tracks

    def plot_items(self,num=40):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        for p in range(num):
            trac=self.__getitem__(p)
            ntrac=trac.numpy()
            ax.plot(ntrac[:, 0], ntrac[:, 2],'-')
            #print(ntrac[:, 0], ntrac[:, 2])
        plt.show()

    def plot_tracks(self, tracklets, contig):
        """
        plot the tracklet and the contiguous track, for example, use get_tracklets_by_ID(data,ID,tracklet_length)
        to get this information.
        :param tracklets:
        :param contig:
        :return:
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2)
        for tracklet in tracklets:
            trac_xxyy = np.array([b['pos'] for b in tracklet])

            ax[0].plot(trac_xxyy[:, 0], trac_xxyy[:, 2])
        if contig:
            for cnt in contig:
                cnt_xxyy = np.array([b['pos'] for b in cnt])
                ax[1].plot(cnt_xxyy[:, 0], cnt_xxyy[:, 2])
                ax[0].set_ylim([0, 300])
                ax[1].set_ylim([0, 300])

        ax[0].set_xlim([0, 900])
        ax[1].set_xlim([0, 900])

        plt.show()


    def _proc_json(self,data,path):
        for ID in data['data']:
            # print(ID)
            tracklets, _ = self._get_tracklets_by_ID(data, ID)
            track_dict = {'ID': ID, 'tracklets': tracklets, 'file': path.stem}
            self.all_tracklets.append(track_dict)
            # trac_xxyy = [b['pos'] for b in tracklets]
            for trac in tracklets:
                self.all_tracklets_xxyy.append([b['pos'] for b in trac])
            # self.all_tracklets_xxyy.extend(trac_xxyy)

    def get_all_data(self):

        if self.single_dataset_file:  #only process a single file
            with open(self.single_dataset_file, 'r') as f:
                data = json.load(f)
            self._proc_json(data,self.single_dataset_file)
            return

        for data_set_path in self.data_set_path:  # scan directory and process all
            for path in data_set_path.iterdir():
                if not path.is_dir():
                    continue
                input_file = path / Path('gt.json')  # load the pre-process data
                logger.info(f"Processing {input_file}")
                with open(input_file, 'r') as f:
                    data = json.load(f)

                self._proc_json(data,path)
                # for ID in data['data']:
                #     #print(ID)
                #     tracklets, _ = self._get_tracklets_by_ID(data,ID)
                #     track_dict = {'ID':ID, 'tracklets':tracklets,'file':path.stem}
                #     self.all_tracklets.append(track_dict)
                #     #trac_xxyy = [b['pos'] for b in tracklets]
                #     for trac in tracklets:
                #         self.all_tracklets_xxyy.append( [b['pos'] for b in trac])
                #     #self.all_tracklets_xxyy.extend(trac_xxyy)
    def _calc_deltas(self):
        deltas = self.all_tracklets_xxyy[:,1:,:] - self.all_tracklets_xxyy[:,:-1,:]
        zero_start = torch.zeros((deltas.shape[0],1,4), dtype=deltas.dtype)
        deltas = torch.cat([zero_start, deltas],dim=1)
        self.all_tracklets_xxyy = torch.cat([self.all_tracklets_xxyy, deltas], dim=2)


if __name__ == '__main__':
    d1 = TrackletDataset(Path('datasets/processed_output/dancetrack/'),'train')
    paths = [Path('datasets/processed_output/dancetrack/'), Path('datasets/processed_output/MOT17'),
             Path('datasets/processed_output/MOT20')]
    d2 = TrackletDataset(paths,'train')
    print(f'Number of tracklets: {len(d1)} and {len(d2)}')