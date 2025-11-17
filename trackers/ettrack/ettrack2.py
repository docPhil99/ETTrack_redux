# ettrack class, was STAR.src.predict
import sys
from pathlib import Path

import torch
import torch.nn as nn

from trackers.ettrack.tcn_transformer2 import tcn_transformer
from trackers.ettrack.utils2 import getLossMask#, Trajectory_Dataloader
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm
from tools.utils.TrackletDataset import TrackletDataset
from loguru import logger
CUDA_LAUNCH_BLOCKING = 1
from trackers.ettrack.network.loss import Loss
from trackers.ettrack.augmentor import FlipUD, FlipTime, FlipLR, Rotate
from torch.utils.tensorboard import SummaryWriter

class Predict(object):
    def __init__(self, args, training=False):
        logger.debug(f'Creating Birch tracker network in torch version {torch.__version__}')
        self.args = args
        self.optimizer = None
        self.criterion = None
        self.tcn_transformer = tcn_transformer(args)
        self.log_file_curve = None
        self.best_validation_loss = float('inf')
        self.best_epoch = 0
        self._augment_in_training_loop = False  # augmentation in the train function, used from MOT that has no validation set
        self.augmentation_prob = 0.1
        if training:
            if args.augmentation:
                self.transforms=[]
                if args.augmentation_list == 'all':
                    #transforms = [FlipTime(), FlipLR(), FlipUD()]
                    self.transforms = [FlipLR(probability=0.1),FlipUD(probability=0.1), FlipTime(probability=self.augmentation_prob)]
                if 'lr' in args.augmentation_list:
                    self.transforms.append(FlipLR(probability=self.augmentation_prob))
                if 'ud' in args.augmentation_list:
                    self.transforms.append(FlipUD(probability=self.augmentation_prob))
                if 'time' in args.augmentation_list:
                    self.transforms.append(FlipTime(probability=self.augmentation_prob))

            else:
                self.transforms = None
            logger.info(f'Using augmentation list: {args.augmentation_list}: {self.transforms}')
            if args.single_dataset_file:
                dataset_paths = [args.single_dataset_file]
            else:
                dataset_paths = [args.dataset_root_path/Path(p) for p in args.dataset]
            logger.info(f'Dataset paths: {dataset_paths}')
            if args.single_dataset_file:
                ds = TrackletDataset(dataset_paths, set_type='train' ,transforms=self.transforms, single_dataset_file=args.single_dataset_file)
                proportions = [.75, .25]
                split_lengths = self._get_split_lengths(ds, proportions)
                dataset, valdataset = torch.utils.data.random_split(ds, lengths=split_lengths)
            elif 'MOT17' in str(dataset_paths):
                logger.info('Proc MOT17')
                if len(dataset_paths)>1:
                    logger.error('Can not mixt MOT17 yet')
                    sys.exit(1)
                ds = TrackletDataset(dataset_paths, set_type='train', transforms=None)
                if args.augmentation:
                    self._augment_in_training_loop = True
                proportions = [.75, .25]
                split_lengths = self._get_split_lengths(ds, proportions)
                dataset, valdataset = torch.utils.data.random_split(ds, split_lengths)  # old version of pytorch needs exact lengths

            else:
                dataset = TrackletDataset(dataset_paths, set_type='train' ,transforms=self.transforms)
                valdataset = TrackletDataset(dataset_paths, set_type='val')

            self.training_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
            self.val_dataloader = DataLoader(valdataset,batch_size=args.batch_size, shuffle=False)
            self.set_optimizer()
            self.loss_fn = Loss(tracklet_length=args.seq_length, use_direction_loss=not args.ignore_direction_loss)
            # save directories
            args.save_dir.mkdir(parents=True, exist_ok=True)
            # save network as text file
            with open(args.save_dir / Path('network.txt'), 'w') as f:
                f.write(str(self.tcn_transformer))
            self.log_file_curve = open(self.args.save_dir/ Path('log_curve.txt'), 'wt')
            self.summary_writer = SummaryWriter(log_dir=self.args.save_dir)

        else:
            self.tcn_transformer.eval()

        if self.args.device == 'gpu' or self.args.device==torch.device('cuda'):
            logger.debug('Using ettrack on GPU')
            self.tcn_transformer = self.tcn_transformer.cuda()
        else:
            logger.debug('Using ettrack on CPU')

    def _get_split_lengths(self, dataset, proportions):
        """
        Gets the correct dataset split lengths. Not needed in new pytorch versions.
        :param dataset:
        :param proportions:
        :return:
        """
        lengths = [int(p * len(dataset)) for p in proportions]
        lengths[-1] = len(dataset) - sum(lengths[:-1])
        logger.debug(lengths)
        return lengths

    def __del__(self):
        logger.info('Deleting tracker network')
        if self.log_file_curve:
            self.log_file_curve.close()

    def save_model(self, epoch ,path:Path =None):
        if path:
            model_path=path
        else:
            model_path = self.args.save_dir / Path(f'{self.args.model_name}_{epoch}.pt')
        logger.info(f'Saving model {model_path}')
        torch.save({
            'epoch': epoch,
            'state_dict': self.tcn_transformer.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, model_path)

    def load_model(self):
        logger.info('Loading ettrack model')
        if self.args.ettrack_model_path:
            model_path = self.args.ettrack_model_path
        else:
            #logger.info(f'Loading model from {self.args.load_model}')
            #if self.args.load_model is not None:
            model_path = self.args.save_dir / Path(f'{self.args.model_name}_{self.args.load_model_epoch}.pt')
            logger.info(f'Load from generated model save path: {model_path}')
            #else:
            #    raise Exception(f'No model epoch provided')
        logger.info(f'Loading ettrack checkpoint file {model_path}')
        checkpoint = torch.load(model_path)
        model_epoch = checkpoint['epoch']
        self.tcn_transformer.load_state_dict(checkpoint['state_dict'])
        logger.info('Loaded checkpoint at epoch', model_epoch)


    def set_optimizer(self):

        self.optimizer = torch.optim.Adam(self.tcn_transformer.parameters(), lr=self.args.learning_rate)
        self.criterion = nn.MSELoss(reduction='none')

    def get_network(self):
        """
        loads network, sets to eval mode and returns network
        :return:  torch network
        """
        self.load_model()
        self.tcn_transformer.eval()
        return self.tcn_transformer

    def test(self):

        logger.info('Testing begin')
        self.load_model()
        self.tcn_transformer.eval()

        test_error, test_final_error = self.test_epoch()
        logger.info(
            f'Set: {self.args.test_set}, epoch: {self.args.load_model},test_error: {test_error} '
            f'test_final_error: {test_final_error}')

    def train(self):
        if not self.args.load_model_epoch == 1:
            self.load_model()
        logger.info('Training begins')
        test_error, test_final_error = 0, 0
        for epoch in range(self.args.load_model_epoch,self.args.num_epochs+1):
            self.tcn_transformer.train()

            train_loss = self.train_epoch(epoch)
            train1_loss = 0 # todo
            self.save_model(epoch)
            val_loss = self.test_epoch()


            self.log_file_curve.write(f'{epoch},{train_loss},{val_loss},{self.args.learning_rate}')
            self.log_file_curve.flush()

            self.summary_writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : train_loss, 'Validation' : val_loss },
                    epoch)
            self.summary_writer.flush()
            print("----epoch {}, train_loss={:.5f}, val_loss={:.5f}".format(epoch, train_loss, val_loss))
            if val_loss < self.best_validation_loss:
                self.best_validation_loss = val_loss
                self.best_epoch = epoch
                self.save_model(epoch,path=self.args.save_dir / Path(f'{self.args.model_name}_best.pt'))
        logger.info(f'Best validation loss: {self.best_validation_loss} in epoch {self.best_epoch}')
    def test_epoch(self):
        self.tcn_transformer.eval()
        running_loss = 0
        with torch.no_grad():
            for index, inputs in enumerate(self.val_dataloader):
                if self.args.device == 'gpu':
                    inputs = inputs.cuda()
                vout = self.tcn_transformer.forward(inputs, validation=True)
                vloss = self.loss_fn(vout, inputs)
                vloss = vloss.mean()
                running_loss += vloss
        ave_loss = running_loss/(index+1)
        return ave_loss
    def train_epoch(self, epoch):

        running_loss = 0.0


        for index, inputs in enumerate(self.training_dataloader):
            self.optimizer.zero_grad()
            if self.args.device == 'gpu':
                inputs = inputs.cuda()
            if self._augment_in_training_loop:
                for ind in range(inputs.shape[0]):
                    for trans in self.transforms:
                        inputs[ind,:,:] = trans(inputs[ind,:,:])

            outputs = self.tcn_transformer.forward(inputs)
            loss = self.loss_fn(outputs, inputs)
            loss = loss.mean()
            loss.backward()  #todo is this correct?
            self.optimizer.step()

            running_loss += loss.item()

        last_loss = running_loss / len(self.training_dataloader)
        return last_loss



    @torch.no_grad()
    def test_epoch_old(self):
        self.dataloader.reset_batch_pointer(set='test')
        error_epoch, final_error_epoch = 0, 0,
        error_cnt_epoch, final_error_cnt_epoch = 1e-5, 1e-5
        loss_epoch = 0
        loss1_epoch = 0
        for batch in tqdm(range(self.dataloader.testbatchnums - 1)):
            loss = torch.zeros(1).cuda()
            loss1 = torch.zeros(1).cuda()
            inputs, batch_id = self.dataloader.get_test_batch(batch)
            inputs = tuple([torch.Tensor(i) for i in inputs])

            if self.args.device == 'gpu':
                inputs = tuple([i.cuda() for i in inputs])

            # batch_abs, batch_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum = inputs
            #
            # inputs_forward = batch_abs[:-1], batch_norm[:-1], shift_value[:-1], seq_list[:-1], nei_list[:-1], nei_num[
            #                                                                                :-1], batch_pednum
            batch_abs, seq_list, batch_pednum = inputs

            batch_offset = batch_abs[1:] - batch_abs[:-1]
            batch_offset_ = batch_offset.cpu().tolist()

            node_index = self.tcn_transformer.get_node_index(seq_list)
            updated_batch_pednum = self.tcn_transformer.update_batch_pednum(batch_pednum, node_index)
            st_ed = self.tcn_transformer.get_st_ed(updated_batch_pednum)
            name = 'test'
            # batch_abs_nom = self.normalize(batch_abs[:, node_index], st_ed, batch_id, name)
            batch_offset_nom, nomlize_all = self.tcn_transformer.mean_normalize_abs_input(
                batch_offset[:, node_index], st_ed)

            # batch_offset = batch_abs[1:] - batch_abs[:-1]

            inputs_forward = batch_abs[:15, :, :], batch_offset[:15], seq_list[:15], batch_pednum

            outputs_infer = self.tcn_transformer.forward(inputs_forward, iftest=True, training_test=True)
            if self.args.device == 'gpu':
                using_cuda = True
            else:
                using_cuda = False
            lossmask, num = getLossMask(outputs_infer, seq_list[0], seq_list[1:16], using_cuda=using_cuda)
            loss_o = torch.sum(F.smooth_l1_loss(outputs_infer, batch_offset[:15, :, :4], reduction='none'), dim=2)
            loss += (torch.sum(loss_o * lossmask / num))
            ##########################################################################

            #####################################################################
            loss_epoch += loss.item()

            # outputs_ = self.normalize_vert(outputs_infer, st_ed, batch_id, name)
            outputs_ = self.tcn_transformer.mean_normalize_abs_input_vert(outputs_infer, st_ed, nomlize_all)
            # loss_1 = torch.sum(self.criterion(outputs_, batch_abs[1:]), dim=2)
            loss1 = F.l1_loss(outputs_, batch_offset[:15, :, :4], reduction="mean")

            loss1_epoch += loss1.item()
            loss1_epoch = 0

        train_loss_epoch = loss_epoch / (self.dataloader.testbatchnums - 1)
        train_loss1_epoch = loss1_epoch / (self.dataloader.testbatchnums - 1)
        return train_loss_epoch, train_loss1_epoch

