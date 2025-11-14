import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from loguru import logger
from trackers.ettrack.network.temporal_conv import TemporalConvNet
from trackers.ettrack.tcn_transformer import TransformerEncoderLayer, TransformerModel, TransformerEncoder, get_noise

class tcn_transformer(torch.nn.Module):
    def __init__(self, args, dropout_prob=0,tcn_only=False):
        """

        :param args:
        :param dropout_prob:
        :param tcn_only: only use tcn, for ablation studies
        """
        super().__init__()
        self._first_run_testing = True # used for printing debug
        self._first_run_training = True  # used for printing debug
        self.args = args
        self._tcn_only = tcn_only
        # set parameters for network architecture
        self.embedding_size = [32]
        self.output_size = 4
        self.dropout_prob = dropout_prob

        self.temporal_encoder_layer = TransformerEncoderLayer(d_model=32, nhead=8)
        self.encode_norm = nn.LayerNorm(32)
        # TCN
        input_channels = 8
        levels = 4
        nhid = 32  # number of hidden units per layer
        channel_sizes = levels * [nhid]
        self.tcn = TemporalConvNet(input_channels, channel_sizes, kernel_size=7, dropout=0.2)

        emsize = 32  # embedding dimension
        nhead = 8  # the number of heads in the multihead-attention models
        nlayers = 6  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        dropout = 0.1  # the dropout value
        self.spatial_encoder_1 = TransformerModel(emsize, nhead, nhid, nlayers, dropout)
        self.spatial_encoder_2 = TransformerModel(emsize, nhead, nhid, nlayers, dropout)

        self.temporal_encoder_1 = TransformerEncoder(self.temporal_encoder_layer, 1, self.encode_norm)
        self.temporal_encoder_2 = TransformerEncoder(self.temporal_encoder_layer,1, self.encode_norm)

        # Linear layer to map input to embedding
        self.input_embedding_layer_temporal = nn.Linear(8, 32)
        self.input_embedding_layer_spatial = nn.Linear(8, 32)
        # Linear layer to output and fusion
        self.output_layer = nn.Linear(48, 4)

        self.fusion_layer = nn.Linear(64, 32)
        # ReLU and dropout init
        self.relu = nn.ReLU(inplace=False)
        self.sigmoid = nn.Sigmoid()
        self.dropout_in = nn.Dropout(self.dropout_prob)
        self.dropout_in2 = nn.Dropout(self.dropout_prob)
        if tcn_only:
            self.tcn_output_layer = nn.Linear(nhid, 4)

    def forward(self, inputs, validation=False):
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, dtype=torch.float)
        if self.args.device == 'gpu':
            inputs = inputs.cuda()
        if self.training or validation:
            return self._training_forwards(inputs)
        if self._first_run_testing:
            logger.debug('testing forwards')
            self._first_run_testing = False

        nodes_abs = inputs[:, :, :9]
        nodes_xywh = inputs[-1, :, :4]  # for prediction batch is always 1, this is just taking the last
        #nodes_xywh = inputs[:, :, :4] # todo but not always for some reason!
        tcn_input = nodes_abs.transpose(1, 2)
        tcn_input  = self.relu(self.tcn(tcn_input))
        tcn_input = tcn_input.transpose(1, 2)  #1,9,32
        trans_input = self.dropout_in(tcn_input.clone())  #make copy here

        trans_input = self.temporal_encoder_1(trans_input)
        # https://github.com/locuslab/TCN/blob/master/TCN/mnist_pixel/model.py
        temporal_input_embedded_last = trans_input[-1]   # last activation of
        noise = get_noise((1, 16), 'gaussian')
        noise_to_cat = noise.repeat(temporal_input_embedded_last.shape[0], 1)  # (323,16)
        temporal_input_embedded_wnoise = torch.cat((temporal_input_embedded_last, noise_to_cat), dim=1)  # (323,48)
        outputs_current = self.output_layer(temporal_input_embedded_wnoise)
        #outputs_current = outputs_current.cpu().detach().numpy()
        outputs_pre = nodes_xywh + outputs_current[:, :4]
        return outputs_pre

    def _training_forwards(self,inputs: Tensor):
        if self._first_run_training:
            logger.debug('training forwards')
            self._first_run_training = False
        output = torch.zeros(inputs.shape[0],inputs.shape[1],4, dtype=inputs.dtype, device=inputs.device)
        for frame_num in range(inputs.shape[0]):
            nodes_abs = torch.unsqueeze(inputs[frame_num, :, :9],0)
            tcn_input = nodes_abs.transpose(1, 2)
            tcn_input = self.relu(self.tcn(tcn_input))
            tcn_input = tcn_input.transpose(1, 2)  # 1,9,32
            trans_input = self.dropout_in(tcn_input.clone())  # make copy here

            trans_input = self.temporal_encoder_1(trans_input)
            # https://github.com/locuslab/TCN/blob/master/TCN/mnist_pixel/model.py
            temporal_input_embedded_last = trans_input[-1]  # last activation of
            noise = get_noise((1, 16), 'gaussian')
            noise_to_cat = noise.repeat(temporal_input_embedded_last.shape[0], 1)  # (323,16)
            temporal_input_embedded_wnoise = torch.cat((temporal_input_embedded_last, noise_to_cat), dim=1)  # (323,48)
            outputs_current = self.output_layer(temporal_input_embedded_wnoise)
            # outputs_current = outputs_current.cpu().detach().numpy()
            output[frame_num,:,:] = outputs_current[:, :4]
        return output