import torch.nn as nn
import torch
import torch.nn.functional as F

class Loss(nn.Module):
    def __init__(self,tracklet_length=20, use_direction_loss=True):
        super().__init__()
        self.tracklet_length = tracklet_length
        self.use_direction_loss = use_direction_loss

    def _computer_direction_loss(self, outputs, targets):
        ######################################################################################
        last_abs = targets[:15, :, :4]  # todo track length hard coded!
        pred_abs = targets[:15, :, :4] + outputs
        true_abs = targets[1:16, :, :4]
        loss_dir = 0
        fram_all = self.tracklet_length - 5
        pedinum = true_abs.shape[1]
        for framenum in range(fram_all):
            last_abs_one = convert_bbox(last_abs[framenum, :, :])
            pred_abs_one = convert_bbox(pred_abs[framenum, :, :])
            true_abs_one = convert_bbox(true_abs[framenum, :, :])

            for pedi in range(pedinum):
                pred_cen = speed_direction(last_abs_one[pedi], pred_abs_one[pedi])
                pred_lt = speed_direction_lt(last_abs_one[pedi], pred_abs_one[pedi])
                pred_rt = speed_direction_rt(last_abs_one[pedi], pred_abs_one[pedi])
                pred_lb = speed_direction_lb(last_abs_one[pedi], pred_abs_one[pedi])
                pred_rb = speed_direction_rb(last_abs_one[pedi], pred_abs_one[pedi])

                true_cen = speed_direction(last_abs_one[pedi], true_abs_one[pedi])
                true_lt = speed_direction_lt(last_abs_one[pedi], true_abs_one[pedi])
                true_rt = speed_direction_rt(last_abs_one[pedi], true_abs_one[pedi])
                true_lb = speed_direction_lb(last_abs_one[pedi], true_abs_one[pedi])
                true_rb = speed_direction_rb(last_abs_one[pedi], true_abs_one[pedi])

                cost_cen = cost_vel(pred_cen, true_cen, weight=0.2)
                cost_lt = cost_vel(pred_lt, true_lt, weight=0.2)
                cost_rt = cost_vel(pred_rt, true_rt, weight=0.2)
                cost_lb = cost_vel(pred_lb, true_lb, weight=0.2)
                cost_rb = cost_vel(pred_rb, true_rb, weight=0.2)

                cost_all = cost_cen + cost_lt + cost_rt + cost_lb + cost_rb
                loss_dir = loss_dir + cost_all
        loss_dir = loss_dir / (pedinum * fram_all)
        return loss_dir

        ##################################################################################################
    def forward(self, outputs, targets):

        delta_targets = targets[:, :, 4:8]
        loss_o = torch.sum(F.smooth_l1_loss(outputs, delta_targets, reduction='none'), dim=2)
        loss_o = loss_o/targets.shape[1]  #num people
        if self.use_direction_loss:
            loss = 0.99 * loss_o + 0.01 * self.compute_direction_loss(outputs, targets)
            return loss
        else:
            return loss_o




def convert_bbox(x):  # predict=None
    """
  Takes a bounding box in the centre form [x,y,w,h] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
    x[:, 0] = x[:, 0] - x[:, 2] / 2
    x[:, 1] = x[:, 1] - x[:, 3] / 2
    x[:, 2] = x[:, 0] + x[:, 2] / 2
    x[:, 3] = x[:, 1] + x[:, 3] / 2

    return x


def speed_direction(bbox1, bbox2):
    cx1, cy1 = (bbox1[0]+bbox1[2]) / 2.0, (bbox1[1]+bbox1[3])/2.0
    cx2, cy2 = (bbox2[0]+bbox2[2]) / 2.0, (bbox2[1]+bbox2[3])/2.0
    speed = np.array([cy2-cy1, cx2-cx1])
    norm = np.sqrt((cy2-cy1)**2 + (cx2-cx1)**2) + 1e-6
    return speed / norm


def speed_direction_lt(bbox1, bbox2):
    cx1, cy1 = bbox1[0], bbox1[1]
    cx2, cy2 = bbox2[0], bbox2[1]
    speed = np.array([cy2-cy1, cx2-cx1])
    norm = np.sqrt((cy2-cy1)**2 + (cx2-cx1)**2) + 1e-6
    return speed / norm


def speed_direction_rt(bbox1, bbox2):
    cx1, cy1 = bbox1[0], bbox1[3]
    cx2, cy2 = bbox2[0], bbox2[3]
    speed = np.array([cy2-cy1, cx2-cx1])
    norm = np.sqrt((cy2-cy1)**2 + (cx2-cx1)**2) + 1e-6
    return speed / norm


def speed_direction_lb(bbox1, bbox2):
    cx1, cy1 = bbox1[2], bbox1[1]
    cx2, cy2 = bbox2[2], bbox2[1]
    speed = np.array([cy2-cy1, cx2-cx1])
    norm = np.sqrt((cy2-cy1)**2 + (cx2-cx1)**2) + 1e-6
    return speed / norm


def speed_direction_rb(bbox1, bbox2):
    cx1, cy1 = bbox1[2], bbox1[3]
    cx2, cy2 = bbox2[2], bbox2[3]
    speed = np.array([cy2-cy1, cx2-cx1])
    norm = np.sqrt((cy2-cy1)**2 + (cx2-cx1)**2) + 1e-6
    return speed / norm


def cost_vel(pred, true, weight):
    # Y, X = speed_direction_batch(detections, previous_obs)
    pred_y, pred_x = pred[0], pred[1]
    true_y, true_x = true[0], true[1]

    angle_cos = pred_y * true_y + pred_x * true_x
    angle_cos = np.clip(angle_cos, a_min=-1, a_max=1)
    diff_angle = np.arccos(angle_cos)
    # diff_angle = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi

    angle_diff_cost = diff_angle * weight

    return angle_diff_cost

if __name__ == '__main__':

    import numpy as np

    test = [10,20,5,6]
    test2 = [15,22,5,6]
    data = np.array([[test,test2]])