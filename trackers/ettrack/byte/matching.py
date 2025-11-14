import cv2
import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from cython_bbox import bbox_overlaps as bbox_ious
#from .kalman_filter import KalmanFilter
import time


def merge_matches(m1, m2, shape):
    O, P, Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1 * M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])

    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def linear_assignment_hungarian(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_a = []
    matched_b = []
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= thresh:
            matches.append([r, c])
            matched_a.append(r)
            matched_b.append(c)

    row_index_array = np.array(range(cost_matrix.shape[0]))
    col_index_array = np.array(range(cost_matrix.shape[1]))

    unmatched_a = np.setdiff1d(row_index_array, np.array(matched_a))
    unmatched_b = np.setdiff1d(col_index_array, np.array(matched_b))
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=float),
        np.ascontiguousarray(btlbrs, dtype=float)
    )

    return ious


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
            len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    # if len(atlbrs)>0 and len(btlbrs)>0:
    #     ct_dist1 = c_dist(atlbrs, btlbrs)
    #     ct_matrix = 1 - ct_dist1
    #     _ious = ious(atlbrs, btlbrs)
    #     cost_matrix = 1 - _ious
    #     all_matrix = cost_matrix + 0.5*ct_matrix
    # else:
    _ious = ious(atlbrs, btlbrs)
    all_matrix = 1 - _ious

    return all_matrix


def v_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
            len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in atracks]
        btlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=float)
    # for i, track in enumerate(tracks):
    # cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=float)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


def fuse_iou(cost_matrix, tracks, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    # fuse_sim = fuse_sim * (1 + det_scores) / 2
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def ct_dist(bboxes1, bboxes2):
    """
        Measure the center distance between two sets of bounding boxes,
        this is a coarse implementation, we don't recommend using it only
        for association, which can be unstable and sensitive to frame rate
        and object speed.
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    centerx1 = (bboxes1[..., 0] + bboxes1[..., 2]) / 2.0
    centery1 = (bboxes1[..., 1] + bboxes1[..., 3]) / 2.0
    centerx2 = (bboxes2[..., 0] + bboxes2[..., 2]) / 2.0
    centery2 = (bboxes2[..., 1] + bboxes2[..., 3]) / 2.0

    ct_dist2 = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2

    ct_dist = np.sqrt(ct_dist2)

    # The linear rescaling is a naive version and needs more study
    if ct_dist.max() == 0:
        return ct_dist
    else:
        ct_dist = ct_dist / ct_dist.max()
        return ct_dist.max() - ct_dist


def c_dist(trackers, detections):
    dist_matrix = np.zeros((len(trackers), len(detections)), dtype=float32)
    for d, trk in enumerate(trackers):
        for t, det in enumerate(detections):
            dist_matrix[d, t] = l2_dist(trk, det)
    dist_matrix = dist_matrix / dist_matrix.max()

    return dist_matrix.max() - dist_matrix


def l2_dist(bb_test, bb_gt):
    """
    Computes center L2 distance between two bboxes in the form [x1,y1,x2,y2]
    """
    test_x = 0.5 * (bb_test[0] + bb_test[2])
    test_y = 0.5 * (bb_test[1] + bb_test[3])
    gt_x = 0.5 * (bb_gt[0] + bb_gt[2])
    gt_y = 0.5 * (bb_gt[1] + bb_gt[3])
    dist = np.sqrt((test_x - gt_x) ** 2 + (test_y - gt_y) ** 2)
    return dist


def add_score_kalman(cost_matrix, strack_pool, detections, interval=1.0, track_thresh=0.6):
    if cost_matrix.size == 0:
        return cost_matrix
    strack_score = np.array([np.clip(strack.pred_score, track_thresh, 1.0) for strack in strack_pool])
    det_score = np.array([det.score for det in detections])
    cost_matrix += (
                abs(np.expand_dims(strack_score, axis=1).repeat(cost_matrix.shape[1], axis=1) - det_score) * interval)
    return cost_matrix


def add_score_kalman_byte_step(cost_matrix, strack_pool, detections, interval=1.0, track_thresh=0.6):
    if cost_matrix.size == 0:
        return cost_matrix
    # strack_score = np.array([np.clip(strack.score_kalman, 0.1, track_thresh) for strack in strack_pool])
    strack_score = np.array(
        [np.clip(strack.score - (strack.pre_score - strack.score), 0.1, track_thresh) for strack in strack_pool])
    # det_score = np.array([np.clip(det.score - (det.pre_score - det.score), 0.1, track_thresh) for det in detections])
    det_score = np.array([det.score for det in detections])
    cost_matrix += (
                abs(np.expand_dims(strack_score, axis=1).repeat(cost_matrix.shape[1], axis=1) - det_score) * interval)
    return cost_matrix