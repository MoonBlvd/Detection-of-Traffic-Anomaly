import numpy as np
import torch
import json

from sklearn.utils.extmath import stable_cumsum
from sklearn.utils.validation import column_or_1d, check_consistent_length, assert_all_finite
from sklearn.utils.multiclass import type_of_target
import warnings
import pdb

def bbox_to_score_map(bboxes, scores, image_size=(1280,720)):
    '''
    Params:
        bboxes: a BoxList object or a tensor bboxes, in x1y1x2y2 format
        scores: scores of each bbox
    Return:
        score_map: (H, W)
    '''
    H = 256 #720
    W = 256 #1280
    if isinstance(bboxes, list):
        bboxes = torch.tensor(bboxes)
    score_map = torch.zeros(H, W)
    if bboxes.max() > 1:
        # normalize then denormalize to correct size
        bboxes[:,[0,2]] /= image_size[0] 
        bboxes[:,[1,3]] /= image_size[1]
    bboxes[:,[0,2]] *= W
    bboxes[:,[1,3]] *= H
    bboxes = bboxes.type(torch.long)
    bboxes[:,[0,2]] = torch.clamp(bboxes[:,[0,2]], min=0, max=W)
    bboxes[:,[1,3]] = torch.clamp(bboxes[:,[1,3]], min=0, max=H)
    
    # Generate gaussian
    for bbox, score in zip(bboxes, scores):
        
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        sigma = torch.tensor([w, h])
        
        x_locs = torch.arange(0, w, 1).type(torch.float)
        y_locs = torch.arange(0, h, 1).type(torch.float)
        y_locs = y_locs[:, np.newaxis]

        x0 = w // 2
        y0 = h // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = np.exp(- (((x_locs - x0) ** 2)/sigma[0] + ((y_locs - y0) ** 2)/sigma[1]) / 2 )        
        score = g * score
        score_map[bbox[1]:bbox[3], bbox[0]:bbox[2]] += score
        
    return score_map


def get_tarr(difference_map, 
             label, 
             bboxes, 
             image_size=(1280, 720), 
             method_type='frame', 
             obj_bboxes=None, 
             obj_scores=None,
             ):
    '''
    Given a difference map and annotations, compute the 
    True Anomaly Region Rate

    difference_map: (H, W)
    bboxes: a list of gt anomalous box, x1y1x2y2, not normalized 
    label: 0/1, normal or abnormal
    input_type: 'object' or 'frame' 
    '''
    K = 300
    if label == 0:
        return 0, []
    elif len(bboxes) == 0:
        return 1, []
    else:
        if method_type == 'object':
            if len(obj_bboxes) == 0 or len(obj_scores) == 0:
                return 1, []
            else:
                difference_map = bbox_to_score_map(obj_bboxes, obj_scores, image_size=image_size)
        H, W = difference_map.shape
        if not isinstance(difference_map, torch.Tensor):
            difference_map = torch.FloatTensor(difference_map)
        if isinstance(bboxes, (list, np.ndarray)):
            bboxes = torch.tensor(bboxes)
        if bboxes.max() > 1:
            # normalize then denormalize to correct size
            bboxes[:,[0,2]] /= image_size[0] 
            bboxes[:,[1,3]] /= image_size[1]
        bboxes[:,[0,2]] *= W
        bboxes[:,[1,3]] *= H
        bboxes = bboxes.type(torch.long)
        
        
        values, indices = torch.topk(difference_map.view(-1), k=K)
        h_coord = (indices // W)#.type(torch.float)
        w_coord = (indices % W)#.type(torch.float)
        
        mask = torch.zeros([H, W])
        for bbox in bboxes:
            mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
        true_anomaly_idx = mask[h_coord, w_coord] == 1
        tarr = values[true_anomaly_idx].sum() / (values.sum() + 1e-6)
        
        return tarr, mask

class ST_AUC():
    def __init__(self, labels, scores, tarrs):
        self.labels = labels
        self.scores = scores
        self.tarrs = tarrs

    def roc_curve(self, pos_label=0, drop_intermediate=True):
        '''
        '''
        fps, tps, thresholds, positives = self._binary_clf_curve(
                                            y_true=self.labels, 
                                            y_score=self.scores, 
                                            pos_label=pos_label, 
                                            sample_weight=self.tarrs)
#         pdb.set_trace()
        if drop_intermediate and len(fps) > 2:
            optimal_idxs = np.where(np.r_[True,
                                          np.logical_or(np.diff(fps, 2),
                                                        np.diff(tps, 2)),
                                          True])[0]
            fps = fps[optimal_idxs]
            tps = tps[optimal_idxs]
            positives = positives[optimal_idxs]
            thresholds = thresholds[optimal_idxs]

        # Add an extra threshold position
        # to make sure that the curve starts at (0, 0)
        tps = np.r_[0, tps]
        fps = np.r_[0, fps]
        positives = np.r_[0, positives]
        thresholds = np.r_[thresholds[0] + 1, thresholds]

        if fps[-1] <= 0:
            warnings.warn("No negative samples in y_true, "
                        "false positive value should be meaningless")
                        # UndefinedMetricWarning)
            fpr = np.repeat(np.nan, fps.shape)
        else:
            fpr = fps / fps[-1]

        if tps[-1] <= 0:
            warnings.warn("No positive samples in y_true, "
                        "true positive value should be meaningless")
                        # UndefinedMetricWarning)
            tpr = np.repeat(np.nan, tps.shape)
            sttpr = np.repeat(np.nan, tps.shape)
        else:
            sttpr = tps / positives[-1] #tps[-1]
            tpr = positives / positives[-1]
        return fpr, tpr, sttpr, thresholds
    
    def _binary_clf_curve(self, y_true, y_score, pos_label=None, sample_weight=None):
        """Calculate true and false positives per binary classification threshold.
        Parameters
        ----------
        y_true : array, shape = [n_samples]
            True targets of binary classification
        y_score : array, shape = [n_samples]
            Estimated probabilities or decision function
        pos_label : int or str, default=None
            The label of the positive class
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
        Returns
        -------
        fps : array, shape = [n_thresholds]
            A count of false positives, at index i being the number of negative
            samples assigned a score >= thresholds[i]. The total number of
            negative samples is equal to fps[-1] (thus true negatives are given by
            fps[-1] - fps).
        tps : array, shape = [n_thresholds <= len(np.unique(y_score))]
            An increasing count of true positives, at index i being the number
            of positive samples assigned a score >= thresholds[i]. The total
            number of positive samples is equal to tps[-1] (thus false negatives
            are given by tps[-1] - tps).
        thresholds : array, shape = [n_thresholds]
            Decreasing score values.
        """
        # Check to make sure y_true is valid
        y_type = type_of_target(y_true)
        if not (y_type == "binary" or
                (y_type == "multiclass" and pos_label is not None)):
            raise ValueError("{0} format is not supported".format(y_type))

        check_consistent_length(y_true, y_score, sample_weight)
        y_true = column_or_1d(y_true)
        y_score = column_or_1d(y_score)
        assert_all_finite(y_true)
        assert_all_finite(y_score)

        if sample_weight is not None:
            sample_weight = column_or_1d(sample_weight)

        # ensure binary classification if pos_label is not specified
        # classes.dtype.kind in ('O', 'U', 'S') is required to avoid
        # triggering a FutureWarning by calling np.array_equal(a, b)
        # when elements in the two arrays are not comparable.
        classes = np.unique(y_true)
        if (pos_label is None and (
                classes.dtype.kind in ('O', 'U', 'S') or
                not (np.array_equal(classes, [0, 1]) or
                    np.array_equal(classes, [-1, 1]) or
                    np.array_equal(classes, [0]) or
                    np.array_equal(classes, [-1]) or
                    np.array_equal(classes, [1])))):
            classes_repr = ", ".join(repr(c) for c in classes)
            raise ValueError("y_true takes value in {{{classes_repr}}} and "
                            "pos_label is not specified: either make y_true "
                            "take value in {{0, 1}} or {{-1, 1}} or "
                            "pass pos_label explicitly.".format(
                                classes_repr=classes_repr))
        elif pos_label is None:
            pos_label = 1.

        # make y_true a boolean vector
        y_true = (y_true == pos_label)

        # sort scores and corresponding truth values
        desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
        y_score = y_score[desc_score_indices]
        y_true = y_true[desc_score_indices]
        if sample_weight is not None:
            weight = sample_weight[desc_score_indices]
        else:
            weight = 1.

        # y_score typically has many tied values. Here we extract
        # the indices associated with the distinct values. We also
        # concatenate a value for the end of the curve.
        distinct_value_indices = np.where(np.diff(y_score))[0]
        threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

        # accumulate the true positives with decreasing threshold
        tps = stable_cumsum(y_true * weight)[threshold_idxs]
        positives = stable_cumsum(y_true)[threshold_idxs] # Note that the number of positive should be computed differently
        if sample_weight is not None:
            # express fps as a cumsum to ensure fps is increasing even in
            # the presence of floating point errors
            fps = stable_cumsum((1 - y_true))[threshold_idxs]
        else:
            fps = 1 + threshold_idxs - tps
        return fps, tps, y_score[threshold_idxs], positives
