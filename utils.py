import numpy as np
import torch


def get_score_for_one_patient(labels, predicts, threshold=0.5):
    '''
    :param truths: [184, 1, 224, 192]
    :param predicts: [184, 1, 224, 192]
    :param threshold: threshold for computing dice score
    :return: score of this patient
    '''
    if labels.size(0) != 184 or predicts.size(0) != 184:
        print('ERROR')
        return 0

    label_positive = labels > threshold
    lp_count = torch.nonzero(label_positive).size(0)
    predict_positive = predicts > threshold
    pp_count = torch.nonzero(predict_positive).size(0)

    TP_count = torch.nonzero(torch.logical_and(label_positive, predict_positive)).size(0)
    FN_count = lp_count - TP_count
    FP_count = pp_count - TP_count

    dice_score = 2 * TP_count / (lp_count + pp_count) if lp_count + pp_count != 0 else 0
    iou_score = TP_count / (lp_count + pp_count - TP_count) if lp_count + pp_count - TP_count != 0 else 0
    precision = TP_count / (TP_count + FP_count) if FP_count + TP_count != 0 else 0
    recall = TP_count / (TP_count + FN_count) if TP_count + FN_count != 0 else 0

    return dice_score, iou_score, precision, recall


def get_score_from_all_slices(labels, predicts, threshold=0.5):
    '''
    :param truths: [n, 1, 224, 192]
    :param predicts: [n, 1, 224, 192]
    :param threshold: threshold for computing dice
    :return: scores
    '''
    if labels.size(0) % 184 != 0 or predicts.size(0) % 184 != 0:
        print('ERROR')
        return 0

    dice_scores = []
    iou_scores = []
    precision_scores = []
    recall_scores = []

    for i in range(labels.size(0) // 184):
        tmp_labels = labels[i * 184:(i + 1) * 184]
        tmp_pred = predicts[i * 184:(i + 1) * 184]
        tmp_dice, tmp_iou, tmp_precision, tmp_recall \
            = get_score_for_one_patient(labels=tmp_labels, predicts=tmp_pred, threshold=threshold)

        dice_scores.append(tmp_dice)
        iou_scores.append(tmp_iou)
        precision_scores.append(tmp_precision)
        recall_scores.append(tmp_recall)

    scores = {}
    scores['dice'] = dice_scores
    scores['iou'] = iou_scores
    scores['precision'] = precision_scores
    scores['recall'] = recall_scores

    return scores


def get_dice_from_all_scores(labels, predicts, threshold=0.5):
    '''
    :param truths: [n, 1, 224, 192]
    :param predicts: [n, 1, 224, 192]
    :param threshold: threshold for computing dice
    :return: a dice scores
    '''
    if labels.size(0) % 184 != 0 or predicts.size(0) % 184 != 0:
        print('ERROR')
        return 0

    dice_scores = []

    for i in range(labels.size(0) // 184):
        tmp_labels = labels[i * 184:(i + 1) * 184]
        tmp_pred = predicts[i * 184:(i + 1) * 184]

        tmp_dice, _, _, _ \
            = get_score_for_one_patient(labels=tmp_labels, predicts=tmp_pred, threshold=threshold)
        dice_scores.append(tmp_dice)

    return np.mean(dice_scores)
