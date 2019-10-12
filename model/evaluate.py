from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import numpy as np
import torch
def decode_model_output(output, blank=0):
    """
    decode the output for every timestamp
    :param output: the size of model_output, [batch_size, w, num_class], w is the final feature map's width
    :param blank:
    :return:
    """
    batch_size, width, num_class = output.size()
    output = output.max(dim=-1)[1].cpu().data.numpy()
    result = []

    for sample in output:
        sample_result = []
        for i in range(width - 1):
            if sample[i] != sample[i - 1]:
                sample_result.append(sample[i])
        sample_result.append(sample[-1])
        sample_result = np.asarray(sample_result, dtype=np.int32)
        result.append(sample_result)

    # filter blank
    decoded_pred_label = [sample[sample != blank] for sample in result]
    return decoded_pred_label


def decode_model_output_verifycode(ctc_outputs, blank=0):
    outputs = ctc_outputs.max(dim=-1)[1].cpu().data.numpy()
    seq_len = outputs.shape[1]

    rank_score = ctc_outputs.sort(dim=-1, descending=True)[0].detach().cpu().numpy()
    rank_score = rank_score[:, :, :10]

    rank_id = ctc_outputs.sort(dim=-1, descending=True)[1].cpu().numpy()
    rank_id = rank_id[:, :, :10]

    saveIdxList = []
    preZeroIdx = -1

    for i, sample in enumerate(outputs):
        for idx in range(0, len(sample)):
            x = sample[idx]
            if x == 0:
                preZeroIdx = idx
                continue
            if len(saveIdxList) == 0:  # 加入第一个确定保留元素的下标
                saveIdxList.append(idx)
                continue

            preIdx = saveIdxList[-1]  # 最新的确定保留元素的下标
            preNum = sample[preIdx]
            if x == preNum and not (
                    preIdx < preZeroIdx < idx):  # 中间没有0隔开，必然有A[preIdx] == A[preIdx + 1] == ... == A[idx - 1] == A[idx]
                continue
            else:  # 否则加入保留下标列表
                saveIdxList.append(idx)

    result_digits = outputs[0][saveIdxList]
    result_id = rank_id[:, saveIdxList, :].squeeze(0)
    result_score = rank_score[:, saveIdxList, :].squeeze(0)

    return result_digits, result_id, result_score


def cal_accuracy(output, labels, length, decoded_pred_label):
    """
    compute the fully match accuary. firstly we need to decode labels through label length array,
    as all labels in min-batch have been concated to list.
    :param output: the size of model_output, [batch_size, w, num_class], w is the final feature map's width
    :param labels: ground truth
    :param length: length list for every sample
    :param decoded_pred_label: decoded from model output
    :return:
    """
    # decode the training labels
    decoded_gt_label = np.split(labels.data.numpy(), length.data.numpy().cumsum())[:-1]

    batch_size, width, num_class = output.size()
    result = []
    for i in range(batch_size):
        result.append(np.array_equal(decoded_gt_label[i], decoded_pred_label[i]))

    accuracy = np.asarray(result).mean()
    return accuracy


def levenshtein(s1, s2):
    """edit distance"""
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[
                             j + 1] + 1  # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1  # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

