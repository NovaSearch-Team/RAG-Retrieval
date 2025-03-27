import torch
import torch.nn as nn
from itertools import product


def pointwise_mse(logits, labels):
    scores = torch.sigmoid(logits)
    scores = scores.to(labels.dtype)
    return nn.MSELoss(reduction="mean")(scores, labels)


def pointwise_bce(logits, labels):
    return nn.BCEWithLogitsLoss(reduction="mean")(logits, labels)


def pairwise_ranknet(logits, labels, group_size):
    grouped_logits = logits.view(-1, group_size)
    grouped_labels = labels.view(-1, group_size)

    # here we generate every pair of indices from the range of document length in the batch
    document_pairs_candidates = list(product(range(grouped_labels.shape[1]), repeat=2))

    pairs_true = grouped_labels[:, document_pairs_candidates]
    selected_pred = grouped_logits[:, document_pairs_candidates]

    # here we calculate the relative true relevance of every candidate pair
    true_diffs = pairs_true[:, :, 0] - pairs_true[:, :, 1]
    pred_diffs = selected_pred[:, :, 0] - selected_pred[:, :, 1]

    # here we filter just the pairs that are 'positive' and did not involve a padded instance
    # we can do that since in the candidate pairs we had symetric pairs so we can stick with
    # positive ones for a simpler loss function formulation
    the_mask = (true_diffs > 0) & (~torch.isinf(true_diffs))

    pred_diffs = pred_diffs[the_mask]

    weight = None
    # true_pow_diffs = torch.pow(pairs_true[:, :, 0], 2) - torch.pow(pairs_true[:, :, 1], 2)
    abs_diff = torch.abs(true_diffs)
    weight = abs_diff[the_mask]

    # here we 'binarize' true relevancy diffs since for a pairwise loss we just need to know
    # whether one document is better than the other and not about the actual difference in
    # their relevancy levels
    true_diffs = (true_diffs > 0).type(torch.float32)
    true_diffs = true_diffs[the_mask]

    return nn.BCEWithLogitsLoss(reduction="mean", weight=weight)(pred_diffs, true_diffs)


def listwise_ce(logits, labels, group_size, tao=0.1):
    grouped_logits = logits.view(-1, group_size)
    grouped_labels = labels.view(-1, group_size)
    grouped_labels = torch.softmax(grouped_labels.detach(), dim=-1).to(
        grouped_logits.device
    )
    loss = -torch.mean(
        torch.sum(
            grouped_labels * torch.log_softmax(grouped_logits / tao, dim=-1), dim=-1
        )
    )
    return loss
