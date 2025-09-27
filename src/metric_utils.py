import numpy as np
import torch

from tqdm.cli import tqdm
from typing import Dict, List

from mask_rcnn_utils import process_outputs

def mask_iou_matrix(masks_pred, masks_true):
    """Compute the IoU matrix between a set of segmentation masks.
    
    Parameters:
        masks_pred (torch.Tensor[bool]): The predicted segmentation masks.
        masks_true (torch.Tensor[bool]): The ground truth segmentation masks.
    
    Returns:
        iou_matrix (torch.Tensor[float]): The IoU Matrix where `iou_matrix[i, j]`
            is the IoU between predicted mask `i` and ground truth mask `j`.
    """
    n = len(masks_pred)
    m = len(masks_true)
    iou_matrix = torch.zeros((n, m), dtype=torch.float, device=masks_pred.device)

    for i, mask_pred in enumerate(masks_pred):
        for j, mask_true in enumerate(masks_true):
            intersection = torch.logical_and(mask_pred, mask_true).sum()
            union = torch.logical_or(mask_pred, mask_true).sum()
            iou_matrix[i, j] = intersection / union
    
    return iou_matrix


def match_predicted_and_true_masks_from_iou_matrix(
    targets_pred: Dict,
    targets_true: Dict,
    iou_matrix,
    iou_threshold: float = 0.5
    ):
    """Similar to `match_predicted_and_true_masks` except we take in a pre-computed
    IoU matrix to reduce computation.
    """
    masks_pred = targets_pred["masks"]
    masks_true = targets_true["masks"]

    labels_pred = targets_pred["labels"]
    labels_true = targets_true["labels"]

    # To avoid double-counting, keep track of which ground-truth indices we've matched
    matched = torch.zeros(len(masks_true), dtype=torch.bool, device=masks_true.device)

    # Iterate through each prediction label and IoU to see if we have a match (TP)
    true_positives = []
    false_positives = []
    for idx_pred, label_pred in enumerate(labels_pred):
        # Our goal is to find the ground truth index that maximizes our IoU
        max_iou = -1
        best_idx_true = -1

        for idx_true, (iou, label_true) in enumerate(zip(iou_matrix[idx_pred], labels_true)):
            if not matched[idx_true] and iou >= iou_threshold and label_pred == label_true:
                if iou > max_iou:
                    max_iou = iou
                    best_idx_true = idx_true
                    matched[idx_true] = True
        
        if best_idx_true >= 0:
            # Match found - TP
            true_positives.append((idx_pred, best_idx_true, max_iou.item()))
        else:
            # No match found - FP
            false_positives.append(idx_pred)
    
    # Remaining unmatched ground truth indices are false negatives
    false_negatives = torch.where(~matched)[0].tolist()

    return true_positives, false_positives, false_negatives


def match_predicted_and_true_masks(
        targets_pred: Dict,
        targets_true: Dict,
        iou_threshold: float = 0.5
    ):
    """Generate the confusion matrix beween the segmentations and labels of
    predicted targets and ground truth targets.

    Parameters:
        targets_pred (Dict[str, torch.Tensor]): The predicted outputs such that...
            - `targets_pred["masks"]` provide the predicted segmentation masks
            - `targets_pred["labels"]` provide the predicted labels
        targets_true (Dict[str, torch.Tensor]): The ground truth outputs. Expected
            to contain similar outputs as `targets_pred`.
        iou_threshold (float): The minimum IoU required for a predicted
            mask to be considered a match (true positive).
    
    Returns:
        true_positives (List[Tuple[int, int, float]]): A list of true positive
            predictions indices. Each list contains a 3-tuple such that, for a
            true postive at index `i`, `idx_pred, idx_true, iou = true_positives[i]`.
        false_positives (List[int]): A list of false positive indices corresponding
            to the predictions.
        false_negatives (List[int]): A list of false negative indices corresponding
            to the ground truth.
    """
    masks_pred = targets_pred["masks"]
    masks_true = targets_true["masks"]

    labels_pred = targets_pred["labels"]
    labels_true = targets_true["labels"]

    iou_matrix = mask_iou_matrix(masks_pred, masks_true)
    return match_predicted_and_true_masks_from_iou_matrix(
        targets_pred,
        targets_true,
        iou_matrix,
        iou_threshold=iou_threshold
    )


def f_score_from_counts(tp: int, fp: int, fn: int, beta: float = 1.0):
    """Compute the F-Score from raw confusion matrix counts.

    Parameters:
        tp (int): The number of true positives (TP).
        fp (int): The number of false positives (FP).
        fn (int): The number of false negatives (FN).
        beta (float): A weighting factor such that the recall is considered
            `beta` times as important as precision 
    
    Returns:
        f_score (float): The computed F-Score.
    """
    factor = (1 + beta**2) * tp
    denominator = beta**2 * fn + fp
    f_score = factor / (factor + denominator)
    return f_score


def f_score_from_matches(tp: List, fp: List, fn: List, beta: float = 1.0):
    """For details, see `f_score_from_counts`. Takes in output from
    `match_predicted_and_true_masks`.
    """
    tp = len(tp)
    fp = len(fp)
    fn = len(fn)
    return f_score_from_counts(tp, fp, fn, beta=beta)


def evaluate_instance_segmentation(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        iou_threshold: float = 0.5
    ) -> Dict[str, float]:
    """ Evaluates instance segmentation model performance. """

    if not (0 <= iou_threshold <= 1):
        raise ValueError("IoU threshold must be between 0 and 1")

    model.eval()
    device = next(model.parameters()).device   # ✅ detect model device
    tp, fp, fn = 0, 0, 0

    all_scores = []
    all_matches = []

    pbar = tqdm(enumerate(dataloader), desc="Evaluating", total=len(dataloader))
    for i, (images, y_trues) in pbar:
        # ✅ move inputs to same device as model
        images = [img.to(device) for img in images]
        y_trues = [{k: v.to(device) for k, v in t.items()} for t in y_trues]

        with torch.no_grad():
            y_preds = model(images)
            y_preds = process_outputs(y_preds)

            iou_matrices = []
            for y_pred, y_true in zip(y_preds, y_trues):
                # ✅ align masks to same device
                y_pred["masks"] = y_pred["masks"].to(device)
                y_true["masks"] = y_true["masks"].to(device)

                iou_matrices.append(mask_iou_matrix(y_pred["masks"], y_true["masks"]))


        for y_pred, y_true, iou_matrix in zip(y_preds, y_trues, iou_matrices):
            tp_idx, fp_idx, fn_idx = match_predicted_and_true_masks_from_iou_matrix(
                y_pred, y_true, iou_matrix, iou_threshold=iou_threshold
            )

            tp += len(tp_idx)
            fp += len(fp_idx)
            fn += len(fn_idx)

            scores = y_pred["scores"].detach().cpu().numpy()
            matches = np.zeros(len(scores))
            matches[[idx[0] for idx in tp_idx]] = 1

            all_scores.extend(scores)
            all_matches.extend(matches)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            pbar.set_postfix({
                "Precision": f"{precision:.3f}",
                "Recall": f"{recall:.3f}",
                "F1-Score": f"{f1:.3f}"
            })

    # mAP calculation
    all_scores = np.array(all_scores)
    all_matches = np.array(all_matches)
    sort_idx = np.argsort(-all_scores)
    all_matches = all_matches[sort_idx]

    precisions = np.cumsum(all_matches) / np.arange(1, len(all_matches) + 1)
    recalls = np.cumsum(all_matches) / (tp + fn)
    mAP = np.trapz(precisions, recalls) if len(recalls) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mAP": mAP,
        "total_predictions": tp + fp,
        "total_ground_truth": tp + fn
    }


def evaluate_instance_segmentation_multiple_thresholds(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        iou_thresholds: List[float],
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ) -> List[Dict[str, float]]:

    if not all(0 <= iou_threshold <= 1 for iou_threshold in iou_thresholds):
        raise ValueError("All IoU thresholds must be between 0 and 1.")

    model.eval()
    tp = np.zeros_like(iou_thresholds)
    fp = np.zeros_like(iou_thresholds)
    fn = np.zeros_like(iou_thresholds)
    
    # Our return dict - each index hold metrics for the IoU threshold at that index
    metrics = [{"iou_threshold": iou_threshold} for iou_threshold in iou_thresholds]

    all_scores = [[] for _ in range(len(iou_thresholds))]
    all_matches = [[] for _ in range(len(iou_thresholds))]

    pbar = tqdm(enumerate(dataloader), desc="Evaluating", total=len(dataloader))
    for i, (images, y_trues) in pbar:
        with torch.no_grad():
            # Make predictions with the testing set
            images = [img.to(device) for img in images]   # move images to GPU
            y_trues = [{k: v.to(device) for k, v in t.items()} for t in y_trues]  # if using targets
            y_preds = model(images)
            y_preds = process_outputs(y_preds)

            # Compute IoU between the predicted and true segmentation masks
            iou_matrices = [
                mask_iou_matrix(y_pred["masks"], y_true["masks"])
                for y_pred, y_true in zip(y_preds, y_trues)
            ]

        for y_pred, y_true, iou_matrix in zip(y_preds, y_trues, iou_matrices):
            for i, iou_threshold in enumerate(iou_thresholds):
                # Compute tp, fp, fn for each IoU threshold
                tp_idx, fp_idx, fn_idx = match_predicted_and_true_masks_from_iou_matrix(
                    y_pred,
                    y_true,
                    iou_matrix
                )

                # Store this in it's row-index
                tp[i] += len(tp_idx)
                fp[i] += len(fp_idx)
                fn[i] += len(fn_idx)

                # Store confidence scores and matches (Matches = 1 => TP; Matches = 0 => FP)
                scores = y_pred["scores"].cpu().numpy()
                matches = np.zeros(len(scores))
                matches[[idx[0] for idx in tp_idx]] = 1

                all_scores[i].extend(scores)
                all_matches[i].extend(matches)
        
    # Iterate through each IoU threshold and store/compute our metrics
    for i, iou_threshold in enumerate(iou_thresholds):
        metrics[i]["tp"] = tp[i]
        metrics[i]["fp"] = fp[i]
        metrics[i]["fn"] = fn[i]

        precision = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0
        recall = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall)

        metrics[i]["precision"] = precision
        metrics[i]["recall"] = recall
        metrics[i]["f1_score"] = f1

        # Compute our mAP
        scores = np.array(all_scores[i])
        matches = np.array(all_matches[i])

        # Sort matches by confidence (descending)
        sort_idx = np.argsort(-scores)
        matches = matches[sort_idx]

        precisions = np.cumsum(matches) / np.arange(1, len(matches) + 1)
        recalls = np.cumsum(matches) / (tp[i] + fn[i])
        mAP = np.trapz(precisions, recalls) if len(recalls) > 0 else 0

        metrics[i]["mAP"] = mAP
        metrics[i]["total_predictions"] = tp[i] + fp[i]
        metrics[i]["total_ground_truth"] = tp[i] + fn[i]
        metrics[i]["mAP_precisions"] = precisions
        metrics[i]["mAP_recalls"] = recalls
        metrics[i]["mAP_scores"] = scores[sort_idx]

    # Now create average metrics as the last index in our list
    metric_keys = ["precision", "recall", "f1_score", "mAP"]
    metrics_avg = {"iou_threshold": "average"}
    for metric_key in metric_keys:
        metrics_avg[metric_key] = np.mean([metric[metric_key] for metric in metrics])
    
    metrics.append(metrics_avg)

    return metrics