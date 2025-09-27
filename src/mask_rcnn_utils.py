import numpy as np
import cv2 as cv
import torch
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2
from torchvision.datasets import (
    CocoDetection,
    wrap_dataset_for_transforms_v2
)

from pathlib import Path
from typing import Union, Tuple, Optional


CATEGORY_ID_TO_NAME = {
    0: "Null",
    1: "Caries",
    2: "Cavity",
    3: "Crack",
    4: "Tooth"
}

TRANSFORMS_AUGMENT = v2.Compose([
    v2.ToTensor(),
    v2.RandomPhotometricDistort(p=0.5),
    v2.ClampBoundingBoxes(),
    v2.ConvertImageDtype()
])

TRANSFORMS_PROCESS = v2.Compose([
    v2.ToTensor(),
    v2.ClampBoundingBoxes(),
    v2.ConvertImageDtype()
])


from typing import Union
from pathlib import Path
from pycocotools import mask as maskUtils
from torchvision.datasets import CocoDetection
from torchvision import tv_tensors
from torchvision.transforms import v2


def convert_coco_poly_to_mask(segmentation, height, width):
    """Convert COCO polygon/RLE segmentation to a binary mask."""
    if isinstance(segmentation, list):  # polygons
        rles = maskUtils.frPyObjects(segmentation, height, width)
        rle = maskUtils.merge(rles)
    elif isinstance(segmentation, dict):  # RLE
        rle = segmentation
    else:
        raise TypeError(f"Unknown segmentation format: {type(segmentation)}")
    mask = maskUtils.decode(rle)
    # Some masks are (H, W, 1), squeeze to (H, W)
    if mask.ndim == 3:
        mask = np.any(mask, axis=2)
    return torch.as_tensor(mask, dtype=torch.uint8)


class CocoDetectionWithMasks(CocoDetection):
    def __getitem__(self, idx):
        img, anns = super().__getitem__(idx)
        if hasattr(img, "size") and not callable(img.size):  
            # PIL image
            w, h = img.size
        else:
            # Tensor image: shape is (C, H, W)
            _, h, w = img.shape


        boxes = []
        labels = []
        masks = []
        segmentations = []
        category_id =[]
        for ann in anns:
            # bbox in COCO is [x, y, w, h]
            boxes.append(ann["bbox"])
            labels.append(ann["category_id"])
            masks.append(convert_coco_poly_to_mask(ann["segmentation"], h, w))
            segmentations.append(ann["segmentation"])
            category_id.append(ann["category_id"])

        # Convert to tensors compatible with torchvision v2
        target = {}
        if boxes:
            target["boxes"] = tv_tensors.BoundingBoxes(
                torch.tensor(boxes, dtype=torch.float32),
                format="XYWH",  # COCO format
                canvas_size=(h, w),
            )
            target["labels"] = torch.tensor(labels, dtype=torch.int64)
            target["masks"] = tv_tensors.Mask(torch.stack(masks))
            target["segmentation"] = segmentations
            target["category_id"] = category_id
        else:
            # Handle images without annotations
            target["boxes"] = tv_tensors.BoundingBoxes(
                torch.zeros((0, 4), dtype=torch.float32),
                format="XYWH",
                canvas_size=(h, w),
            )
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
            target["masks"] = tv_tensors.Mask(torch.zeros((0, h, w), dtype=torch.uint8))
            target["segmentation"] = segmentations
            target["category_id"] = category_id

        return img, target


def get_cocodetection_dataset(
    data_path: Union[Path, str],
    annotation_path: Union[Path, str],
    train: bool = True
):
    transforms = TRANSFORMS_AUGMENT if train else TRANSFORMS_PROCESS
    return CocoDetectionWithMasks(
        data_path,
        annotation_path,
        transforms=transforms
    )

def fix_boxes(boxes):
    """Ensure x_min < x_max and y_min < y_max."""
    boxes = boxes.clone()
    x_min = boxes[:, 0]
    y_min = boxes[:, 1]
    x_max = boxes[:, 2]
    y_max = boxes[:, 3]

    # swap if needed
    x1 = torch.min(x_min, x_max)
    x2 = torch.max(x_min, x_max)
    y1 = torch.min(y_min, y_max)
    y2 = torch.max(y_min, y_max)

    return torch.stack([x1, y1, x2, y2], dim=1)

def custom_collate_function(batch, to_cuda: bool = True):
    images, targets = tuple(zip(*batch))

    if not to_cuda:
        return (images, targets)

    images = torch.stack(images).cuda()
    for target in targets:
        
        target["boxes"] = fix_boxes(target["boxes"]).cuda()
        target["bbox"] = fix_boxes(target["boxes"]).cuda()
        target["labels"] = torch.Tensor(target["labels"]).cuda()
        
    return (images, targets)

def to_xywh(boxes):
    # boxes in XYXY -> XYWH
    x_min, y_min, x_max, y_max = boxes.unbind(1)
    return torch.stack([x_min, y_min, x_max - x_min, y_max - y_min], dim=1)

def to_xyxy(boxes):
    # boxes in XYWH -> XYXY
    x, y, w, h = boxes.unbind(1)
    return torch.stack([x, y, x + w, y + h], dim=1)

def process_output(
        output,
        spatial_size: Optional[Tuple[int, int]] = None,
        iou_threshold: float = 0.6,
        score_threshold: float = 0.5
    ):
    # Filter out low confidence bounding boxes: `score < score_threshold`
    indices_keep_scores = (output["scores"] >= score_threshold)
    output["boxes"] = output["boxes"][indices_keep_scores]
    output["labels"] = output["labels"][indices_keep_scores]
    output["scores"] = output["scores"][indices_keep_scores]
    output["masks"] = output["masks"][indices_keep_scores]

    # Perform Non-Max Suppression to help remove duplicate bounding boxes
    indices_keep = torchvision.ops.nms(output["boxes"], output["scores"], iou_threshold).cpu()

    # Move to CPU and remove unnecessary boxes
    for key, val in output.items():
        if isinstance(val, torch.Tensor):
            output[key] = val.cpu()

    output["boxes"] = output["boxes"][indices_keep]
    output["labels"] = output["labels"][indices_keep]
    output["scores"] = output["scores"][indices_keep]
    output["masks"] = output["masks"][indices_keep]

    if spatial_size is None:
        spatial_size = output["boxes"].shape[2:]

    # Add in the "bbox" key
    #bbox = torchvision.datapoints.BoundingBox(output["boxes"], format="XYXY", spatial_size=spatial_size)
    output["bbox"] = to_xywh(output["boxes"])#v2.ConvertBoundingBoxFormat("XYWH")(bbox)

    # We can remove the 2nd dimension of our masks.
    output["masks"] = output["masks"].squeeze(1)
    output["masks_raw"] = output["masks"]
    output["masks"] = output["masks"] >= 0.5
    
    # Extract segmentation information from our masks
    output["segmentation"] = []
    for mask in output["masks"]:
        mask = np.array(mask).astype(np.uint8)
        contours, hierarchy = cv.findContours(
            mask,
            mode=cv.RETR_EXTERNAL,
            method=cv.CHAIN_APPROX_NONE
        )
    
        segmentation = np.expand_dims(contours[-1].flatten(), 0)
        output["segmentation"].append(segmentation)

    # category_id is just a simple transformation of our labels
    output["category_id"] = [label.item() for label in output["labels"]]

    print([label.item() for label in output["labels"]])

    return output   
    

def process_outputs(
        outputs,
        spatial_size: Optional[Tuple[int, int]] = None,
        iou_threshold: float = 0.6,
        score_threshold: float = 0.5
    ):
    return [
        process_output(output, spatial_size, iou_threshold, score_threshold)
        for output in outputs
    ]