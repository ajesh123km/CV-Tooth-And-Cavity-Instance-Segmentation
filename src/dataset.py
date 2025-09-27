import torch
import torchvision.transforms as T
from PIL import Image
import json
import os
import numpy as np
from pycocotools import mask as coco_mask


class COCODataset(torch.utils.data.Dataset):
    def __init__(self, root, annFile, transforms=None):
        """
        root: folder with images
        annFile: path to COCO JSON annotation file
        """
        self.root = root
        self.transforms = transforms

        # Load COCO-style annotations
        with open(annFile, "r") as f:
            self.coco = json.load(f)

        self.images = self.coco["images"]
        self.annotations = self.coco["annotations"]

        # Build lookup for annotations per image
        self.img_to_anns = {img["id"]: [] for img in self.images}
        for ann in self.annotations:
            self.img_to_anns[ann["image_id"]].append(ann)

        # Lookup categories (optional, in case you need class names)
        self.catid_to_name = {cat["id"]: cat["name"] for cat in self.coco.get("categories", [])}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get image info
        img_info = self.images[idx]
        img_path = os.path.join(self.root, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")

        # Get annotations for this image
        anns = self.img_to_anns[img_info["id"]]

        boxes = []
        labels = []
        masks = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            x_min, y_min = x, y
            x_max, y_max = x + w, y + h

            if w > 0 and h > 0:
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(ann["category_id"])

                # 🔑 Build binary mask from segmentation
                if "segmentation" in ann:
                    if isinstance(ann["segmentation"], list):
                        # Polygon format
                        rles = coco_mask.frPyObjects(ann["segmentation"], img_info["height"], img_info["width"])
                        rle = coco_mask.merge(rles)
                        m = coco_mask.decode(rle)
                    elif isinstance(ann["segmentation"], dict):  
                        # RLE format
                        m = coco_mask.decode(ann["segmentation"])
                    else:
                        raise ValueError("Unknown segmentation format in annotation")
                    masks.append(m)

        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        if masks:
            masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        else:
            masks = torch.zeros((0, img_info["height"], img_info["width"]), dtype=torch.uint8)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([img_info["id"]]),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) if len(boxes) > 0 else torch.tensor([]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64)
        }

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target


# Example transforms for images
def get_transform(train=True):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)
