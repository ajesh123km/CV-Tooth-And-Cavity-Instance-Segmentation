import matplotlib
import matplotlib.pyplot as plt
import torch

from typing import Optional


def plot_coco_image(
        image,
        target,
        plot_masks: bool = True,
        plot_bboxes: bool = True,
        plot_segmentations: bool = True,
        plot_category_id: bool = True,
        category_names: Optional[dict[int, str]] = None
    ):
    if image.shape[0] == 3:
        image = torch.moveaxis(image, 0, 2)

    bboxes = target["boxes"]
    masks = target["masks"]
    segmentations = target["segmentation"]
    categories = target["category_id"]

    fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])

    if plot_masks:
        axis_add_masks(ax, masks)

    if plot_bboxes:
        axis_add_bboxes(ax, bboxes)

    if plot_segmentations:
        axis_add_segmentations(ax, segmentations)

    if plot_category_id:
        axis_add_category_id(ax, bboxes, categories, category_names)
        
    return fig


def axis_add_masks(ax, masks):
    for i, mask in enumerate(masks):
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        # squeeze channel dim if shape is (1, H, W)
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)
        ax.imshow(mask + 0.5 * i, alpha=mask * 0.25, cmap="tab10", vmin=0, vmax=len(masks) / 2)




def axis_add_bboxes(ax, bboxes):
    for bbox in bboxes:
        x, y, width, height = bbox
        patch = matplotlib.patches.Rectangle(
            (x, y),
            width,
            height,
            alpha=1,
            fill=False,
            edgecolor="red",
            linewidth=1,
            mouseover=True
        )
        ax.add_patch(patch)


def axis_add_segmentations(ax, segmentations):
    for segmentation in segmentations:
        x = [val for i, val in enumerate(segmentation[0]) if i % 2 == 0]
        y = [val for i, val in enumerate(segmentation[0]) if i % 2 != 0]
        ax.plot(x, y, color="black")


def axis_add_category_id(ax, bboxes, categories, category_names):
    for i, bbox in enumerate(bboxes):
        category_id = categories[i]
        if isinstance(category_id, torch.Tensor):
            category_id = category_id.item()  # ✅ convert to int
        if category_names is not None:
            display_text = category_names[category_id]
        else:
            display_text = str(category_id)
        # then draw text
        x, y = bbox[0], bbox[1]
        ax.text(x, y, display_text, color="white", fontsize=8,
                bbox=dict(facecolor="red", alpha=0.5))



def plot_coco_image_prediction(
        image,
        target_pred,
        target_true,
        plot_masks: bool = True,
        plot_bboxes: bool = True,
        plot_segmentations: bool = True,
        plot_category_id: bool = True,
        category_names: Optional[dict[int, str]] = None
    ):
    if image.shape[0] == 3:
        image = torch.moveaxis(image, 0, 2)

    bboxes_pred = target_pred["bbox"]
    bboxes_true = target_true["bbox"]

    masks_pred = target_pred["masks"]
    masks_true = target_true["masks"]

    segmentations_pred = target_pred["segmentation"]
    segmentations_true = target_true["segmentation"]
    
    categories_pred = target_pred["category_id"]
    print(categories_pred)
    categories_true = target_true["category_id"]
    

    fig, axes = plt.subplots(figsize=(12, 8), ncols=3, tight_layout=True)
    axes[0].set_title("Image")
    axes[1].set_title("Truth")
    axes[2].set_title("Prediction")
    for ax in axes:
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])

    if plot_masks:
        axis_add_masks(axes[1], masks_true)
        axis_add_masks(axes[2], masks_pred)

    if plot_bboxes:
        axis_add_bboxes(axes[1], bboxes_true)
        axis_add_bboxes(axes[2], bboxes_pred)

    if plot_segmentations:
        axis_add_segmentations(axes[1], segmentations_true)
        axis_add_segmentations(axes[2], segmentations_pred)

    if plot_category_id:
        axis_add_category_id(axes[1], bboxes_true, categories_true, category_names)
        axis_add_category_id(axes[2], bboxes_pred, categories_pred, category_names)
        
    return fig


# def plot_coco_image_predictions(
#         images,
#         targets_pred,
#         targets_true,
#         plot_masks: bool = True,
#         plot_bboxes: bool = True,
#         plot_segmentations: bool = True,
#         plot_category_id: bool = True,
#         category_names: Optional[dict[int, str]] = None
#     ):
#     # ✅ Ensure images is a list of tensors
#     if isinstance(images, torch.Tensor):
#         images = list(images)  

#     n_images = len(images)

#     figsize = (12, 4 * n_images)
#     fig, axes = plt.subplots(figsize=figsize, nrows=n_images, ncols=3, tight_layout=True)

#     # Handle single image case
#     if n_images == 1:
#         axes = [axes]

#     axes[0][0].set_title("Image")
#     axes[0][1].set_title("Truth")
#     axes[0][2].set_title("Prediction")

#     for i, (image, target_pred, target_true) in enumerate(zip(images, targets_pred, targets_true)):
#         if image.shape[0] == 3:  # C, H, W → H, W, C
#             image = torch.moveaxis(image, 0, 2)

#         bboxes_pred = target_pred.get("boxes", [])
#         bboxes_true = target_true.get("boxes", [])

#         masks_pred = target_pred.get("masks", [])
#         masks_true = target_true.get("masks", [])

#         segmentations_pred = target_pred.get("segmentation", [])
#         segmentations_true = target_true.get("segmentation", [])

#         categories_pred = target_pred.get("labels", [])
#         categories_true = target_true.get("labels", [])

#         for ax in axes[i]:
#             ax.imshow(image)
#             ax.set_xticks([])
#             ax.set_yticks([])

#         if plot_masks:
#             axis_add_masks(axes[i][1], masks_true)
#             axis_add_masks(axes[i][2], masks_pred)

#         if plot_bboxes:
#             axis_add_bboxes(axes[i][1], bboxes_true)
#             axis_add_bboxes(axes[i][2], bboxes_pred)

#         if plot_segmentations:
#             axis_add_segmentations(axes[i][1], segmentations_true)
#             axis_add_segmentations(axes[i][2], segmentations_pred)

#         if plot_category_id:
#             axis_add_category_id(axes[i][1], bboxes_true, categories_true, category_names)
#             axis_add_category_id(axes[i][2], bboxes_pred, categories_pred, category_names)

#     return fig


def plot_coco_image_predictions(
        images,
        targets_pred,
        targets_true,
        plot_masks: bool = True,
        plot_bboxes: bool = True,
        plot_segmentations: bool = True,
        plot_category_id: bool = True,
        category_names: Optional[dict[int, str]] = None,
        score_threshold: float = 0.7   # ✅ NEW: confidence threshold
    ):
    # ✅ Ensure images is a list of tensors
    if isinstance(images, torch.Tensor):
        images = list(images)  

    n_images = len(images)

    figsize = (12, 4 * n_images)
    fig, axes = plt.subplots(figsize=figsize, nrows=n_images, ncols=3, tight_layout=True)

    # Handle single image case
    if n_images == 1:
        axes = [axes]

    axes[0][0].set_title("Image")
    axes[0][1].set_title("Truth")
    axes[0][2].set_title("Prediction")

    for i, (image, target_pred, target_true) in enumerate(zip(images, targets_pred, targets_true)):
        if image.shape[0] == 3:  # C, H, W → H, W, C
            image = torch.moveaxis(image, 0, 2)

        # ✅ Filter predictions by score threshold
        scores = target_pred.get("scores", torch.tensor([]))
        if len(scores) > 0:
            keep = scores >= score_threshold
            bboxes_pred = target_pred.get("boxes", [])[keep]
            masks_pred = target_pred.get("masks", [])[keep] if "masks" in target_pred else []
            segmentations_pred = target_pred.get("segmentation", [])
            categories_pred = target_pred.get("labels", [])[keep]
        else:
            bboxes_pred, masks_pred, segmentations_pred, categories_pred = [], [], [], []

        # Truth (no filtering)
        bboxes_true = target_true.get("boxes", [])
        masks_true = target_true.get("masks", [])
        segmentations_true = target_true.get("segmentation", [])
        categories_true = target_true.get("labels", [])

        for ax in axes[i]:
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])

        if plot_masks:
            axis_add_masks(axes[i][1], masks_true)
            axis_add_masks(axes[i][2], masks_pred)

        if plot_bboxes:
            axis_add_bboxes(axes[i][1], bboxes_true)
            axis_add_bboxes(axes[i][2], bboxes_pred)

        if plot_segmentations:
            axis_add_segmentations(axes[i][1], segmentations_true)
            axis_add_segmentations(axes[i][2], segmentations_pred)

        if plot_category_id:
            axis_add_category_id(axes[i][1], bboxes_true, categories_true, category_names)
            axis_add_category_id(axes[i][2], bboxes_pred, categories_pred, category_names)

    return fig
