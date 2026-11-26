# Adapted from Qualcomm AI Hub Models v0.61.0. BSD-3-Clause.
# Arithmetic function bodies copied unchanged; source files and SHA256 manifest retained.
from __future__ import annotations
import math
import cv2
import numpy as np
import torch
from torch.nn.functional import interpolate,pad
from torchvision.ops import nms,batched_nms as tv_batched_nms


def batched_nms(
    iou_threshold: float,
    score_threshold: float | None,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_indices: torch.Tensor | None = None,
    *gather_additional_args: torch.Tensor,
) -> tuple[list[torch.Tensor], ...]:
    """
    Non maximum suppression over several batches.

    Parameters
    ----------
    iou_threshold
        Intersection over union (IoU) threshold.
    score_threshold
        Score threshold (throw away any boxes with scores under this threshold).
    boxes
        Boxes to run NMS on. Shape is [B, N, 4], B == batch, N == num boxes,
        and 4 == (x1, x2, y1, y2).
    scores
        Scores for each box. Shape is [B, N], range is [0:1].
    class_indices
        Class for each box. Shape is [B, N].
        If set, NMS is applied per-class rather than globally.
    *gather_additional_args
        Additional tensor(s) to be gathered in the same way as boxes and scores.
        In other words, each arg is returned with only the elements for the boxes
        selected by NMS. Should be shape [B, N, ...].

    Returns
    -------
    tuple[list[torch.Tensor], ...]
        boxes_out
            Output boxes. This is list of tensors--one tensor per batch.
            Each tensor is shape [S, 4], where S == number of selected boxes,
            and 4 == (x1, x2, y1, y2).
        scores_out
            Output scores. This is list of tensors--one tensor per batch.
            Each tensor is shape [S], where S == number of selected boxes.
        class_indices_out
            Output classes. This is list of tensors--one tensor per batch.
            Each tensor is shape [S], where S == number of selected boxes.
            Only returned when class_indices is provided; omitted from return tuple otherwise.
        *args_out
            Filtered versions of gather_additional_args, containing only elements
            corresponding to boxes selected by NMS. Each is a list of tensors
            (one per batch). Only returned when gather_additional_args is provided.
    """
    scores_out: list[torch.Tensor] = []
    boxes_out: list[torch.Tensor] = []
    class_indices_out: list[torch.Tensor] = []
    args_out: list[list[torch.Tensor]] = (
        [[] for _ in gather_additional_args] if gather_additional_args else []
    )

    for batch_idx in range(boxes.shape[0]):
        # Index to current batch.
        batch_scores = scores[batch_idx]
        batch_boxes = boxes[batch_idx]
        batch_args = [arg[batch_idx] for arg in gather_additional_args or []]
        batch_class_indices = (
            class_indices[batch_idx] if class_indices is not None else None
        )

        # Clip outputs to valid scores
        if score_threshold is not None:
            scores_idx = torch.nonzero(scores[batch_idx] >= score_threshold).squeeze(-1)
            batch_scores = batch_scores[scores_idx]
            batch_boxes = batch_boxes[scores_idx]
            batch_class_indices = (
                batch_class_indices[scores_idx]
                if batch_class_indices is not None
                else None
            )
            batch_args = [arg[scores_idx] for arg in batch_args or []]

        if len(batch_scores > 0):
            # Apply NMS
            if batch_class_indices is not None:
                # class dependent
                nms_indices = tv_batched_nms(
                    batch_boxes[..., :4],
                    batch_scores,
                    batch_class_indices,
                    iou_threshold,
                )
            else:
                # class agnostic
                nms_indices = nms(batch_boxes[..., :4], batch_scores, iou_threshold)

            # Apply NMS indices
            batch_boxes = batch_boxes[nms_indices]
            batch_scores = batch_scores[nms_indices]
            batch_class_indices = (
                batch_class_indices[nms_indices]
                if batch_class_indices is not None
                else None
            )
            batch_args = [arg[nms_indices] for arg in batch_args]

        # Append to outputs
        boxes_out.append(batch_boxes)
        scores_out.append(batch_scores)
        if batch_class_indices is not None:
            class_indices_out.append(batch_class_indices)
        for arg_idx, arg in enumerate(batch_args):
            args_out[arg_idx].append(arg)

    if class_indices is None:
        return boxes_out, scores_out, *args_out
    return boxes_out, scores_out, class_indices_out, *args_out

def compute_box_corners_with_rotation(
    xc: torch.Tensor,
    yc: torch.Tensor,
    w: torch.Tensor,
    h: torch.Tensor,
    theta: torch.Tensor,
) -> torch.Tensor:
    """
    From the provided information, compute the (x, y) coordinates of the box's corners.

    Parameters
    ----------
    xc
        Center of box (x). Shape is [Batch].
    yc
        Center of box (y). Shape is [Batch].
    w
        Width of box. Shape is [Batch].
    h
        Height of box. Shape is [Batch].
    theta
        Rotation of box (in radians). Shape is [Batch].

    Returns
    -------
    corners : torch.Tensor
        Computed corners. Shape is (B x 4 x 2), where 2 == (x, y).
    """
    batch_size = xc.shape[0]

    # Construct unit square
    points = torch.tensor([[-1, -1, 1, 1], [-1, 1, -1, 1]], dtype=torch.float32).repeat(
        batch_size, 1, 1
    )  # Construct Unit Square. Shape [B, 2, 4], where 2 == (X, Y)
    points *= torch.stack((w / 2, h / 2), dim=-1).unsqueeze(
        dim=2
    )  # Scale unit square to appropriate height and width

    # Rotate unit square to new coordinate system
    R = torch.stack(
        (
            torch.stack((torch.cos(theta), -torch.sin(theta)), dim=1),
            torch.stack((torch.sin(theta), torch.cos(theta)), dim=1),
        ),
        dim=1,
    )  # Construct rotation matrix
    points = R @ points  # Apply Rotation

    # Adjust box to center around the original center
    points = points + torch.stack((xc, yc), dim=1).unsqueeze(dim=2)

    return points.transpose(-1, -2)

def compute_box_affine_crop_resize_matrix(
    box_corners: torch.Tensor, output_image_size: tuple[int, int]
) -> list[np.ndarray]:
    """
    Computes the affine transform matrices required to crop, rescale,
    and pad the box described by box_corners to fit into an image of the given size without warping.

    Parameters
    ----------
    box_corners
        Bounding box corners. These coordinates will be mapped to the output image.
        Shape is [B, 3, 2], where B = batch, 3 = (top left point, bottom left point,
        top right point), and 2 = (x, y).
    output_image_size
        Size of image to which the box should be resized and cropped. Layout is (W, H).

    Returns
    -------
    affines : list[np.ndarray]
        Computed affine transform matrices. Shape is (2 x 3).
    """
    # Define coordinates for translated image
    network_input_points = np.array(
        [[0, 0], [0, output_image_size[1] - 1], [output_image_size[0] - 1, 0]],
        dtype=np.float32,
    )

    # Compute affine transformation that will map the square to the point
    affines: list[np.ndarray] = []
    for batch in range(box_corners.shape[0]):
        src = box_corners[batch][..., :3].detach().numpy()
        affines.append(cv2.getAffineTransform(src, network_input_points))
    return affines

def box_xywh_to_xyxy(box_cwh: torch.Tensor) -> torch.Tensor:
    """
    Convert center, W, H to top left / bottom right bounding box values.

    Parameters
    ----------
    box_cwh
        Bounding box.

        Shape is either...
        [..., 4]
        Box layout is [xc, yc, w, h]

        [..., 2, 2]
        Box layout is [[xc, yc], [w, h]]

    Returns
    -------
    box_xyxy: torch.Tensor
        if input shape is [..., 4]:
            Output shape is [..., 4]
            Output box layout is [x0, y0, x1, y1]
        if input shape is [..., 2, 2]:
            Output shape is [..., 2, 2]
            Output box layout is [[x0, y0], [x1, y1]]
    """
    if box_cwh.shape[-1] == 4:
        # box_cwh is [..., 4] with [cx, cy, w, h]
        # Output is [..., 4] with [x0, y0, x1, y1]
        center = box_cwh[..., :2]  # [cx, cy]
        wh_half = box_cwh[..., 2:] * 0.5  # [w/2, h/2]
        return torch.cat((center - wh_half, center + wh_half), dim=-1)

    center = box_cwh[..., 0, :]  # [cx, cy]
    wh_half = box_cwh[..., 1, :] * 0.5  # [w/2, h/2]
    return torch.stack((center - wh_half, center + wh_half), dim=-2)

def box_xyxy_to_xywh(box_xy: torch.Tensor) -> torch.Tensor:
    """
    Converts box coordinates to center / width / height notation.

    Parameters
    ----------
    box_xy
        Bounding box.

        Shape is either...
        [..., 4]
        Box layout is [x0, y0, x1, y1]

        [..., 2, 2]
        Box layout is [[x0, y0], [x1, y1]]

    Returns
    -------
    box_cwh: torch.Tensor
        if input shape is [..., 4]:
            Output shape is [..., 4]
            Output box layout is [xc, yc, w, h]
        if input shape is [..., 2, 2]:
            Output shape is [..., 2, 2]
            Output box layout is [[xc, yc], [w, h]]
    """
    if box_xy.shape[-1] == 4:
        # Input is [..., 4] with [x0, y0, x1, y1]
        # Output is [..., 4] with [xc, yc, w, h]
        xy1 = box_xy[..., :2]
        xy2 = box_xy[..., 2:]
        center = (xy1 + xy2) * 0.5
        wh = xy2 - xy1
        return torch.cat((center, wh), dim=-1)

    # Input is [..., 2, 2] with [[x0, y0], [x1, y1]]
    # Output is [..., 2, 2] with [[xc, yc], [w, h]]
    xy1 = box_xy[..., 0, :]
    xy2 = box_xy[..., 1, :]
    center = (xy1 + xy2) * 0.5
    wh = xy2 - xy1
    return torch.stack((center, wh), dim=-2)

def apply_directional_box_offset(
    offset: float | torch.Tensor,
    vec_start: torch.Tensor,
    vec_end: torch.Tensor,
    xc: torch.Tensor,
    yc: torch.Tensor,
) -> None:
    """
    Offset the bounding box defined by [xc, yc] by a pre-determined length.
    The offset will be applied in the direction of the supplied vector.

    Parameters
    ----------
    offset
        Floating point offset to apply to the bounding box, in absolute values.
    vec_start
        Starting point of the vector. Shape is [B, 2], where 2 == (x, y).
    vec_end
        Ending point of the vector. Shape is [B, 2], where 2 == (x, y).
    xc
        X center of box. Modified in place.
    yc
        Y center of box. Modified in place.

    Returns
    -------
    None
        No return value; xc and yc are modified in place.
    """
    xlen = vec_end[..., 0] - vec_start[..., 0]
    ylen = vec_end[..., 1] - vec_start[..., 1]
    vec_len = torch.sqrt(torch.float_power(xlen, 2) + torch.float_power(ylen, 2))

    xc += offset * (xlen / vec_len)
    yc += offset * (ylen / vec_len)

def numpy_image_to_torch(image: np.ndarray, to_float: bool = True) -> torch.Tensor:
    """Convert a Numpy image (dtype uint8, shape [H W C] or [N H W C]) into a pyTorch tensor with range [0, 1] and shape NCHW."""
    image_torch = torch.from_numpy(image)

    if len(image.shape) == 3:
        image_torch = image_torch.unsqueeze(0)
    image_torch = image_torch.permute(0, 3, 1, 2)
    if to_float:
        return image_torch.float() / 255.0
    return image_torch

def resize_pad(
    image: torch.Tensor,
    dst_size: tuple[int, ...],
    pad_mode: str = "constant",
    pad_value: float = 0.0,
    vertical_float: Literal["center", "top", "bottom"] = "center",
    horizontal_float: Literal["center", "left", "right"] = "center",
) -> tuple[torch.Tensor, float, tuple[int, int]]:
    """
    Resize and pad image to be shape [..., dst_size[0], dst_size[1]].

    This will not warp or crop the image. It will be resized as large as it can
    possibly be without being cropped and while maintaining aspect ratio.
    The image is then padded so that it's in the center of the returned image "frame"
    of the desired size (dst_size).

    Parameters
    ----------
    image
        Image to reshape with shape (..., H, W).
    dst_size
        Size to which the image should be reshaped (height, width).
    pad_mode
        Padding mode.
    pad_value
        Padding value.
    vertical_float
        Where the image should float vertically in the resulting canvas.
        For example, if "float" is bottom, the image will be written
        in the bottom of the frame, and the remainder of the frame
        (above the image) is padded.
    horizontal_float
        Where the image should float horizontally in the resulting canvas.
        For example, if "float" is left, the image will be written
        in the left of the frame, and the remainder of the frame
        (to the right of the image) is padded.

    Returns
    -------
    rescaled_padded_image : torch.Tensor
        Tensor with shape (..., dst_size[0], dst_size[1]).
    scale : float
        Scale factor between original image and dst_size image.
    pad : tuple[int, int]
        Pixels of padding added to the rescaled image (left_padding, top_padding).

    Notes
    -----
    Based on https://github.com/zmurez/MediaPipePyTorch/blob/master/blazebase.py
    """
    height, width = image.shape[-2:]
    dst_frame_height, dst_frame_width = dst_size

    h_ratio = dst_frame_height / height
    w_ratio = dst_frame_width / width
    scale = min(h_ratio, w_ratio)
    if h_ratio < w_ratio:
        scale = h_ratio
        new_height = dst_frame_height
        new_width = math.floor(width * scale)
    else:
        scale = w_ratio
        new_height = math.floor(height * scale)
        new_width = dst_frame_width

    new_height = math.floor(height * scale)
    new_width = math.floor(width * scale)

    def _split_padding(
        float_img_in_frame: Literal["center", "left", "right", "top", "bottom"],
        pad_size: int,
    ) -> tuple[int, int]:
        """
        Split padding into (Left, Rright), or (Top, Bottom)
        splits based on where the image should float in the frame.
        """
        if float_img_in_frame == "center":
            return (int(pad_size // 2), int(pad_size // 2 + pad_size % 2))
        if float_img_in_frame in ["right", "bottom"]:
            return (pad_size, 0)
        if float_img_in_frame in ["left", "top"]:
            return (0, pad_size)
        raise ValueError(f"Invalid pad type: {float_img_in_frame}")

    pad_top, pad_bottom = _split_padding(vertical_float, dst_frame_height - new_height)
    pad_left, pad_right = _split_padding(horizontal_float, dst_frame_width - new_width)

    rescaled_image = interpolate(
        image, size=[int(new_height), int(new_width)], mode="bilinear"
    )
    rescaled_padded_image = pad(
        rescaled_image,
        (pad_left, pad_right, pad_top, pad_bottom),
        mode=pad_mode,
        value=pad_value,
    )
    padding = (pad_left, pad_top)

    return rescaled_padded_image, scale, padding

def denormalize_coordinates(
    coordinates: torch.Tensor,
    input_img_size: tuple[int, int],
    scale: float = 1.0,
    pad: tuple[int, int] = (0, 0),
) -> None:
    """
    Maps detection coordinates from [0,1] to coordinates in the original image.

    This function can be exported and run inside inference frameworks if desired.

    Note: If included in the model, this code is likely to be unfriendly to quantization.
          This is because of the high range and variability of the output tensor.
          For best quantization accuracy, this code should be run separately from the model,
          or the model should de-quantize activations before running these layers.

    Parameters
    ----------
    coordinates
        Coordinate tensor with shape [..., 2]. Range must be [0, 1].
    input_img_size
        The size of the tensor that was fed to the NETWORK (NOT the original image size).
        H / W is the same order as coordinates.
    scale
        Scale factor used to resize the image to be fed to the network.
    pad
        Padding used during resizing of input image to network input tensor.
        This is the absolute number of padding pixels in the network input tensor, NOT in the original image.
        H / W is in the same order as coordinates.

    Notes
    -----
    The coordinates input is modified in place.
    The output coordinates are in the format [..., 2] where 2 is (y0, x0).
    """
    img_0, img_1 = input_img_size
    pad_0, pad_1 = pad

    coordinates[..., 0] = ((coordinates[..., 0] * img_0 - pad_0) / scale).int()
    coordinates[..., 1] = ((coordinates[..., 1] * img_1 - pad_1) / scale).int()

def apply_batched_affines_to_frame(
    frame: np.ndarray, affines: list[np.ndarray], output_image_size: tuple[int, int]
) -> np.ndarray:
    """
    Generate one image per affine applied to the given frame.

    I/O is numpy since this uses cv2 APIs under the hood.

    Parameters
    ----------
    frame
        Frame on which to apply the affine. Shape is [H W C], dtype must be np.byte.
    affines
        List of 2x3 affine matrices to apply to the frame.
    output_image_size
        Size of each output frame.

    Returns
    -------
    transformed_images : np.ndarray
        Computed images with shape [B H W C].
    """
    assert (
        frame.dtype == np.byte or frame.dtype == np.uint8  # noqa: PLR1714 Using a set for comparison is not equivalent to using == on both of these individually.
    )  # cv2 does not work correctly otherwise. Don't remove this assertion.

    imgs = []
    for affine in affines:
        img = cv2.warpAffine(frame, affine, output_image_size)
        imgs.append(img)
    return np.stack(imgs)

def apply_affine_to_coordinates(
    coordinates: ndarrayOrTensor, affine: ndarrayOrTensor
) -> ndarrayOrTensor:
    """
    Apply the given affine matrix to the given coordinates.

    Parameters
    ----------
    coordinates
        Coordinates on which to apply the affine. Shape is [..., 2], where 2 == [X, Y].
    affine
        Affine matrix to apply to the coordinates.

    Returns
    -------
    ndarrayOrTensor
        Transformed coordinates with shape [..., 2], where 2 == [X, Y].
    """
    return (affine[:, :2] @ coordinates.T + affine[:, 2:]).T

def compute_vector_rotation(
    vec_start: torch.Tensor,
    vec_end: torch.Tensor,
    offset_rads: float | torch.Tensor = 0,
) -> torch.Tensor:
    """
    From the given vector, compute the rotation of the vector with added offset.

    Parameters
    ----------
    vec_start
        Starting point of the vector. Shape is [B, 2], where 2 == (x, y).
    vec_end
        Ending point of the vector. Shape is [B, 2], where 2 == (x, y).
    offset_rads
        Offset to subtract from the rotation calculation. Can be size [1] or [Batch].

    Returns
    -------
    rotation_angle : torch.Tensor
        Computed rotation angle in radians. Shape is [Batch].
    """
    return (
        torch.atan2(
            vec_start[..., 1] - vec_end[..., 1], vec_start[..., 0] - vec_end[..., 0]
        )
        - offset_rads
    )

def decode_preds_from_anchors(
    boxes_and_coordinates: torch.Tensor,
    img_size: tuple[int, int],
    anchors: torch.Tensor,
) -> torch.Tensor:
    """
    Decode predictions using the provided anchors.

    This function can be exported and run inside inference frameworks if desired.

    Parameters
    ----------
    boxes_and_coordinates
        Coordiantes in pixel space. Shape (B, N, K, 2), where N is number of detections, and K is the number of coordinates.
        In the second to last dimension:
        - indices [0-1] are the box coordinates ((x_min, y_min), (width, height))
        - indices [2-K-1] are the K keypoint coordinates ((x1,y1), (x2,y2), (x3,y3), (x4,y4), (x5,y5), (x6,y6), (x7,y7)).
    img_size
        The size of the tensor that was fed to the NETWORK (NOT the original image size). Layout is [H, W]
        H / W is the same order as coordinates.
    anchors
        Box anchors. Range must be [0, 1]. Shape is [Batch, Num Anchors, 2, 2],
        where [2, 2] == [[x_offset, y_offset], [x_scale, y_scale]]

    Returns
    -------
    decoded_coords: torch.Tensor
        The "boxes_and_coordinates" input decoded using the provided anchors.
    """
    assert boxes_and_coordinates.shape[-1] == anchors.shape[-1] == 2
    assert boxes_and_coordinates.shape[-3] == anchors.shape[-3]

    h_size, w_size = img_size

    # Convert offset from normalized [0,1] to image space
    offset = anchors[..., 0:1, :] * torch.tensor(
        [[w_size, h_size]], dtype=anchors.dtype
    )
    scale = anchors[..., 1:2, :]

    # Create mask to zero out offset at index 1 (wh doesn't get offset added)
    K = boxes_and_coordinates.shape[-2]
    mask = (torch.arange(K) != 1).view(K, 1)

    # Scale all coordinates, then add masked offset
    return boxes_and_coordinates * scale + (offset * mask)

def mediapipe_detector_postprocess(
    coords: torch.Tensor,
    scores: torch.Tensor,
    score_clipping_threshold: float,
    img_size: tuple[int, int],
    anchors: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Mediapipe detector postprocessing.

    Parameters
    ----------
    coords
        Detected boxes and coordinates in pixel space. Can be either:
        * shape (B, N, K, 2). Where N is number of detections, and K is the number of coordinates.
            In the second to last dimension:
            - indices [0:1] are the box coordinates ((x_min, y_min), (width, height))
            - indices [2:K-1] are the K keypoint coordinates ((x1,y1), (x2,y2), (x3,y3), (x4,y4), (x5,y5), (x6,y6), (x7,y7)).
        * shape (B, N, M). Where N is number of detections, and M is total number of coordinates K * 2.
            In the last dimension:
            - indices [0:3] are the box coordinates ((x_min, y_min), (width, height))
            - indices [4:M] are the K keypoint coordinates ((x1,y1), (x2,y2), (x3,y3), ...
    scores
        Raw model output scores of shape (B, N, 1).
    score_clipping_threshold
        Score clipping threshold for postprocessing.
    img_size
        Shape of the network image input (H, W).
    anchors
        Anchors used for decoding bounding box predictions. Shape [N, 2, 2],
        where [2, 2] == [[x_scale, y_scale], [w_offset, h_offset]]

    Returns
    -------
    coords: torch.Tensor
        Detected boxes and coordinates in pixel space.
        Shape (B, N, M), where N is number of detections, and M is total number of coordinates K * 2.
        In the last dimension:
        - indices [0-3] are the box coordinates ((x_min, y_min), (x_max, y_max))
        - indices [4-M] are the K keypoint coordinates ((x1,y1), (x2,y2), (x3,y3), ...).e.

    scores: torch.Tensor
        Clipped and sigmoid activated scores of shape (B, N).
    """
    scores = scores.clamp(
        -score_clipping_threshold,
        score_clipping_threshold,
    )
    scores = scores.sigmoid()

    # Ensure scores are of shape [B, N]
    if scores.dim() == 3 and scores.size(-1) == 1:
        scores = scores.squeeze(dim=-1)

    # Reshape outputs so that they have shape [..., # of coordinates, 2], where 2 == (x, y)
    if coords.dim() == 3:
        coords = coords.view([*list(coords.shape)[:-1], -1, 2])

    # Decode to output coordinates using the model's trained anchors.
    coords = decode_preds_from_anchors(coords, img_size, anchors)

    # flatten coords (remove final [2] dim)
    coords = (
        coords.view([*list(coords.shape)[:-2], -1]) if len(coords.shape) > 3 else coords
    )

    # Convert box coordinates from CWH -> XYXY format.
    xyxy_boxes = box_xywh_to_xyxy(coords[..., :4])
    coords = torch.cat([xyxy_boxes, coords[..., 4:]], dim=-1)

    return coords, scores

class NativeROI:
    def _compute_object_roi(
            self,
            batched_selected_boxes: list[torch.Tensor],
            batched_selected_keypoints: list[torch.Tensor],
        ) -> list[torch.Tensor]:
            """
            From the provided bounding boxes and keypoints, compute the region of interest (ROI) that should be used
            as input to the landmark detection model.
    
            Parameters
            ----------
            batched_selected_boxes
                Selected object bounding box coordinates. Empty tensor if batch had no bounding boxes with a score above the threshold.
                Shape of each list element is [num_selected_boxes, 2, 2].
                    Layout is
                        [[box_x1, box_y1],
                            [box_x2, box_y2]]
    
            batched_selected_keypoints
                Selected object bounding box keypoints. Empty tensor if batch had no bounding boxes with a score above the threshold.
                Shape of each list element is [num_selected_boxes, # of keypoints, 2].
                    Layout is
                        [[keypoint_0_x, keypoint_0_y],
                            ...,
                            [keypoint_max_x, keypoint_max_y]]
    
            Returns
            -------
            batched_roi_4corners : list[torch.Tensor]
                Selected object "region of interest" (region used as input to the landmark detector) corner coordinates.
                Empty tensor if batch had no bounding boxes with a score above the threshold.
                Shape of each list element is [num_selected_boxes, 4, 2], where 2 == (x, y).
                The order of points is (top left point, bottom left point, top right point, bottom right point).
            """
            batched_selected_roi: list[torch.Tensor] = []
            for boxes, keypoints in zip(
                batched_selected_boxes, batched_selected_keypoints, strict=False
            ):
                if boxes.nelement() == 0 or keypoints.nelement() == 0:
                    batched_selected_roi.append(torch.Tensor())
                    continue
    
                # Compute bounding box center and rotation
                theta = compute_vector_rotation(
                    keypoints[:, self.keypoint_rotation_vec_start_idx, ...],
                    keypoints[:, self.keypoint_rotation_vec_end_idx, ...],
                    self.rotation_offset_rads,
                )
                selected_boxes_cwh = box_xyxy_to_xywh(boxes)
                xc = selected_boxes_cwh[..., 0, 0]
                yc = selected_boxes_cwh[..., 0, 1]
                w = selected_boxes_cwh[..., 1, 0]
                h = selected_boxes_cwh[..., 1, 1]
    
                # The bounding box often misses the entire object.
                # Move the bounding box slightly (if necessary) to center it with the object.
                apply_directional_box_offset(
                    self.detect_box_offset_xy * w,
                    keypoints[..., self.keypoint_rotation_vec_start_idx, :],
                    keypoints[..., self.keypoint_rotation_vec_end_idx, :],
                    xc,
                    yc,
                )
    
                # Apply scaling to enlargen the bounding box
                w *= self.detect_box_scale
                h *= self.detect_box_scale
    
                # Compute box corners from box center, width, height
                batched_selected_roi.append(
                    compute_box_corners_with_rotation(xc, yc, w, h, theta)
                )
    
            return batched_selected_roi
