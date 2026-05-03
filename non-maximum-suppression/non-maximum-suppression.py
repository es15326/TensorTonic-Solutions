import torch

def compute_iou(box1, boxes):
    x_left = torch.max(box1[0], boxes[:, 0])
    y_top = torch.max(box1[1], boxes[:, 1])
    x_right = torch.min(box1[2], boxes[:, 2])
    y_bottom = torch.min(box1[3], boxes[:, 3])

    intersection = torch.clamp(x_right - x_left, 0) * torch.clamp(y_bottom - y_top, 0)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    other_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + other_areas - intersection

    return intersection / union
    
def nms(boxes, scores, iou_threshold):
    """
    Apply Non-Maximum Suppression.
    """
    # Write code here
    if not boxes:
        return []
    boxes, scores = torch.tensor(boxes, dtype=torch.float32), torch.tensor(scores, dtype=torch.float32)

    indices = torch.argsort(scores, descending=True)

    keep = []
    while len(indices) > 0:
        curr_index = indices[0]
        keep.append(curr_index.item())
        if len(indices) == 1:
            return keep
        other_indices = indices[1:]

        ious = compute_iou(boxes[curr_index], boxes[other_indices])

        filtered = ious < iou_threshold
        indices = other_indices[filtered]

    return keep
    
        