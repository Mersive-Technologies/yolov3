from utils.utils import compute_loss
from fastai.vision import *
from fastai.vision import *

from utils.utils import compute_loss


# Define a custom loss function that translate between FastAI and Ultralytics
def loss_func(model, predicted, boxes, classes):
    if not model.training:
        predicted = predicted[1]
    targets = []
    bs = classes.shape[0]
    max_detections = classes.shape[1]
    for img_idx in range(bs):
        for detect_idx in range(max_detections):
            clazz = classes[img_idx, detect_idx]
            if clazz == 0: continue
            l, t, r, b = boxes[img_idx, detect_idx] * 0.5 + 0.5
            w = r - l
            h = b - t
            targets.append([img_idx, float(clazz-1), float(l), float(t), float(w), float(h)])
    ft = torch.cuda.FloatTensor if predicted[0].is_cuda else torch.Tensor
    targets = ft(targets)
    loss, _ = compute_loss(predicted, targets, model)
    return loss[0]


# https://github.com/cedrickchee/knowledge/blob/master/courses/fast.ai/deep-learning-part-2/2018-edition/lesson-8-object-detection.md#convert-vocs-bounding-box
def hw_bb(bb):
    return np.array([ bb[1], bb[0], bb[3] + bb[1] - 1, bb[2] + bb[0] - 1 ])


def filename_to_key(filename):
    trunk = os.path.splitext(filename)[0]
    txt = trunk.replace("_", "")
    id = int(txt)
    return id


# Create function for FastAI to get labels
def get_y_func(images, category, path):
    key = filename_to_key(path.name)
    image = images[key]
    boxes = [hw_bb(a['bbox']) for a in image['annotations'] if a['category_id'] == category]
    classes = ['person'] * len(boxes)
    return [boxes, classes]
