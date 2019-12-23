from models import Darknet
from utils import torch_utils

device = 'cpu'
arc = 'default'
cfg = 'cfg/yolov3-tiny-anchors.cfg'
device = torch_utils.select_device(device, apex=False, batch_size=64)
model = Darknet(cfg, arc=arc).to(device)
