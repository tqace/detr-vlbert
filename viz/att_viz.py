import math

from PIL import Image
import requests
import matplotlib.pyplot as plt
%config InlineBackend.figure_format = 'retina'

import ipywidgets as widgets
from IPython.display import display, clear_output

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
torch.set_grad_enabled(False);







transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def viz_detr(img_path,model_path):
    im = Image.open(img_path)
    img = transform(im).unsqueeze(0)
    model = torch.load(model_path)
    
    conv_features, enc_attn_weights, dec_attn_weights = [], [], []

    hooks = [
	model.backbone[-2].register_forward_hook(
	    lambda self, input, output: conv_features.append(output)
	),
	model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
	    lambda self, input, output: enc_attn_weights.append(output[1])
	),
	model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
	    lambda self, input, output: dec_attn_weights.append(output[1])
	),
    ]

    # propagate through the model
    outputs = model(img)

    for hook in hooks:
	hook.remove()

    # don't need the list anymore
    conv_features = conv_features[0]
    enc_attn_weights = enc_attn_weights[0]
    dec_attn_weights = dec_attn_weights[0]
