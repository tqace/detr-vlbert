import torch
from typing import Optional, List
import torchvision
from torch import Tensor
from ForkedPdb import ForkedPdb
class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

def clip_pad_images(tensor, pad_shape, pad=0):
    """
    Clip clip_pad_images of the pad area.
    :param tensor: [c, H, W]
    :param pad_shape: [h, w]
    :return: [c, h, w]
    """
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    H, W = tensor.shape[1:]
    h = pad_shape[1]
    w = pad_shape[2]

    tensor_ret = torch.zeros((tensor.shape[0], h, w), dtype=tensor.dtype) + pad
    tensor_ret[:, :min(h, H), :min(w, W)] = tensor[:, :min(h, H), :min(w, W)]

    return tensor_ret


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)

def clip_pad_boxes(tensor, pad_length, pad=0):
    """
        Clip boxes of the pad area.
        :param tensor: [k, d]
        :param pad_shape: K
        :return: [K, d]
    """
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    k = tensor.shape[0]
    d = tensor.shape[1]
    K = pad_length
    tensor_ret = torch.zeros((K, d), dtype=tensor.dtype) + pad
    tensor_ret[:min(k, K), :] = tensor[:min(k, K), :]

    return tensor_ret


def clip_pad_1d(tensor, pad_length, pad=0):
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    tensor_ret = torch.zeros((pad_length, ), dtype=tensor.dtype) + pad
    tensor_ret[:min(tensor.shape[0], pad_length)] = tensor[:min(tensor.shape[0], pad_length)]

    return tensor_ret


def clip_pad_2d(tensor, pad_shape, pad=0):
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    tensor_ret = torch.zeros(*pad_shape, dtype=tensor.dtype) + pad
    tensor_ret[:min(tensor.shape[0], pad_shape[0]), :min(tensor.shape[1], pad_shape[1])] \
        = tensor[:min(tensor.shape[0], pad_shape[0]), :min(tensor.shape[1], pad_shape[1])]

    return tensor_ret

def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes
