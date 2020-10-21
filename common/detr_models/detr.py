# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import requests
from torch import nn
from PIL import Image
from common.utils import box_ops
from common.utils.bbox import coordinate_embeddings
from common.utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer

from ForkedPdb import ForkedPdb
import matplotlib
import random
matplotlib.use('Agg')

class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self,args, aux_loss=False):
        """ Initializes the model.
        Parameters:
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        num_classes = 80 if args.dataset_file != 'coco' else 91
        if args.dataset_file == "coco_panoptic":
            num_classes = 250
        self.backbone = build_backbone(args)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.matcher = build_matcher(args)
        self.transformer = build_transformer(args)
        for p in self.transformer.parameters():
            p.requires_grad = False
        self.num_queries = args.num_queries
        hidden_dim = self.transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        for p in self.class_embed.parameters():
            p.requires_grad = False
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        for p in self.bbox_embed.parameters():
            p.requires_grad = False
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)
        for p in self.query_embed.parameters():
            p.requires_grad = False
        self.input_proj = nn.Conv2d(self.backbone.num_channels, hidden_dim, kernel_size=1)
        for p in self.input_proj.parameters():
            p.requires_grad = False
        self.aux_loss = aux_loss
        self.num_classes = num_classes
        self.args = args
        self.obj_upsample = torch.nn.Sequential(
                torch.nn.Linear(256,768),
                torch.nn.ReLU(inplace=True),
                )
        self.obj_upsample_ve = torch.nn.Sequential(
                torch.nn.Linear(256,768),
                torch.nn.ReLU(inplace=True),
                )

    def forward(self, samples: NestedTensor,boxes,box_mask,im_info,mvrc_ops,boxes_cls_scores,mask_visual_embed=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        box_inds = box_mask.nonzero()
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        samples=samples.to("cuda")
        features, pos = self.backbone(samples)
    
        src, mask = features[-1].decompose()
        assert mask is not None
        hs, mem = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        #model_ = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True).cuda()
        #model_.eval()
        #transform = T.Compose([
    	#	T.Resize(800),
        #    	T.ToTensor(),
	#	T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	#    ])
        #url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
        #im = Image.open(requests.get(url, stream=True).raw)
        #img = transform(im).unsqueeze(0)
        

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        #return out
        weight_dict = {'loss_ce': 1, 'loss_bbox': self.args.bbox_loss_coef}
        weight_dict['loss_giou'] = self.args.giou_loss_coef
        if self.args.masks:
            weight_dict["loss_mask"] = self.args.mask_loss_coef
            weight_dict["loss_dice"] = self.args.dice_loss_coef
        # TODO this is a hack
        if self.args.aux_loss:
            aux_weight_dict = {}
            for i in range(self.args.dec_layers - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses2cal = ['labels', 'boxes', 'cardinality']
        if self.args.masks:
            losses += ["masks"]
        criterion = SetCriterion(self.num_classes,self.matcher,weight_dict,self.args.eos_coef,losses2cal)
        targets = self.get_targets(boxes,boxes_cls_scores,im_info)
        losses,indices = criterion(out,targets)
        max_boxes_len = max([len(iindices[0]) for iindices in indices])
        rep_all = hs[-1]
        obj_reps = {}
        bs,_,_ = rep_all.shape
        '''
        obj_rep_ = torch.zeros((bs,max_boxes_len,256)).cuda()
        #coord_embeds = coordinate_embeddings(torch.cat((boxes[box_inds[:, 0], box_inds[:, 1]], im_info[box_inds[:, 0], :2]), 1),256)
        for i in range(bs):    
        #    obj_rep_[i][:len(indices[i][0])] = torch.cat((hs[-1][i][(indices[i][0])].clone(),coord_embeds[:len(indices[i][0])].view(coord_embeds[:len(indices[i][0])].shape[0],-1)),-1)
        #    coord_embeds = coord_embeds[len(indices[i][0]):]
            #batch_matched_boxes = out['pred_boxes'][i][(indices[i][0])]
            #encoder_boxes = self.get_encoder_boxes(batch_matched_boxes,im_info[i],mem,samples)
            #for j in range(len(encoder_boxes)):
            #    roi = mem[i,:,max(0,int(encoder_boxes[j][1])-1):min(mem.shape[-2],int(encoder_boxes[j][3])+2),max(0,int(encoder_boxes[j][0])-1):min(mem.shape[-1],int(encoder_boxes[j][2])+2)]
            #    m = nn.AdaptiveMaxPool2d((1,1))
            #    roi_vec = m(roi)
            #    obj_rep_[i][j] = roi_vec.view(1,-1)
            obj_rep_[i][:len(indices[i][0])] = hs[-1][i][(indices[i][0])].clone()
        if mask_visual_embed is not None:
            for i in range(obj_rep_.shape[0]):
                obj_rep_[i][((mvrc_ops == 1)[i])] =  mask_visual_embed
        final_feats = self.obj_upsample(obj_rep_)
        obj_reps_padded = final_feats.new_zeros((final_feats.shape[0], boxes.shape[1], final_feats.shape[2]))
        obj_reps_padded[:, :final_feats.shape[1]] = final_feats
        obj_reps_ = obj_reps_padded
        '''
        mem_ = mem.view(bs,256,-1)
        mem_ = torch.transpose(mem_,1,2)
        mem_ = self.obj_upsample(mem_.view(-1,256))
        obj_reps["obj_reps"] = mem_.view(bs,-1,768)
        
        gap = nn.AdaptiveAvgPool2d((1,1))

        visual_embedding = self.obj_upsample_ve(gap(mem).view(bs,-1))

        return losses,obj_reps, visual_embedding

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
    
    def get_targets(self,boxes,boxes_cls_scores,im_info):
        targets= []
        for i in range(len(boxes)):
            w = im_info[i][0]
            h = im_info[i][1]
            is_pad = boxes[i]==-2
            iboxes_keep = boxes[i][is_pad.sum(dim=1) == 0]
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device="cuda")
            iboxes_keep = iboxes_keep / image_size_xyxy
            iboxes_keep = box_ops.box_xyxy_to_cxcywh(iboxes_keep)
            ilabels_keep = torch.argmax(boxes_cls_scores[i],dim=1)[is_pad.sum(dim=1) == 0]
            targets.append({'boxes':iboxes_keep.cuda(),'labels':ilabels_keep.cuda()})

        return targets

    def get_encoder_boxes(self,pred_boxes,im_info,mem,samples):
        w = im_info[0]
        h = im_info[1]
        boxes = box_ops.box_cxcywh_to_xyxy(pred_boxes)*torch.as_tensor([w, h, w, h], dtype=torch.float, device="cuda")
        ratio = mem.shape[-1]/samples.decompose()[0].shape[-1]
        boxes*=ratio
        return boxes


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1).cuda()
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        #permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses,indices


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    num_classes = 20 if args.dataset_file != 'coco' else 80
    if args.dataset_file == "coco_panoptic":
        num_classes = 250
    device = torch.device(args.device)


    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )
    

    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
