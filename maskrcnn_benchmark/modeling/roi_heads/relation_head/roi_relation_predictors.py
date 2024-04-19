# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import numpy as np
import random
import torch
from maskrcnn_benchmark.modeling import registry
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from maskrcnn_benchmark.layers import smooth_l1_loss, kl_div_loss, entropy_loss, Label_Smoothing_Regression
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.roi_heads.relation_head.classifier import build_classifier

from maskrcnn_benchmark.modeling.utils import cat
from .model_msg_passing import IMPContext
from .model_vtranse import VTransEFeature

from .model_motifs import LSTMContext, FrequencyBias
from .model_motifs import LSTMContext_split
from .model_vctree import VCTreeLSTMContext
from .model_vctree import VCTreeLSTMContext_split
from .relation_sampling import Relation_Sampling

from .model_motifs_with_attribute import AttributeLSTMContext
from .model_transformer import TransformerContext
from .model_transformer import TransformerContext_split

from .utils_relation import layer_init, get_box_info, get_box_pair_info
from maskrcnn_benchmark.data import get_dataset_statistics

from .rel_proposal_network.loss import (
    FocalLossFGBGNormalization,
    RelAwareLoss,
)


@registry.ROI_RELATION_PREDICTOR.register("IMPPredictor")
class IMPPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(IMPPredictor, self).__init__()
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.use_bias = False

        assert in_channels is not None

        self.context_layer = IMPContext(config, self.num_obj_cls, self.num_rel_cls, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        
        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        # freq 
        if self.use_bias:
            statistics = get_dataset_statistics(config)
            self.freq_bias = FrequencyBias(config, statistics)


    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        if self.union_single_not_match:
            union_features = self.up_dim(union_features)

        # encode context infomation
        obj_dists, rel_dists = self.context_layer(roi_features, proposals, union_features, rel_pair_idxs, logger)

        num_objs = [len(b) for b in proposals]
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        assert len(num_rels) == len(num_objs)

        if self.use_bias:
            obj_preds = obj_dists.max(-1)[1]
            obj_preds = obj_preds.split(num_objs, dim=0)

            pair_preds = []
            for pair_idx, obj_pred in zip(rel_pair_idxs, obj_preds):
                pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
            pair_pred = cat(pair_preds, dim=0)

            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        return obj_dists, rel_dists, add_losses
    
    
@registry.ROI_RELATION_PREDICTOR.register("MotifPredictor")
class MotifPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(MotifPredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics[
            'att_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_att_cls == len(att_classes)
        assert self.num_rel_cls == len(rel_classes)
        # init contextual lstm encoding
        if self.attribute_on:
            self.context_layer = AttributeLSTMContext(config, obj_classes, att_classes, rel_classes, in_channels)
        else:
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=True)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)
        layer_init(self.rel_compress, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        # encode context infomation
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        else:
            obj_dists, obj_preds, edge_ctx, _ = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        if self.use_vision:
            if self.union_single_not_match:
                prod_rep = prod_rep * self.up_dim(union_features)
            else:
                prod_rep = prod_rep * union_features

        rel_dists = self.rel_compress(prod_rep)

        if self.use_bias:
            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        if self.attribute_on:
            att_dists = att_dists.split(num_objs, dim=0)
            return (obj_dists, att_dists), rel_dists, add_losses
        else:
            return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("VCTreePredictor")
class VCTreePredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(VCTreePredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics[
            'att_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_att_cls == len(att_classes)
        assert self.num_rel_cls == len(rel_classes)
        # init contextual lstm encoding
        self.context_layer = VCTreeLSTMContext(config, obj_classes, rel_classes, statistics, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        # learned-mixin
        # self.uni_gate = nn.Linear(self.pooling_dim, self.num_rel_cls)
        # self.frq_gate = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        # self.uni_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        # layer_init(self.uni_gate, xavier=True)
        # layer_init(self.frq_gate, xavier=True)
        layer_init(self.ctx_compress, xavier=True)
        # layer_init(self.uni_compress, xavier=True)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        self.freq_bias = FrequencyBias(config, statistics)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        # encode context infomation
        obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs,
                                                                          logger)

        # post decode
        edge_rep = F.relu(self.post_emb(edge_ctx))
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        # learned-mixin Gate
        # uni_gate = torch.tanh(self.uni_gate(self.drop(prod_rep)))
        # frq_gate = torch.tanh(self.frq_gate(self.drop(prod_rep)))

        if self.union_single_not_match:
            union_features = self.up_dim(union_features)

        ctx_dists = self.ctx_compress(prod_rep * union_features)
        # uni_dists = self.uni_compress(self.drop(union_features))
        frq_dists = self.freq_bias.index_with_labels(pair_pred.long())

        rel_dists = ctx_dists + frq_dists
        # rel_dists = ctx_dists + uni_gate * uni_dists + frq_gate * frq_dists

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        if self.training:
            binary_loss = []
            for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                bi_gt = (bi_gt > 0).float()
                binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
            add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)

        return obj_dists, rel_dists, add_losses

@registry.ROI_RELATION_PREDICTOR.register("CausalAnalysisPredictor")
class CausalAnalysisPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(CausalAnalysisPredictor, self).__init__()
        self.cfg = config
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.spatial_for_vision = config.MODEL.ROI_RELATION_HEAD.CAUSAL.SPATIAL_FOR_VISION
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.fusion_type = config.MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE
        self.separate_spatial = config.MODEL.ROI_RELATION_HEAD.CAUSAL.SEPARATE_SPATIAL
        self.use_vtranse = config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "vtranse"
        self.effect_type = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE

        assert in_channels is not None
        num_inputs = in_channels

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)
        # init contextual lstm encoding
        if config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "motifs":
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)
        elif config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "vctree":
            self.context_layer = VCTreeLSTMContext(config, obj_classes, rel_classes, statistics, in_channels)
        elif config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "vtranse":
            self.context_layer = VTransEFeature(config, obj_classes, rel_classes, in_channels)
        else:
            print('ERROR: Invalid Context Layer')

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        if self.use_vtranse:
            self.edge_dim = self.pooling_dim
            self.post_emb = nn.Linear(self.hidden_dim, self.pooling_dim * 2)
            self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=False)
        else:
            self.edge_dim = self.hidden_dim
            self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
            self.post_cat = nn.Sequential(*[nn.Linear(self.hidden_dim * 2, self.pooling_dim),
                                            nn.ReLU(inplace=True), ])
            self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.vis_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)

        if self.fusion_type == 'gate':
            self.ctx_gate_fc = nn.Linear(self.pooling_dim, self.num_rel_cls)
            layer_init(self.ctx_gate_fc, xavier=True)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        if not self.use_vtranse:
            layer_init(self.post_cat[0], xavier=True)
            layer_init(self.ctx_compress, xavier=True)
        layer_init(self.vis_compress, xavier=True)

        assert self.pooling_dim == config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM

        # convey statistics into FrequencyBias to avoid loading again
        self.freq_bias = FrequencyBias(config, statistics)

        # add spatial emb for visual feature
        if self.spatial_for_vision:
            self.spt_emb = nn.Sequential(*[nn.Linear(32, self.hidden_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(self.hidden_dim, self.pooling_dim),
                                           nn.ReLU(inplace=True)
                                           ])
            layer_init(self.spt_emb[0], xavier=True)
            layer_init(self.spt_emb[2], xavier=True)

        self.label_smooth_loss = Label_Smoothing_Regression(e=1.0)

        # untreated average features
        self.effect_analysis = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS
        self.average_ratio = 0.0005

        self.register_buffer("untreated_spt", torch.zeros(32))
        self.register_buffer("untreated_conv_spt", torch.zeros(self.pooling_dim))
        self.register_buffer("avg_post_ctx", torch.zeros(self.pooling_dim))
        self.register_buffer("untreated_feat", torch.zeros(self.pooling_dim))

    def pair_feature_generate(self, roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger,
                              ctx_average=False):
        # encode context infomation
        obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs,
                                                                          logger, ctx_average=ctx_average)
        obj_dist_prob = F.softmax(obj_dists, dim=-1)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.edge_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.edge_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.edge_dim)
        # split
        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        obj_prob_list = obj_dist_prob.split(num_objs, dim=0)
        obj_dist_list = obj_dists.split(num_objs, dim=0)
        ctx_reps = []
        pair_preds = []
        pair_obj_probs = []
        pair_bboxs_info = []
        for pair_idx, head_rep, tail_rep, obj_pred, obj_box, obj_prob in zip(rel_pair_idxs, head_reps, tail_reps,
                                                                             obj_preds, obj_boxs, obj_prob_list):
            if self.use_vtranse:
                ctx_reps.append(head_rep[pair_idx[:, 0]] - tail_rep[pair_idx[:, 1]])
            else:
                ctx_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
            pair_obj_probs.append(torch.stack((obj_prob[pair_idx[:, 0]], obj_prob[pair_idx[:, 1]]), dim=2))
            pair_bboxs_info.append(get_box_pair_info(obj_box[pair_idx[:, 0]], obj_box[pair_idx[:, 1]]))
        pair_obj_probs = cat(pair_obj_probs, dim=0)
        pair_bbox = cat(pair_bboxs_info, dim=0)
        pair_pred = cat(pair_preds, dim=0)
        ctx_rep = cat(ctx_reps, dim=0)
        if self.use_vtranse:
            post_ctx_rep = ctx_rep
        else:
            post_ctx_rep = self.post_cat(ctx_rep)

        return post_ctx_rep, pair_pred, pair_bbox, pair_obj_probs, binary_preds, obj_dist_prob, edge_rep, obj_dist_list

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        obj_boxs = [get_box_info(p.bbox, need_norm=True, proposal=p) for p in proposals]

        assert len(num_rels) == len(num_objs)

        post_ctx_rep, pair_pred, pair_bbox, pair_obj_probs, binary_preds, obj_dist_prob, edge_rep, obj_dist_list = self.pair_feature_generate(
            roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger)

        if (not self.training) and self.effect_analysis:
            with torch.no_grad():
                avg_post_ctx_rep, _, _, avg_pair_obj_prob, _, _, _, _ = self.pair_feature_generate(roi_features,
                                                                                                   proposals,
                                                                                                   rel_pair_idxs,
                                                                                                   num_objs, obj_boxs,
                                                                                                   logger,
                                                                                                   ctx_average=True)

        if self.separate_spatial:
            union_features, spatial_conv_feats = union_features
            post_ctx_rep = post_ctx_rep * spatial_conv_feats
        
        if self.spatial_for_vision:
            post_ctx_rep = post_ctx_rep * self.spt_emb(pair_bbox)

        rel_dists = self.calculate_logits(union_features, post_ctx_rep, pair_pred, use_label_dist=False)
        rel_dist_list = rel_dists.split(num_rels, dim=0)

        add_losses = {}
        # additional loss
        if self.training:
            rel_labels = cat(rel_labels, dim=0)

            # binary loss for VCTree
            if binary_preds is not None:
                binary_loss = []
                for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                    bi_gt = (bi_gt > 0).float()
                    binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
                add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)

            # branch constraint: make sure each branch can predict independently
            add_losses['auxiliary_ctx'] = F.cross_entropy(self.ctx_compress(post_ctx_rep), rel_labels)
            if not (self.fusion_type == 'gate'):
                add_losses['auxiliary_vis'] = F.cross_entropy(self.vis_compress(union_features), rel_labels)
                add_losses['auxiliary_frq'] = F.cross_entropy(self.freq_bias.index_with_labels(pair_pred.long()),
                                                              rel_labels)

            # untreated average feature
            if self.spatial_for_vision:
                self.untreated_spt = self.moving_average(self.untreated_spt, pair_bbox)
            if self.separate_spatial:
                self.untreated_conv_spt = self.moving_average(self.untreated_conv_spt, spatial_conv_feats)
            self.avg_post_ctx = self.moving_average(self.avg_post_ctx, post_ctx_rep)
            self.untreated_feat = self.moving_average(self.untreated_feat, union_features)

        elif self.effect_analysis:
            with torch.no_grad():
                # untreated spatial
                if self.spatial_for_vision:
                    avg_spt_rep = self.spt_emb(self.untreated_spt.clone().detach().view(1, -1))
                # untreated context
                avg_ctx_rep = avg_post_ctx_rep * avg_spt_rep if self.spatial_for_vision else avg_post_ctx_rep
                avg_ctx_rep = avg_ctx_rep * self.untreated_conv_spt.clone().detach().view(1,
                                                                                          -1) if self.separate_spatial else avg_ctx_rep
                # untreated visual
                avg_vis_rep = self.untreated_feat.clone().detach().view(1, -1)
                # untreated category dist
                avg_frq_rep = avg_pair_obj_prob

            if self.effect_type == 'TDE':  # TDE of CTX
                rel_dists = self.calculate_logits(union_features, post_ctx_rep, pair_obj_probs) - self.calculate_logits(
                    union_features, avg_ctx_rep, pair_obj_probs)
            elif self.effect_type == 'NIE':  # NIE of FRQ
                rel_dists = self.calculate_logits(union_features, avg_ctx_rep, pair_obj_probs) - self.calculate_logits(
                    union_features, avg_ctx_rep, avg_frq_rep)
            elif self.effect_type == 'TE':  # Total Effect
                rel_dists = self.calculate_logits(union_features, post_ctx_rep, pair_obj_probs) - self.calculate_logits(
                    union_features, avg_ctx_rep, avg_frq_rep)
            else:
                assert self.effect_type == 'none'
                pass
            rel_dist_list = rel_dists.split(num_rels, dim=0)

        return obj_dist_list, rel_dist_list, add_losses

    def moving_average(self, holder, input):
        assert len(input.shape) == 2
        with torch.no_grad():
            holder = holder * (1 - self.average_ratio) + self.average_ratio * input.mean(0).view(-1)
        return holder

    def calculate_logits(self, vis_rep, ctx_rep, frq_rep, use_label_dist=True, mean_ctx=False):
        if use_label_dist:
            frq_dists = self.freq_bias.index_with_probability(frq_rep)
        else:
            frq_dists = self.freq_bias.index_with_labels(frq_rep.long())

        if mean_ctx:
            ctx_rep = ctx_rep.mean(-1).unsqueeze(-1)
        vis_dists = self.vis_compress(vis_rep)
        ctx_dists = self.ctx_compress(ctx_rep)

        if self.fusion_type == 'gate':
            ctx_gate_dists = self.ctx_gate_fc(ctx_rep)
            union_dists = ctx_dists * torch.sigmoid(vis_dists + frq_dists + ctx_gate_dists)
            # union_dists = (ctx_dists.exp() * torch.sigmoid(vis_dists + frq_dists + ctx_constraint) + 1e-9).log()    # improve on zero-shot, but low mean recall and TDE recall
            # union_dists = ctx_dists * torch.sigmoid(vis_dists * frq_dists)                                          # best conventional Recall results
            # union_dists = (ctx_dists.exp() + vis_dists.exp() + frq_dists.exp() + 1e-9).log()                        # good zero-shot Recall
            # union_dists = ctx_dists * torch.max(torch.sigmoid(vis_dists), torch.sigmoid(frq_dists))                 # good zero-shot Recall
            # union_dists = ctx_dists * torch.sigmoid(vis_dists) * torch.sigmoid(frq_dists)                           # balanced recall and mean recall
            # union_dists = ctx_dists * (torch.sigmoid(vis_dists) + torch.sigmoid(frq_dists)) / 2.0                   # good zero-shot Recall
            # union_dists = ctx_dists * torch.sigmoid((vis_dists.exp() + frq_dists.exp() + 1e-9).log())               # good zero-shot Recall, bad for all of the rest

        elif self.fusion_type == 'sum':
            union_dists = vis_dists + ctx_dists + frq_dists
        else:
            print('invalid fusion type')

        return union_dists

    def binary_ce_loss(self, logits, gt):
        batch_size, num_cat = logits.shape
        answer = torch.zeros((batch_size, num_cat), device=gt.device).float()
        answer[torch.arange(batch_size, device=gt.device), gt.long()] = 1.0
        return F.binary_cross_entropy_with_logits(logits, answer) * num_cat

    def fusion(self, x, y):
        return F.relu(x + y) - (x - y) ** 2


@registry.ROI_RELATION_PREDICTOR.register("TransformerPredictor")
class TransformerPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(TransformerPredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        # load parameters
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.num_experts = config.MODEL.ROI_RELATION_HEAD.NUM_EXPERTS

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics['att_classes']
        assert self.num_obj_cls==len(obj_classes)
        assert self.num_att_cls==len(att_classes)
        assert self.num_rel_cls==len(rel_classes)
        # module construct
        if self.num_experts > 1:
            self.context_layer = TransformerContext_split(config, obj_classes, rel_classes, in_channels)
        else:
            self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.hidden_dim * 2, self.num_rel_cls)

        # initialize layer parameters 
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.rel_compress, xavier=True)
        layer_init(self.ctx_compress, xavier=True)
        layer_init(self.post_cat, xavier=True)
        
        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        else:
            if self.num_experts > 1:
                obj_dists, obj_preds, edge_ctx_info = self.context_layer(roi_features, proposals, logger)
                outs = edge_ctx_info['logits']
                edge_ctx = edge_ctx_info['output']

            else:
                obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)


        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        rel_dists_lst = []
        obj_preds = obj_preds.split(num_objs, dim=0)
        rel_dists = self.post_decode(edge_ctx, rel_pair_idxs, num_objs, num_rels, obj_preds, union_features)
        obj_dists = obj_dists.split(num_objs, dim=0)

        if self.num_experts > 1:
            for i in range(self.num_experts):
                rel_dists_lst.append(self.post_decode(outs[:, i, :], rel_pair_idxs, num_objs, num_rels, obj_preds, union_features))

        add_losses = {}

        if self.attribute_on:
            att_dists = att_dists.split(num_objs, dim=0)
            return (obj_dists, att_dists), rel_dists, add_losses
        elif self.num_experts > 1:
            return obj_dists, rel_dists, add_losses, rel_dists_lst
        elif self.num_experts == 1:
            return obj_dists, rel_dists, add_losses


    # post decode
    def post_decode(self, edge_ctx, rel_pair_idxs, num_objs, num_rels, obj_preds, union_features):
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)
        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)

        # from object level feature to pairwise relation level feature
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        ctx_gate = self.post_cat(prod_rep)

        # use union box and mask convolution
        if self.use_vision:
            if self.union_single_not_match:
                visual_rep = ctx_gate * self.up_dim(union_features)
            else:
                visual_rep = ctx_gate * union_features

        rel_dists = self.rel_compress(visual_rep) + self.ctx_compress(prod_rep)

        # use frequence bias
        if self.use_bias:
            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred)

        rel_dists = rel_dists.split(num_rels, dim=0)

        return rel_dists


@registry.ROI_RELATION_PREDICTOR.register("MotifPredictor_PL")
class MotifPredictor_PL(nn.Module):
    def __init__(self, config, in_channels):
        super(MotifPredictor_PL, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        # CAME
        self.num_experts = config.MODEL.ROI_RELATION_HEAD.NUM_EXPERTS
        self.use_relation_aware_gating = config.MODEL.ROI_RELATION_HEAD.USE_RELATION_AWARE_GATING
        self.use_relation_sampling = config.MODEL.ROI_RELATION_HEAD.RELATION_SAMPLING
        self.expert_voting = config.MODEL.ROI_RELATION_HEAD.EXPERT_VOTING

        if self.use_relation_sampling:
            self.relation_sampling = Relation_Sampling()

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION  # True
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS  # True

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics[
            'att_classes']
        # print('obj_classes:', len(obj_classes), obj_classes)        # 151,
        # print('rel_classes:', len(rel_classes), rel_classes)        # 51,
        # print('att_classes:', len(att_classes), att_classes)        # 201, includes: '__background__'

        assert self.num_obj_cls == len(obj_classes)
        assert self.num_att_cls == len(att_classes)
        assert self.num_rel_cls == len(rel_classes)

        # init contextual lstm encoding
        if self.attribute_on:  # False
            self.context_layer = AttributeLSTMContext(config, obj_classes, att_classes, rel_classes, in_channels)

        elif self.num_experts > 1:
            # print('self.context_layer:', self.context_layer)
            #self.context_layer = LSTMContext_split(config, obj_classes, rel_classes, in_channels)
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)

        elif self.num_experts == 1:
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)

        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM             # 512
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM           # 4096

        # post decoding
        if (self.num_experts > 1):
            self.post_emb_lst = nn.ModuleList(
                [nn.Linear(self.hidden_dim, self.hidden_dim * 2) for _ in range(self.num_experts)]       # 512, 1024
            )
            self.post_cat_lst = nn.ModuleList(
                [nn.Linear(self.hidden_dim * 2, self.pooling_dim) for _ in range(self.num_experts)]      # 1024, 4096
            )
            self.rel_compress_lst = nn.ModuleList(
                [nn.Linear(self.pooling_dim, self.num_rel_cls, bias=True) for _ in range(self.num_experts)] # 4096, 51   (rel_classifier)
            )

            for i in range(self.num_experts):
                layer_init(self.post_emb_lst[i], 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
                layer_init(self.post_cat_lst[i], xavier=True)
                layer_init(self.rel_compress_lst[i], xavier=True)


        else:
            self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)                 # 512, 1024
            self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)                # 1024, 4096
            self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=True)    # 4096, 51   (rel_classifier)

            # initialize layer parameters
            layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
            layer_init(self.post_cat, xavier=True)
            layer_init(self.rel_compress, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:      # 4096 != 4096
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)       # 4096, 4096
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

        counts = [6712.0, 171.0, 208.0, 379.0, 504.0, 1829.0, 1413.0, 10011.0, 644.0, 394.0,
                  1603.0, 397.0, 460.0, 565.0, 4.0, 809.0, 163.0, 157.0, 663.0, 67144.0,
                  10764.0, 21748.0, 3167.0, 752.0, 676.0, 364.0, 114.0, 234.0, 15300.0, 31347.0,
                  109355.0, 333.0, 793.0, 151.0, 601.0, 429.0, 71.0, 4260.0, 44.0, 5086.0,
                  2273.0, 299.0, 3757.0, 551.0, 270.0, 1225.0, 352.0, 47326.0, 4810.0, 11059.0]

        sorted_ids = np.argsort(counts)[::-1]
        total_num = sum(counts)
        p_y = [i / total_num for i in counts]
        print('p_y:', p_y)
        for i in sorted_ids:
            if (i <= 15):
                p_y[i] = p_y[i] * 3
            elif (i > 35):
                p_y[i] = p_y[i] / 10
        print('p_y:', p_y)
        p_y = torch.tensor(p_y, dtype=torch.float)
        p_y.log()
        self.rel_adjustments = (0.50 * p_y).to('cuda')


    def edge_ctx_process(self, expert_num, edge_ctx, rel_pair_idxs, obj_preds, union_features, num_objs, selected_info_matrix=None):
        # post decode
        edge_rep = self.post_emb_lst[expert_num](edge_ctx)
        # print('edge_rep:', edge_rep.shape)          # [x, 1024], x = 480

        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        # print('edge_rep:', edge_rep.shape)          # [x, 2, 512], x = 480

        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        # print('head_rep:', head_rep.shape)          # [x, 512]

        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)
        # print('tail_rep:', tail_rep.shape)          # [x, 512]

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        # print('num_rels:', num_rels)    # num_rels: [272, 182, 110, 30, 90, 156]

        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        # print('head_rep:', head_rep.shape)          # [x, 512]

        tail_reps = tail_rep.split(num_objs, dim=0)
        # print('tail_rep:', tail_rep.shape)          # [x, 512]

        obj_preds = obj_preds.split(num_objs, dim=0)
        # print('obj_preds:', obj_preds)

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat_lst[expert_num](prod_rep)

        if self.use_vision:
            if self.union_single_not_match:
                prod_rep = prod_rep * self.up_dim(union_features)
            else:
                prod_rep = prod_rep * union_features

        if selected_info_matrix is not None:
            # print('prod_rep:', prod_rep.shape)
            # print('selected_info_matrix[expert_num]:', max(selected_info_matrix[expert_num]), selected_info_matrix[expert_num])

            prod_rep = prod_rep[selected_info_matrix[expert_num]]
            pair_pred = pair_pred[selected_info_matrix[expert_num]]

        rel_dists = self.rel_compress_lst[expert_num](prod_rep)

        if self.use_bias:
            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())

        if selected_info_matrix is None:
            # rel_dists[:, 1:] = rel_dists[:, 1:] - self.rel_adjustments
            rel_dists = rel_dists.split(num_rels, dim=0)

        # print('rel_dists:', len(rel_dists), rel_dists[0])   # tuple contains 6 elements, [X, 51], X is arbitrary number

        # print('prod_rep:', len(prod_rep), prod_rep)
        # print('pair_pred:', len(pair_pred), pair_pred)



        return rel_dists

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor]): logits of relation label distribution
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """


        # print('rel_labels:', len(rel_labels), rel_labels)
        # rel_labelsx = cat(rel_labels, dim=0)

        # print('rel_labelsx:', len(rel_labelsx), rel_labelsx)

        # print('rel_pair_idxs:', len(rel_pair_idxs), rel_pair_idxs)

        # encode context infomation
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
            # print('obj_dists:', obj_dists)
            # print('obj_preds:', obj_preds)
            # print('att_dists:', att_dists)
            # print('edge_ctx:', edge_ctx)

        else:
            if self.use_relation_aware_gating:
                obj_dists, obj_preds, edge_ctx_info, beta_relation_aware_gating, _ = self.context_layer(roi_features,
                                                                                                        proposals,
                                                                                                        logger)
            else:
                obj_dists, obj_preds, edge_ctx_info, _ = self.context_layer(roi_features, proposals, logger)

            if self.num_experts > 1:
                edge_ctx = edge_ctx_info

                # outs = edge_ctx_info['logits']
                # edge_ctx = edge_ctx_info['output']
                # print('edge_ctx:', edge_ctx.shape)      # [x, 512]
                # print('obj_dists:', obj_dists.shape)    # [x, 151], x is the number of objects
                # print('obj_preds:', obj_preds)          #  x
            elif self.num_experts == 1:
                edge_ctx = edge_ctx_info

        num_objs = [len(b) for b in proposals]
        # print('num_objs:', num_objs)    # num_objs: [80, 80, 80, 80, 80, 80]
        obj_dists = obj_dists.split(num_objs, dim=0)
        # print('obj_dists:', len(obj_dists), obj_dists[0])   # tuple contains 6 elements, each with [80, 151]


        if self.num_experts > 1:
            rel_dists_lst = []

            # if self.training:
            if self.use_relation_sampling:
                if rel_labels is not None:
                    selected_info_matrix, rel_labels_lst,  num_activated_experts = self.relation_sampling.obtain_selected_info_matrix(rel_labels)
                else:
                    selected_info_matrix = None
                    num_activated_experts = self.num_experts
                    rel_labels_lst = None
            else:
                num_activated_experts = self.num_experts
                selected_info_matrix = None

            rel_dists = self.edge_ctx_process(0, edge_ctx, rel_pair_idxs, obj_preds, union_features, num_objs, selected_info_matrix)
            rel_dists_lst.append(rel_dists)

            # print('num_activated_experts:', num_activated_experts)
            for i in range(num_activated_experts-1):
                # edge_info = self.edge_ctx_process(i+1, outs[:, i+1, :], rel_pair_idxs, obj_preds, union_features, num_objs, selected_info_matrix)
                edge_info = self.edge_ctx_process(i+1, edge_ctx, rel_pair_idxs, obj_preds, union_features, num_objs, selected_info_matrix)

                # print('edge_info:', edge_info)

                rel_dists_lst.append(edge_info)

        # print('rel_dists_lst:', rel_dists_lst)
        # print('rel_labels_lst:', rel_labels_lst)

        # print('rel_dists:', len(rel_dists), rel_dists[0].shape)
        # print('rel_dists_lst:', len(rel_dists_lst), rel_dists_lst[0][0].shape)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        if self.attribute_on:
            att_dists = att_dists.split(num_objs, dim=0)
            return (obj_dists, att_dists), rel_dists, add_losses

        elif self.num_experts > 1:
            if self.use_relation_sampling:
                return obj_dists, rel_dists, add_losses, rel_labels_lst, rel_dists_lst, beta_relation_aware_gating

            elif self.use_relation_aware_gating:
                return obj_dists, rel_dists, add_losses, rel_dists_lst, beta_relation_aware_gating

            else:
                return obj_dists, rel_dists, add_losses, rel_dists_lst

        elif self.num_experts == 1:
            return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("VCTreePredictor_PL")
class VCTreePredictor_PL(nn.Module):
    def __init__(self, config, in_channels):
        super(VCTreePredictor_PL, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON   
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.num_experts = config.MODEL.ROI_RELATION_HEAD.NUM_EXPERTS
        self.use_relation_aware_gating = config.MODEL.ROI_RELATION_HEAD.USE_RELATION_AWARE_GATING

        print('self.num_experts:', self.num_experts)
        print('self.use_relation_aware_gating:', self.use_relation_aware_gating)

        assert in_channels is not None
        num_inputs = in_channels

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics[
            'att_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_att_cls == len(att_classes)
        assert self.num_rel_cls == len(rel_classes)
        # init contextual lstm encoding

        if self.num_experts == 1:
            self.context_layer = VCTreeLSTMContext(config, obj_classes, rel_classes, statistics, in_channels)
        elif self.num_experts > 1:
            # self.context_layer = VCTreeLSTMContext_split(config, obj_classes, rel_classes, statistics, in_channels)
            self.context_layer = VCTreeLSTMContext(config, obj_classes, rel_classes, statistics, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM         # 512
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM       # 4096

        if self.num_experts > 1:
            self.post_emb_lst = nn.ModuleList(
                [nn.Linear(self.hidden_dim, self.hidden_dim * 2) for _ in range(self.num_experts)]      # 512, 1024
            )
            self.post_cat_lst = nn.ModuleList(
                [nn.Linear(self.hidden_dim * 2, self.pooling_dim) for _ in range(self.num_experts)]     # 1024, 4096
            )
            self.ctx_compress_lst = nn.ModuleList(
                [nn.Linear(self.pooling_dim, self.num_rel_cls) for _ in range(self.num_experts)]        # 4096, 51
            )

            for i in range(self.num_experts):
                # initialize layer parameters
                layer_init(self.ctx_compress_lst[i], xavier=True)
                layer_init(self.post_emb_lst[i], 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
                layer_init(self.post_cat_lst[i], xavier=True)

        else:
            self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)             # 512, 1024
            self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)            # 1024, 4096
            self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)           # 4096, 51

            # initialize layer parameters
            layer_init(self.ctx_compress, xavier=True)
            layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
            layer_init(self.post_cat, xavier=True)


        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        self.freq_bias = FrequencyBias(config, statistics)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        # print('rel_labels:', len(rel_labels), rel_labels)
        # print('rel_pair_idxs:', len(rel_pair_idxs), rel_pair_idxs)


        # encode context infomation
        if self.num_experts == 1:
            obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs,
                                                                              logger)
        elif self.num_experts > 1:
            obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs,
                                                                              logger)

            # if self.use_relation_aware_gating:
            #     obj_dists, obj_preds, edge_ctx_info, binary_preds, beta_relation_aware_gating = self.context_layer(
            #         roi_features, proposals, rel_pair_idxs, logger)
            # else:
            #     obj_dists, obj_preds, edge_ctx_info, binary_preds = self.context_layer(roi_features, proposals,
            #                                                                            rel_pair_idxs, logger)

            # outs = edge_ctx_info['logits']
            # edge_ctx = edge_ctx_info['output']

        num_objs = [len(b) for b in proposals]
        obj_preds = obj_preds.split(num_objs, dim=0)
        obj_dists = obj_dists.split(num_objs, dim=0)

        rel_dists_lst = []
        rel_dists_full_lst = []
        if self.num_experts > 1:
            rel_dists, rel_dists_full = self.edge_ctx_process(0, edge_ctx, rel_pair_idxs, obj_preds, union_features, num_objs)
            rel_dists_lst.append(rel_dists)
            rel_dists_full_lst.append(rel_dists_full)

            for i in range(self.num_experts-1):
                edge_info, rel_dists_full = self.edge_ctx_process(i+1, edge_ctx, rel_pair_idxs, obj_preds, union_features, num_objs)
                # print('edge_info.shape:', edge_info)
                rel_dists_lst.append(edge_info)
                rel_dists_full_lst.append(rel_dists_full)
                
        elif self.num_experts == 1:
            rel_dists = self.edge_ctx_process(0, edge_ctx, rel_pair_idxs, obj_preds, union_features, num_objs)
            
            
        if self.num_experts > 1:
            weighted_rel_dists_full_lst = []
            weight_lst = np.zeros(self.num_experts)
            sigma = 0.5
            for i in range(self.num_experts):
                if(i == 0):
                    weight_lst[i] = 1
                else:
                    weight_lst[i] = 1 / self.num_experts * sigma
            
            # print('weight_lst:', weight_lst)
            # print('rel_dists_full_lst:', len(rel_dists_full_lst))
            # print('rel_dists_full_lst:', rel_dists_full_lst[0].shape)

            for i in range(self.num_experts):
                weighted_rel_dists_full = weight_lst[i] * rel_dists_full_lst[i] # torch.mul(weight_lst[i], rel_dists_full_lst[i])
                weighted_rel_dists_full_lst.append(weighted_rel_dists_full)

            weighted_rel_dists_full_sum = torch.stack(weighted_rel_dists_full_lst, dim=0).sum(dim=0)
            rel_dists_full_mean = torch.stack(rel_dists_full_lst, dim=0).mean(dim=0)

            # print('weighted_rel_dists_full.shape:', weighted_rel_dists_full.shape)
            num_rels = [r.shape[0] for r in rel_pair_idxs]
            weighted_rel_dists_full_sum = weighted_rel_dists_full_sum.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        if self.training:
            binary_loss = []
            for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                bi_gt = (bi_gt > 0).float()
                binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
            add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)

        if self.num_experts > 1:
            if self.use_relation_aware_gating:
                return obj_dists, rel_dists, add_losses, rel_dists_lst #, beta_relation_aware_gating
            else:
                return obj_dists, rel_dists, add_losses, rel_dists_lst # , weighted_rel_dists_full_sum, rel_dists_full_mean

        elif self.num_experts == 1:
            return obj_dists, rel_dists, add_losses

    def edge_ctx_process(self, expert_num, edge_ctx, rel_pair_idxs, obj_preds, union_features, num_objs):
        # post decode
        edge_rep = F.relu(self.post_emb_lst[expert_num](edge_ctx))
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)

        pair_pred = cat(pair_preds, dim=0)                          # idx of object pairs
        prod_rep = self.post_cat_lst[expert_num](prod_rep)          # probability among object pairs

        # learned-mixin Gate
        # uni_gate = torch.tanh(self.uni_gate(self.drop(prod_rep)))
        # frq_gate = torch.tanh(self.frq_gate(self.drop(prod_rep)))

        if self.union_single_not_match:
            union_features = self.up_dim(union_features)

        ctx_dists = self.ctx_compress_lst[expert_num](prod_rep * union_features)
        # uni_dists = self.uni_compress(self.drop(union_features))
        frq_dists = self.freq_bias.index_with_labels(pair_pred.long())

        rel_dists_full = ctx_dists + frq_dists
        # rel_dists = ctx_dists + uni_gate * uni_dists + frq_gate * frq_dists

        rel_dists = rel_dists_full.split(num_rels, dim=0)


        return rel_dists, rel_dists_full
    
    
@registry.ROI_RELATION_PREDICTOR.register("TransformerPredictor_PL")
class TransformerPredictor_PL(nn.Module):
    def __init__(self, config, in_channels):
        super(TransformerPredictor_PL, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        # load parameters
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        # PLME
        self.num_experts = config.MODEL.ROI_RELATION_HEAD.NUM_EXPERTS

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics[
            'att_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_att_cls == len(att_classes)
        assert self.num_rel_cls == len(rel_classes)
        # module construct
        if self.num_experts > 1:
            self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)
        else:
            self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)



        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        # post decoding
        if (self.num_experts > 1):
            self.post_emb_lst = nn.ModuleList(
                [nn.Linear(self.hidden_dim, self.hidden_dim * 2) for _ in range(self.num_experts)]       # 512, 1024
            )
            self.post_cat_lst = nn.ModuleList(
                [nn.Linear(self.hidden_dim * 2, self.pooling_dim) for _ in range(self.num_experts)]      # 1024, 4096
            )
            self.rel_compress_lst = nn.ModuleList(
                [nn.Linear(self.pooling_dim, self.num_rel_cls, bias=True) for _ in range(self.num_experts)] # 4096, 51   (rel_classifier)
            )
            self.ctx_compress_lst = nn.ModuleList(
                [nn.Linear(self.hidden_dim * 2, self.num_rel_cls) for _ in range(self.num_experts)]
            )

            for i in range(self.num_experts):
                layer_init(self.post_emb_lst[i], 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
                layer_init(self.post_cat_lst[i], xavier=True)
                layer_init(self.rel_compress_lst[i], xavier=True)
                layer_init(self.ctx_compress_lst[i], xavier=True)

        elif self.num_experts == 1:
            self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
            self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
            self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
            self.ctx_compress = nn.Linear(self.hidden_dim * 2, self.num_rel_cls)

            # initialize layer parameters
            layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
            layer_init(self.post_cat, xavier=True)
            layer_init(self.rel_compress, xavier=True)
            layer_init(self.ctx_compress, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        else:
            if self.num_experts > 1:
                obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)

            else:
                obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        obj_preds = obj_preds.split(num_objs, dim=0)
        obj_dists = obj_dists.split(num_objs, dim=0)

        if self.num_experts == 1:
            rel_dists = self.post_decode(0, edge_ctx, rel_pair_idxs, num_objs, num_rels, obj_preds, union_features)

        elif self.num_experts > 1:
            rel_dists_lst = []
            if self.num_experts > 1:
                for i in range(self.num_experts):
                    rel_dists_lst.append(
                        self.post_decode(i, edge_ctx, rel_pair_idxs, num_objs, num_rels, obj_preds, union_features))

            rel_dists = rel_dists_lst[0]


        add_losses = {}
        if self.attribute_on:
            att_dists = att_dists.split(num_objs, dim=0)
            return (obj_dists, att_dists), rel_dists, add_losses
        elif self.num_experts > 1:
            return obj_dists, rel_dists, add_losses, rel_dists_lst
        elif self.num_experts == 1:
            return obj_dists, rel_dists, add_losses

    # post decode
    def post_decode(self, expert_index, edge_ctx, rel_pair_idxs, num_objs, num_rels, obj_preds, union_features):
        edge_rep = self.post_emb_lst[expert_index](edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)
        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)

        # from object level feature to pairwise relation level feature
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        ctx_gate = self.post_cat_lst[expert_index](prod_rep)

        # use union box and mask convolution
        if self.use_vision:
            if self.union_single_not_match:
                visual_rep = ctx_gate * self.up_dim(union_features)
            else:
                visual_rep = ctx_gate * union_features

        rel_dists = self.rel_compress_lst[expert_index](visual_rep) + self.ctx_compress_lst[expert_index](prod_rep)

        # use frequence bias
        if self.use_bias:
            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred)

        rel_dists = rel_dists.split(num_rels, dim=0)

        return rel_dists


@registry.ROI_RELATION_PREDICTOR.register("MotifPredictor_CAME")
class MotifPredictor_CAME(nn.Module):
    def __init__(self, config, in_channels):
        super(MotifPredictor_CAME, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        # CAME
        self.num_experts = config.MODEL.ROI_RELATION_HEAD.NUM_EXPERTS
        self.use_relation_aware_gating = config.MODEL.ROI_RELATION_HEAD.USE_RELATION_AWARE_GATING
        self.use_relation_sampling = config.MODEL.ROI_RELATION_HEAD.RELATION_SAMPLING
        self.expert_voting = config.MODEL.ROI_RELATION_HEAD.EXPERT_VOTING

        if self.use_relation_sampling:
            self.relation_sampling = Relation_Sampling()

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION  # True
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS  # True

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics[
            'att_classes']
        # print('obj_classes:', len(obj_classes), obj_classes)        # 151,
        # print('rel_classes:', len(rel_classes), rel_classes)        # 51,
        # print('att_classes:', len(att_classes), att_classes)        # 201, includes: '__background__'

        print('self.num_obj_cls:', self.num_obj_cls)
        print('obj_classes:', len(obj_classes), obj_classes)
        
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_att_cls == len(att_classes)
        assert self.num_rel_cls == len(rel_classes)

        # init contextual lstm encoding
        if self.attribute_on:  # False
            self.context_layer = AttributeLSTMContext(config, obj_classes, att_classes, rel_classes, in_channels)

        elif self.num_experts > 1:
            # print('self.context_layer:', self.context_layer)
            # self.context_layer = LSTMContext_split(config, obj_classes, rel_classes, in_channels)
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)

        elif self.num_experts == 1:
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)

        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM             # 512
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM           # 4096

        # post decoding
        if (self.num_experts > 1):
            # for relation encoder
            self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)                 # 512, 1024
            self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)                # 1024, 4096
            self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=True)    # 4096, 51   (rel_classifier)
            self.rel_prob_encoder = nn.Linear(self.num_rel_cls, self.num_experts)
            self.bn = nn.BatchNorm1d(self.num_experts)
            self.alpha = config.MODEL.ROI_RELATION_HEAD.PER_CLASS_ALPHA
            self.softmax = nn.Softmax()

            # initialize layer parameters
            layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
            layer_init(self.post_cat, xavier=True)
            layer_init(self.rel_compress, xavier=True)
            layer_init(self.rel_prob_encoder, xavier=True)

            self.post_emb_lst = nn.ModuleList(
                [nn.Linear(self.hidden_dim, self.hidden_dim * 2) for _ in range(self.num_experts)]       # 512, 1024
            )
            self.post_cat_lst = nn.ModuleList(
                [nn.Linear(self.hidden_dim * 2, self.pooling_dim) for _ in range(self.num_experts)]      # 1024, 4096
            )
            self.rel_compress_lst = nn.ModuleList(
                [nn.Linear(self.pooling_dim, self.num_rel_cls, bias=True) for _ in range(self.num_experts)] # 4096, 51   (rel_classifier)
            )

            for i in range(self.num_experts):
                layer_init(self.post_emb_lst[i], 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
                layer_init(self.post_cat_lst[i], xavier=True)
                layer_init(self.rel_compress_lst[i], xavier=True)


        else:
            self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)                 # 512, 1024
            self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)                # 1024, 4096
            self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=True)    # 4096, 51   (rel_classifier)

            # initialize layer parameters
            layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
            layer_init(self.post_cat, xavier=True)
            layer_init(self.rel_compress, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:      # 4096 != 4096
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)       # 4096, 4096
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)


    def context_encoder(self, edge_ctx, rel_pair_idxs, obj_preds, union_features, num_objs):
        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]

        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        if self.use_vision:
            if self.union_single_not_match:
                prod_rep = prod_rep * self.up_dim(union_features)
            else:
                prod_rep = prod_rep * union_features

        rel_dists = self.rel_compress(prod_rep)

        if self.use_bias:
            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())

        rel_prob_matrix = self.rel_prob_encoder(rel_dists)
        # print('rel_prob_matrix:', rel_prob_matrix.shape, rel_prob_matrix[1:10, :])

        rel_prob_matrix = self.bn(rel_prob_matrix)
        # print('rel_prob_matrix bn:', rel_prob_matrix.shape, rel_prob_matrix[1:10, :])

        rel_prob_matrix = self.alpha * rel_prob_matrix
        # print('rel_prob_matrix bn:', rel_prob_matrix.shape, rel_prob_matrix[1:10, :])

        rel_prob_matrix = self.softmax(rel_prob_matrix)
        # print('rel_prob_matrix soft:', rel_prob_matrix.shape, rel_prob_matrix[1:10, :])

        # print('len(rel_dists):', len(rel_dists))
        return rel_prob_matrix

    def edge_ctx_process(self, expert_num, edge_ctx, rel_pair_idxs, obj_preds, union_features, num_objs, rel_prob_matrix):
        # post decode
        edge_rep = self.post_emb_lst[expert_num](edge_ctx)
        # print('edge_rep:', edge_rep.shape)          # [x, 1024], x = 480

        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        # print('edge_rep:', edge_rep.shape)          # [x, 2, 512], x = 480

        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        # print('head_rep:', head_rep.shape)          # [x, 512]

        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)
        # print('tail_rep:', tail_rep.shape)          # [x, 512]

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        # print('num_rels:', num_rels)    # num_rels: [272, 182, 110, 30, 90, 156]

        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        # print('head_rep:', head_rep.shape)          # [x, 512]

        tail_reps = tail_rep.split(num_objs, dim=0)
        # print('tail_rep:', tail_rep.shape)          # [x, 512]

        obj_preds = obj_preds.split(num_objs, dim=0)
        # print('obj_preds:', obj_preds)

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat_lst[expert_num](prod_rep)

        if self.use_vision:
            if self.union_single_not_match:
                prod_rep = prod_rep * self.up_dim(union_features)
            else:
                prod_rep = prod_rep * union_features

        rel_dists = self.rel_compress_lst[expert_num](prod_rep)

        if self.use_bias:
            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())

        rel_dists_full = rel_dists

        # add weight to rel_dists
        weight = torch.unsqueeze(rel_prob_matrix[:, expert_num], dim=1)
        rel_dists = torch.mul(weight, rel_dists)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # print('rel_dists:', len(rel_dists), rel_dists[0])   # tuple contains 6 elements, [X, 51], X is arbitrary number

        # print('prod_rep:', len(prod_rep), prod_rep)
        # print('pair_pred:', len(pair_pred), pair_pred)

        return rel_dists, rel_dists_full

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor]): logits of relation label distribution
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        # encode context infomation
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, 
                                                                           proposals, 
                                                                           logger)


        else:
            if self.use_relation_aware_gating:
                # print('use_relation_aware_gating')
                obj_dists, obj_preds, edge_ctx_info, beta_relation_aware_gating, _ = self.context_layer(roi_features,
                                                                                                        proposals,
                                                                                                        logger)
            else:
                obj_dists, obj_preds, edge_ctx_info, _ = self.context_layer(roi_features, 
                                                                            proposals, 
                                                                            logger)

            # if self.num_experts > 1:
            #     outs = edge_ctx_info['logits']
            #     # print('outs.shape:', outs.shape)
            #     edge_ctx = edge_ctx_info['output']
            #     # print('edge_ctx:', edge_ctx.shape)      # [x, 512]
            #
            #     # print('obj_dists:', obj_dists.shape)    # [x, 151], x is the number of objects
            #     # print('obj_preds:', obj_preds)          #  x
            # elif self.num_experts == 1:
            #     edge_ctx = edge_ctx_info

            edge_ctx = edge_ctx_info

        num_objs = [len(b) for b in proposals]
        num_rels = [r.shape[0] for r in rel_pair_idxs]

        # print('num_objs:', num_objs)    # num_objs: [80, 80, 80, 80, 80, 80]
        obj_dists = obj_dists.split(num_objs, dim=0)
        # print('obj_dists:', len(obj_dists), obj_dists[0])   # tuple contains 6 elements, each with [80, 151]

        rel_dists_lst = []
        rel_dists_full_lst = []
        weighted_rel_dists_full_lst = []

        if self.num_experts > 1:
            # if self.training:
            if self.use_relation_sampling:
                if rel_labels is not None:
                    selected_info_matrix, rel_labels_lst,  num_activated_experts = self.relation_sampling.obtain_selected_info_matrix(rel_labels)
                else:
                    selected_info_matrix = None
                    num_activated_experts = self.num_experts
                    rel_labels_lst = None
            else:
                num_activated_experts = self.num_experts
                selected_info_matrix = None

            rel_prob_matrix = self.context_encoder(edge_ctx, rel_pair_idxs, obj_preds, union_features, num_objs)

            rel_dists, rel_dists_full = self.edge_ctx_process(0, edge_ctx, rel_pair_idxs, obj_preds, union_features, num_objs, rel_prob_matrix)
            rel_dists_lst.append(rel_dists)
            rel_dists_full_lst.append(rel_dists_full)
            # print('num_activated_experts:', num_activated_experts)
            for i in range(num_activated_experts-1):
                # edge_info = self.edge_ctx_process(i+1, outs[:, i+1, :], rel_pair_idxs, obj_preds, union_features, num_objs, selected_info_matrix)
                edge_info, rel_dists_full = self.edge_ctx_process(i+1, edge_ctx, rel_pair_idxs, obj_preds, union_features, num_objs, rel_prob_matrix)

                # print('edge_info:', edge_info)

                rel_dists_lst.append(edge_info)
                rel_dists_full_lst.append(rel_dists_full)


            for i in range(len(rel_dists_full_lst)):
                weight = torch.unsqueeze(rel_prob_matrix[:, i], dim=1)
                weighted_rel_dists_full = torch.mul(weight, rel_dists_full_lst[i])
                weighted_rel_dists_full_lst.append(weighted_rel_dists_full)

            weighted_rel_dists_full_sum = torch.stack(weighted_rel_dists_full_lst, dim=0).sum(dim=0)
            # print('weighted_rel_dists_full.shape:', weighted_rel_dists_full.shape)
            weighted_rel_dists_full_sum = weighted_rel_dists_full_sum.split(num_rels, dim=0)

        # print('rel_dists_lst:', rel_dists_lst)
        # print('rel_labels_lst:', rel_labels_lst)

        # print('rel_dists:', len(rel_dists), rel_dists[0].shape)
        # print('rel_dists_lst:', len(rel_dists_lst), rel_dists_lst[0][0].shape)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        if self.attribute_on:
            att_dists = att_dists.split(num_objs, dim=0)
            return (obj_dists, att_dists), rel_dists, add_losses

        elif self.num_experts > 1:
            if self.use_relation_sampling:
                # print('use_relation_sampling')
                return obj_dists, rel_dists, add_losses, rel_labels_lst, rel_dists_lst, beta_relation_aware_gating

            elif self.use_relation_aware_gating:
                # print('use_relation_aware_gating')
                return obj_dists, rel_dists, add_losses, rel_dists_lst, beta_relation_aware_gating, weighted_rel_dists_full_sum

            else:
                return obj_dists, rel_dists, add_losses, rel_dists_lst

        elif self.num_experts == 1:
            return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("VCTreePredictor_CAME")
class VCTreePredictor_CAME(nn.Module):
    def __init__(self, config, in_channels):
        super(VCTreePredictor_CAME, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.num_experts = config.MODEL.ROI_RELATION_HEAD.NUM_EXPERTS
        self.use_relation_aware_gating = config.MODEL.ROI_RELATION_HEAD.USE_RELATION_AWARE_GATING
        self.use_per_class_content_aware_matrix = config.MODEL.ROI_RELATION_HEAD.USE_PER_CLASS_CONTEXT_AWARE

        print('self.num_experts:', self.num_experts)
        print('self.use_relation_aware_gating:', self.use_relation_aware_gating)
        print('self.use_per_class_content_aware_matrix:', self.use_per_class_content_aware_matrix)

        assert in_channels is not None
        num_inputs = in_channels

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics[
            'att_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_att_cls == len(att_classes)
        assert self.num_rel_cls == len(rel_classes)
        # init contextual lstm encoding

        if self.num_experts == 1:
            self.context_layer = VCTreeLSTMContext(config, obj_classes, rel_classes, statistics, in_channels)
        elif self.num_experts > 1:
            #self.context_layer = VCTreeLSTMContext_split(config, obj_classes, rel_classes, statistics, in_channels)
            self.context_layer = VCTreeLSTMContext(config, obj_classes, rel_classes, statistics, in_channels)


        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM         # 512
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM       # 4096

        if self.num_experts > 1:
            self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)             # 512, 1024
            self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)            # 1024, 4096
            self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)           # 4096, 51
            self.rel_prob_encoder = nn.Linear(self.num_rel_cls, self.num_experts)

            self.bn = nn.BatchNorm1d(self.num_experts)
            self.alpha = config.MODEL.ROI_RELATION_HEAD.PER_CLASS_ALPHA
            self.softmax = nn.Softmax()

            # initialize layer parameters
            layer_init(self.ctx_compress, xavier=True)
            layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
            layer_init(self.post_cat, xavier=True)


            self.post_emb_lst = nn.ModuleList(
                [nn.Linear(self.hidden_dim, self.hidden_dim * 2) for _ in range(self.num_experts)]      # 512, 1024
            )
            self.post_cat_lst = nn.ModuleList(
                [nn.Linear(self.hidden_dim * 2, self.pooling_dim) for _ in range(self.num_experts)]     # 1024, 4096
            )
            self.ctx_compress_lst = nn.ModuleList(
                [nn.Linear(self.pooling_dim, self.num_rel_cls) for _ in range(self.num_experts)]        # 4096, 51
            )

            for i in range(self.num_experts):
                # initialize layer parameters
                layer_init(self.ctx_compress_lst[i], xavier=True)
                layer_init(self.post_emb_lst[i], 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
                layer_init(self.post_cat_lst[i], xavier=True)

        else:
            self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)             # 512, 1024
            self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)            # 1024, 4096
            self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)           # 4096, 51

            # initialize layer parameters
            layer_init(self.ctx_compress, xavier=True)
            layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
            layer_init(self.post_cat, xavier=True)


        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        self.freq_bias = FrequencyBias(config, statistics)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        # encode context infomation
        if self.num_experts == 1:
            obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs,
                                                                              logger)
        elif self.num_experts > 1:
            if self.use_relation_aware_gating:
                obj_dists, obj_preds, edge_ctx_info, binary_preds, beta_relation_aware_gating = self.context_layer(
                    roi_features, proposals, rel_pair_idxs, logger)
            else:
                obj_dists, obj_preds, edge_ctx_info, binary_preds = self.context_layer(roi_features, proposals,
                                                                                       rel_pair_idxs, logger)

            edge_ctx = edge_ctx_info

        num_objs = [len(b) for b in proposals]
        num_rels = [r.shape[0] for r in rel_pair_idxs]

        obj_preds = obj_preds.split(num_objs, dim=0)
        obj_dists = obj_dists.split(num_objs, dim=0)

        rel_dists_lst = []
        rel_dists_full_lst = []
        weighted_rel_dists_full_lst = []

        if self.num_experts > 1:
            if self.use_per_class_content_aware_matrix:
                rel_prob_matrix = self.context_encoder(edge_ctx, rel_pair_idxs, obj_preds, union_features, num_objs)
            else:
                rel_prob_matrix = None

            rel_dists, rel_dists_full = self.edge_ctx_process(0, edge_ctx, rel_pair_idxs, obj_preds, union_features, num_objs, rel_prob_matrix)
            rel_dists_lst.append(rel_dists)
            rel_dists_full_lst.append(rel_dists_full)

            for i in range(self.num_experts-1):
                # edge_info = self.edge_ctx_process(i+1, outs[:, i+1, :], rel_pair_idxs, obj_preds, union_features, num_objs)

                edge_info, rel_dists_full = self.edge_ctx_process(i + 1, edge_ctx, rel_pair_idxs, obj_preds, union_features, num_objs, rel_prob_matrix)
                # print('edge_info.shape:', edge_info)
                rel_dists_lst.append(edge_info)
                rel_dists_full_lst.append(rel_dists_full)

            if self.use_per_class_content_aware_matrix:
                for i in range(len(rel_dists_full_lst)):
                    weight = torch.unsqueeze(rel_prob_matrix[:, i], dim=1)
                    weighted_rel_dists_full = torch.mul(weight, rel_dists_full_lst[i])
                    weighted_rel_dists_full_lst.append(weighted_rel_dists_full)

                weighted_rel_dists_full_sum = torch.stack(weighted_rel_dists_full_lst, dim=0).sum(dim=0)
                # print('weighted_rel_dists_full.shape:', weighted_rel_dists_full.shape)
                weighted_rel_dists_full_sum = weighted_rel_dists_full_sum.split(num_rels, dim=0)

        elif self.num_experts == 1:
            rel_dists = self.edge_ctx_process_original(edge_ctx, rel_pair_idxs, obj_preds, union_features, num_objs)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        if self.training:
            binary_loss = []
            for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                bi_gt = (bi_gt > 0).float()
                binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
            add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)

        if self.num_experts > 1:
            if self.use_relation_aware_gating:
                return obj_dists, rel_dists, add_losses, rel_dists_lst, beta_relation_aware_gating, weighted_rel_dists_full_sum
            elif self.use_per_class_content_aware_matrix:
                return obj_dists, rel_dists, add_losses, rel_dists_lst, weighted_rel_dists_full_sum
            else:
                return obj_dists, rel_dists, add_losses, rel_dists_lst

        elif self.num_experts == 1:
            return obj_dists, rel_dists, add_losses

    def context_encoder(self, edge_ctx, rel_pair_idxs, obj_preds, union_features, num_objs):
        # post decode
        edge_rep = F.relu(self.post_emb(edge_ctx))
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)

        pair_pred = cat(pair_preds, dim=0)                          # idx of object pairs
        prod_rep = self.post_cat(prod_rep)          # probability among object pairs

        # learned-mixin Gate
        # uni_gate = torch.tanh(self.uni_gate(self.drop(prod_rep)))
        # frq_gate = torch.tanh(self.frq_gate(self.drop(prod_rep)))

        if self.union_single_not_match:
            union_features = self.up_dim(union_features)

        ctx_dists = self.ctx_compress(prod_rep * union_features)
        # uni_dists = self.uni_compress(self.drop(union_features))
        frq_dists = self.freq_bias.index_with_labels(pair_pred.long())

        rel_dists = ctx_dists + frq_dists
        rel_prob_matrix = self.rel_prob_encoder(rel_dists)
        rel_prob_matrix = self.bn(rel_prob_matrix)
        rel_prob_matrix = self.alpha * rel_prob_matrix
        rel_prob_matrix = self.softmax(rel_prob_matrix)

        return rel_prob_matrix

    def edge_ctx_process(self, expert_num, edge_ctx, rel_pair_idxs, obj_preds, union_features, num_objs, rel_prob_matrix):
        # post decode
        edge_rep = F.relu(self.post_emb_lst[expert_num](edge_ctx))
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)

        pair_pred = cat(pair_preds, dim=0)                          # idx of object pairs
        prod_rep = self.post_cat_lst[expert_num](prod_rep)          # probability among object pairs

        # learned-mixin Gate
        # uni_gate = torch.tanh(self.uni_gate(self.drop(prod_rep)))
        # frq_gate = torch.tanh(self.frq_gate(self.drop(prod_rep)))

        if self.union_single_not_match:
            union_features = self.up_dim(union_features)

        ctx_dists = self.ctx_compress_lst[expert_num](prod_rep * union_features)
        # uni_dists = self.uni_compress(self.drop(union_features))
        frq_dists = self.freq_bias.index_with_labels(pair_pred.long())

        rel_dists = ctx_dists + frq_dists
        # rel_dists = ctx_dists + uni_gate * uni_dists + frq_gate * frq_dists

        rel_dists_full = rel_dists

        # add weight to rel_dists
        if self.use_per_class_content_aware_matrix:
            weight = torch.unsqueeze(rel_prob_matrix[:, expert_num], dim=1)
            rel_dists = torch.mul(weight, rel_dists)

        rel_dists = rel_dists.split(num_rels, dim=0)


        return rel_dists, rel_dists_full


    def edge_ctx_process_original(self, edge_ctx, rel_pair_idxs, obj_preds, union_features, num_objs):
        # post decode
        edge_rep = F.relu(self.post_emb(edge_ctx))
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)

        pair_pred = cat(pair_preds, dim=0)                          # idx of object pairs
        prod_rep = self.post_cat(prod_rep)          # probability among object pairs

        # learned-mixin Gate
        # uni_gate = torch.tanh(self.uni_gate(self.drop(prod_rep)))
        # frq_gate = torch.tanh(self.frq_gate(self.drop(prod_rep)))

        if self.union_single_not_match:
            union_features = self.up_dim(union_features)

        ctx_dists = self.ctx_compress(prod_rep * union_features)
        # uni_dists = self.uni_compress(self.drop(union_features))
        frq_dists = self.freq_bias.index_with_labels(pair_pred.long())

        rel_dists = ctx_dists + frq_dists
        # rel_dists = ctx_dists + uni_gate * uni_dists + frq_gate * frq_dists

        rel_dists = rel_dists.split(num_rels, dim=0)


        return rel_dists


@registry.ROI_RELATION_PREDICTOR.register("TransformerPredictor_CAME")
class TransformerPredictor_CAME(nn.Module):
    def __init__(self, config, in_channels):
        super(TransformerPredictor_CAME, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        # load parameters
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        # CAME
        self.num_experts = config.MODEL.ROI_RELATION_HEAD.NUM_EXPERTS
        self.use_relation_aware_gating = config.MODEL.ROI_RELATION_HEAD.USE_RELATION_AWARE_GATING

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics[
            'att_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_att_cls == len(att_classes)
        assert self.num_rel_cls == len(rel_classes)
        # module construct
        if self.num_experts > 1:
            self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)
        else:
            self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)

        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        # post decoding
        if (self.num_experts > 1):

            self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
            self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
            self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
            self.ctx_compress = nn.Linear(self.hidden_dim * 2, self.num_rel_cls)
            self.rel_prob_encoder = nn.Linear(self.num_rel_cls, self.num_experts)

            self.bn = nn.BatchNorm1d(self.num_experts)
            self.softmax = nn.Softmax()

            # initialize layer parameters
            layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
            layer_init(self.post_cat, xavier=True)
            layer_init(self.rel_compress, xavier=True)
            layer_init(self.ctx_compress, xavier=True)
            layer_init(self.rel_prob_encoder, xavier=True)

            self.post_emb_lst = nn.ModuleList(
                [nn.Linear(self.hidden_dim, self.hidden_dim * 2) for _ in range(self.num_experts)]  # 512, 1024
            )
            self.post_cat_lst = nn.ModuleList(
                [nn.Linear(self.hidden_dim * 2, self.pooling_dim) for _ in range(self.num_experts)]  # 1024, 4096
            )
            self.rel_compress_lst = nn.ModuleList(
                [nn.Linear(self.pooling_dim, self.num_rel_cls, bias=True) for _ in range(self.num_experts)]
                # 4096, 51   (rel_classifier)
            )
            self.ctx_compress_lst = nn.ModuleList(
                [nn.Linear(self.hidden_dim * 2, self.num_rel_cls, bias=True) for _ in range(self.num_experts)]
            )

            for i in range(self.num_experts):
                layer_init(self.post_emb_lst[i], 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
                layer_init(self.post_cat_lst[i], xavier=True)
                layer_init(self.rel_compress_lst[i], xavier=True)
                layer_init(self.ctx_compress_lst[i], xavier=True)

        else:
            self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
            self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
            self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
            self.ctx_compress = nn.Linear(self.hidden_dim * 2, self.num_rel_cls)

            # initialize layer parameters
            layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
            layer_init(self.rel_compress, xavier=True)
            layer_init(self.ctx_compress, xavier=True)
            layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        else:
            if self.use_relation_aware_gating:
                obj_dists, obj_preds, edge_ctx, relation_aware_gating = self.context_layer(roi_features, proposals, logger)
            else:
                obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        # print('np.sum(num_rels):', num_rels, np.sum(num_rels))

        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        obj_preds = obj_preds.split(num_objs, dim=0)
        obj_dists = obj_dists.split(num_objs, dim=0)

        rel_dists_lst = []
        rel_dists_full_lst = []
        weighted_rel_dists_full_lst = []
        rel_prob_matrix = self.context_encoder(edge_ctx, rel_pair_idxs, num_objs, num_rels, obj_preds, union_features)

        if self.num_experts > 1:
            for i in range(self.num_experts):
                rel_dists, rel_dists_full = self.post_decode(edge_ctx, rel_pair_idxs, num_objs, num_rels, obj_preds, union_features, expert_num=i)
                rel_dists_lst.append(rel_dists)
                rel_dists_full_lst.append(rel_dists_full)

            rel_dists = rel_dists_lst[0]
            # print('rel_prob_matrix.shape:', rel_prob_matrix.shape)
            # print('rel_dists_full_lst.shape:', rel_dists_full_lst[0].shape)

            for i in range(len(rel_dists_full_lst)):
                temp = torch.unsqueeze(rel_prob_matrix[:, i], dim=1)
                weighted_rel_dists_full = torch.mul(temp, rel_dists_full_lst[i])
                weighted_rel_dists_full_lst.append(weighted_rel_dists_full)

            weighted_rel_dists_full_sum = torch.stack(weighted_rel_dists_full_lst, dim=0).sum(dim=0)
            # print('weighted_rel_dists_full.shape:', weighted_rel_dists_full.shape)
            weighted_rel_dists_full_sum = weighted_rel_dists_full_sum.split(num_rels, dim=0)

            # total_size = 0.0
            # for i in range(len(weighted_rel_dists_full)):
            #     tensize = weighted_rel_dists_full[i].shape
            #     print('tensize_tt:', tensize)
            #     total_size += tensize[0]
            # print('total_size_tt:', total_size)

        else:
            rel_dists = self.post_decode(edge_ctx, rel_pair_idxs, num_objs, num_rels, obj_preds, union_features)

        add_losses = {}

        if self.attribute_on:
            att_dists = att_dists.split(num_objs, dim=0)
            return (obj_dists, att_dists), rel_dists, add_losses

        if self.use_relation_aware_gating:
            return obj_dists, rel_dists, add_losses, rel_dists_lst, relation_aware_gating, weighted_rel_dists_full_sum
        elif self.num_experts > 1:
            return obj_dists, rel_dists, add_losses, rel_dists_lst
        elif self.num_experts == 1:
            return obj_dists, rel_dists, add_losses

    # post decode
    def context_encoder(self, edge_ctx, rel_pair_idxs, num_objs, num_rels, obj_preds, union_features):

        # print('union_features:', union_features.shape)  # the same as bs [x, 4096]
        # print('edge_ctx:', edge_ctx.shape)
        edge_rep = self.post_emb(edge_ctx)
        # print('edge_ctx:', edge_ctx.shape)

        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        # print('edge_rep:', edge_rep.shape)

        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        # print('head_rep:', len(head_rep), head_rep[0].shape)

        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)
        # print('tail_rep:', tail_rep.shape)

        head_reps = head_rep.split(num_objs, dim=0)
        # print('head_reps:', len(head_reps), head_reps[0].shape)

        tail_reps = tail_rep.split(num_objs, dim=0)
        # print('tail_reps:', len(tail_reps), tail_reps[0].shape)

        # from object level feature to pairwise relation level feature
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            # print('prod_reps:', len(prod_reps), prod_reps[0].shape)
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
            # print('pair_preds:', len(pair_preds), pair_preds[0].shape)

        prod_rep = cat(prod_reps, dim=0)        # the same as bs [x, 1024]
        # print('prod_rep:', prod_rep.shape)

        pair_pred = cat(pair_preds, dim=0)      # the same as bs [x, 2]
        # print('pair_pred:', pair_pred.shape)

        ctx_gate = self.post_cat(prod_rep)      # the same as bs [x, 4096]
        # print('ctx_gate:', ctx_gate.shape)

        # use union box and mask convolution
        if self.use_vision:
            if self.union_single_not_match:
                visual_rep = ctx_gate * self.up_dim(union_features)     # the same as bs [x, 4096]
            else:
                visual_rep = ctx_gate * union_features                  # the same as bs [x, 4096]

        # print('visual_rep:', visual_rep.shape)

        rel_dists = self.rel_compress(visual_rep) + self.ctx_compress(prod_rep) # the same as bs [x, 51]
        # print('rel_dists:', rel_dists.shape)

        # use frequence bias
        if self.use_bias:
            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred)                                 # the same as bs [x, 51]


        # rel_dists = rel_dists.split(num_rels, dim=0)

        rel_prob_matrix = self.rel_prob_encoder(rel_dists)
        # print('rel_prob_matrix:', rel_prob_matrix.shape, rel_prob_matrix[1:10, :])

        rel_prob_matrix = self.bn(rel_prob_matrix)
        # print('rel_prob_matrix bn:', rel_prob_matrix.shape, rel_prob_matrix[1:10, :])

        rel_prob_matrix = self.softmax(rel_prob_matrix)
        # print('rel_prob_matrix soft:', rel_prob_matrix.shape, rel_prob_matrix[1:10, :])

        # print('len(rel_dists):', len(rel_dists))
        return rel_prob_matrix

    # post decode
    def post_decode(self, edge_ctx, rel_pair_idxs, num_objs, num_rels, obj_preds, union_features, expert_num):
        # print('edge_ctx:', edge_ctx.shape)
        edge_rep = self.post_emb_lst[expert_num](edge_ctx)
        # print('edge_ctx:', edge_ctx.shape)

        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        # print('edge_rep:', edge_rep.shape)

        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        # print('head_rep:', len(head_rep), head_rep[0].shape)

        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)
        # print('tail_rep:', tail_rep.shape)

        head_reps = head_rep.split(num_objs, dim=0)
        # print('head_reps:', len(head_reps), head_reps[0].shape)

        tail_reps = tail_rep.split(num_objs, dim=0)
        # print('tail_reps:', len(tail_reps), tail_reps[0].shape)

        # from object level feature to pairwise relation level feature
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            # print('prod_reps:', len(prod_reps), prod_reps[0].shape)
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
            # print('pair_preds:', len(pair_preds), pair_preds[0].shape)

        prod_rep = cat(prod_reps, dim=0)
        # print('prod_rep:', prod_rep.shape)

        pair_pred = cat(pair_preds, dim=0)
        # print('pair_pred:', pair_pred.shape)

        ctx_gate = self.post_cat_lst[expert_num](prod_rep)
        # print('ctx_gate:', ctx_gate.shape)

        # use union box and mask convolution
        if self.use_vision:
            if self.union_single_not_match:
                visual_rep = ctx_gate * self.up_dim(union_features)
            else:
                visual_rep = ctx_gate * union_features

        # print('visual_rep:', visual_rep.shape)

        rel_dists = self.rel_compress_lst[expert_num](visual_rep) + self.ctx_compress_lst[expert_num](prod_rep)
        # print('rel_dists:', rel_dists.shape)

        # use frequence bias
        if self.use_bias:
            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred)

        # print('len(rel_dists:', len(rel_dists))
        # print('(num_rels:', (num_rels))

        rel_dists_full = rel_dists
        # print('rel_dists:', rel_dists.shape)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # print('len(rel_dists):', len(rel_dists))

        return rel_dists, rel_dists_full


# for test
@registry.ROI_RELATION_PREDICTOR.register("MotifPredictor_CAME_ENCODER")
class MotifPredictor_CAME_ENCODER(nn.Module):
    def __init__(self, config, in_channels):
        super(MotifPredictor_CAME_ENCODER, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.num_experts = config.MODEL.ROI_RELATION_HEAD.NUM_EXPERTS
        self.use_relation_aware_gating = config.MODEL.ROI_RELATION_HEAD.USE_RELATION_AWARE_GATING

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION     # True
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS         # True

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics['att_classes']
        # print('obj_classes:', len(obj_classes), obj_classes)        # 151,
        # print('rel_classes:', len(rel_classes), rel_classes)        # 51,
        # print('att_classes:', len(att_classes), att_classes)        # 201, includes: '__background__'

        assert self.num_obj_cls==len(obj_classes)
        assert self.num_att_cls==len(att_classes)
        assert self.num_rel_cls==len(rel_classes)

        # init contextual lstm encoding
        if self.attribute_on:               # False
            self.context_layer = AttributeLSTMContext(config, obj_classes, att_classes, rel_classes, in_channels)

        elif self.num_experts > 1:
            # print('self.context_layer:', self.context_layer)
            self.context_layer = LSTMContext_split(config, obj_classes, rel_classes, in_channels)

        elif self.num_experts == 1:
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=True)

        # initialize layer parameters 
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)
        layer_init(self.rel_compress, xavier=True)
        
        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

    def edge_ctx_process(self, edge_ctx, rel_pair_idxs, obj_preds, union_features, num_objs):
        # post decode
        edge_rep = self.post_emb(edge_ctx)
        # print('edge_rep:', edge_rep.shape)          # [x, 1024], x = 480

        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        # print('edge_rep:', edge_rep.shape)          # [x, 2, 512], x = 480

        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        # print('head_rep:', head_rep.shape)          # [x, 512]


        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)
        # print('tail_rep:', tail_rep.shape)          # [x, 512]

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        # print('num_rels:', num_rels)    # num_rels: [272, 182, 110, 30, 90, 156]


        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        # print('head_rep:', head_rep.shape)          # [x, 512]

        tail_reps = tail_rep.split(num_objs, dim=0)
        # print('tail_rep:', tail_rep.shape)          # [x, 512]

        obj_preds = obj_preds.split(num_objs, dim=0)
        # print('obj_preds:', obj_preds)

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append( torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1) )
            pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        if self.use_vision:
            if self.union_single_not_match:
                prod_rep = prod_rep * self.up_dim(union_features)
            else:
                prod_rep = prod_rep * union_features

        rel_dists = self.rel_compress(prod_rep)

        if self.use_bias:
            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())

        rel_dists = rel_dists.split(num_rels, dim=0)
        # print('rel_dists:', len(rel_dists), rel_dists[0])   # tuple contains 6 elements, [X, 51], X is arbitrary number

        return rel_dists

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor]): logits of relation label distribution
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        print('proposals:', proposals)
        print('rel_pair_idxs:', rel_pair_idxs)
        print('rel_labels:', rel_labels)

        # encode context infomation
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
            # print('obj_dists:', obj_dists)
            # print('obj_preds:', obj_preds)
            # print('att_dists:', att_dists)
            # print('edge_ctx:', edge_ctx)

        else:
            if self.use_relation_aware_gating:
                obj_dists, obj_preds, edge_ctx_info, beta_relation_aware_gating, _ = self.context_layer(roi_features, proposals, logger)
            else:
                obj_dists, obj_preds, edge_ctx_info, _ = self.context_layer(roi_features, proposals, logger)


            if self.num_experts > 1:
                outs = edge_ctx_info['logits']
                # print('outs.shape:', outs.shape)
                edge_ctx = edge_ctx_info['output']
                # print('edge_ctx:', edge_ctx.shape)      # [x, 512]

                # print('obj_dists:', obj_dists.shape)    # [x, 151], x is the number of objects
                #print('obj_preds:', obj_preds)          #  x
            elif self.num_experts == 1:
                edge_ctx = edge_ctx_info


        num_objs = [len(b) for b in proposals]
        # print('num_objs:', num_objs)    # num_objs: [80, 80, 80, 80, 80, 80]
        obj_dists = obj_dists.split(num_objs, dim=0)
        # print('obj_dists:', len(obj_dists), obj_dists[0])   # tuple contains 6 elements, each with [80, 151]

        rel_dists_lst = []

        if self.num_experts > 1:
            rel_dists = self.edge_ctx_process(edge_ctx, rel_pair_idxs, obj_preds, union_features, num_objs)
            for i in range(self.num_experts):
                edge_info = self.edge_ctx_process(outs[:, i, :], rel_pair_idxs, obj_preds, union_features, num_objs)
                # print('edge_info.shape:', edge_info)
                rel_dists_lst.append(edge_info)

        elif self.num_experts == 1:
            rel_dists = self.edge_ctx_process(edge_ctx, rel_pair_idxs, obj_preds, union_features, num_objs)

        # print('rel_dists:', len(rel_dists), rel_dists[0].shape)
        # print('rel_dists_lst:', len(rel_dists_lst), rel_dists_lst[0][0].shape)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        if self.attribute_on:
            att_dists = att_dists.split(num_objs, dim=0)
            return (obj_dists, att_dists), rel_dists, add_losses
        
        elif self.num_experts > 1:
            if self.use_relation_aware_gating:
                return obj_dists, rel_dists, add_losses, rel_dists_lst, beta_relation_aware_gating
            else:
                return obj_dists, rel_dists, add_losses, rel_dists_lst

        elif self.num_experts == 1:
            return obj_dists, rel_dists, add_losses

# for test
@registry.ROI_RELATION_PREDICTOR.register("VCTreePredictor_CAME_ENCODER")
class VCTreePredictor_CAME_ENCODER(nn.Module):
    def __init__(self, config, in_channels):
        super(VCTreePredictor_CAME_ENCODER, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.num_experts = config.MODEL.ROI_RELATION_HEAD.NUM_EXPERTS
        self.use_relation_aware_gating = config.MODEL.ROI_RELATION_HEAD.USE_RELATION_AWARE_GATING

        print('self.num_experts:', self.num_experts)
        print('self.use_relation_aware_gating:', self.use_relation_aware_gating)

        assert in_channels is not None
        num_inputs = in_channels

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics['att_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_att_cls == len(att_classes)
        assert self.num_rel_cls == len(rel_classes)
        # init contextual lstm encoding

        if self.num_experts == 1:
            self.context_layer = VCTreeLSTMContext(config, obj_classes, rel_classes, statistics, in_channels)
        elif self.num_experts > 1:
            self.context_layer = VCTreeLSTMContext_split(config, obj_classes, rel_classes, statistics, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        # learned-mixin
        #self.uni_gate = nn.Linear(self.pooling_dim, self.num_rel_cls)
        #self.frq_gate = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        #self.uni_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        #layer_init(self.uni_gate, xavier=True)
        #layer_init(self.frq_gate, xavier=True)
        layer_init(self.ctx_compress, xavier=True)
        #layer_init(self.uni_compress, xavier=True)

        # initialize layer parameters 
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)
        
        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        self.freq_bias = FrequencyBias(config, statistics)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        # encode context infomation
        if self.num_experts == 1:
            obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs, logger)
        elif self.num_experts > 1:
            if self.use_relation_aware_gating:
                obj_dists, obj_preds, edge_ctx_info, binary_preds, beta_relation_aware_gating = self.context_layer(roi_features, proposals, rel_pair_idxs, logger)
            else:
                obj_dists, obj_preds, edge_ctx_info, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs, logger)

            outs = edge_ctx_info['logits']
            edge_ctx = edge_ctx_info['output']
  
        num_objs = [len(b) for b in proposals]
        obj_preds = obj_preds.split(num_objs, dim=0)
        obj_dists = obj_dists.split(num_objs, dim=0)

        rel_dists_lst = []
        if self.num_experts > 1:
            rel_dists = self.edge_ctx_process(edge_ctx, rel_pair_idxs, obj_preds, union_features, num_objs)
            for i in range(self.num_experts):
                edge_info = self.edge_ctx_process(outs[:, i, :], rel_pair_idxs, obj_preds, union_features, num_objs)
                # print('edge_info.shape:', edge_info)
                rel_dists_lst.append(edge_info)

        elif self.num_experts == 1:
            rel_dists = self.edge_ctx_process(edge_ctx, rel_pair_idxs, obj_preds, union_features, num_objs)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        if self.training:
            binary_loss = []
            for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                bi_gt = (bi_gt > 0).float()
                binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
            add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)

        if self.num_experts > 1:
            if self.use_relation_aware_gating:
                return obj_dists, rel_dists, add_losses, rel_dists_lst, beta_relation_aware_gating
            else:
                return obj_dists, rel_dists, add_losses, rel_dists_lst

        elif self.num_experts == 1:
            return obj_dists, rel_dists, add_losses


    def edge_ctx_process(self, edge_ctx, rel_pair_idxs, obj_preds, union_features, num_objs):
        # post decode
        edge_rep = F.relu(self.post_emb(edge_ctx))
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append( torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1) )
            pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        # learned-mixin Gate
        #uni_gate = torch.tanh(self.uni_gate(self.drop(prod_rep)))
        #frq_gate = torch.tanh(self.frq_gate(self.drop(prod_rep)))

        if self.union_single_not_match:
            union_features = self.up_dim(union_features)

        ctx_dists = self.ctx_compress(prod_rep * union_features)
        #uni_dists = self.uni_compress(self.drop(union_features))
        frq_dists = self.freq_bias.index_with_labels(pair_pred.long())

        rel_dists = ctx_dists + frq_dists
        #rel_dists = ctx_dists + uni_gate * uni_dists + frq_gate * frq_dists

        rel_dists = rel_dists.split(num_rels, dim=0)

        return rel_dists

# for test
@registry.ROI_RELATION_PREDICTOR.register("CausalAnalysisPredictor_CAME")
class CausalAnalysisPredictor_CAME(nn.Module):
    def __init__(self, config, in_channels):
        super(CausalAnalysisPredictor_CAME, self).__init__()
        self.cfg = config
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.spatial_for_vision = config.MODEL.ROI_RELATION_HEAD.CAUSAL.SPATIAL_FOR_VISION
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.fusion_type = config.MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE
        self.separate_spatial = config.MODEL.ROI_RELATION_HEAD.CAUSAL.SEPARATE_SPATIAL
        self.use_vtranse = config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "vtranse"
        self.effect_type = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE

        self.num_experts = config.MODEL.ROI_RELATION_HEAD.NUM_EXPERTS
        self.use_relation_aware_gating = config.MODEL.ROI_RELATION_HEAD.USE_RELATION_AWARE_GATING
        self.model_type = config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER

        assert in_channels is not None
        num_inputs = in_channels

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls==len(obj_classes)
        assert self.num_rel_cls==len(rel_classes)


        # init contextual lstm encoding
        if self.model_type == "motifs":
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)
        elif  self.model_type == "motifs-came":
            self.context_layer = LSTMContext_split(config, obj_classes, rel_classes, in_channels)
        elif self.model_type == "vctree":
            self.context_layer = VCTreeLSTMContext(config, obj_classes, rel_classes, statistics, in_channels)
        elif self.model_type == "vctree-came":
            self.context_layer = VCTreeLSTMContext_split(config, obj_classes, rel_classes, statistics, in_channels)
        elif self.model_type == "vtranse":
            self.context_layer = VTransEFeature(config, obj_classes, rel_classes, in_channels)

        else:
            print('ERROR: Invalid Context Layer')

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        
        if self.use_vtranse:
            self.edge_dim = self.pooling_dim
            self.post_emb = nn.Linear(self.hidden_dim, self.pooling_dim * 2)
            self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=False)
        else:
            self.edge_dim = self.hidden_dim
            self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
            self.post_cat = nn.Sequential(*[nn.Linear(self.hidden_dim * 2, self.pooling_dim),
                                            nn.ReLU(inplace=True),])
            self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.vis_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)

        if self.fusion_type == 'gate':
            self.ctx_gate_fc = nn.Linear(self.pooling_dim, self.num_rel_cls)
            layer_init(self.ctx_gate_fc, xavier=True)
        
        # initialize layer parameters 
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        if not self.use_vtranse:
            layer_init(self.post_cat[0], xavier=True)
            layer_init(self.ctx_compress, xavier=True)
        layer_init(self.vis_compress, xavier=True)
        
        assert self.pooling_dim == config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM

        # convey statistics into FrequencyBias to avoid loading again
        self.freq_bias = FrequencyBias(config, statistics)

        # add spatial emb for visual feature
        if self.spatial_for_vision:
            self.spt_emb = nn.Sequential(*[nn.Linear(32, self.hidden_dim), 
                                            nn.ReLU(inplace=True),
                                            nn.Linear(self.hidden_dim, self.pooling_dim),
                                            nn.ReLU(inplace=True)
                                        ])
            layer_init(self.spt_emb[0], xavier=True)
            layer_init(self.spt_emb[2], xavier=True)

        self.label_smooth_loss = Label_Smoothing_Regression(e=1.0)

        # untreated average features
        self.effect_analysis = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS
        self.average_ratio = 0.0005

        self.register_buffer("untreated_spt", torch.zeros(32))
        self.register_buffer("untreated_conv_spt", torch.zeros(self.pooling_dim))
        self.register_buffer("avg_post_ctx", torch.zeros(self.pooling_dim))
        self.register_buffer("untreated_feat", torch.zeros(self.pooling_dim))

    def edge_ctx_process(self, union_features, edge_ctx, num_rels, num_objs, obj_preds, obj_dist_prob, obj_boxs, rel_pair_idxs):
        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.edge_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.edge_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.edge_dim)
        # split
        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)

        obj_preds = obj_preds.split(num_objs, dim=0)
        obj_prob_list = obj_dist_prob.split(num_objs, dim=0)

        ctx_reps = []
        pair_preds = []
        pair_obj_probs = []
        pair_bboxs_info = []
        for pair_idx, head_rep, tail_rep, obj_pred, obj_box, obj_prob in zip(rel_pair_idxs, head_reps, tail_reps,
                                                                             obj_preds, obj_boxs, obj_prob_list):
            if self.use_vtranse:
                ctx_reps.append(head_rep[pair_idx[:, 0]] - tail_rep[pair_idx[:, 1]])
            else:
                ctx_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
            pair_obj_probs.append(torch.stack((obj_prob[pair_idx[:, 0]], obj_prob[pair_idx[:, 1]]), dim=2))
            pair_bboxs_info.append(get_box_pair_info(obj_box[pair_idx[:, 0]], obj_box[pair_idx[:, 1]]))

        pair_bbox = cat(pair_bboxs_info, dim=0)
        pair_pred = cat(pair_preds, dim=0)
        pair_obj_probs = cat(pair_obj_probs, dim=0)

        ctx_rep = cat(ctx_reps, dim=0)
        if self.use_vtranse:
            post_ctx_rep = ctx_rep
        else:
            post_ctx_rep = self.post_cat(ctx_rep)

        if self.separate_spatial:
            union_features, spatial_conv_feats = union_features
            post_ctx_rep = post_ctx_rep * spatial_conv_feats

        if self.spatial_for_vision:
            post_ctx_rep = post_ctx_rep * self.spt_emb(pair_bbox)

        rel_dists = self.calculate_logits(union_features, post_ctx_rep, pair_pred, use_label_dist=False)
        rel_dist_list = rel_dists.split(num_rels, dim=0)

        return post_ctx_rep, pair_pred, pair_bbox, pair_obj_probs, edge_rep, rel_dist_list

    def pair_feature_generate(self, union_features, num_rels, roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger, ctx_average=False):
        # encode context infomation
        if self.model_type == 'motifs' or self.model_type == 'vctree':
            obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs, logger, ctx_average=ctx_average)

        elif self.model_type == 'motifs-came':
            if self.use_relation_aware_gating:
                obj_dists, obj_preds, edge_ctx, beta_relation_aware_gating, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs, logger, ctx_average=ctx_average)
            else:
                obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs, logger, ctx_average=ctx_average)

        elif self.model_type == 'vctree-came':
            if self.use_relation_aware_gating:
                obj_dists, obj_preds, edge_ctx, binary_preds, beta_relation_aware_gating = self.context_layer(roi_features, proposals, rel_pair_idxs, logger, ctx_average=ctx_average)
            else:
                obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs, logger, ctx_average=ctx_average)

        if self.num_experts > 1:
            outs = edge_ctx['logits']
            edge_ctx = edge_ctx['output']

        obj_dist_prob = F.softmax(obj_dists, dim=-1)
        obj_dist_list = obj_dists.split(num_objs, dim=0)

        post_ctx_rep, pair_pred, pair_bbox, pair_obj_probs, edge_rep, rel_dist_list =\
            self.edge_ctx_process(union_features, edge_ctx, num_rels, num_objs, obj_preds, obj_dist_prob, obj_boxs, rel_pair_idxs)
        
        
        post_ctx_rep_temp_lst=[]
        pair_obj_probs_temp_lst=[]

        if self.model_type == 'motifs-came' or self.model_type == 'vctree-came':
            rel_dists_lst = []
            # if not ctx_average:
            for i in range(self.num_experts):
                post_ctx_rep_temp, _, _, pair_obj_probs_temp, _, rel_dist_list = \
                    self.edge_ctx_process(union_features, outs[:, i, :], num_rels, num_objs, obj_preds, obj_dist_prob, obj_boxs, rel_pair_idxs)
                rel_dists_lst.append(rel_dist_list)

                post_ctx_rep_temp_lst.append(beta_relation_aware_gating[i]*post_ctx_rep_temp)
                pair_obj_probs_temp_lst.append(beta_relation_aware_gating[i]*pair_obj_probs_temp)

            post_ctx_rep_rame=torch.stack(post_ctx_rep_temp_lst, dim=0).sum(dim=0)
            pair_obj_probs_rame=torch.stack(pair_obj_probs_temp_lst, dim=0).sum(dim=0)

            return post_ctx_rep, pair_pred, pair_bbox, pair_obj_probs, binary_preds, obj_dist_prob, edge_rep, obj_dist_list, rel_dist_list, rel_dists_lst, beta_relation_aware_gating, post_ctx_rep_rame, pair_obj_probs_rame
        

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        obj_boxs = [get_box_info(p.bbox, need_norm=True, proposal=p) for p in proposals]

        assert len(num_rels) == len(num_objs)

        post_ctx_rep, pair_pred, pair_bbox, pair_obj_probs, binary_preds, obj_dist_prob, edge_rep, obj_dist_list, rel_dist_list, rel_dists_lst, beta_relation_aware_gating, post_ctx_rep_rame, pair_obj_probs_rame = self.pair_feature_generate(
            union_features, num_rels, roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger)


        if (not self.training) and self.effect_analysis:
            with torch.no_grad():
                avg_post_ctx_rep, _, _, avg_pair_obj_prob, _, _, _, _, _, _, _, avg_post_ctx_rep_rame, avg_pair_obj_probs_rame = self.pair_feature_generate(
                    union_features, num_rels, roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger, ctx_average=True)



        add_losses = {}
        # additional loss
        if self.training:
            rel_labels = cat(rel_labels, dim=0)

            # binary loss for VCTree
            if binary_preds is not None:
                binary_loss = []
                for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                    bi_gt = (bi_gt > 0).float()
                    binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
                add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)

            # branch constraint: make sure each branch can predict independently
            add_losses['auxiliary_ctx'] = F.cross_entropy(self.ctx_compress(post_ctx_rep), rel_labels)
            if not (self.fusion_type == 'gate'):
                add_losses['auxiliary_vis'] = F.cross_entropy(self.vis_compress(union_features), rel_labels)
                add_losses['auxiliary_frq'] = F.cross_entropy(self.freq_bias.index_with_labels(pair_pred.long()), rel_labels)

            # untreated average feature
            if self.spatial_for_vision:
                self.untreated_spt = self.moving_average(self.untreated_spt, pair_bbox)
            if self.separate_spatial:
                self.untreated_conv_spt = self.moving_average(self.untreated_conv_spt, spatial_conv_feats)

            self.avg_post_ctx = self.moving_average(self.avg_post_ctx, post_ctx_rep)
            self.untreated_feat = self.moving_average(self.untreated_feat, union_features)

        elif self.effect_analysis:
            
            if "came" in self.effect_type:
                with torch.no_grad():
                    # untreated spatial
                    if self.spatial_for_vision:
                        avg_spt_rep = self.spt_emb(self.untreated_spt.clone().detach().view(1, -1))
                    # untreated context
                    avg_ctx_rep_rame = avg_post_ctx_rep_rame * avg_spt_rep if self.spatial_for_vision else avg_post_ctx_rep_rame  
                    avg_ctx_rep_rame = avg_ctx_rep_rame * self.untreated_conv_spt.clone().detach().view(1, -1) if self.separate_spatial else avg_ctx_rep_rame
                    # untreated visual
                    avg_vis_rep_rame = self.untreated_feat.clone().detach().view(1, -1)
                    # untreated category dist
                    avg_frq_rep_rame = avg_pair_obj_probs_rame

                if self.effect_type == 'TDE-came':   # TDE of CTX
                    rel_dists = self.calculate_logits(union_features, post_ctx_rep_rame, pair_obj_probs_rame) - self.calculate_logits(union_features, avg_ctx_rep_rame, pair_obj_probs_rame)
                    rel_dist_list = rel_dists.split(num_rels, dim=0)

                elif self.effect_type == 'NIE-came': # NIE of FRQ
                    rel_dists = self.calculate_logits(union_features, avg_ctx_rep_rame, pair_obj_probs_rame) - self.calculate_logits(union_features, avg_ctx_rep_rame, avg_frq_rep_rame)
                    rel_dist_list = rel_dists.split(num_rels, dim=0)

                elif self.effect_type == 'TE-came':  # Total Effect
                    rel_dists = self.calculate_logits(union_features, post_ctx_rep_rame, pair_obj_probs_rame) - self.calculate_logits(union_features, avg_ctx_rep_rame, avg_frq_rep_rame)
                    rel_dist_list = rel_dists.split(num_rels, dim=0)
                else:
                    assert self.effect_type == 'none'
                    pass

            else:
                with torch.no_grad():
                    # untreated spatial
                    if self.spatial_for_vision:
                        avg_spt_rep = self.spt_emb(self.untreated_spt.clone().detach().view(1, -1))
                    # untreated context
                    avg_ctx_rep = avg_post_ctx_rep * avg_spt_rep if self.spatial_for_vision else avg_post_ctx_rep  
                    avg_ctx_rep = avg_ctx_rep * self.untreated_conv_spt.clone().detach().view(1, -1) if self.separate_spatial else avg_ctx_rep
                    # untreated visual
                    avg_vis_rep = self.untreated_feat.clone().detach().view(1, -1)
                    # untreated category dist
                    avg_frq_rep = avg_pair_obj_prob

                if self.effect_type == 'TDE':   # TDE of CTX
                    rel_dists = self.calculate_logits(union_features, post_ctx_rep, pair_obj_probs) - self.calculate_logits(union_features, avg_ctx_rep, pair_obj_probs)
                    rel_dist_list = rel_dists.split(num_rels, dim=0)

                elif self.effect_type == 'NIE': # NIE of FRQ
                    rel_dists = self.calculate_logits(union_features, avg_ctx_rep, pair_obj_probs) - self.calculate_logits(union_features, avg_ctx_rep, avg_frq_rep)
                    rel_dist_list = rel_dists.split(num_rels, dim=0)

                elif self.effect_type == 'TE':  # Total Effect
                    rel_dists = self.calculate_logits(union_features, post_ctx_rep, pair_obj_probs) - self.calculate_logits(union_features, avg_ctx_rep, avg_frq_rep)
                    rel_dist_list = rel_dists.split(num_rels, dim=0)
                else:
                    assert self.effect_type == 'none'
                    pass

        return obj_dist_list, rel_dist_list, add_losses, rel_dists_lst, beta_relation_aware_gating

    def moving_average(self, holder, input):
        assert len(input.shape) == 2
        with torch.no_grad():
            holder = holder * (1 - self.average_ratio) + self.average_ratio * input.mean(0).view(-1)
        return holder

    def calculate_logits(self, vis_rep, ctx_rep, frq_rep, use_label_dist=True, mean_ctx=False):
        if use_label_dist:
            frq_dists = self.freq_bias.index_with_probability(frq_rep)
        else:
            frq_dists = self.freq_bias.index_with_labels(frq_rep.long())

        if mean_ctx:
            ctx_rep = ctx_rep.mean(-1).unsqueeze(-1)
        vis_dists = self.vis_compress(vis_rep)
        ctx_dists = self.ctx_compress(ctx_rep)

        if self.fusion_type == 'gate':
            ctx_gate_dists = self.ctx_gate_fc(ctx_rep)
            union_dists = ctx_dists * torch.sigmoid(vis_dists + frq_dists + ctx_gate_dists)
            #union_dists = (ctx_dists.exp() * torch.sigmoid(vis_dists + frq_dists + ctx_constraint) + 1e-9).log()    # improve on zero-shot, but low mean recall and TDE recall
            #union_dists = ctx_dists * torch.sigmoid(vis_dists * frq_dists)                                          # best conventional Recall results
            #union_dists = (ctx_dists.exp() + vis_dists.exp() + frq_dists.exp() + 1e-9).log()                        # good zero-shot Recall
            #union_dists = ctx_dists * torch.max(torch.sigmoid(vis_dists), torch.sigmoid(frq_dists))                 # good zero-shot Recall
            #union_dists = ctx_dists * torch.sigmoid(vis_dists) * torch.sigmoid(frq_dists)                           # balanced recall and mean recall
            #union_dists = ctx_dists * (torch.sigmoid(vis_dists) + torch.sigmoid(frq_dists)) / 2.0                   # good zero-shot Recall
            #union_dists = ctx_dists * torch.sigmoid((vis_dists.exp() + frq_dists.exp() + 1e-9).log())               # good zero-shot Recall, bad for all of the rest
            
        elif self.fusion_type == 'sum':
            union_dists = vis_dists + ctx_dists + frq_dists
        else:
            print('invalid fusion type')

        return union_dists

    def binary_ce_loss(self, logits, gt):
        batch_size, num_cat = logits.shape
        answer = torch.zeros((batch_size, num_cat), device=gt.device).float()
        answer[torch.arange(batch_size, device=gt.device), gt.long()] = 1.0
        return F.binary_cross_entropy_with_logits(logits, answer) * num_cat

    def fusion(self, x, y):
        return F.relu(x + y) - (x - y) ** 2



def make_roi_relation_predictor(cfg, in_channels):
    func = registry.ROI_RELATION_PREDICTOR[cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR]
    return func(cfg, in_channels)
