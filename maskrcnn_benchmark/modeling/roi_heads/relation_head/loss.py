# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import numpy.random as npr

from maskrcnn_benchmark.layers import smooth_l1_loss, Label_Smoothing_Regression
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat
from scipy import stats
from .relation_sampling import label_grouping


class RelationLossComputation(object):
    """
    Computes the loss for relation triplet.
    Also supports FPN
    """

    def __init__(
        self,
        config,
        attri_on,
        num_attri_cat,
        max_num_attri,
        attribute_sampling,
        attribute_bgfg_ratio,
        loss_option,
        predicate_proportion,
        rel_class_num_lst,
        obj_class_num_lst,
        use_context_aware_gating,
    ):
        """
        Arguments:
            bbox_proposal_matcher (Matcher)
            rel_fg_bg_sampler (RelationPositiveNegativeSampler)
        """
        self.attri_on = attri_on
        self.num_attri_cat = num_attri_cat
        self.max_num_attri = max_num_attri
        self.attribute_sampling = attribute_sampling
        self.attribute_bgfg_ratio = attribute_bgfg_ratio
        self.pred_weight = (1.0 / torch.FloatTensor([0.5,] + predicate_proportion)).cuda()

        # CAME
        self.loss_option = loss_option
        self.use_context_aware_gating = use_context_aware_gating
        self.use_relation_sampling = config.MODEL.ROI_RELATION_HEAD.RELATION_SAMPLING

        print('self.loss_option:', self.loss_option)
        if self.loss_option == 'FOCAL_LOSS':
            print('starting FOCAL_LOSS ')
            self.criterion_loss = FocalLoss(gamma=2, alpha=0.75)

        elif self.loss_option == 'LDAM_LOSS':
            print('starting LDAM_LOSS')
            self.rel_criterion_loss = LDAMLoss(cls_num_list=rel_class_num_lst, max_m=0.5, s=30)
            self.obj_criterion_loss = LDAMLoss(cls_num_list=obj_class_num_lst, max_m=0.5, s=30)
        
        elif self.loss_option == 'LDAM_LOSS_PN':
            print('starting LDAM_LOSS_PN')
            self.rel_criterion_loss = LDAMLoss_PN(config, cls_num_list=rel_class_num_lst, max_m=0.5, s=30)
            self.obj_criterion_loss = LDAMLoss_PN(config, cls_num_list=obj_class_num_lst, max_m=0.5, s=30)
                    

        elif self.loss_option == 'LABEL_SMOOTHING_LOSS':
            print('starting LABEL_SMOOTHING_LOSS')
            self.criterion_loss = Label_Smoothing_Regression(e=0.01)

        elif self.loss_option == 'CROSS_ENTROPY_LOSS':
            print('starting CROSS_ENTROPY_LOSS')
            self.criterion_loss = nn.CrossEntropyLoss()

        elif self.loss_option == 'RIDE_LOSS':
            print('starting RIDE_LOSS')
            self.rel_criterion_loss = RIDELoss(cls_num_list=rel_class_num_lst,
                                               use_context_aware_gating=self.use_context_aware_gating,
                                               use_relation_sampling=self.use_relation_sampling)

            self.obj_criterion_loss = RIDELoss(cls_num_list=obj_class_num_lst,
                                               use_context_aware_gating=self.use_context_aware_gating,
                                               use_relation_sampling=self.use_relation_sampling)

        elif self.loss_option == 'CAME_LOSS':
            print('starting CAME_LOSS')
            self.rel_criterion_loss = CAMELoss(config,
                                               cls_num_list=rel_class_num_lst,
                                               use_context_aware_gating=self.use_context_aware_gating,
                                               use_relation_sampling=self.use_relation_sampling)

            self.obj_criterion_loss = CAMELoss(config,
                                               cls_num_list=obj_class_num_lst,
                                               use_context_aware_gating=self.use_context_aware_gating,
                                               use_relation_sampling=self.use_relation_sampling)
            
        
        elif self.loss_option == 'PLME_LOSS':
            print('starting PLME_LOSS')
            self.rel_criterion_loss = PLMELoss(config,
                                               cls_num_list=rel_class_num_lst,
                                               use_context_aware_gating=self.use_context_aware_gating,
                                               use_relation_sampling=self.use_relation_sampling)

            self.obj_criterion_loss = PLMELoss(config,
                                               cls_num_list=obj_class_num_lst,
                                               use_context_aware_gating=self.use_context_aware_gating,
                                               use_relation_sampling=self.use_relation_sampling)

        
        elif self.loss_option == 'CAME_LOSS_WO_RW':
            print('starting CAME_LOSS')
            self.rel_criterion_loss = CAMELoss(config,
                                               cls_num_list=None,
                                               use_context_aware_gating=self.use_context_aware_gating,
                                               use_relation_sampling=self.use_relation_sampling)

            self.obj_criterion_loss = CAMELoss(config,
                                               cls_num_list=None,
                                               use_context_aware_gating=self.use_context_aware_gating,
                                               use_relation_sampling=self.use_relation_sampling)

        elif self.loss_option == 'CB_LOSS':
            print('starting CAME_LOSS')
            self.rel_criterion_loss = CAMELoss(config,
                                               cls_num_list=rel_class_num_lst,
                                               use_context_aware_gating=False,
                                               use_relation_sampling=False,
                                               reweight=True)

            self.obj_criterion_loss = CAMELoss(config,
                                               cls_num_list=obj_class_num_lst,
                                               use_context_aware_gating=False,
                                               use_relation_sampling=False,
                                               reweight=True)


        elif self.loss_option == 'CAME_LOSS_WORW':
            print('starting CAME_LOSS_WORW')
            self.rel_criterion_loss = CAMELoss(config,
                                               cls_num_list=rel_class_num_lst,
                                               use_context_aware_gating=self.use_context_aware_gating,
                                               use_relation_sampling=self.use_relation_sampling,
                                               reweight=False)

            self.obj_criterion_loss = CAMELoss(config,
                                               cls_num_list=obj_class_num_lst,
                                               use_context_aware_gating=self.use_context_aware_gating,
                                               use_relation_sampling=self.use_relation_sampling,
                                               reweight=False)


        else:
            print('starting use_focal_loss ')
            self.criterion_loss = FocalLoss(gamma=2, alpha=0.75)

    def __call__(self, proposals, rel_labels, relation_logits, refine_logits, beta_relation_aware_gating=None, extra_info=None, extra_labels=None):
        """
        Computes the loss for relation triplet.
        This requires that the subsample method has been called beforehand.

        Arguments:
            relation_logits (list[Tensor])
            refine_obj_logits (list[Tensor])

        Returns:
            predicate_loss (Tensor)
            finetune_obj_loss (Tensor)
        """
        if self.attri_on:
            if isinstance(refine_logits[0], (list, tuple)):
                refine_obj_logits, refine_att_logits = refine_logits
            else:
                # just use attribute feature, do not actually predict attribute
                self.attri_on = False
                refine_obj_logits = refine_logits
        else:
            refine_obj_logits = refine_logits

        if not self.use_relation_sampling:
            relation_logits = cat(relation_logits, dim=0)
            rel_labels = cat(rel_labels, dim=0)

        refine_obj_logits = cat(refine_obj_logits, dim=0)
        fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)

        if self.loss_option == 'CAME_LOSS':
            # relation loss
            if self.use_relation_sampling:
                loss_relation = self.rel_criterion_loss(relation_logits, 
                                                        rel_labels, 
                                                        beta_relation_aware_gating,
                                                        extra_info=extra_info, 
                                                        extra_labels=extra_labels)

                loss_refine_obj = self.obj_criterion_loss(refine_obj_logits,
                                                          fg_labels.long(),
                                                          beta_relation_aware_gating)


            elif self.use_context_aware_gating:
                loss_relation = self.rel_criterion_loss(relation_logits, 
                                                        rel_labels.long(), 
                                                        beta_relation_aware_gating,
                                                        extra_info=extra_info)
                # obj loss refinement
                loss_refine_obj = self.obj_criterion_loss(refine_obj_logits,
                                                          fg_labels.long(),
                                                          beta_relation_aware_gating)

            else:
                loss_relation = self.rel_criterion_loss(relation_logits, rel_labels.long(),
                                                        extra_info=extra_info)
                # obj loss refinement
                loss_refine_obj = self.obj_criterion_loss(refine_obj_logits, fg_labels.long())
        
        elif self.loss_option == 'PLME_LOSS':
            # relation loss
            loss_relation = self.rel_criterion_loss(relation_logits,
                                                    rel_labels,
                                                    beta_relation_aware_gating,
                                                    extra_info=extra_info,
                                                    extra_labels=extra_labels)

            loss_refine_obj = self.obj_criterion_loss(refine_obj_logits,
                                                      fg_labels.long(),
                                                      beta_relation_aware_gating)

        elif self.loss_option == 'CAME_LOSS_WO_RW':
            loss_relation = self.rel_criterion_loss(relation_logits, rel_labels.long(), beta_relation_aware_gating, extra_info=extra_info)
            loss_refine_obj = self.obj_criterion_loss(refine_obj_logits, fg_labels.long(), beta_relation_aware_gating)

        elif self.loss_option == 'CB_LOSS':
            loss_relation = self.rel_criterion_loss(relation_logits,
                                                    rel_labels.long(),
                                                    extra_info=None)
            # obj loss refinement
            loss_refine_obj = self.obj_criterion_loss(refine_obj_logits,
                                                      fg_labels.long(),
                                                      extra_info=None)

        elif self.loss_option == 'CAME_LOSS_WORW':
            # relation loss
            if self.use_relation_sampling:
                loss_relation = self.rel_criterion_loss(relation_logits, rel_labels, beta_relation_aware_gating,
                                                        extra_info=extra_info, extra_labels=extra_labels)

                loss_refine_obj = self.obj_criterion_loss(refine_obj_logits,
                                                          fg_labels.long(),
                                                          beta_relation_aware_gating)


            elif self.use_context_aware_gating:
                loss_relation = self.rel_criterion_loss(relation_logits, rel_labels.long(), beta_relation_aware_gating,
                                                        extra_info=extra_info)
                # obj loss refinement
                loss_refine_obj = self.obj_criterion_loss(refine_obj_logits,
                                                          fg_labels.long(),
                                                          beta_relation_aware_gating)

            else:
                loss_relation = self.rel_criterion_loss(relation_logits, rel_labels.long(), extra_info=extra_info)
                # obj loss refinement
                loss_refine_obj = self.obj_criterion_loss(refine_obj_logits, fg_labels.long())

        elif self.loss_option == 'RIDE_LOSS':
            # relation loss
            if self.use_relation_sampling:
                loss_relation = self.rel_criterion_loss(relation_logits, rel_labels.long(), beta_relation_aware_gating,
                                                        extra_info=extra_info, extra_labels=extra_labels)
                loss_refine_obj = self.obj_criterion_loss(refine_obj_logits, fg_labels.long(), beta_relation_aware_gating)

            elif self.use_context_aware_gating:
                loss_relation = self.rel_criterion_loss(relation_logits, rel_labels.long(), beta_relation_aware_gating,
                                                        extra_info=extra_info)
                # obj loss refinement
                loss_refine_obj = self.obj_criterion_loss(refine_obj_logits, fg_labels.long(), beta_relation_aware_gating)

            else:
                loss_relation = self.rel_criterion_loss(relation_logits, rel_labels.long(),
                                                        extra_info=extra_info)
                # obj loss refinement
                loss_refine_obj = self.obj_criterion_loss(refine_obj_logits, fg_labels.long())
            
        elif self.loss_option == 'LDAM_LOSS':
            # relation loss
            loss_relation = self.rel_criterion_loss(relation_logits, rel_labels.long())
            # obj loss refinement
            loss_refine_obj = self.obj_criterion_loss(refine_obj_logits, fg_labels.long())
            
        elif self.loss_option == 'LDAM_LOSS_PN':
            # relation loss
            loss_relation = self.rel_criterion_loss(relation_logits, 
                                                    rel_labels.long(), 
                                                    extra_info=extra_info)
            # obj loss refinement
            loss_refine_obj = self.obj_criterion_loss(refine_obj_logits, 
                                                      fg_labels.long(),
                                                      extra_info=extra_info)    
            
        else:
            # relation loss
            loss_relation = self.criterion_loss(relation_logits, rel_labels.long())
            # obj loss refinement
            loss_refine_obj = self.criterion_loss(refine_obj_logits, fg_labels.long())

        # The following code is used to calcaulate sampled attribute loss
        if self.attri_on:
            refine_att_logits = cat(refine_att_logits, dim=0)
            fg_attributes = cat([proposal.get_field("attributes") for proposal in proposals], dim=0)

            attribute_targets, fg_attri_idx = self.generate_attributes_target(fg_attributes)
            if float(fg_attri_idx.sum()) > 0:
                # have at least one bbox got fg attributes
                refine_att_logits = refine_att_logits[fg_attri_idx > 0]
                attribute_targets = attribute_targets[fg_attri_idx > 0]
            else:
                refine_att_logits = refine_att_logits[0].view(1, -1)
                attribute_targets = attribute_targets[0].view(1, -1)

            loss_refine_att = self.attribute_loss(refine_att_logits, attribute_targets, 
                                             fg_bg_sample=self.attribute_sampling, 
                                             bg_fg_ratio=self.attribute_bgfg_ratio)
            return loss_relation, (loss_refine_obj, loss_refine_att)
        else:
            return loss_relation, loss_refine_obj

    def generate_attributes_target(self, attributes):
        """
        from list of attribute indexs to [1,0,1,0,0,1] form
        """
        assert self.max_num_attri == attributes.shape[1]
        device = attributes.device
        num_obj = attributes.shape[0]

        fg_attri_idx = (attributes.sum(-1) > 0).long()
        attribute_targets = torch.zeros((num_obj, self.num_attri_cat), device=device).float()

        for idx in torch.nonzero(fg_attri_idx).squeeze(1).tolist():
            for k in range(self.max_num_attri):
                att_id = int(attributes[idx, k])
                if att_id == 0:
                    break
                else:
                    attribute_targets[idx, att_id] = 1
        return attribute_targets, fg_attri_idx

    def attribute_loss(self, logits, labels, fg_bg_sample=True, bg_fg_ratio=3):
        if fg_bg_sample:
            loss_matrix = F.binary_cross_entropy_with_logits(logits, labels, reduction='none').view(-1)
            fg_loss = loss_matrix[labels.view(-1) > 0]
            bg_loss = loss_matrix[labels.view(-1) <= 0]

            num_fg = fg_loss.shape[0]
            # if there is no fg, add at least one bg
            num_bg = max(int(num_fg * bg_fg_ratio), 1)   
            perm = torch.randperm(bg_loss.shape[0], device=bg_loss.device)[:num_bg]
            bg_loss = bg_loss[perm]

            return torch.cat([fg_loss, bg_loss], dim=0).mean()
        else:
            attri_loss = F.binary_cross_entropy_with_logits(logits, labels)
            attri_loss = attri_loss * self.num_attri_cat / 20.0
            return attri_loss



class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):
        target = target.view(-1)

        logpt = F.log_softmax(input)
        logpt = logpt.index_select(-1, target).diag()
        logpt = logpt.view(-1)
        pt = logpt.exp()

        logpt = logpt * self.alpha * (target > 0).float() + logpt * (1 - self.alpha) * (target <= 0).float()

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class FocalLoss_PN(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):
        target = target.view(-1)

        logpt = F.log_softmax(input)
        logpt = logpt.index_select(-1, target).diag()
        logpt = logpt.view(-1)
        pt = logpt.exp()

        logpt = logpt * self.alpha * (target > 0).float() + logpt * (1 - self.alpha) * (target <= 0).float()

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
        
# class LDAMLoss(nn.Module):
#
#     def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
#         super(LDAMLoss, self).__init__()
#         m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
#         print('m_list:', m_list.shape, m_list)
#
#         m_list = m_list * (max_m / np.max(m_list[1:]))
#         print('m_list:', m_list.shape, m_list)
#
#         m_list = torch.cuda.FloatTensor(m_list)
#         print('m_list:', m_list.shape, m_list)
#         m_list[0] = 0.0
#         self.m_list = m_list
#         print('self.m_list:', self.m_list.shape, self.m_list)
#         assert s > 0
#         self.s = s
#         self.weight = weight
#
#     def forward(self, x, target):
#         index = torch.zeros_like(x, dtype=torch.uint8)
#         index.scatter_(1, target.data.view(-1, 1), 1)
#
#         index_float = index.type(torch.cuda.FloatTensor)
#
#         batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
#         batch_m = batch_m.view((-1, 1))
#
#         x_m = x - batch_m
#
#         output = torch.where(index, x_m, x)
#
#         return F.cross_entropy(self.s * output, target, weight=self.weight)
#

class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list=None, max_m=0.5, s=30, reweight_epoch=-1):
        super().__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if cls_num_list is None:
            # No cls_num_list is provided, then we cannot adjust cross entropy with LDAM.
            self.m_list = None
        else:
            self.reweight_epoch = reweight_epoch
            m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
            m_list = m_list * (max_m / np.max(m_list[1:]))

            # m_list = m_list * (max_m / np.max(m_list))

            #m_list[0] = 0.0
            m_list = torch.tensor(m_list, dtype=torch.float, requires_grad=False)
            self.m_list = m_list
            self.m_list = self.m_list.to(device)
            # print('self.m_list:', len(self.m_list), self.m_list)

            assert s > 0
            self.s = s
            if reweight_epoch != -1:
                idx = 1  # condition could be put in order to set idx
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], cls_num_list)
                # print('effective_num:', effective_num)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                # print('per_cls_weights:',  per_cls_weights)

                per_cls_weights[0] = 0

                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
                # print('per_cls_weights:', per_cls_weights)

                self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
                self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)

            else:
                self.per_cls_weights_enabled = None
                self.per_cls_weights = None

    def to(self, device):
        super().to(device)
        if self.m_list is not None:
            self.m_list = self.m_list.to(device)

        if self.per_cls_weights_enabled is not None:
            self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)

        return self

    def _hook_before_epoch(self, epoch):
        if self.reweight_epoch != -1:
            self.epoch = epoch

            if epoch > self.reweight_epoch:
                self.per_cls_weights = self.per_cls_weights_enabled
            else:
                self.per_cls_weights = None

    def get_final_output(self, output_logits, target):
        x = output_logits       # [1140, 51], has value
        # print('x:', x.shape, x)

        index = torch.zeros_like(x, dtype=torch.uint8, device=x.device)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index_float = index.float()

        # print('index_float:', index_float.shape, index_float)   # [74, 151], okay
        # print('self.m_list:', self.m_list[None, :].shape, self.m_list[None, :]) # 1, 151

        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        # print('batch_m:', batch_m.shape, batch_m)       # [1, 74],  all nan

        batch_m = batch_m.view((-1, 1))
        # print('batch_m:', batch_m.shape, batch_m)

        x_m = x - batch_m * self.s
        # print('x_m:', x_m.shape, x_m)

        final_output = torch.where(index, x_m, x)
        # print('final_output:', final_output.shape, final_output)

        return final_output

    def forward(self, output_logits, target):
        if self.m_list is None:
            return F.cross_entropy(output_logits, target)

        final_output = self.get_final_output(output_logits, target)
        return F.cross_entropy(final_output, target, weight=self.per_cls_weights)


class LDAMLoss_PN(nn.Module):
    def __init__(self, config,
                 cls_num_list=None, max_m=0.5, s=30, reweight_epoch=-1):
        super().__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_experts = config.MODEL.ROI_RELATION_HEAD.NUM_EXPERTS

        if cls_num_list is None:
            # No cls_num_list is provided, then we cannot adjust cross entropy with LDAM.
            self.m_list = None
        else:
            self.reweight_epoch = reweight_epoch
            m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
            m_list = m_list * (max_m / np.max(m_list[1:]))

            # m_list = m_list * (max_m / np.max(m_list))

            #m_list[0] = 0.0
            m_list = torch.tensor(m_list, dtype=torch.float, requires_grad=False)
            self.m_list = m_list
            self.m_list = self.m_list.to(device)
            # print('self.m_list:', len(self.m_list), self.m_list)

            assert s > 0
            self.s = s
            if reweight_epoch != -1:
                idx = 1  # condition could be put in order to set idx
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], cls_num_list)
                # print('effective_num:', effective_num)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                # print('per_cls_weights:',  per_cls_weights)

                # per_cls_weights[0] = 15000

                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
                # print('per_cls_weights:', per_cls_weights)

                self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
                self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)

            else:
                self.per_cls_weights_enabled = None
                self.per_cls_weights = None

    def to(self, device):
        super().to(device)
        if self.m_list is not None:
            self.m_list = self.m_list.to(device)

        if self.per_cls_weights_enabled is not None:
            self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)

        return self

    def _hook_before_epoch(self, epoch):
        if self.reweight_epoch != -1:
            self.epoch = epoch

            if epoch > self.reweight_epoch:
                self.per_cls_weights = self.per_cls_weights_enabled
            else:
                self.per_cls_weights = None


    def get_final_output(self, output_logits, target):
        x = output_logits       # [1140, 51], has value
        # print('x:', x.shape, x)

        index = torch.zeros_like(x, dtype=torch.uint8, device=x.device)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index_float = index.float()

        # print('index_float:', index_float.shape, index_float)   # [74, 151], okay
        # print('self.m_list:', self.m_list[None, :].shape, self.m_list[None, :]) # 1, 151

        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        # print('batch_m:', batch_m.shape, batch_m)       # [1, 74],  all nan

        batch_m = batch_m.view((-1, 1))
        # print('batch_m:', batch_m.shape, batch_m)

        x_m = x - batch_m * self.s
        # print('x_m:', x_m.shape, x_m)

        final_output = torch.where(index, x_m, x)
        # print('final_output:', final_output.shape, final_output)

        return final_output

    def forward(self, output_logits, target, extra_info=None):
        loss = 0.0
        print('len(extra_info):', len(extra_info))
        for logits_item in extra_info:         
            for logit in logits_item:
                print('logit.shape:', logit.shape)
               
            logits_item = torch.cat(logits_item, dim=0)
            
            print('logits_item:', logits_item.shape)
            print('target shape:', target.shape)
                
            loss += F.cross_entropy(logits_item, target, weight=self.per_cls_weights)

        return loss / self.num_experts

    # def forward(self, output_logits, target):
    #     if self.m_list is None:
    #         return F.cross_entropy(output_logits, target)

    #     final_output = self.get_final_output(output_logits, target)
    #     return F.cross_entropy(final_output, target, weight=self.per_cls_weights)
    
class CAMELoss(nn.Module):
    def __init__(self, config,
                 cls_num_list=None, use_context_aware_gating=False, use_relation_sampling=False,
                 base_diversity_temperature=0.2, max_m=0.5, s=30, reweight=True,
                 # reweight_epoch=-1,
                 reweight_epoch=1,
                 base_loss_factor=1.0, additional_diversity_factor=-2.0, reweight_factor=0.05):
        super().__init__()
        self.base_loss = F.cross_entropy
        self.base_loss_factor = base_loss_factor
        # CAME
        self.use_context_aware_gating = use_context_aware_gating
        self.num_experts = config.MODEL.ROI_RELATION_HEAD.NUM_EXPERTS
        self.use_relation_sampling = config.MODEL.ROI_RELATION_HEAD.RELATION_SAMPLING
        self.m_list = []
        self.loss_option = config.MODEL.ROI_RELATION_HEAD.LOSS_OPTION

        if not reweight:
            self.reweight_epoch = -1
        else:
            self.reweight_epoch = reweight_epoch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # LDAM is a variant of cross entropy and we handle it with self.m_list.
        if cls_num_list is None:
            print('cls_num_list:', (cls_num_list))
        else:
            # cls_num_list[0] = 150000.0
            # cls_num_list[0] = 619237.0
            cls_num_list[0] = 33864.0
            # cls_num_list[0] = 0.0
            
            # cls_num_list[0] = 150000.0
            print('cls_num_list:', cls_num_list)
            print(len(cls_num_list))


        self.cls_num_lst = cls_num_list
        # self.per_cls_gaussian_weights = self.gaussian_reweight()

        if cls_num_list is None:
            # No cls_num_list is provided, then we cannot adjust cross entropy with LDAM.

            self.m_list = None
            self.per_cls_weights_enabled = None
            self.per_cls_weights_enabled_diversity = None
        else:
            # We will use LDAM loss if we provide cls_num_list.

            m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
            print('m_list:', m_list)
            m_list = m_list * (max_m / np.max(m_list[:]))
            m_list = torch.tensor(m_list, dtype=torch.float, requires_grad=False)
            self.m_list = m_list
            self.m_list = self.m_list.to(device)
            print('self.m_list:', len(self.m_list), self.m_list)

            self.s = s
            assert s > 0


            if self.reweight_epoch != -1:
                idx = 1  # condition could be put in order to set idx
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], cls_num_list)
                print('effective_num:', effective_num)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                print('per_cls_weights:', per_cls_weights)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
                print('per_cls_weights:', len(per_cls_weights), per_cls_weights)

                self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
                self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)

            else:
                self.per_cls_weights_enabled = None

            cls_num_list = np.array(cls_num_list) / np.sum(cls_num_list)
            C = len(cls_num_list)
            per_cls_weights = C * cls_num_list * reweight_factor + 1 - reweight_factor

            # Experimental normalization: This is for easier hyperparam tuning, the effect can be described in the learning rate so the math formulation keeps the same.
            # At the same time, the 1 - max trick that was previously used is not required since weights are already adjusted.
            per_cls_weights = per_cls_weights / np.max(per_cls_weights)

            assert np.all(per_cls_weights > 0), "reweight factor is too large: out of bounds"
            # save diversity per_cls_weights
            self.per_cls_weights_enabled_diversity = torch.tensor(per_cls_weights, dtype=torch.float,
                                                                  requires_grad=False).cuda()

            self.per_cls_weights_base = self.per_cls_weights_enabled
            self.per_cls_weights_diversity = self.per_cls_weights_enabled_diversity


        self.base_diversity_temperature = base_diversity_temperature
        self.additional_diversity_factor = additional_diversity_factor

    def hand_rw(self, cls_num_list):
        topk = 16
        cls_num = len(cls_num_list)
        head_inds = np.argsort(cls_num_list[0:cls_num])[::-1][:topk]
        body_inds = np.argsort(cls_num_list[0:cls_num])[::-1][topk:topk + 20]
        tail_inds = np.argsort(cls_num_list[0:cls_num])[::-1][topk + 20:]

        body_num = 16960.0
        tail_num = 2935.0

        if self.num_experts == 3:
            self.expert_body_per_cls_weights = []
            self.expert_tail_per_cls_weights = []
            for i in range(cls_num):
                if i in head_inds:
                    self.expert_body_per_cls_weights.append(self.per_cls_weights_base[i] * 0.0)
                    self.expert_tail_per_cls_weights.append(self.per_cls_weights_base[i] * 0.0)
                elif i in body_inds:
                    self.expert_body_per_cls_weights.append(self.per_cls_weights_base[i] * body_num/tail_num)
                    self.expert_tail_per_cls_weights.append(self.per_cls_weights_base[i] / 3.0)
                elif i in tail_inds:
                    self.expert_body_per_cls_weights.append(self.per_cls_weights_base[i] * tail_num/body_num)
                    self.expert_tail_per_cls_weights.append(self.per_cls_weights_base[i] * 3.0)


            self.expert_body_per_cls_weights = torch.tensor(self.expert_body_per_cls_weights, dtype=torch.float, requires_grad=False).cuda()
            self.expert_tail_per_cls_weights = torch.tensor(self.expert_tail_per_cls_weights, dtype=torch.float, requires_grad=False).cuda()
            print('expert_body_per_cls_weights:', self.expert_body_per_cls_weights)
            print('expert_tail_per_cls_weights:', self.expert_tail_per_cls_weights)


    def gaussian_reweight(self):
        topk = 16
        num_len = len(self.cls_num_lst)

        ranked_inds = np.argsort(self.cls_num_lst[0:num_len])[::-1]
        head_inds = np.argsort(self.cls_num_lst[0:num_len])[::-1][:topk]
        body_inds = np.argsort(self.cls_num_lst[0:num_len])[::-1][topk:topk + 20]
        tail_inds = np.argsort(self.cls_num_lst[0:num_len])[::-1][topk + 20:]

        # head = np.array(self.cls_label_lst[0:num_len])[head_inds.tolist()]
        # body = np.array(self.cls_label_lst[0:num_len])[body_inds.tolist()]
        # tail = np.array(self.cls_label_lst[0:num_len])[tail_inds.tolist()]
        # label_lst = []
        # label_lst.extend(head)
        # label_lst.extend(body)
        # label_lst.extend(tail)

        head_value_list = np.array(self.cls_num_lst[0:num_len])[head_inds.tolist()]
        body_value_list = np.array(self.cls_num_lst[0:num_len])[body_inds.tolist()]
        tail_value_list = np.array(self.cls_num_lst[0:num_len])[tail_inds.tolist()]


        if self.m_list is None:
            self.m_list = None
            self.per_cls_weights_enabled = None
            self.per_cls_weights_enabled_diversity = None
        else:
            first_weight = 0.01
            head_gaussian_lst, _ = self.gaussian_transform(head_value_list, first_weight)
            first_weight = np.max(head_gaussian_lst)
            body_gaussian_lst, _ = self.gaussian_transform(body_value_list, first_weight)
            first_weight = np.max(body_gaussian_lst)
            tail_gaussian_lst, _ = self.gaussian_transform(tail_value_list, first_weight)

            # head_sum = np.sum(head_value_list)
            # body_sum = np.sum(body_value_list)
            # tail_sum = np.sum(tail_value_list)

            #importance_ratio = head_sum / body_sum
            #print('importance_ratio:', importance_ratio)

            #importance_ratio = body_sum / tail_sum
            #print('importance_ratio:', importance_ratio)

            gaussian_lst = []
            gaussian_lst.extend(head_gaussian_lst)
            gaussian_lst.extend(body_gaussian_lst)
            gaussian_lst.extend(tail_gaussian_lst)
            #print('gaussian_lst:', len(gaussian_lst), gaussian_lst)

            per_cls_gaussian_weights = np.zeros([len(self.cls_num_lst)])
            for idx in range(num_len):
                per_cls_gaussian_weights[ranked_inds[idx]] = gaussian_lst[idx]

            print('per_cls_gaussian_weights:', per_cls_gaussian_weights)
            per_cls_gaussian_weights = torch.tensor(per_cls_gaussian_weights, dtype=torch.float, requires_grad=False).cuda()
            print('per_cls_gaussian_weights:', per_cls_gaussian_weights)

            return per_cls_gaussian_weights

    def gaussian_transform(self, value_list, first_weight):
        value_prob_list = 1.0 / np.array(value_list)
        # value_prob_list =  np.array(value_list)

        # print('value_prob_list:', value_prob_list)
        gaussian_lst, lmbda = stats.boxcox(value_prob_list)
        # print('gaussian_lst:', gaussian_lst)
        std = np.std(gaussian_lst)
        mean = np.mean(gaussian_lst)
        ratio = np.max(gaussian_lst) / np.min(gaussian_lst)
        # print('std:', std, 'mean:', mean, 'ratio:', ratio)

        # normalize gaussian list

        gaussian_lst = (gaussian_lst - np.min(gaussian_lst) + np.std(gaussian_lst))  + first_weight
        ratio = np.max(gaussian_lst) / np.min(gaussian_lst)
        print('gaussian_lst:', gaussian_lst)
        print('mean:', np.mean(gaussian_lst), 'std:', np.std(gaussian_lst), 'ratio:', ratio)

        return gaussian_lst, lmbda

    def to(self, device):
        super().to(device)
        if self.m_list is not None:
            self.m_list = self.m_list.to(device)

        if self.per_cls_weights_enabled is not None:
            self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)

        if self.per_cls_weights_enabled_diversity is not None:
            self.per_cls_weights_enabled_diversity = self.per_cls_weights_enabled_diversity.to(device)

        return self

    def _hook_before_epoch(self, epoch):
        if self.reweight_epoch != -1:
            self.epoch = epoch

            if epoch > self.reweight_epoch:
                self.per_cls_weights_base = self.per_cls_weights_enabled
                self.per_cls_weights_diversity = self.per_cls_weights_enabled_diversity
            else:
                self.per_cls_weights_base = None
                self.per_cls_weights_diversity = None

    def get_final_output(self, output_logits, target):
        x = output_logits

        x = torch.cat(x, dim=0)
        # index = torch.zeros_like(x, dtype=torch.uint8, device=x.device)

        index = torch.zeros_like(x, dtype=torch.uint8, device=x.device)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index_float = index.float()
        # print('index_float:', index_float.shape)
        a = index_float.transpose(0, 1)
        # print('a.shape:', a.shape)
        # print('self.m_list[None, :]:', self.m_list[None, :], len(self.m_list[None, :]))
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        # print('batch_m:', batch_m.shape)

        batch_m = batch_m.view((-1, 1))
        # print('batch_m2:', batch_m.shape)

        x_m = x - batch_m * self.s
        # print('x_m:', x_m.shape)

        final_output = torch.where(index, x_m, x)

        return final_output

    def kl_loss(self, extra_info, logits_item):
        logits_lst = []
        for logits in extra_info:
            logits = torch.cat(logits, dim=0)
            logits_lst.append(logits)

        logits_lst_torch = torch.stack(logits_lst, dim=0)
        logits_mean = torch.mean(logits_lst_torch, dim=0)

        output_dist = F.log_softmax(logits_item, dim=1)
        with torch.no_grad():
            # Using the mean takes only linear instead of quadratic time in computing and has only a slight difference so using the mean is preferred here
            mean_output_dist = F.softmax(logits_mean, dim=1)

        kl_loss = self.additional_diversity_factor * F.kl_div(output_dist,
                                                              mean_output_dist,
                                                              reduction='batchmean')
        kl_loss = kl_loss / logits_mean.shape[0]
        # print('kl_loss:', kl_loss)

        return kl_loss


    def forward(self, output_logits, target, beta_relation_aware_gating=None, extra_info=None, extra_labels=None):
        if self.loss_option == 'CB_LOSS':
            return self.base_loss(output_logits, target, weight=self.per_cls_weights_base)
        elif extra_info is None:
            return self.base_loss(output_logits, target)

        # Adding CAME Individual Loss for each expert
        # self.num_experts = len(extra_info)
        loss = 0
        expert_num = 0

        # print('self.m_list:', self.m_list)
        for logits_item in extra_info:
            
            if self.m_list is None:
                # print('logits_item:', logits_item, len(logits_item), logits_item[0].shape, logits_item[1].shape)
                logits_item = torch.cat(logits_item, dim=0)
                # print('logits_item:', logits_item.shape)

                # print('target:', target.shape)
               

                loss += self.base_loss_factor * self.base_loss(logits_item, target)

            else:
                if not self.use_relation_sampling:
                    # print('logits_item:', logits_item[0].shape)
                    logits_item = self.get_final_output(logits_item, target)
                    # print('logits_item final:', logits_item.shape)

                if self.use_relation_sampling:
                    if self.use_context_aware_gating:
                        # if self.num_experts == 3:
                        #     if expert_num == 0:
                        #         loss += self.base_loss_factor * self.base_loss(logits_item, extra_labels[expert_num], weight=self.per_cls_weights_base) * beta_relation_aware_gating[expert_num]
                        #     elif expert_num == 1:
                        #         loss += self.base_loss_factor * self.base_loss(logits_item, extra_labels[expert_num], weight=self.expert_body_per_cls_weights) * beta_relation_aware_gating[expert_num]
                        #     elif expert_num == 2:
                        #         loss += self.base_loss_factor * self.base_loss(logits_item, extra_labels[expert_num], weight=self.expert_tail_per_cls_weights) * beta_relation_aware_gating[expert_num]
                        #
                        # else:
                        loss += self.base_loss_factor * self.base_loss(logits_item, extra_labels[expert_num],
                                                                       weight=self.per_cls_weights_base) * \
                                beta_relation_aware_gating[expert_num]

                    else:
                        loss += self.base_loss_factor * self.base_loss(logits_item, extra_labels[expert_num],
                                                                       weight=self.per_cls_weights_base)

                # no relation sampling
                else:
                    if self.use_context_aware_gating:
                        # if self.num_experts == 3:
                        #     if expert_num == 0:
                        #         loss += self.base_loss_factor * self.base_loss(logits_item, target) * beta_relation_aware_gating[expert_num]
                        #     elif expert_num == 1:
                        #         loss += self.base_loss_factor * self.base_loss(logits_item, target, weight=self.expert_body_per_cls_weights) * beta_relation_aware_gating[expert_num]
                        #     elif expert_num == 2:
                        #         loss += self.base_loss_factor * self.base_loss(logits_item, target, weight=self.expert_tail_per_cls_weights) * beta_relation_aware_gating[expert_num]
                        # else:
                        #     loss += self.base_loss_factor * self.base_loss(logits_item, target,
                        #                                                    weight=self.per_cls_weights_base) * beta_relation_aware_gating[expert_num]

                        loss += self.base_loss_factor * self.base_loss(logits_item,
                                                                       target,
                                                                       weight=self.per_cls_weights_base) * beta_relation_aware_gating[expert_num]

                        # loss += self.base_loss_factor * self.base_loss(logits_item,
                        #                                                target,
                        #                                                weight=self.per_cls_gaussian_weights) * \
                        #         beta_relation_aware_gating[expert_num]
                        # print('loss:', loss)

                    else:
                        # CB Loss
                        # loss += self.base_loss_factor * self.base_loss(logits_item,
                        #                                                target,
                        #                                                weight=self.per_cls_weights_base)
                        # Pure CE Loss
                        loss += self.base_loss_factor * self.base_loss(logits_item,
                                                                        target)

            expert_num = expert_num + 1

            # base_diversity_temperature = self.base_diversity_temperature

            # if self.per_cls_weights_diversity is not None:
            #     diversity_temperature = base_diversity_temperature * self.per_cls_weights_diversity.view((1, -1))
            #     # print('per_cls_weights_diversity:', self.per_cls_weights_diversity.view((1, -1)))   # [51x1]
            #     # print('diversity_temperature:', diversity_temperature)  # [51x1]
            #     temperature_mean = diversity_temperature.mean().item()
            #     # print('temperature_mean:', temperature_mean)            # 0.58
            # else:
            #     diversity_temperature = base_diversity_temperature
            #     temperature_mean = base_diversity_temperature


            # print('diversity_temperature:', diversity_temperature)
            # print('output_dist.shape:', output_dist.shape)




        # loss = loss / (self.num_experts * self.batch_size)

        self.batch_size = len(extra_info[0])
        if self.use_context_aware_gating:
            return loss / self.batch_size
            # return loss
        else:
            return loss / (self.num_experts * self.batch_size)

        return loss


class PLMELoss(nn.Module):
    def __init__(self, config,
                 cls_num_list=None, use_context_aware_gating=False, use_relation_sampling=False,
                 base_diversity_temperature=0.2, max_m=0.5, s=30, reweight=True,
                 # reweight_epoch=-1,
                 reweight_epoch=1,
                 base_loss_factor=1.0, additional_diversity_factor=-2.0, reweight_factor=0.05):
        super().__init__()
        self.base_loss = F.cross_entropy
        self.base_loss_factor = base_loss_factor
        self.expert_mode = config.MODEL.ROI_RELATION_HEAD.EXPERT_MODE

        if not reweight:
            self.reweight_epoch = -1
        else:
            self.reweight_epoch = reweight_epoch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # LDAM is a variant of cross entropy and we handle it with self.m_list.
        print('cls_num_lst:', cls_num_list)
        cls_num_list[0] = 200000.0

        self.label_grouping = label_grouping(config)
        self.label_group_dic = self.label_grouping.obtain_group_labels()
        print('self.label_group_dic:', self.label_group_dic)

        if cls_num_list is None:
            # No cls_num_list is provided, then we cannot adjust cross entropy with LDAM.

            self.m_list = None
            self.per_cls_weights_enabled = None
            self.per_cls_weights_enabled_diversity = None
        else:
            # We will use LDAM loss if we provide cls_num_list.

            m_list  = 1.0 / np.sqrt(np.sqrt(cls_num_list))
            print('m_list:', m_list)
            # m_list = m_list * (max_m / np.max(m_list[1:]))
            # m_list[0] = 0.0

            m_list = m_list * (max_m / np.max(m_list))
            m_list = torch.tensor(m_list, dtype=torch.float, requires_grad=False)

            self.m_list = m_list
            self.m_list = self.m_list.to(device)
            print('self.m_list:', len(self.m_list), self.m_list)

            self.s = s
            assert s > 0

            if reweight_epoch != -1:
                idx = 1  # condition could be put in order to set idx
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], cls_num_list)
                print('effective_num:', effective_num)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                print('per_cls_weights:', per_cls_weights)

                # per_cls_weights[0] = 0

                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
                print('per_cls_weights:', len(per_cls_weights), per_cls_weights)

                self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
                self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)
            else:
                self.per_cls_weights_enabled = None

            cls_num_list = np.array(cls_num_list) / np.sum(cls_num_list)
            C = len(cls_num_list)
            per_cls_weights = C * cls_num_list * reweight_factor + 1 - reweight_factor

            # Experimental normalization: This is for easier hyperparam tuning, the effect can be described in the learning rate so the math formulation keeps the same.
            # At the same time, the 1 - max trick that was previously used is not required since weights are already adjusted.
            per_cls_weights = per_cls_weights / np.max(per_cls_weights)

            assert np.all(per_cls_weights > 0), "reweight factor is too large: out of bounds"
            # save diversity per_cls_weights
            self.per_cls_weights_enabled_diversity = torch.tensor(per_cls_weights, dtype=torch.float,
                                                                  requires_grad=False).cuda()

            self.per_cls_weights_base = self.per_cls_weights_enabled
            self.per_cls_weights_diversity = self.per_cls_weights_enabled_diversity

        self.base_diversity_temperature = base_diversity_temperature
        self.additional_diversity_factor = additional_diversity_factor

        # CAME
        self.use_context_aware_gating = use_context_aware_gating
        self.use_relation_sampling = config.MODEL.ROI_RELATION_HEAD.RELATION_SAMPLING

    def to(self, device):
        super().to(device)
        if self.m_list is not None:
            self.m_list = self.m_list.to(device)

        if self.per_cls_weights_enabled is not None:
            self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)

        if self.per_cls_weights_enabled_diversity is not None:
            self.per_cls_weights_enabled_diversity = self.per_cls_weights_enabled_diversity.to(device)

        return self

    def _hook_before_epoch(self, epoch):
        if self.reweight_epoch != -1:
            self.epoch = epoch

            if epoch > self.reweight_epoch:
                self.per_cls_weights_base = self.per_cls_weights_enabled
                self.per_cls_weights_diversity = self.per_cls_weights_enabled_diversity
            else:
                self.per_cls_weights_base = None
                self.per_cls_weights_diversity = None

    def get_final_output(self, output_logits, target):
        x = output_logits

        x = torch.cat(x, dim=0)
        # index = torch.zeros_like(x, dtype=torch.uint8, device=x.device)

        index = torch.zeros_like(x, dtype=torch.uint8, device=x.device)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index_float = index.float()
        # print('index_float:', index_float.shape)
        a = index_float.transpose(0, 1)
        # print('a.shape:', a.shape)
        # print('self.m_list[None, :]:', self.m_list[None, :], len(self.m_list[None, :]))
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        # print('batch_m:', batch_m.shape)

        batch_m = batch_m.view((-1, 1))
        # print('batch_m2:', batch_m.shape)

        x_m = x - batch_m * self.s
        # print('x_m:', x_m.shape)

        final_output = torch.where(index, x_m, x)

        return final_output

    def label_assignment(self, target):
        target_h = []
        target_m = []
        target_t = []
        # expert_mode = 'hmt_mt_t'     # 'h_m_t', 'hmt_m_t', 'hmt_mt_t'
        # expert_mode = 'hmt_m_t'
        # expert_mode = 'h_m_t'
        # expert_mode = 'hm_ht_mt'
        # expert_mode = 'hmt_ht_mt'

        if self.num_experts == 2:
            if self.expert_mode == 'hbt_b' or self.expert_mode == 'hbt_t':
                for i in range(target.shape[0]):
                    label = target[i]

                    # print('label:', label.item())
                    if self.label_group_dic[label.item()] == 'head':
                        target_h.append(label.item())
                        target_m.append(0)
                        target_t.append(0)

                    elif self.label_group_dic[label.item()] == 'body':
                        target_h.append(0)
                        target_m.append(label.item())
                        target_t.append(0)

                    elif self.label_group_dic[label.item()] == 'tail':
                        target_h.append(0)
                        target_m.append(0)
                        target_t.append(label.item())

                if self.expert_mode == 'hbt_t':
                    return target, target_t

                elif self.expert_mode == 'hbt_b':
                    return target, target_m

            elif self.expert_mode == 'hbt_ht' or self.expert_mode == 'hbt_bt' or self.expert_mode == 'ht_bt':
                for i in range(target.shape[0]):
                    label = target[i]

                    if self.label_group_dic[label.item()] == 'head' or self.label_group_dic[label.item()] == 'tail':    # ht
                        target_m.append(label.item())
                    else:
                        target_m.append(0)

                    if self.label_group_dic[label.item()] == 'body' or self.label_group_dic[label.item()] == 'tail':    # bt
                        target_t.append(label.item())

                    else:
                        target_t.append(0)

                if self.expert_mode == 'hbt_ht':
                    return target, target_m

                elif self.expert_mode == 'hbt_bt':
                    return target, target_t

                elif self.expert_mode == 'ht_bt':
                    return target_m, target_t


        if self.num_experts == 3:
            if self.expert_mode == 'h_b_t' or self.expert_mode == 'hbt_b_t':
                for i in range(target.shape[0]):
                    label = target[i]

                    # print('label:', label.item())
                    if self.label_group_dic[label.item()] == 'head':
                        target_h.append(label.item())
                        target_m.append(0)
                        target_t.append(0)

                    elif self.label_group_dic[label.item()] == 'body':
                        target_h.append(0)
                        target_m.append(label.item())
                        target_t.append(0)

                    elif self.label_group_dic[label.item()] == 'tail':
                        target_h.append(0)
                        target_m.append(0)
                        target_t.append(label.item())

                if self.expert_mode == 'h_b_t':
                    return target_h, target_m, target_t
                else:
                    return target, target_m, target_t

            elif self.expert_mode == 'hbt_bt_t':
                for i in range(target.shape[0]):
                    label = target[i]

                    if self.label_group_dic[label.item()] == 'body' or self.label_group_dic[label.item()] == 'tail':
                        target_m.append(label.item())
                    else:
                        target_m.append(0)

                    if self.label_group_dic[label.item()] == 'tail':
                        target_t.append(label.item())

                    else:
                        target_t.append(0)
                return target, target_m, target_t


            elif self.expert_mode == 'hb_ht_bt' or self.expert_mode == 'hbt_ht_bt':
                for i in range(target.shape[0]):
                    label = target[i]

                    if self.label_group_dic[label.item()] == 'head' or self.label_group_dic[label.item()] == 'body':
                        target_h.append(label.item())
                    else:
                        target_h.append(0)

                    if self.label_group_dic[label.item()] == 'head' or self.label_group_dic[label.item()] == 'tail':
                        target_m.append(label.item())

                    else:
                        target_m.append(0)

                    if self.label_group_dic[label.item()] == 'body' or self.label_group_dic[label.item()] == 'tail':
                        target_t.append(label.item())

                    else:
                        target_t.append(0)

                if self.expert_mode == 'hb_ht_bt':
                    return target_h, target_m, target_t
                elif self.expert_mode == 'hbt_ht_bt':
                    return target, target_m, target_t

        elif self.num_experts == 4:
            if self.expert_mode == 'hbt_h_b_t':
                for i in range(target.shape[0]):
                    label = target[i]

                    # print('label:', label.item())
                    if self.label_group_dic[label.item()] == 'head':
                        target_h.append(label.item())
                        target_m.append(0)
                        target_t.append(0)

                    elif self.label_group_dic[label.item()] == 'body':
                        target_h.append(0)
                        target_m.append(label.item())
                        target_t.append(0)

                    elif self.label_group_dic[label.item()] == 'tail':
                        target_h.append(0)
                        target_m.append(0)
                        target_t.append(label.item())

                return target, target_h, target_m, target_t

            elif self.expert_mode == 'hbt_hb_ht_bt':
                for i in range(target.shape[0]):
                    label = target[i]
                    if self.label_group_dic[label.item()] == 'head' or self.label_group_dic[label.item()] == 'body':
                        target_h.append(label.item())
                    else:
                        target_h.append(0)

                    if self.label_group_dic[label.item()] == 'head' or self.label_group_dic[label.item()] == 'tail':
                        target_m.append(label.item())

                    else:
                        target_m.append(0)

                    if self.label_group_dic[label.item()] == 'body' or self.label_group_dic[label.item()] == 'tail':
                        target_t.append(label.item())

                    else:
                        target_t.append(0)

                return target, target_h, target_m, target_t



    def forward(self, output_logits, target, beta_relation_aware_gating=None, extra_info=None, extra_labels=None):

        if extra_info is None:
            return self.base_loss(output_logits, target)

        # Adding CAME Individual Loss for each expert
        self.num_experts = len(extra_info)
        loss = 0
        expert_num = 0

        logits_lst = []
        for logits_item in extra_info:
            logits_item = torch.cat(logits_item, dim=0)

            logits_lst.append(logits_item)

        logits_lst_torch = torch.stack(logits_lst, dim=0)
        logits_mean = torch.mean(logits_lst_torch, dim=0)
        # print('target:', type(target), target.shape)


        if self.num_experts == 3:
            target_h, target_m, target_t = self.label_assignment(target)
            target_h = torch.tensor(target_h).cuda()
            target_m = torch.tensor(target_m).cuda()
            target_t = torch.tensor(target_t).cuda()

        elif self.num_experts == 2:
            target_h, target_m = self.label_assignment(target)
            target_h = torch.tensor(target_h).cuda()
            target_m = torch.tensor(target_m).cuda()

        elif self.num_experts == 4:
            target_h, target_m, target_t, target_l = self.label_assignment(target)
            target_h = torch.tensor(target_h).cuda()
            target_m = torch.tensor(target_m).cuda()
            target_t = torch.tensor(target_t).cuda()
            target_l = torch.tensor(target_l).cuda()

        # target_h = torch.cat(target_h, dim=0)
        # target_m = torch.cat(target_m, dim=0)
        # target_t = torch.cat(target_t, dim=0)

        # print('target_h:', len(target_h), target_h)
        # print('target_m:', len(target_m), target_m)
        # print('target_t:', len(target_t), target_t)


        # print('target_h:', type(target), target_h.shape)
        # print('target_m:', type(target), target_m.shape)
        # print('target_t:', type(target), target_t.shape)


        for logits_item in extra_info:

            if self.m_list is None:
                loss += self.base_loss_factor * self.base_loss(logits_item, target)

            else:
                if not self.use_relation_sampling:
                    # print('logits_item:', logits_item[0].shape)
                    logits_item = self.get_final_output(logits_item, target)
                    # print('logits_item final:', logits_item.shape)


                # print('len of extra_labels[expert_num]:', len(extra_labels[expert_num]), extra_labels[expert_num])
                # print('len of logits_item:', len(logits_item), len(logits_item[0]), logits_item)

                if self.use_relation_sampling:
                    if self.use_context_aware_gating:
                        loss += self.base_loss_factor * self.base_loss(logits_item, extra_labels[expert_num],
                                                                       weight=self.per_cls_weights_base) * beta_relation_aware_gating[expert_num]

                    else:
                        loss += self.base_loss_factor * self.base_loss(logits_item, extra_labels[expert_num],
                                                                       weight=self.per_cls_weights_base)
                else:
                    if self.use_context_aware_gating:
                        # print('logits_item.shape:', logits_item.shape)
                        # print('target.shape:', target.shape)
                        if (expert_num == 0):
                            target = target_h
                        elif(expert_num == 1):
                            target = target_m
                        elif(expert_num == 2):
                            target = target_t

                        loss += self.base_loss_factor * self.base_loss(logits_item, target,
                                                                       weight=self.per_cls_weights_base) * beta_relation_aware_gating[expert_num]


                    else:
                        if self.num_experts == 3:
                            if (expert_num == 0):
                                target = target_h
                                self.base_loss_factor = 2.0

                            elif (expert_num == 1):
                                target = target_m
                                self.base_loss_factor = 1.0

                            elif (expert_num == 2):
                                target = target_t
                                self.base_loss_factor = 1.0

                        elif self.num_experts == 2:
                            if (expert_num == 0):
                                target = target_h
                                self.base_loss_factor = 1.0

                            elif (expert_num == 1):
                                target = target_m
                                self.base_loss_factor = 1.0

                        elif self.num_experts == 4:
                            if (expert_num == 0):
                                target = target_h
                                self.base_loss_factor = 2.0

                            elif (expert_num == 1):
                                target = target_m
                                self.base_loss_factor = 1.0

                            elif (expert_num == 2):
                                target = target_t
                                self.base_loss_factor = 1.0

                            elif (expert_num == 3):
                                target = target_l
                                self.base_loss_factor = 1.0

                        loss += self.base_loss_factor * self.base_loss(logits_item, target,
                                                                       weight=self.per_cls_weights_base)

            expert_num = expert_num + 1

            base_diversity_temperature = self.base_diversity_temperature

            if self.per_cls_weights_diversity is not None:
                diversity_temperature = base_diversity_temperature * self.per_cls_weights_diversity.view((1, -1))
                # print('per_cls_weights_diversity:', self.per_cls_weights_diversity.view((1, -1)))   # [51x1]
                # print('diversity_temperature:', diversity_temperature)  # [51x1]
                temperature_mean = diversity_temperature.mean().item()
                # print('temperature_mean:', temperature_mean)            # 0.58
            else:
                diversity_temperature = base_diversity_temperature
                temperature_mean = base_diversity_temperature

            # print('diversity_temperature:', diversity_temperature)
            output_dist = F.log_softmax(logits_item, dim=1)
            # print('output_dist.shape:', output_dist.shape)

            with torch.no_grad():
                # Using the mean takes only linear instead of quadratic time in computing and has only a slight difference so using the mean is preferred here
                mean_output_dist = F.softmax(logits_mean, dim=1)
                # print('mean_output_dist:', mean_output_dist.shape)

            # print('temperature_mean:', temperature_mean)
            # kl_loss = self.additional_diversity_factor * F.kl_div(output_dist,
            #                                                     mean_output_dist,
            #                                                     reduction='batchmean')
            #
            # kl_loss = kl_loss / logits_mean.shape[0]
            # print('kl_loss:', kl_loss)
            # loss += kl_loss

            # print('logits_mean.shape:', logits_mean)
            # print('kl_loss:', kl_loss) # ~-0.19

            # kl_loss = self.additional_diversity_factor * F.kl_div(output_dist,
            #                                                       mean_output_dist,
            #                                                       reduction='batchmean')



        # loss = loss / (self.num_experts * self.batch_size)

        self.batch_size = len(extra_info[0])
        if self.use_context_aware_gating:
            return loss / self.batch_size
            # return loss
        else:
            return loss / (self.num_experts * self.batch_size)

        return loss


class RIDELoss(nn.Module):
    def __init__(self, cls_num_list=None, use_context_aware_gating=False, use_relation_sampling=False,
                 base_diversity_temperature=0.2, max_m=0.5, s=30, reweight=True,
                 # reweight_epoch=-1,
                 reweight_epoch=1,
                 base_loss_factor=1.0, additional_diversity_factor=-0.2, reweight_factor=0.05):
        super().__init__()
        self.base_loss = F.cross_entropy
        self.base_loss_factor = base_loss_factor
        if not reweight:
            self.reweight_epoch = -1
        else:
            self.reweight_epoch = reweight_epoch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # LDAM is a variant of cross entropy and we handle it with self.m_list.
        print('cls_num_lst:', cls_num_list)
        cls_num_list[0] = 200000.0

        if cls_num_list is None:
            # No cls_num_list is provided, then we cannot adjust cross entropy with LDAM.

            self.m_list = None
            self.per_cls_weights_enabled = None
            self.per_cls_weights_enabled_diversity = None
        else:
            # We will use LDAM loss if we provide cls_num_list.

            m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
            print('m_list:', m_list)
            m_list = m_list * (max_m / np.max(m_list[1:]))
            m_list[0] = 0.0
            m_list = torch.tensor(m_list, dtype=torch.float, requires_grad=False)
            self.m_list = m_list
            self.m_list = self.m_list.to(device)
            print('self.m_list:', len(self.m_list), self.m_list)

            self.s = s
            assert s > 0

            if reweight_epoch != -1:
                idx = 1  # condition could be put in order to set idx
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], cls_num_list)
                print('effective_num:',  effective_num)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                print('per_cls_weights:',  per_cls_weights)
                #per_cls_weights[0] = 0
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
                print('per_cls_weights:', len(per_cls_weights), per_cls_weights)

                self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
                self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)
            else:
                self.per_cls_weights_enabled = None

            cls_num_list = np.array(cls_num_list) / np.sum(cls_num_list)
            C = len(cls_num_list)
            per_cls_weights = C * cls_num_list * reweight_factor + 1 - reweight_factor

            # Experimental normalization: This is for easier hyperparam tuning, the effect can be described in the learning rate so the math formulation keeps the same.
            # At the same time, the 1 - max trick that was previously used is not required since weights are already adjusted.
            per_cls_weights = per_cls_weights / np.max(per_cls_weights)

            assert np.all(per_cls_weights > 0), "reweight factor is too large: out of bounds"
            # save diversity per_cls_weights
            self.per_cls_weights_enabled_diversity = torch.tensor(per_cls_weights, dtype=torch.float,
                                                                  requires_grad=False).cuda()

            self.per_cls_weights_base = self.per_cls_weights_enabled
            self.per_cls_weights_diversity = self.per_cls_weights_enabled_diversity

        self.base_diversity_temperature = base_diversity_temperature
        self.additional_diversity_factor = additional_diversity_factor
        self.use_relation_aware_gating = use_context_aware_gating
        self.use_relation_sampling = use_relation_sampling

    def to(self, device):
        super().to(device)
        if self.m_list is not None:
            self.m_list = self.m_list.to(device)

        if self.per_cls_weights_enabled is not None:
            self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)

        if self.per_cls_weights_enabled_diversity is not None:
            self.per_cls_weights_enabled_diversity = self.per_cls_weights_enabled_diversity.to(device)

        return self

    def _hook_before_epoch(self, epoch):
        if self.reweight_epoch != -1:
            self.epoch = epoch

            if epoch > self.reweight_epoch:
                self.per_cls_weights_base = self.per_cls_weights_enabled
                self.per_cls_weights_diversity = self.per_cls_weights_enabled_diversity
            else:
                self.per_cls_weights_base = None
                self.per_cls_weights_diversity = None

    def get_final_output(self, output_logits, target):
        x = output_logits
        # for element in x:
        #     print('element:', element.shape)
        x = torch.cat(x, dim=0)
        #print('x:', x.shape)
        #print('target:', target.shape, target)
        # index = torch.zeros_like(x, dtype=torch.uint8, device=x.device)
        index = torch.zeros_like(x, dtype=torch.uint8, device=x.device)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index_float = index.float()
        #print('index_float:', index_float.shape)
        a = index_float.transpose(0, 1)
        #print('a.shape:', a.shape)
        #print('self.m_list[None, :]:', self.m_list[None, :], len(self.m_list[None, :]))
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        #print('batch_m:', batch_m.shape)

        batch_m = batch_m.view((-1, 1))
        #print('batch_m2:', batch_m.shape)

        x_m = x - batch_m * self.s
        #print('x_m:', x_m.shape)

        final_output = torch.where(index, x_m, x)
        #print('final_output:', final_output.shape, final_output)

        return final_output

    def forward(self, output_logits, target, beta_relation_aware_gating=None, extra_info=None, extra_labels=None):
        # print('start ride loss')
        # print('output_logits:', len(output_logits), output_logits[0].shape)

        # for output_element in output_logits:
        #     print('output_element:', output_element.shape)
        # for extra in extra_info:
        #     print('extra_element:',  extra.shape)

        # print('target:', target.shape)

        if extra_info is None:
            return self.base_loss(output_logits, target)

        loss = 0
        # print('extra_info:', extra_info[0][0].shape)
        # print('extra_info:', extra_info[0][1].shape)
        # print('extra_info:', extra_info[1][0].shape)
        # print('extra_info:', extra_info[1][1].shape)

        # Adding RIDE Individual Loss for each expert
        expert_num = 0
        self.num_experts = len(extra_info)

        for logits_item in extra_info:
            ride_loss_logits = output_logits if self.additional_diversity_factor == 0 else logits_item

            if self.m_list is None:
                loss += self.base_loss_factor * self.base_loss(ride_loss_logits, target)
            else:
                if self.use_relation_sampling:
                    final_output = self.get_final_output(ride_loss_logits, extra_labels[expert_num])

                else:
                    final_output = self.get_final_output(ride_loss_logits, target)


                if self.use_relation_sampling:
                    if self.use_relation_aware_gating:
                        loss += self.base_loss_factor * self.base_loss(final_output, extra_labels[expert_num],
                                                                       weight=self.per_cls_weights_base) * \
                                beta_relation_aware_gating[expert_num]
                        # loss += self.base_loss_factor * self.base_loss(final_output, target, weight=self.per_cls_weights_base)
                        # expert_num = expert_num + 1
                    else:
                        loss += self.base_loss_factor * self.base_loss(final_output, extra_labels[expert_num],
                                                                       weight=self.per_cls_weights_base)
                else:
                    if self.use_relation_aware_gating:
                        loss += self.base_loss_factor * self.base_loss(final_output, target, weight=self.per_cls_weights_base) * beta_relation_aware_gating[expert_num]
                        # loss += self.base_loss_factor * self.base_loss(final_output, target, weight=self.per_cls_weights_base)
                        # expert_num = expert_num + 1
                    else:
                        loss += self.base_loss_factor * self.base_loss(final_output, target, weight=self.per_cls_weights_base)

            expert_num = expert_num + 1

            base_diversity_temperature = self.base_diversity_temperature

            if self.per_cls_weights_diversity is not None:
                diversity_temperature = base_diversity_temperature * self.per_cls_weights_diversity.view((1, -1))
                # print('per_cls_weights_diversity:', self.per_cls_weights_diversity.view((1, -1)))   # [51x1]
                # print('diversity_temperature:', diversity_temperature)  # [51x1]
                temperature_mean = diversity_temperature.mean().item()
                # print('temperature_mean:', temperature_mean)            # 0.58
            else:
                diversity_temperature = base_diversity_temperature
                temperature_mean = base_diversity_temperature

            #print('logits_item:', logits_item)
            # for element in logits_item:
                # print('element.shape:', element.shape)
            # print('diversity_temperature:', diversity_temperature.shape)
            logits_item = torch.cat(logits_item, dim=0)
            #print('logits_item:', logits_item.shape)

            output_dist = F.log_softmax(logits_item / diversity_temperature, dim=1)

            # print('output_dist:', output_dist)


            with torch.no_grad():
                # Using the mean takes only linear instead of quadratic time in computing and has only a slight difference so using the mean is preferred here
                mean_output_dist = F.softmax(output_logits / diversity_temperature, dim=1)

            # print('self.additional_diversity_factor:', self.additional_diversity_factor)
            # print('temperature_mean:', temperature_mean)
            kl_loss = self.additional_diversity_factor * temperature_mean * temperature_mean * F.kl_div(output_dist,
                                                                                                      mean_output_dist,
                                                                                                      reduction='batchmean')
            # print('kl_loss:', kl_loss) # ~-0.19

            # loss += kl_loss
            
        # loss = loss / (self.num_experts * self.batch_size)

        self.batch_size = len(extra_info[0])
        if self.use_relation_aware_gating:
            return loss / self.batch_size
            # return loss
        else:
            return loss / (self.num_experts * self.batch_size)

        return loss

def make_roi_relation_loss_evaluator(cfg):

    loss_evaluator = RelationLossComputation(
        cfg,
        cfg.MODEL.ATTRIBUTE_ON,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.MAX_ATTRIBUTES,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_SAMPLE,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_RATIO,
        cfg.MODEL.ROI_RELATION_HEAD.LOSS_OPTION,
        cfg.MODEL.ROI_RELATION_HEAD.REL_PROP,
        cfg.MODEL.REL_CLASS_NUM_LST,
        cfg.MODEL.OBJ_CLASS_NUM_LST,
        cfg.MODEL.ROI_RELATION_HEAD.USE_RELATION_AWARE_GATING,
    )

    return loss_evaluator
