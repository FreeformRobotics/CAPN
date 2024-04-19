# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.bounding_box import BoxList

from ..attribute_head.roi_attribute_feature_extractors import make_roi_attribute_feature_extractor
from ..box_head.roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_relation_feature_extractors import make_roi_relation_feature_extractor
from .roi_relation_predictors import make_roi_relation_predictor
from .inference import make_roi_relation_post_processor
from .loss import make_roi_relation_loss_evaluator
from .sampling import make_roi_relation_samp_processor
import numpy as np
from maskrcnn_benchmark.structures.bounding_box import BoxList
from .relation_sampling import label_grouping

class ROIRelationHead(torch.nn.Module):
    """
    Generic Relation Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIRelationHead, self).__init__()
        self.cfg = cfg.clone()
        # same structure with box head, but different parameters
        # these param will be trained in a slow learning rate, while the parameters of box head will be fixed
        # Note: there is another such extractor in uniton_feature_extractor
        self.union_feature_extractor = make_roi_relation_feature_extractor(cfg, in_channels)
        if cfg.MODEL.ATTRIBUTE_ON:
            self.box_feature_extractor = make_roi_box_feature_extractor(cfg, in_channels, half_out=True)
            self.att_feature_extractor = make_roi_attribute_feature_extractor(cfg, in_channels, half_out=True)
            feat_dim = self.box_feature_extractor.out_channels * 2
        else:
            self.box_feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
            feat_dim = self.box_feature_extractor.out_channels
        self.predictor = make_roi_relation_predictor(cfg, feat_dim)
        self.post_processor = make_roi_relation_post_processor(cfg)
        self.loss_evaluator = make_roi_relation_loss_evaluator(cfg)
        self.samp_processor = make_roi_relation_samp_processor(cfg)
        self.checkpoint = self.cfg.MODEL.PRETRAINED_DETECTOR_CKPT
        self.expert_mode = self.cfg.MODEL.ROI_RELATION_HEAD.EXPERT_MODE


        # parameters
        self.use_union_box = self.cfg.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION

        # CAME
        self.num_experts = self.cfg.MODEL.ROI_RELATION_HEAD.NUM_EXPERTS
        self.use_relation_aware_gating = cfg.MODEL.ROI_RELATION_HEAD.USE_RELATION_AWARE_GATING
        self.use_per_class_content_aware_matrix = cfg.MODEL.ROI_RELATION_HEAD.USE_PER_CLASS_CONTEXT_AWARE
        self.use_relation_sampling = cfg.MODEL.ROI_RELATION_HEAD.RELATION_SAMPLING
        self.expert_voting_flag = cfg.MODEL.ROI_RELATION_HEAD.EXPERT_VOTING
        self.loss_option = cfg.MODEL.ROI_RELATION_HEAD.LOSS_OPTION  
        
        self.label_grouping = label_grouping(self.cfg)
        self.peer_knowledge_lst = self.label_grouping.obtain_peer_knowledge(self.expert_mode)       # hbt_b_t, h_b_t, hbt_ht_bt
        
        #self.label_group_dic = self.label_grouping.obtain_group_labels()
        #print('self.label_group_dic:', self.label_group_dic)
        #self.peer_knowledge_lst = self.obtain_peer_knowledge('hbt_b_t')       # hbt_b_t, h_b_t


    def vote_max_score(self, labels, scores):
        max_score = 0.0

        cnt_dict = {}
        for key in labels:
            cnt_dict[key] = cnt_dict.get(key, 0) + 1

        for key, value in cnt_dict.items():
            vote_score = 0.0

            if value == len(labels):
                max_score = np.max(scores)
                max_label = key
                return max_label, max_score

            elif value > 1:
                for i in range(len(labels)):
                    if (labels[i] == key):
                        vote_score = scores[i]

                        if vote_score > max_score:
                            max_score = vote_score
                            max_label = labels[i]

            elif value == 1:
                vote_score = scores[labels.index(key)]
                if vote_score > max_score:
                    max_score = vote_score
                    max_label = key

        return max_label, max_score

    def vote_accumulate_max_score(self, labels, scores):
        max_score = 0.0

        cnt_dict = {}
        for key in labels:
            cnt_dict[key] = cnt_dict.get(key, 0) + 1

        for key, value in cnt_dict.items():
            vote_score = 0.0

            if value == len(labels):
                max_score = np.sum(scores)
                max_label = key
                return max_label, max_score

            elif value > 1:
                for i in range(len(labels)):
                    if (labels[i] == key):
                        vote_score += scores[i]

                        if vote_score > max_score:
                            max_score = vote_score
                            max_label = labels[i]

            elif value == 1:
                vote_score = scores[labels.index(key)]
                if vote_score > max_score:
                    max_score = vote_score
                    max_label = key

        return max_label, max_score
    
    # def obtain_peer_knowledge(self, dataset_name, expert_mode):

    #     if dataset_name == 'vg':
    #         REL_CLASS_NUM_LST = [371741.0, 6712.0, 171.0, 208.0, 379.0, 504.0, 1829.0, 1413.0, 10011.0, 644.0, 394.0,
    #                             1603.0, 397.0, 460.0, 565.0, 4.0, 809.0, 163.0, 157.0, 663.0, 67144.0, 10764.0,
    #                             21748.0, 3167.0, 752.0, 676.0, 364.0, 114.0, 234.0, 15300.0, 31347.0, 109355.0, 333.0,
    #                             793.0, 151.0, 601.0, 429.0, 71.0, 4260.0, 44.0, 5086.0, 2273.0, 299.0, 3757.0, 551.0,
    #                             270.0, 1225.0, 352.0, 47326.0, 4810.0, 11059.0]
    #     elif dataset_name == 'oiv6':
    #         REL_CLASS_NUM_LST = [200000.0, 115251.0, 33018.0, 102653.0, 240.0, 1332.0, 189.0, 67.0, 34684.0, 12223.0, 
    #                              3460.0, 287.0, 96.0, 10.0, 3916.0, 82.0, 11.0, 20149.0, 87.0, 1797.0, 
    #                              11.0, 4192.0, 1988.0, 151.0, 54.0, 950.0, 22.0, 524.0, 75.0, 10881.0, 160.0]


    #     print('len:', len(REL_CLASS_NUM_LST))

    #     sorted_indices = np.argsort(REL_CLASS_NUM_LST)[::-1]
    #     sorted_counts = np.sort(REL_CLASS_NUM_LST)[::-1]

    #     sorted_class_ids = sorted_indices.tolist()
    #     sorted_counts = sorted_counts.tolist()

    #     peer_knowledge_lst = []
    #     if expert_mode == 'hbt_b_t':
    #         head_labels = sorted_class_ids
    #         body_labels = sorted_class_ids[16:36]
    #         tail_labels = sorted_class_ids[36:]
    #         peer_knowledge_lst.append(head_labels)
    #         peer_knowledge_lst.append(body_labels)
    #         peer_knowledge_lst.append(tail_labels)
            
    #     elif expert_mode == 'h_b_t':
    #         head_labels = sorted_class_ids[:16]
    #         body_labels = sorted_class_ids[16:36]
    #         tail_labels = sorted_class_ids[36:]
    #         peer_knowledge_lst.append(head_labels)
    #         peer_knowledge_lst.append(body_labels)
    #         peer_knowledge_lst.append(tail_labels)
            
    #     elif expert_mode == 'hbt_ht_bt':
    #         head_labels = sorted_class_ids
    #         body_tail_labels = sorted_class_ids[16:]  # Body and tail together
    #         head_tail_labels = sorted_class_ids[:16] + sorted_class_ids[36:]  # Head and tail together

    #         peer_knowledge_lst.append(head_labels)
    #         peer_knowledge_lst.append(head_tail_labels)
    #         peer_knowledge_lst.append(body_tail_labels)
                
    #     return peer_knowledge_lst

    def obtain_result_lst_peer_network(self, relation_logits_lst, obj_refine_logits, rel_pair_idxs, proposals):
        result_lst = []
        
        for i in range(self.num_experts):
            result = self.post_processor((relation_logits_lst[i], obj_refine_logits), rel_pair_idxs, proposals)
            # print('result:', len(result), result[0].fields())
            
            result_lst.append(result)
            
        return result_lst
    
    def obtain_idx_pair_rel_score_lst(self, result_lst, j):
        rel_pair_idxs_lst = []
        pred_rel_labels_lst = []
        pred_rel_scores_lst = []
        pred_rel_scores_vector_lst = []
        
        for i in range(self.num_experts):
            # Get relevant fields from the result
            rel_pair_idx_expert = result_lst[i][j].get_field('rel_pair_idxs')
            pred_rel_labels_expert = result_lst[i][j].get_field('pred_rel_labels')
            # print('pred_rel_labels_expert:', len(pred_rel_labels_expert), pred_rel_labels_expert)
            pred_rel_scores_expert = result_lst[i][j].get_field('pred_rel_scores')[np.arange(pred_rel_labels_expert.shape[0]), pred_rel_labels_expert]
            # print('pred_rel_scores_expert:', len(pred_rel_scores_expert), pred_rel_scores_expert)
            pred_rel_scores_vector_expert = result_lst[i][j].get_field('pred_rel_scores')
            # print('pred_rel_scores_vector_expert:', pred_rel_scores_vector_expert.shape, pred_rel_scores_vector_expert[0])

            #print('pred_rel_scores_vector_expert:', np.sum(pred_rel_scores_vector_expert[0].detach().cpu().numpy()), np.argmax(pred_rel_scores_vector_expert[0].detach().cpu().numpy()), pred_rel_scores_vector_expert[0], 
            #      np.sum(pred_rel_scores_vector_expert[5].detach().cpu().numpy()), np.argmax(pred_rel_scores_vector_expert[5].detach().cpu().numpy()), pred_rel_scores_vector_expert[5])

            rel_pair_idxs_lst.append(rel_pair_idx_expert.detach().cpu().numpy())
            pred_rel_labels_lst.append(pred_rel_labels_expert.detach().cpu().numpy())
            pred_rel_scores_lst.append(pred_rel_scores_expert.detach().cpu().numpy())
            pred_rel_scores_vector_lst.append(pred_rel_scores_vector_expert.detach().cpu().numpy())

        return rel_pair_idxs_lst, pred_rel_labels_lst, pred_rel_scores_lst, pred_rel_scores_vector_lst
            
            
    # def combine_expert_scores(self, rel_pair_idxs_lst, expert_scores_lst):
    #     all_pairs = np.concatenate(rel_pair_idxs_lst)
    #     unique_pairs, unique_idx = np.unique(all_pairs, axis=0, return_inverse=True)
        
    #     # print('test:', expert_scores_lst[0].shape[1])
    #     combined_scores = np.zeros((len(unique_pairs), expert_scores_lst[0].shape[1]))
    #     expert_scores = [np.zeros((len(unique_pairs), expert_scores_lst[0].shape[1])) for _ in range(len(expert_scores_lst))]
    #     # print('combined_scores.shape:', combined_scores.shape)
        
        
    #     for idx, rel_pair_idxs in enumerate(rel_pair_idxs_lst):
    #         # print('idx:', idx, 'rel_pair_idxs:', len(rel_pair_idxs), rel_pair_idxs[0])
    #         # print('unique_pairs:', len(unique_pairs), unique_pairs[0])
    #         # print('unique_pairs == rel_pair_idx:', unique_pairs == rel_pair_idx)
                  
    #         for rel_pair_idx, rel_score in zip(rel_pair_idxs, expert_scores_lst[idx]):
    #             # print('unique_pairs == rel_pair_idx:', unique_pairs == rel_pair_idx)
    #             # print('np.where unique_pairs == rel_pair_idx:', np.where((unique_pairs == rel_pair_idx)))
    #             unique_idx = np.where((unique_pairs == rel_pair_idx).all(axis=1))[0]
    #             expert_scores[idx][unique_idx] = rel_score
    #             # print('[unique_idx]:', unique_idx)
    #             # print('expert_scores[0][unique_idx].shape:', expert_scores[0][unique_idx].shape)
    #             # peer1 = np.argmax(expert_scores[0][unique_idx][0, 1:]) + 1
    #             # peer2 = np.argmax(expert_scores[1][unique_idx][0, 1:]) + 1
    #             # peer3 = np.argmax(expert_scores[2][unique_idx][0, 1:]) + 1
    #             # print('perrs:', peer1, peer2, peer3)
        
    #     # Assign the scores according to their area of expertise
    #     for i, peer_knowledge in enumerate(self.peer_knowledge_lst):
    #         if i == 0:
    #             combined_scores[:, peer_knowledge] = expert_scores[i][:, peer_knowledge] 
    #         else:
    #             combined_scores[:, peer_knowledge] = expert_scores[i][:, peer_knowledge] 
            
    #     return unique_pairs, combined_scores

    def create_pred_idx_pair_label_score_dictionary(self, rel_pair_idxs_lst, pred_rel_labels_lst, pred_rel_scores_lst, pred_rel_scores_vector_lst):
        expert_dic_lst = []
        score_matrix = np.zeros((rel_pair_idxs_lst[0].shape[0], len(pred_rel_scores_vector_lst[0][1])))
        # print("score_matrix:", score_matrix.shape)
        
        # unique_pairs, combined_score_matrix = self.combine_expert_scores(rel_pair_idxs_lst, pred_rel_scores_vector_lst)

        for i in range(self.num_experts):
            expert_dic = {}
            for j in range(rel_pair_idxs_lst[i].shape[0]):
                idx_pair = rel_pair_idxs_lst[i][j, :]
                label = pred_rel_labels_lst[i][j]
                score = pred_rel_scores_lst[i][j]    
                score_vec = pred_rel_scores_vector_lst[i][j] 
                expert_dic[str(idx_pair)] = [label, score, score_vec]
                
            expert_dic_lst.append(expert_dic)
        
        return expert_dic_lst #, unique_pairs, combined_score_matrix

    def combine_score_matrix(self, rel_pair_idxs_lst, expert_dic_lst):
        combined_scores = np.zeros( (len(rel_pair_idxs_lst[0]), 51) )
        num_ones_first_expert = 0
        num_ones_second_expert = 0
        num_ones_third_expert = 0
        # knowledge_weights = [1.0, 6.5, 11.6]
        knowledge_weights = [1.0, 5.0, 5.0]

        for j in range(rel_pair_idxs_lst[0].shape[0]):
            idx_pair = rel_pair_idxs_lst[0][j, :]
            vector_scores = []
            
            for i in range(self.num_experts):
                label_score_info = expert_dic_lst[i][str(idx_pair)]
                vector_scores.append(label_score_info[2])


            confidences = np.zeros(self.num_experts)
            max_confidences = np.zeros(self.num_experts)

            for i in range(self.num_experts):
                if i == 0:
                    # Adjust indices by subtracting 1
                    adjusted_indices = np.array(self.peer_knowledge_lst[i][1:]) - 1
                    confidences[i] = np.mean(vector_scores[i][1:][adjusted_indices])
                    max_confidences[i] = np.max(vector_scores[i][1:][adjusted_indices])
                else:
                    confidences[i] = np.mean(vector_scores[i][self.peer_knowledge_lst[i]])
                    max_confidences[i] = np.max(vector_scores[i][self.peer_knowledge_lst[i]])

            normalized_confidences = confidences / np.min(confidences)
            normalized_max_confidences = max_confidences / np.min(max_confidences)
            # print('normalized_confidences:', normalized_confidences, confidences)
            
            if self.num_experts == 2:
                num_ones_first_expert += np.count_nonzero(normalized_confidences[0] == 1.0)
                num_ones_second_expert += np.count_nonzero(normalized_confidences[1] == 1.0)
                
            elif self.num_experts == 3:
                num_ones_first_expert += np.count_nonzero(normalized_confidences[0] == 1.0)
                num_ones_second_expert += np.count_nonzero(normalized_confidences[1] == 1.0)
                num_ones_third_expert += np.count_nonzero(normalized_confidences[2] == 1.0)
            
            #print('normalized_max_confidences:', normalized_max_confidences, max_confidences)
            
            # Find the index of the expert with the highest confidence
            highest_confidence_index = np.argmax(normalized_confidences)


            for i, peer_knowledge in enumerate(self.peer_knowledge_lst):
                    # if i == highest_confidence_index:
                    #     weight = knowledge_weights[i]  
                    # else: 
                    #     weight = 1.0
                    weight = 1.0

                    # print('i:', i, 'peer_knowledge:', peer_knowledge)
                    if (i == 0):
                        # print('vector_scores[0][peer_knowledge]:', 
                        #       np.max(vector_scores[i][peer_knowledge]), 
                        #       np.mean(vector_scores[i][peer_knowledge]),
                        #       vector_scores[i][peer_knowledge])
                        #[1:][1:]
                        combined_scores[j, peer_knowledge] += vector_scores[i][peer_knowledge] * knowledge_weights[i] 
                    elif (i == 1):
                        # print('vector_scores[1][peer_knowledge]:', 
                        #       np.max(vector_scores[i][peer_knowledge]),
                        #       np.mean(vector_scores[i][peer_knowledge]),
                        #       vector_scores[i][peer_knowledge])
                        combined_scores[j, peer_knowledge] += vector_scores[i][peer_knowledge] * knowledge_weights[i]  
                    else:
                        # print('vector_scores[2][peer_knowledge]:', 
                        #       np.max(vector_scores[i][peer_knowledge]),
                        #       np.mean(vector_scores[i][peer_knowledge]),
                        #       vector_scores[i][peer_knowledge])
                        combined_scores[j, peer_knowledge] += vector_scores[i][peer_knowledge] * knowledge_weights[i] 
        
        # print('num_ones_first_expert:', num_ones_first_expert, num_ones_second_expert, num_ones_third_expert)
            
        return combined_scores
    
    
    def generate_predicate_labels(self, scores_matrix):
        # Find the index of maximum value in each row
        # The 'axis=1' parameter means the operation is performed across each row
        predicate_labels = np.argmax(scores_matrix[:, 1:], axis=1) + 1

        return predicate_labels

    def obtain_max_label_score_lst(self, rel_pair_idxs_lst, expert_dic_lst):
        max_label_lst = []
        max_score_lst = []
        vector_scores_lst = []
        for j in range(rel_pair_idxs_lst[0].shape[0]):
            idx_pair = rel_pair_idxs_lst[0][j, :]
            labels = []
            scores = []
            vector_scores = []
            
            for i in range(self.num_experts):
                label_score_info = expert_dic_lst[i][str(idx_pair)]
                labels.append(label_score_info[0])
                scores.append(label_score_info[1])
                vector_scores.append(label_score_info[2])
            
            # choose to use max or accumulate max
            # max_label, max_score = self.vote_max_score(labels, scores)
            max_label, max_score = self.vote_accumulate_max_score(labels, scores, vector_scores)
            # print('max_label:', max_label, 'max_score:', max_score)
            max_label_lst.append(max_label)
            max_score_lst.append(max_score)
            vector_scores_lst.append(vector_scores)

        return max_label_lst, max_score_lst
    
    def obtain_predicate_score_matrix(self, result_lst, j, predicate_label_lst, predicate_score_lst):
        
        # clear score matrix of first peer
        predicate_scores_matrix = result_lst[0][j].get_field('pred_rel_scores')
        # print('predicate_scores_matrix.shape:', predicate_scores_matrix.shape)  # (#, 51)
        # print('np.arange(predicate_label_lst[0].shape):', np.arange(len(predicate_label_lst)))  # generate a list from 0 to # -1
        # print('predicate_label_lst[0]:', len(predicate_label_lst))  # the length of #
        
        predicate_scores_matrix[np.arange(len(predicate_label_lst)), predicate_label_lst] = 0.0
        predicate_scores_matrix = predicate_scores_matrix.cpu().numpy()
        # print('len(predicate_label_lst):', len(predicate_label_lst), predicate_label_lst)
        
        for i in range(len(predicate_label_lst)):
            predicate_scores_matrix[i, predicate_label_lst[i]] = predicate_score_lst[i]
            
        return predicate_scores_matrix

    def expert_voting(self, relation_logits_lst, obj_refine_logits, rel_pair_idxs, proposals):

        # Obtain result_lst from peer network
        result_lst = self.obtain_result_lst_peer_network(relation_logits_lst, obj_refine_logits, rel_pair_idxs, proposals)
        # return result_lst[0], {}
        # fields = result_lst[1][0].fields()
        # fields = fields[:-3]
        # # print('fields:', fields)
        # result_copy = result_lst[1][0].copy_with_fields(fields)
        # print('has_field pred_labels:', result_lst[1][0].has_field('pred_labels'))
        # print('has_field pred_scores:', result_lst[1][0].has_field('pred_scores'))
  

        for j in range(len(result_lst[0])): 
            # Create the pred_rel_scores and pred_rel_labels
            rel_pair_idxs_lst, pred_rel_labels_lst, pred_rel_scores_lst, pred_rel_scores_vector_lst = self.obtain_idx_pair_rel_score_lst(result_lst, j)

            # Create dictionary for fast search
            expert_dic_lst = self.create_pred_idx_pair_label_score_dictionary(rel_pair_idxs_lst, pred_rel_labels_lst, pred_rel_scores_lst, pred_rel_scores_vector_lst)
            # predicate_labels = self.generate_predicate_labels(scores_matrix)
            
            
            # One branch to post processing the relational scores
            # Obtian the labels of combined_scores
            predicate_scores = self.combine_score_matrix(rel_pair_idxs_lst, expert_dic_lst)
            predicate_labels = self.generate_predicate_labels(predicate_scores)

            # Another branch to post processing the relational scores

            # obtain predicate_label_lst and predicate_score_lst
            # predicate_label_lst, predicate_score_lst = self.obtain_max_label_score_lst(rel_pair_idxs_lst, expert_dic_lst)
            # predicate_scores_matrix = self.obtain_predicate_score_matrix(result_lst, j, predicate_label_lst, predicate_score_lst)
            
            # print('combined_scores.shape:', combined_scores.shape)
            # print('predicate_labels:', predicate_labels)

            # Clear dictionaries
            for i in range(self.num_experts):
                expert_dic_lst[i].clear()

            result_lst[0][j].remove_field('rel_pair_idxs')
            result_lst[0][j].remove_field('pred_rel_labels')
            result_lst[0][j].remove_field('pred_rel_scores')                
                
            rel_pair_tensor = torch.tensor(rel_pair_idxs_lst[0]).cuda()
            rel_label_tensor = torch.tensor(predicate_labels).cuda()
            scores_matrix_tensor = torch.tensor(predicate_scores).cuda()
            
            #rel_label_tensor = torch.tensor(predicate_label_lst).cuda()
            #scores_matrix_tensor = torch.tensor(predicate_scores_matrix).cuda()
            
            result_lst[0][j].add_field('rel_pair_idxs', rel_pair_tensor)  # (#rel, 2)
            result_lst[0][j].add_field('pred_rel_labels', rel_label_tensor)  # (#rel, )
            result_lst[0][j].add_field('pred_rel_scores', scores_matrix_tensor)  # (#rel, )
    
            
        return result_lst[0], {}
    
        # Obtain the indices from highest score to lowest score
        # ind_lst = np.argsort(max_score_lst)[::-1]
        # print('ind_lst:', ind_lst)

        # add the new labels to results
        # rel_label_lst = []
        # rel_pair_lst = []

        # # Add the new labels and relation pair indices to results
        # for i in range(rel_pair_idxs_lst[0].shape[0]):
        #     rel_label = max_label_lst[ind_lst[i]]
        #     rel_label_lst.append(rel_label)
            
        # for i in range(rel_pair_idxs_lst[0].shape[0]):
        #     rel_pair = rel_pair_idxs_lst[0][ind_lst[i]]
        #     rel_pair_lst.append(rel_pair)


        # obtain the max value of each row as the labels
        # check_labels = np.argmax(scores_matrix, axis=1)
        # print('rel_label_lst:', len(rel_label_lst), rel_label_lst)
        # print('check_labels:', check_labels.shape, check_labels)

        # Update the fields in the result
        # rel_pair_tensor = torch.tensor(unique_pairs).cuda()
        # rel_label_tensor = torch.tensor(predicate_labels).cuda()
        # scores_matrix_tensor = torch.tensor(scores_matrix).cuda()
    
        # print('rel_pair_tensor:', rel_pair_tensor.shape)
        # print('rel_pair_tensor:', rel_label_tensor.shape)
        # print('pred_rel_scores_matrix_tensor:', pred_rel_scores_matrix_tensor.shape)

        # for test
        # rel_pair_idxs_first =  result_lst[0][0].get_field('rel_pair_idxs')  # (#rel, 2)
        # pred_rel_labels_first = result_lst[0][0].get_field('pred_rel_labels')
        # pred_rel_scores_first = result_lst[0][0].get_field('pred_rel_scores')
        
        #return result_lst[0], {}
        
        
 

   

    def forward(self, features, proposals, targets=None, logger=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes. Note: it has been post-processed (regression, nms) in sgdet mode
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        if self.training:
            # relation subsamples and assign ground truth label during training
            with torch.no_grad():
                if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
                    proposals, rel_labels, rel_pair_idxs, rel_binarys = self.samp_processor.gtbox_relsample(proposals, targets)

                else:
                    proposals, rel_labels, rel_pair_idxs, rel_binarys = self.samp_processor.detect_relsample(proposals, targets)

        else:
            rel_labels, rel_binarys = None, None
            rel_pair_idxs = self.samp_processor.prepare_test_pairs(features[0].device, proposals)

        # use box_head to extract features that will be fed to the later predictor processing
        roi_features = self.box_feature_extractor(features, proposals)

        if self.cfg.MODEL.ATTRIBUTE_ON:
            att_features = self.att_feature_extractor(features, proposals)
            roi_features = torch.cat((roi_features, att_features), dim=-1)

        if self.use_union_box:
            union_features = self.union_feature_extractor(features, proposals, rel_pair_idxs)
        else:
            union_features = None
        
        # final classifier that converts the features into predictions
        # should corresponding to all the functions and layers after the self.context class

        # for i in range(len(rel_pair_idxs)):
        #     print('rel_pair_idxs:', rel_pair_idxs[i].shape)

        # add multiple experts
        if self.num_experts > 1:
            weighted_expert_relation_logits_lst = []    # list for storing the weighted expert info
            expert_relation_logits_lst = []             # merge individual experts into one list

            if self.use_relation_sampling:
                # print('self.use_relation_sampling')
                refine_logits, relation_logits, add_losses, rel_labels_lst, relation_logits_lst, beta_relation_aware_gating = self.predictor(proposals, 
                                                                                                                                             rel_pair_idxs, 
                                                                                                                                             rel_labels, 
                                                                                                                                             rel_binarys, 
                                                                                                                                             roi_features, 
                                                                                                                                             union_features, 
                                                                                                                                             logger)
            
            # if using the context-aware mixture-of-experts
            elif self.use_relation_aware_gating:
                # print('self.use_relation_aware_gating')
                refine_logits, relation_logits, add_losses, relation_logits_lst, beta_relation_aware_gating, weighted_rel_dists_full_sum = self.predictor(proposals, 
                                                                                                                                                          rel_pair_idxs, 
                                                                                                                                                          rel_labels, 
                                                                                                                                                          rel_binarys, 
                                                                                                                                                          roi_features, 
                                                                                                                                                          union_features, 
                                                                                                                                                          logger)
            elif self.use_per_class_content_aware_matrix:
                # print('self.use_per_class_content_aware_matrix')
                refine_logits, relation_logits, add_losses, relation_logits_lst, weighted_rel_dists_full_sum = self.predictor(proposals, 
                                                                                                                              rel_pair_idxs, 
                                                                                                                              rel_labels, 
                                                                                                                              rel_binarys, 
                                                                                                                              roi_features, 
                                                                                                                              union_features, 
                                                                                                                              logger)


            # if using the mixture-of-experts
            else:
                # print('self.mixture-of-experts')

                # refine_logits, relation_logits, add_losses, relation_logits_lst, weighted_rel_dists_full_sum, rel_dists_full_mean = self.predictor(proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger)

                refine_logits, relation_logits, add_losses, relation_logits_lst = self.predictor(proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger)


                # Initialize an empty list to store the results
                relation_logits_mean = []
                # print('relation_logits:', relation_logits.shape)
                # print('relation_logits:', relation_logits_lst[0].shape, relation_logits_lst[1].shape, relation_logits_lst[2].shape)

                # Loop over the tensors with the same shape
                for i in range(len(relation_logits_lst[0])):
                    # Stack the tensors with the same shape into a single tensor
                    
                    relation_logits_all = torch.stack([relation_logits_lst[j][i] for j in range(len(relation_logits_lst))])
            
                    # Compute the mean along the first dimension of the stacked tensor
                    relation_logits_mean.append(torch.mean(relation_logits_all, dim=0))

        else:
            refine_logits, relation_logits, add_losses = self.predictor(proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger)

        # for test
        if not self.training:
            if self.num_experts > 1:
                # showing the performance of weighted experts
                if self.use_relation_sampling:
                    # print('use_relation_sampling')
                    result, _ = self.expert_voting(relation_logits_lst, refine_logits, rel_pair_idxs, proposals)
                
                elif self.loss_option == 'PLME_LOSS':
                    print('PLME_LOSS')
                    # result = self.post_processor((relation_logits_mean, refine_logits), rel_pair_idxs, proposals)
                    # result = self.post_processor((relation_logits, refine_logits), rel_pair_idxs, proposals)
                    # print('relation_logits_lst:', len(relation_logits_lst))
                    result, _ = self.expert_voting(relation_logits_lst, refine_logits, rel_pair_idxs, proposals)
                    # result = self.post_processor((relation_logits_lst[2], refine_logits), rel_pair_idxs, proposals)
                else:
                    # print('using others')
                    # result = self.post_processor((weighted_rel_dists_full_sum, refine_logits), rel_pair_idxs, proposals)
                    # result = self.post_processor((expert_relation_logits_lst, refine_logits), rel_pair_idxs, proposals)
                    # result = self.post_processor((relation_logits_lst[2], refine_logits), rel_pair_idxs, proposals)
                    
                    # old one
                    # result = self.post_processor((weighted_rel_dists_full_sum, refine_logits), rel_pair_idxs, proposals)
                    
                    # test ce_pn loss
                    result = self.post_processor((relation_logits_mean, refine_logits), rel_pair_idxs, proposals)
                    
                # showing the performance of all experts
                # result = self.post_processor((expert_relation_logits_lst, refine_logits), rel_pair_idxs, proposals)

                # showing the performance of each individual expert
                # result = self.post_processor((relation_logits_lst[3], refine_logits), rel_pair_idxs, proposals)


            else:
                result = self.post_processor((relation_logits, refine_logits), rel_pair_idxs, proposals)

            return roi_features, result, {}

        # add multiple experts for loss evaluation
        if self.num_experts > 1:
            if self.use_relation_sampling:
                # print('use_relation_sampling')
                loss_relation, loss_refine = self.loss_evaluator(proposals, rel_labels, relation_logits, refine_logits,
                                                                 beta_relation_aware_gating=beta_relation_aware_gating,
                                                                 extra_info=relation_logits_lst,
                                                                 extra_labels=rel_labels_lst)
            elif self.use_relation_aware_gating:
                # loss_relation, loss_refine = self.loss_evaluator(proposals, rel_labels, relation_logits, refine_logits,
                #                                                  beta_relation_aware_gating=beta_relation_aware_gating,
                #                                                  extra_info=weighted_expert_relation_logits_lst)
                loss_relation, loss_refine = self.loss_evaluator(proposals, rel_labels, relation_logits, refine_logits,
                                                                 beta_relation_aware_gating=beta_relation_aware_gating,
                                                                 extra_info=relation_logits_lst)
                
            elif self.loss_option == 'PLME_LOSS':
                loss_relation, loss_refine = self.loss_evaluator(proposals, rel_labels, relation_logits_mean, refine_logits,
                                                                extra_info=relation_logits_lst) 
                
            else:
                loss_relation, loss_refine = self.loss_evaluator(proposals, rel_labels, relation_logits, refine_logits,
                                                                 extra_info=relation_logits_lst)
        else:
            loss_relation, loss_refine = self.loss_evaluator(proposals, rel_labels, relation_logits, refine_logits)


        if self.cfg.MODEL.ATTRIBUTE_ON and isinstance(loss_refine, (list, tuple)):
            output_losses = dict(loss_rel=loss_relation, loss_refine_obj=loss_refine[0], loss_refine_att=loss_refine[1])
        else:
            output_losses = dict(loss_rel=loss_relation, loss_refine_obj=loss_refine)

        output_losses.update(add_losses)

        return roi_features, proposals, output_losses


def build_roi_relation_head(cfg, in_channels):
    """
    Constructs a new relation head.
    By default, uses ROIRelationHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIRelationHead(cfg, in_channels)
