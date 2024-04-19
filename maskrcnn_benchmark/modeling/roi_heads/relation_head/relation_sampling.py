# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 11:37:16 2022

@author: Administrator
"""
import numpy as np
import torch
import random

class label_grouping(object):
    def __init__(self, config):
        checkpoint = config.MODEL.PRETRAINED_DETECTOR_CKPT
        self.checkpoint_name = checkpoint.split('/')[-1]
        if self.checkpoint_name == 'oiv6_det.pth':
            #  31 relations, 10 head, 10 body, 11 tail
            self.rel_lst =  [200000.0, 115251.0, 33018.0, 102653.0, 240.0, 1332.0, 189.0, 67.0, 34684.0, 12223.0, 
                             3460.0, 287.0, 96.0, 10.0, 3916.0, 82.0, 11.0, 20149.0, 87.0, 1797.0, 
                             11.0, 4192.0, 1988.0, 151.0, 54.0, 950.0, 22.0, 524.0, 75.0, 10881.0, 160.0]
        else:
            #  51 relations, 15 head, 15 body, 21 tail
            self.rel_lst = [371741.0, 6712.0, 171.0, 208.0, 379.0, 504.0, 1829.0, 1413.0, 10011.0, 644.0, 394.0, 1603.0,
                            397.0, 460.0, 565.0, 4.0, 809.0, 163.0, 157.0, 663.0, 67144.0, 10764.0, 21748.0, 3167.0, 752.0, 676.0,
                            364.0, 114.0, 234.0, 15300.0, 31347.0, 109355.0, 333.0, 793.0, 151.0, 601.0, 429.0, 71.0,
                            4260.0, 44.0, 5086.0, 2273.0, 299.0, 3757.0, 551.0, 270.0, 1225.0, 352.0, 47326.0, 4810.0, 11059.0]

    def obtain_group_labels(self):
        rel_dic = {}
        for i in range(len(self.rel_lst)):
            rel_dic.update({i: self.rel_lst[i]})

        # print('rel_dic:', rel_dic)
        rel_dic_sorted = sorted(rel_dic.items(), key=lambda x: x[1], reverse=True)
        # print('rel_dic_sorted:', rel_dic_sorted)

        # rel_head_labels = []
        # rel_medium_labels = []
        # rel_tail_labels = []
        if self.checkpoint_name == 'oiv6_det.pth':
            for k, v in enumerate(rel_dic_sorted):
                if (k >= 0 and k < 10):
                    rel_dic[v[0]] = 'head'
                elif (k >= 10 and k < 20):
                    
                    rel_dic[v[0]] = 'body'
                else:
                    rel_dic[v[0]] = 'tail'
        
        else:
            for k, v in enumerate(rel_dic_sorted):
                if (k >= 0 and k < 15):
                    # rel_head_labels.append(v[0])
                    rel_dic[v[0]] = 'head'

                    # print(rel_dic[v[0]])
                elif (k >= 15 and k < 30):
                    # rel_medium_labels.append(v[0])
                    rel_dic[v[0]] = 'body'

                    # print(rel_dic[v[0]])

                else:
                    # rel_tail_labels.append(v[0])
                    rel_dic[v[0]] = 'tail'

                # print(rel_dic[v[0]])

        # print('rel_head_labels:', rel_head_labels)
        # print('rel_medium_labels:', rel_medium_labels)
        # print('rel_tail_labels:', rel_tail_labels)

        return rel_dic

    def obtain_peer_knowledge(self, expert_mode):
        sorted_indices = np.argsort(self.rel_lst)[::-1]
        sorted_class_ids = sorted_indices.tolist()

        peer_knowledge_lst = []
        
        if self.checkpoint_name == 'oiv6_det.pth':
            if expert_mode == 'hbt_b_t':
                head_labels = sorted_class_ids
                body_labels = sorted_class_ids[10:20]
                tail_labels = sorted_class_ids[20:]
                peer_knowledge_lst.append(head_labels)
                peer_knowledge_lst.append(body_labels)
                peer_knowledge_lst.append(tail_labels)

            elif expert_mode == 'h_b_t':
                head_labels = sorted_class_ids[:10]
                body_labels = sorted_class_ids[10:20]
                tail_labels = sorted_class_ids[20:]
                peer_knowledge_lst.append(head_labels)
                peer_knowledge_lst.append(body_labels)
                peer_knowledge_lst.append(tail_labels)

            elif expert_mode == 'hbt_ht_bt':
                head_labels = sorted_class_ids
                head_tail_labels = sorted_class_ids[:10] + sorted_class_ids[20:]  # Head and tail together
                body_tail_labels = sorted_class_ids[10:]                          # Body and tail together

                peer_knowledge_lst.append(head_labels)
                peer_knowledge_lst.append(head_tail_labels)
                peer_knowledge_lst.append(body_tail_labels)
        
        else:
            if expert_mode == 'hbt_b_t':
                print('expert_mode:', expert_mode, '[]-[15:30]-[30:]', '1:1:1')
                head_labels = sorted_class_ids
                body_labels = sorted_class_ids[15:30]
                tail_labels = sorted_class_ids[30:]
                
                peer_knowledge_lst.append(head_labels)
                peer_knowledge_lst.append(body_labels)
                peer_knowledge_lst.append(tail_labels)
                
                tail_weight = np.sum(tail_labels) / np.sum(peer_knowledge_lst)

            elif expert_mode == 'h_b_t':
                print('expert_mode:', expert_mode, '[:15]-[15:30]-[30:]')
                head_labels = sorted_class_ids[:15]
                body_labels = sorted_class_ids[15:30]
                tail_labels = sorted_class_ids[30:]
                peer_knowledge_lst.append(head_labels)
                peer_knowledge_lst.append(body_labels)
                peer_knowledge_lst.append(tail_labels)

            elif expert_mode == 'hbt_ht_bt':
                head_labels = sorted_class_ids
                head_tail_labels = sorted_class_ids[:15] + sorted_class_ids[30:]  # Head and tail together
                body_tail_labels = sorted_class_ids[15:]  # Body and tail together

                peer_knowledge_lst.append(head_labels)
                peer_knowledge_lst.append(head_tail_labels)
                peer_knowledge_lst.append(body_tail_labels)
            
            elif expert_mode == 'hb_ht_bt':
                head_body_labels = sorted_class_ids[0:30]
                head_tail_labels = sorted_class_ids[:15] + sorted_class_ids[30:]  # Head and tail together
                body_tail_labels = sorted_class_ids[15:]  # Body and tail together

                peer_knowledge_lst.append(head_body_labels)
                peer_knowledge_lst.append(head_tail_labels)
                peer_knowledge_lst.append(body_tail_labels)
            
            elif expert_mode == 'hbt_h_b_t':
                all_labels = sorted_class_ids
                head_labels = sorted_class_ids[:10]
                body_labels = sorted_class_ids[10:20]
                tail_labels = sorted_class_ids[20:]
                
                peer_knowledge_lst.append(all_labels)
                peer_knowledge_lst.append(head_labels)
                peer_knowledge_lst.append(body_labels)
                peer_knowledge_lst.append(tail_labels)
                
            elif expert_mode == 'hbt_hb_ht_bt':
                all_labels = sorted_class_ids
                head_body_labels = sorted_class_ids[0:30]
                head_tail_labels = sorted_class_ids[:15] + sorted_class_ids[30:]  # Head and tail together
                body_tail_labels = sorted_class_ids[15:]  # Body and tail together
                
                peer_knowledge_lst.append(all_labels)
                peer_knowledge_lst.append(head_body_labels)
                peer_knowledge_lst.append(head_tail_labels)
                peer_knowledge_lst.append(body_tail_labels)    
                
            elif expert_mode == 'hbt_bt_t':
                head_labels = sorted_class_ids
                body_tail_labels = sorted_class_ids[15:]  # Body and tail together
                tail_labels = sorted_class_ids[30:]

                peer_knowledge_lst.append(head_labels)
                peer_knowledge_lst.append(body_tail_labels)
                peer_knowledge_lst.append(tail_labels)
                
            elif expert_mode == 'hbt_b':
                head_labels = sorted_class_ids
                body_labels = sorted_class_ids[15:30]
                peer_knowledge_lst.append(head_labels)
                peer_knowledge_lst.append(body_labels)
                
            elif expert_mode == 'hbt_t':
                head_labels = sorted_class_ids
                tail_labels = sorted_class_ids[30:]
                peer_knowledge_lst.append(head_labels)
                peer_knowledge_lst.append(tail_labels)    
                    
        return peer_knowledge_lst

            
        # if self.checkpoint_name == 'oiv6_det.pth':
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
        #         head_body_labels = sorted_class_ids[:36]  # Head and body together
        #         head_tail_labels = sorted_class_ids[:16] + sorted_class_ids[36:]  # Head and tail together
        #         body_tail_labels = sorted_class_ids[16:]  # Body and tail together

        #         peer_knowledge_lst.append(head_body_labels)
        #         peer_knowledge_lst.append(head_tail_labels)
        #         peer_knowledge_lst.append(body_tail_labels)
        
        # else:
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
        #         head_body_labels = sorted_class_ids[:36]  # Head and body together
        #         head_tail_labels = sorted_class_ids[:16] + sorted_class_ids[36:]  # Head and tail together
        #         body_tail_labels = sorted_class_ids[16:]  # Body and tail together

        #         peer_knowledge_lst.append(head_body_labels)
        #         peer_knowledge_lst.append(head_tail_labels)
        #         peer_knowledge_lst.append(body_tail_labels)




class Relation_Sampling(object):
    def __init__(self):
        self.zero_label_padding_mode = 'rand_insert'
        self.num_of_group_element_list, self.predicate_stage_count = self.get_group_splits('VG', 'divide4')
        self.max_group_element_number_list = self.generate_num_stage_vector(self.num_of_group_element_list)
        self.incre_idx_list, self.max_elemnt_list, self.group_matrix, self.kd_matrix = self.get_current_predicate_idx(self.num_of_group_element_list,
                                                                                             0.1, 'VG')
        self.sample_rate_matrix = self.generate_sample_rate_vector('VG', self.max_group_element_number_list)
        self.bias_for_group_split = self.generate_current_sequence_for_bias(self.num_of_group_element_list, 'VG')


    def get_group_splits(self, Dataset_name, split_name):
        assert Dataset_name in ['VG', 'GQA_200']
        incremental_stage_list = None
        predicate_stage_count = None
        if Dataset_name == 'VG':
            assert split_name in ['divide3', 'divide4', 'divide5', 'average']
            if split_name == 'divide3':#[]
                incremental_stage_list = [[1, 2, 3],
                                          [4, 5, 6],
                                          [7, 8, 9, 10, 11, 12, 13, 14],
                                          [15, 16, 17, 18, 19, 20],
                                          [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
                                          [41, 42, 43, 44, 45, 46, 47, 48, 49, 50]]
                predicate_stage_count = [3, 3, 8, 6, 20, 10]
            elif split_name == 'divide4':#[4,4,9,19,12]
                incremental_stage_list = [[1, 2, 3, 4],
                                          [5, 6, 7, 8, 9, 10],
                                          [11, 12, 13, 14, 15, 16, 17, 18, 19],
                                          [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38],
                                          [39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]]
                predicate_stage_count = [4, 6, 9, 19, 12]
            elif split_name == 'divide5':
                incremental_stage_list = [[1, 2, 3, 4],
                                          [5, 6, 7, 8, 9, 10, 11, 12],
                                          [13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
                                          [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]]
                predicate_stage_count = [4, 8, 10, 28]
            elif split_name == 'average':
                incremental_stage_list = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                          [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                                          [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
                                          [31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
                                          [41, 42, 43, 44, 45, 46, 47, 48, 49, 50]]
                predicate_stage_count = [10, 10, 10, 10, 10]
            else:
                exit('wrong mode in group split!')
            assert sum(predicate_stage_count) == 50

        elif Dataset_name == 'GQA_200':
            assert split_name in ['divide3', 'divide4', 'divide5', 'average']
            if split_name == 'divide3':  # []
                incremental_stage_list = [[1, 2, 3, 4],
                                          [5, 6, 7, 8],
                                          [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                                          [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
                                          [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66],
                                          [67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]]
                predicate_stage_count = [4, 4, 11, 16, 31, 34]
            elif split_name == 'divide4':  # [4,4,9,19,12]
                incremental_stage_list = [[1, 2, 3, 4, 5],
                                          [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                                          [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
                                          [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]]
                predicate_stage_count = [5, 10, 20, 65]
            elif split_name == 'divide5':
                incremental_stage_list = [[1, 2, 3, 4, 5, 6, 7],
                                          [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
                                          [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
                                          [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]]
                predicate_stage_count = [7, 14, 28, 51]
            elif split_name == 'average':
                incremental_stage_list = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                                          [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
                                          [41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
                                          [61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80],
                                          [81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]]
                predicate_stage_count = [20, 20, 20, 20, 20]
            else:
                exit('wrong mode in group split!')
            assert sum(predicate_stage_count) == 100

        else:
            exit('wrong mode in group split!')

        return incremental_stage_list, predicate_stage_count

    def generate_num_stage_vector(self, incremental_stage_list):
        num_stage_vector = []
        n_p = 0
        for isl in incremental_stage_list:
            n_p += len(isl)
            num_stage_vector.append(n_p)

        return num_stage_vector

    def get_current_predicate_idx(self, incremental_stage_list, zeros_vector_penalty, Dataset_choice):
        data_long = 0
        if Dataset_choice == 'VG':
            data_long = 51
        elif Dataset_choice == 'GQA_200':
            data_long = 101
        else:
            exit('wrong mode in Dataset choice')
        outp = []
        for i in range(data_long):
            outp.append(0)
        for i in range(len(incremental_stage_list)):
            for num in incremental_stage_list[i]:
                outp[num] = i+1
        max_p = []
        for i in incremental_stage_list:
            max_p.append(max(i))

        idx_search_p = []
        kd_p = []
        for i in range(len(incremental_stage_list)):
            p1 = []
            p2 = []
            for j in range(data_long):
                p1.append(0)
                p2.append(zeros_vector_penalty)
            max_l = max_p[i]
            for j in range(max_l):
                p1[j+1] = j+1
                p2[j+1] = 1.0
            idx_search_p.append(p1)
            kd_p.append(p2)

        return outp, max_p, idx_search_p, kd_p

    def generate_sample_rate_vector(self, Dataset_choice, num_stage_predicate):
        if Dataset_choice == 'VG':
            predicate_new_order_count = [3024465, 109355, 67144, 47326, 31347, 21748, 15300, 10011, 11059, 10764, 6712,
                                         5086, 4810, 3757, 4260, 3167, 2273, 1829, 1603, 1413, 1225, 793, 809, 676, 352,
                                         663, 752, 565, 504, 644, 601, 551, 460, 394, 379, 397, 429, 364, 333, 299, 270,
                                         234, 171, 208, 163, 157, 151, 71, 114, 44, 4]
            assert len(predicate_new_order_count) == 51
        elif Dataset_choice == 'GQA_200':
            predicate_new_order_count = [200000, 64218, 47205, 32126, 25203, 21104, 15890, 15676, 7688, 6966, 6596, 6044, 5250, 4260, 4180, 4131, 2859, 2559, 2368, 2351, 2134, 1673, 1532, 1373, 1273, 1175, 1139, 1123, 1077, 941, 916, 849, 835, 808, 782, 767, 628, 603, 569, 540, 494, 416, 412, 412, 398, 395, 394, 390, 345, 327, 302, 301, 292, 275, 270, 267, 267, 264, 258, 251, 233, 233, 229, 224, 215, 214, 209, 204, 198, 195, 192, 191, 185, 181, 176, 158, 158, 154, 151, 148, 143, 136, 131, 130, 130, 128, 127, 125, 124, 124, 121, 118, 112, 112, 106, 105, 104, 103, 102, 52, 52]
            assert len(predicate_new_order_count) == 101
        else:
            exit('wrong mode in Dataset_choice')
        outp = []
        for i in range(len(num_stage_predicate)):
            opiece = []
            for j in range(len(predicate_new_order_count)):
                opiece.append(0.0)
            num_list = predicate_new_order_count[0:(num_stage_predicate[i]+1)]
            median = np.median(num_list[1:])
            for j in range(len(num_list)):
                if num_list[j] > median:
                    num = median / num_list[j]
                    if j == 0:
                        num = num * 10.0
                    if num < 0.01:
                        num = 0.01
                    opiece[j] = num
                else:
                    opiece[j] = 1.0
            outp.append(opiece)
        return outp

    def generate_current_sequence_for_bias(self, incremental_stage_list, Dataset_choice):
        data_long = 0
        if Dataset_choice == 'VG':
            data_long = 51
        elif Dataset_choice == 'GQA_200':
            data_long = 101
        else:
            exit('wrong mode in Dataset choice')
        outp = []
        for i in range(len(incremental_stage_list)):
            opiece = []
            for j in range(data_long):
                opiece.append(0)
            for j in range(i+1):
                for k in incremental_stage_list[j]:
                    opiece[k] = k
            outp.append(opiece)

        return outp


    def cat(self, tensors, dim=0):
        """
        Efficient version of torch.cat that avoids a copy if there is only a single element in a list
        """
        assert isinstance(tensors, (list, tuple))
        if len(tensors) == 1:
            return tensors[0]
        return torch.cat(tensors, dim)

    def obtain_selected_info_matrix(self, rel_labels):
        rel_labels = self.cat(rel_labels, dim=0)
        max_label = max(rel_labels)
        num_groups = self.incre_idx_list[max_label.item()]
        if num_groups == 0:
            num_groups = max(self.incre_idx_list)
        cur_chosen_matrix = []
        for i in range(num_groups):
            cur_chosen_matrix.append([])

        for i in range(len(rel_labels)):
            rel_tar = rel_labels[i].item()
            if rel_tar == 0:
                if self.zero_label_padding_mode == 'rand_insert':
                    random_idx = random.randint(0, num_groups - 1)
                    cur_chosen_matrix[random_idx].append(i)

            else:
                rel_idx = self.incre_idx_list[rel_tar]
                random_num = random.random()
                for j in range(num_groups):
                    act_idx = num_groups - j
                    threshold_cur = self.sample_rate_matrix[act_idx - 1][rel_tar]
                    if random_num <= threshold_cur or act_idx < rel_idx:
                        # print('%d-%d-%d-%.2f-%.2f'%(i, rel_idx, act_idx, random_num, threshold_cur))
                        for k in range(act_idx):
                            cur_chosen_matrix[k].append(i)
                        break

        rel_labels_lst = []
        for i in range(len(cur_chosen_matrix)):
            expert_labels = rel_labels[cur_chosen_matrix[i]]
            rel_labels_lst.append(expert_labels)

        return cur_chosen_matrix, rel_labels_lst, num_groups


class Relation_Sampling(object):
    def __init__(self):
        self.zero_label_padding_mode = 'rand_insert'
        self.num_of_group_element_list, self.predicate_stage_count = self.get_group_splits('VG', 'divide4')
        self.max_group_element_number_list = self.generate_num_stage_vector(self.num_of_group_element_list)
        self.incre_idx_list, self.max_elemnt_list, self.group_matrix, self.kd_matrix = self.get_current_predicate_idx(self.num_of_group_element_list,
                                                                                             0.1, 'VG')
        self.sample_rate_matrix = self.generate_sample_rate_vector('VG', self.max_group_element_number_list)
        self.bias_for_group_split = self.generate_current_sequence_for_bias(self.num_of_group_element_list, 'VG')


    def get_group_splits(self, Dataset_name, split_name):
        assert Dataset_name in ['VG', 'GQA_200']
        incremental_stage_list = None
        predicate_stage_count = None
        if Dataset_name == 'VG':
            assert split_name in ['divide3', 'divide4', 'divide5', 'average']
            if split_name == 'divide3':#[]
                incremental_stage_list = [[1, 2, 3],
                                          [4, 5, 6],
                                          [7, 8, 9, 10, 11, 12, 13, 14],
                                          [15, 16, 17, 18, 19, 20],
                                          [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
                                          [41, 42, 43, 44, 45, 46, 47, 48, 49, 50]]
                predicate_stage_count = [3, 3, 8, 6, 20, 10]
            elif split_name == 'divide4':#[4,4,9,19,12]
                incremental_stage_list = [[1, 2, 3, 4],
                                          [5, 6, 7, 8, 9, 10],
                                          [11, 12, 13, 14, 15, 16, 17, 18, 19],
                                          [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38],
                                          [39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]]
                predicate_stage_count = [4, 6, 9, 19, 12]
            elif split_name == 'divide5':
                incremental_stage_list = [[1, 2, 3, 4],
                                          [5, 6, 7, 8, 9, 10, 11, 12],
                                          [13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
                                          [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]]
                predicate_stage_count = [4, 8, 10, 28]
            elif split_name == 'average':
                incremental_stage_list = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                          [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                                          [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
                                          [31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
                                          [41, 42, 43, 44, 45, 46, 47, 48, 49, 50]]
                predicate_stage_count = [10, 10, 10, 10, 10]
            else:
                exit('wrong mode in group split!')
            assert sum(predicate_stage_count) == 50

        elif Dataset_name == 'GQA_200':
            assert split_name in ['divide3', 'divide4', 'divide5', 'average']
            if split_name == 'divide3':  # []
                incremental_stage_list = [[1, 2, 3, 4],
                                          [5, 6, 7, 8],
                                          [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                                          [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
                                          [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66],
                                          [67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]]
                predicate_stage_count = [4, 4, 11, 16, 31, 34]
            elif split_name == 'divide4':  # [4,4,9,19,12]
                incremental_stage_list = [[1, 2, 3, 4, 5],
                                          [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                                          [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
                                          [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]]
                predicate_stage_count = [5, 10, 20, 65]
            elif split_name == 'divide5':
                incremental_stage_list = [[1, 2, 3, 4, 5, 6, 7],
                                          [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
                                          [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
                                          [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]]
                predicate_stage_count = [7, 14, 28, 51]
            elif split_name == 'average':
                incremental_stage_list = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                                          [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
                                          [41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
                                          [61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80],
                                          [81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]]
                predicate_stage_count = [20, 20, 20, 20, 20]
            else:
                exit('wrong mode in group split!')
            assert sum(predicate_stage_count) == 100

        else:
            exit('wrong mode in group split!')

        return incremental_stage_list, predicate_stage_count

    def generate_num_stage_vector(self, incremental_stage_list):
        num_stage_vector = []
        n_p = 0
        for isl in incremental_stage_list:
            n_p += len(isl)
            num_stage_vector.append(n_p)

        return num_stage_vector

    def get_current_predicate_idx(self, incremental_stage_list, zeros_vector_penalty, Dataset_choice):
        data_long = 0
        if Dataset_choice == 'VG':
            data_long = 51
        elif Dataset_choice == 'GQA_200':
            data_long = 101
        else:
            exit('wrong mode in Dataset choice')
        outp = []
        for i in range(data_long):
            outp.append(0)
        for i in range(len(incremental_stage_list)):
            for num in incremental_stage_list[i]:
                outp[num] = i+1
        max_p = []
        for i in incremental_stage_list:
            max_p.append(max(i))

        idx_search_p = []
        kd_p = []
        for i in range(len(incremental_stage_list)):
            p1 = []
            p2 = []
            for j in range(data_long):
                p1.append(0)
                p2.append(zeros_vector_penalty)
            max_l = max_p[i]
            for j in range(max_l):
                p1[j+1] = j+1
                p2[j+1] = 1.0
            idx_search_p.append(p1)
            kd_p.append(p2)

        return outp, max_p, idx_search_p, kd_p

    def generate_sample_rate_vector(self, Dataset_choice, num_stage_predicate):
        if Dataset_choice == 'VG':
            predicate_new_order_count = [3024465, 109355, 67144, 47326, 31347, 21748, 15300, 10011, 11059, 10764, 6712,
                                         5086, 4810, 3757, 4260, 3167, 2273, 1829, 1603, 1413, 1225, 793, 809, 676, 352,
                                         663, 752, 565, 504, 644, 601, 551, 460, 394, 379, 397, 429, 364, 333, 299, 270,
                                         234, 171, 208, 163, 157, 151, 71, 114, 44, 4]
            assert len(predicate_new_order_count) == 51
        elif Dataset_choice == 'GQA_200':
            predicate_new_order_count = [200000, 64218, 47205, 32126, 25203, 21104, 15890, 15676, 7688, 6966, 6596, 6044, 5250, 4260, 4180, 4131, 2859, 2559, 2368, 2351, 2134, 1673, 1532, 1373, 1273, 1175, 1139, 1123, 1077, 941, 916, 849, 835, 808, 782, 767, 628, 603, 569, 540, 494, 416, 412, 412, 398, 395, 394, 390, 345, 327, 302, 301, 292, 275, 270, 267, 267, 264, 258, 251, 233, 233, 229, 224, 215, 214, 209, 204, 198, 195, 192, 191, 185, 181, 176, 158, 158, 154, 151, 148, 143, 136, 131, 130, 130, 128, 127, 125, 124, 124, 121, 118, 112, 112, 106, 105, 104, 103, 102, 52, 52]
            assert len(predicate_new_order_count) == 101
        else:
            exit('wrong mode in Dataset_choice')
        outp = []
        for i in range(len(num_stage_predicate)):
            opiece = []
            for j in range(len(predicate_new_order_count)):
                opiece.append(0.0)
            num_list = predicate_new_order_count[0:(num_stage_predicate[i]+1)]
            median = np.median(num_list[1:])
            for j in range(len(num_list)):
                if num_list[j] > median:
                    num = median / num_list[j]
                    if j == 0:
                        num = num * 10.0
                    if num < 0.01:
                        num = 0.01
                    opiece[j] = num
                else:
                    opiece[j] = 1.0
            outp.append(opiece)
        return outp

    def generate_current_sequence_for_bias(self, incremental_stage_list, Dataset_choice):
        data_long = 0
        if Dataset_choice == 'VG':
            data_long = 51
        elif Dataset_choice == 'GQA_200':
            data_long = 101
        else:
            exit('wrong mode in Dataset choice')
        outp = []
        for i in range(len(incremental_stage_list)):
            opiece = []
            for j in range(data_long):
                opiece.append(0)
            for j in range(i+1):
                for k in incremental_stage_list[j]:
                    opiece[k] = k
            outp.append(opiece)

        return outp


    def cat(self, tensors, dim=0):
        """
        Efficient version of torch.cat that avoids a copy if there is only a single element in a list
        """
        assert isinstance(tensors, (list, tuple))
        if len(tensors) == 1:
            return tensors[0]
        return torch.cat(tensors, dim)

    def obtain_selected_info_matrix(self, rel_labels):
        rel_labels = self.cat(rel_labels, dim=0)
        max_label = max(rel_labels)
        num_groups = self.incre_idx_list[max_label.item()]
        if num_groups == 0:
            num_groups = max(self.incre_idx_list)
        cur_chosen_matrix = []
        for i in range(num_groups):
            cur_chosen_matrix.append([])

        for i in range(len(rel_labels)):
            rel_tar = rel_labels[i].item()
            if rel_tar == 0:
                if self.zero_label_padding_mode == 'rand_insert':
                    random_idx = random.randint(0, num_groups - 1)
                    cur_chosen_matrix[random_idx].append(i)

            else:
                rel_idx = self.incre_idx_list[rel_tar]
                random_num = random.random()
                for j in range(num_groups):
                    act_idx = num_groups - j
                    threshold_cur = self.sample_rate_matrix[act_idx - 1][rel_tar]
                    if random_num <= threshold_cur or act_idx < rel_idx:
                        # print('%d-%d-%d-%.2f-%.2f'%(i, rel_idx, act_idx, random_num, threshold_cur))
                        for k in range(act_idx):
                            cur_chosen_matrix[k].append(i)
                        break

        rel_labels_lst = []
        for i in range(len(cur_chosen_matrix)):
            expert_labels = rel_labels[cur_chosen_matrix[i]]
            rel_labels_lst.append(expert_labels)

        return cur_chosen_matrix, rel_labels_lst, num_groups







