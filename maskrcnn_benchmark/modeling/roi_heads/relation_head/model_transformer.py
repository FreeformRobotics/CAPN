"""
Based on the implementation of https://github.com/jadore801120/attention-is-all-you-need-pytorch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from maskrcnn_benchmark.modeling.utils import cat
from .utils_motifs import obj_edge_vectors, to_onehot, nms_overlaps, encode_box_info

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q (bsz, len_q, dim_q)
            k (bsz, len_k, dim_k)
            v (bsz, len_v, dim_v)
            Note: len_k==len_v, and dim_q==dim_k
        Returns:
            output (bsz, len_q, dim_v)
            attn (bsz, len_q, len_k)
        """
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):
        """
        Args:
            q (bsz, len_q, dim_q)
            k (bsz, len_k, dim_k)
            v (bsz, len_v, dim_v)
            Note: len_k==len_v, and dim_q==dim_k
        Returns:
            output (bsz, len_q, d_model)
            attn (bsz, len_q, len_k)
        """
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size() # len_k==len_v

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Merge adjacent information. Equal to linear layer if kernel size is 1
        Args:
            x (bsz, len, dim)
        Returns:
            output (bsz, len, dim)
        """
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask.float()

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask.float()

        return enc_output, enc_slf_attn


class TransformerEncoder(nn.Module):
    """
    A encoder model with self attention mechanism.
    """
    def __init__(self, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):
        super().__init__()
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, input_feats, num_objs):
        """
        Args:
            input_feats [Tensor] (#total_box, d_model) : bounding box features of a batch
            num_objs [list of int] (bsz, ) : number of bounding box of each image
        Returns:
            enc_output [Tensor] (#total_box, d_model)
        """
        original_input_feats = input_feats
        input_feats = input_feats.split(num_objs, dim=0)
        input_feats = nn.utils.rnn.pad_sequence(input_feats, batch_first=True)

        # -- Prepare masks
        bsz = len(num_objs)
        device = input_feats.device
        pad_len = max(num_objs)
        num_objs_ = torch.LongTensor(num_objs).to(device).unsqueeze(1).expand(-1, pad_len)
        slf_attn_mask = torch.arange(pad_len, device=device).view(1, -1).expand(bsz, -1).ge(num_objs_).unsqueeze(1).expand(-1, pad_len, -1) # (bsz, pad_len, pad_len)
        non_pad_mask = torch.arange(pad_len, device=device).to(device).view(1, -1).expand(bsz, -1).lt(num_objs_).unsqueeze(-1) # (bsz, pad_len, 1)

        # -- Forward
        enc_output = input_feats
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

        enc_output = enc_output[non_pad_mask.squeeze(-1)]
        return enc_output


class Relation_aware_gating(nn.Module):
    def __init__(self, in_fea_dim, num_experts):
        super(Relation_aware_gating, self).__init__()

        self.fc1 = nn.Linear(in_features=in_fea_dim, out_features=1024, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(in_features=1024, out_features=512, bias=True)
        self.fc3 = nn.Linear(in_features=512, out_features=num_experts)
        self.softmax = nn.Softmax()

    def forward(self, relation_inputs):
        # print('relation_inputs.shape:', relation_inputs.shape)

        relation_inputs = relation_inputs.reshape(relation_inputs.shape[0], relation_inputs.shape[1])
        # print('relation_inputs.shape:', relation_inputs.shape)

        out = self.relu(self.bn1(self.fc1(relation_inputs)))
        # print('out.shape:', out.shape)

        out = self.relu(self.bn2(self.fc2(out)))
        # print('out.shape:', out.shape)

        out = self.fc3(out)
        # print('out.shape:', out.shape)
        # print('out.shape:', out.shape)

        out = out.mean(0)
        #print('out.shape:', out.shape, out)

        out = self.softmax(out)
        # print('out.shape:', out.shape, out)

        return out

class TransformerContext(nn.Module):
    def __init__(self, config, obj_classes, rel_classes, in_channels):
        super().__init__()
        self.cfg = config
        # setting parameters
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            self.mode = 'predcls' if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL else 'sgcls'
        else:
            self.mode = 'sgdet'
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_cls = len(obj_classes)
        self.num_rel_cls = len(rel_classes)
        self.in_channels = in_channels
        self.obj_dim = in_channels
        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES

        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE   
        self.obj_layer = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.OBJ_LAYER      
        self.edge_layer = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.REL_LAYER        
        self.num_head = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD         
        self.inner_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM     
        self.k_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM         
        self.v_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM

        # CAME
        self.num_experts = self.cfg.MODEL.ROI_RELATION_HEAD.NUM_EXPERTS
        self.use_relation_aware_gating = config.MODEL.ROI_RELATION_HEAD.USE_RELATION_AWARE_GATING
        if self.use_relation_aware_gating:
            self.relation_aware_gating = Relation_aware_gating(in_fea_dim=4424, num_experts=self.num_experts)


        # the following word embedding layer should be initalize by glove.6B before using
        embed_vecs = obj_edge_vectors(self.obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        self.obj_embed1 = nn.Embedding(self.num_obj_cls, self.embed_dim)
        self.obj_embed2 = nn.Embedding(self.num_obj_cls, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(embed_vecs, non_blocking=True)
            self.obj_embed2.weight.copy_(embed_vecs, non_blocking=True)

        # position embedding
        self.bbox_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(32, 128), nn.ReLU(inplace=True), nn.Dropout(0.1),
        ])
        self.lin_obj = nn.Linear(self.in_channels + self.embed_dim + 128, self.hidden_dim)
        self.lin_edge = nn.Linear(self.embed_dim + self.hidden_dim + self.in_channels, self.hidden_dim)
        self.out_obj = nn.Linear(self.hidden_dim, self.num_obj_cls)
        self.context_obj = TransformerEncoder(self.obj_layer, self.num_head, self.k_dim, 
                                                self.v_dim, self.hidden_dim, self.inner_dim, self.dropout_rate)
        self.context_edge = TransformerEncoder(self.edge_layer, self.num_head, self.k_dim, 
                                                self.v_dim, self.hidden_dim, self.inner_dim, self.dropout_rate)

    
    def forward(self, roi_features, proposals, logger=None):
        # labels will be used in DecoderRNN during training
        use_gt_label = self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
        obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0) if use_gt_label else None

        # label/logits embedding will be used as input
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_embed = self.obj_embed1(obj_labels)
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed1.weight
        
        # bbox embedding will be used as input
        assert proposals[0].mode == 'xyxy'
        pos_embed = self.bbox_embed(encode_box_info(proposals))

        # encode objects with transformer
        obj_pre_rep = cat((roi_features, obj_embed, pos_embed), -1)

        # add relation aware gating
        if self.use_relation_aware_gating:
            beta_relation_aware_gating = self.relation_aware_gating(obj_pre_rep)

        num_objs = [len(p) for p in proposals]
        obj_pre_rep = self.lin_obj(obj_pre_rep)

        obj_feats = self.context_obj(obj_pre_rep, num_objs)



        # predict obj_dists and obj_preds
        if self.mode == 'predcls':
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_cls)
            edge_pre_rep = cat((roi_features, obj_feats, self.obj_embed2(obj_labels)), dim=-1)

        else:
            obj_dists = self.out_obj(obj_feats)
            use_decoder_nms = self.mode == 'sgdet' and not self.training
            if use_decoder_nms:
                boxes_per_cls = [proposal.get_field('boxes_per_cls') for proposal in proposals]
                obj_preds = self.nms_per_cls(obj_dists, boxes_per_cls, num_objs)
            else:
                obj_preds = obj_dists[:, 1:].max(1)[1] + 1
            edge_pre_rep = cat((roi_features, obj_feats, self.obj_embed2(obj_preds)), dim=-1)

        # edge context
        edge_pre_rep = self.lin_edge(edge_pre_rep)
        # print('edge_pre_rep:', edge_pre_rep.shape)
        edge_ctx = self.context_edge(edge_pre_rep, num_objs)
        # print('edge_ctx:', edge_ctx.shape)

        if self.use_relation_aware_gating:
            return obj_dists, obj_preds, edge_ctx, beta_relation_aware_gating
        else:
            return obj_dists, obj_preds, edge_ctx

    def nms_per_cls(self, obj_dists, boxes_per_cls, num_objs):
        obj_dists = obj_dists.split(num_objs, dim=0)
        obj_preds = []
        for i in range(len(num_objs)):
            is_overlap = nms_overlaps(boxes_per_cls[i]).cpu().numpy() >= self.nms_thresh # (#box, #box, #class)

            out_dists_sampled = F.softmax(obj_dists[i], -1).cpu().numpy()
            out_dists_sampled[:, 0] = -1

            out_label = obj_dists[i].new(num_objs[i]).fill_(0)

            for i in range(num_objs[i]):
                box_ind, cls_ind = np.unravel_index(out_dists_sampled.argmax(), out_dists_sampled.shape)
                out_label[int(box_ind)] = int(cls_ind)
                out_dists_sampled[is_overlap[box_ind,:,cls_ind], cls_ind] = 0.0
                out_dists_sampled[box_ind] = -1.0 # This way we won't re-sample

            obj_preds.append(out_label.long())
        obj_preds = torch.cat(obj_preds, dim=0)
        return obj_preds


class TransformerContext_split(nn.Module):
    def __init__(self, config, obj_classes, rel_classes, in_channels):
        super().__init__()
        self.cfg = config
        # setting parameters
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            self.mode = 'predcls' if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL else 'sgcls'
        else:
            self.mode = 'sgdet'
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_cls = len(obj_classes)
        self.num_rel_cls = len(rel_classes)
        self.in_channels = in_channels
        self.obj_dim = in_channels
        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES

        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        self.obj_layer = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.OBJ_LAYER
        self.edge_layer = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.REL_LAYER
        self.num_head = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD
        self.inner_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM
        self.k_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM
        self.v_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM
        self.num_experts = config.MODEL.ROI_RELATION_HEAD.NUM_EXPERTS


        # the following word embedding layer should be initalize by glove.6B before using
        embed_vecs = obj_edge_vectors(self.obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        self.obj_embed1 = nn.Embedding(self.num_obj_cls, self.embed_dim)
        self.obj_embed2 = nn.Embedding(self.num_obj_cls, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(embed_vecs, non_blocking=True)
            self.obj_embed2.weight.copy_(embed_vecs, non_blocking=True)

        # position embedding
        self.bbox_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(32, 128), nn.ReLU(inplace=True), nn.Dropout(0.1),
        ])
        self.lin_obj = nn.Linear(self.in_channels + self.embed_dim + 128, self.hidden_dim)
        self.lin_edge = nn.Linear(self.embed_dim + self.hidden_dim + self.in_channels, self.hidden_dim)
        self.out_obj = nn.Linear(self.hidden_dim, self.num_obj_cls)

        if self.num_experts > 1:
            self.context_obj = TransformerEncoder(self.obj_layer, self.num_head, self.k_dim,
                                                  self.v_dim, self.hidden_dim, self.inner_dim, self.dropout_rate)

            self.context_edge = nn.ModuleList(
                [TransformerEncoder(self.edge_layer, self.num_head, self.k_dim,
                                   self.v_dim, self.hidden_dim, self.inner_dim, self.dropout_rate) for _ in range(self.num_experts)]
            )
        else:
            self.context_obj = TransformerEncoder(self.obj_layer, self.num_head, self.k_dim,
                                                  self.v_dim, self.hidden_dim, self.inner_dim, self.dropout_rate)
            self.context_edge = TransformerEncoder(self.edge_layer, self.num_head, self.k_dim,
                                                   self.v_dim, self.hidden_dim, self.inner_dim, self.dropout_rate)

    def forward(self, roi_features, proposals, logger=None):
        # labels will be used in DecoderRNN during training
        use_gt_label = self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
        obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0) if use_gt_label else None

        # label/logits embedding will be used as input
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_embed = self.obj_embed1(obj_labels)
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed1.weight

        # bbox embedding will be used as input
        assert proposals[0].mode == 'xyxy'
        pos_embed = self.bbox_embed(encode_box_info(proposals))

        # encode objects with transformer
        obj_pre_rep = cat((roi_features, obj_embed, pos_embed), -1)
        num_objs = [len(p) for p in proposals]
        obj_pre_rep = self.lin_obj(obj_pre_rep)
        obj_feats = self.context_obj(obj_pre_rep, num_objs)

        # predict obj_dists and obj_preds
        if self.mode == 'predcls':
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_cls)
            edge_pre_rep = cat((roi_features, obj_feats, self.obj_embed2(obj_labels)), dim=-1)

        else:
            obj_dists = self.out_obj(obj_feats)
            use_decoder_nms = self.mode == 'sgdet' and not self.training
            if use_decoder_nms:
                boxes_per_cls = [proposal.get_field('boxes_per_cls') for proposal in proposals]
                obj_preds = self.nms_per_cls(obj_dists, boxes_per_cls, num_objs)
            else:
                obj_preds = obj_dists[:, 1:].max(1)[1] + 1
            edge_pre_rep = cat((roi_features, obj_feats, self.obj_embed2(obj_preds)), dim=-1)

        # edge context
        edge_pre_rep = self.lin_edge(edge_pre_rep)
        outs = []
        if self.num_experts > 1:
            for experts_index in range(self.num_experts):
                out = self.context_edge[experts_index](edge_pre_rep, num_objs)
                outs.append(out)

            final_out = torch.stack(outs, dim=1).mean(dim=1)
            edge_ctx = {"output": final_out, "logits": torch.stack(outs, dim=1)}

        else:
            edge_ctx = self.context_edge(edge_pre_rep, num_objs)

        return obj_dists, obj_preds, edge_ctx

    def nms_per_cls(self, obj_dists, boxes_per_cls, num_objs):
        obj_dists = obj_dists.split(num_objs, dim=0)
        obj_preds = []
        for i in range(len(num_objs)):
            is_overlap = nms_overlaps(boxes_per_cls[i]).cpu().numpy() >= self.nms_thresh  # (#box, #box, #class)

            out_dists_sampled = F.softmax(obj_dists[i], -1).cpu().numpy()
            out_dists_sampled[:, 0] = -1

            out_label = obj_dists[i].new(num_objs[i]).fill_(0)

            for i in range(num_objs[i]):
                box_ind, cls_ind = np.unravel_index(out_dists_sampled.argmax(), out_dists_sampled.shape)
                out_label[int(box_ind)] = int(cls_ind)
                out_dists_sampled[is_overlap[box_ind, :, cls_ind], cls_ind] = 0.0
                out_dists_sampled[box_ind] = -1.0  # This way we won't re-sample

            obj_preds.append(out_label.long())
        obj_preds = torch.cat(obj_preds, dim=0)
        return obj_preds
