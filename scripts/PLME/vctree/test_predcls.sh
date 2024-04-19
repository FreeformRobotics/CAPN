#!/bin/bash
export CUDA_VISIBLE_DEVICES="6,7"

python -m torch.distributed.launch --master_port 22275 --nproc_per_node=2 tools/relation_test_net.py \
       --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
       --loss_option PLME_LOSS \
       --num_experts 3 \
       MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
       MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
       MODEL.ROI_RELATION_HEAD.USE_RELATION_AWARE_GATING False \
       MODEL.ROI_RELATION_HEAD.USE_PER_CLASS_CONTEXT_AWARE False \
       MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor_PL \
       MODEL.ROI_RELATION_HEAD.EXPERT_MODE 'hbt_b_t' \
       TEST.IMS_PER_BATCH 12 \
       DTYPE "float16" \
       GLOVE_DIR ./glove \
       MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
       OUTPUT_DIR ./checkpoints/vctree-PL-precls-hmt_m_t/ \
       logs/vctree-PL-predcls-hbt_predicate_label_lst_predicate_scores_matrix.txt 2>&1 &
