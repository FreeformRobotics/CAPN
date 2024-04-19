#!/bin/bash
export OMP_NUM_THREADS=1
export gpu_num=2
export CUDA_VISIBLE_DEVICES="2,3"

python -m torch.distributed.launch --master_port 23285 --nproc_per_node=2 \
      tools/relation_test_net.py \
      --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
      --loss_option PLME_LOSS \
      --num_experts 3 \
      MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
      MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
      MODEL.ROI_RELATION_HEAD.USE_RELATION_AWARE_GATING False \
      MODEL.ROI_RELATION_HEAD.USE_PER_CLASS_CONTEXT_AWARE False \
      MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor_PL \
      MODEL.ROI_RELATION_HEAD.EXPERT_MODE 'hb_ht_bt' \
      TEST.IMS_PER_BATCH 2 \
      DTYPE "float16" \
      GLOVE_DIR ../Scene-Graph-Benchmark.pytorch/glove \
      MODEL.PRETRAINED_DETECTOR_CKPT ../Scene-Graph-Benchmark.pytorch/checkpoints/pretrained_faster_rcnn/model_final.pth \
      OUTPUT_DIR ./checkpoints/motif-PL-sgcls-hbt_bt_t/ > logs/train_motif-PL-sgcls-hbt_bt_t_predicate_labels_predicate_scores.txt 2>&1 &


