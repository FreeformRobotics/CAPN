#!/bin/bash
# source activate sg_benchmark
export CUDA_VISIBLE_DEVICES=4,5

python -m torch.distributed.launch --master_port 45541 --nproc_per_node=2 tools/relation_train_net.py \
       --config-file "configs/e2e_rel_ovi6.yaml" \
       --loss_option CB_LOSS \
       --num_experts 1 \
       MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
       MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
       MODEL.ROI_RELATION_HEAD.USE_RELATION_AWARE_GATING False \
       MODEL.ROI_RELATION_HEAD.USE_PER_CLASS_CONTEXT_AWARE False \
       MODEL.ROI_RELATION_HEAD.PER_CLASS_ALPHA 1.0 \
       MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor \
       SOLVER.IMS_PER_BATCH 12 \
       TEST.IMS_PER_BATCH 2 \
       DTYPE "float16" \
       SOLVER.MAX_ITER 50000 \
       SOLVER.VAL_PERIOD 2000 \
       SOLVER.CHECKPOINT_PERIOD 2000 \
       SOLVER.PRE_VAL False \
       SOLVER.BASE_LR 0.001 \
       GLOVE_DIR ./glove \
       MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_detector/oiv6_det.pth \
       OUTPUT_DIR ./checkpoints/transformer-sgdet-CB_LOSS-oviv6-33864 > logs/ECB/transformer/sgdet-CB_LOSS-oviv6-33864.log 2>&1 & 

