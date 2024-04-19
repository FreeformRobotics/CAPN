#!/bin/bash
# source activate sg_benchmark
export CUDA_VISIBLE_DEVICES=6,7

nohup python -m torch.distributed.launch --master_port 67761 --nproc_per_node=2 tools/relation_train_net.py \
       --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
       --loss_option CB_LOSS \
       --num_experts 1 \
       MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
       MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
       MODEL.ROI_RELATION_HEAD.USE_RELATION_AWARE_GATING False \
       MODEL.ROI_RELATION_HEAD.USE_PER_CLASS_CONTEXT_AWARE False \
       MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor \
       SOLVER.IMS_PER_BATCH 12 \
       TEST.IMS_PER_BATCH 2 \
       DTYPE "float16" \
       SOLVER.MAX_ITER 20000 \
       SOLVER.VAL_PERIOD 2000 \
       SOLVER.CHECKPOINT_PERIOD 2000 \
       SOLVER.PRE_VAL False \
       GLOVE_DIR ./glove \
       MODEL.PRETRAINED_DETECTOR_CKPT ../Scene-Graph-Benchmark.pytorch/checkpoints/pretrained_faster_rcnn/model_final.pth \
       OUTPUT_DIR ./checkpoints/vctree-sgdet-CB_LOSS_33864 > logs/vctree-sgdet-CB_LOSS_33864.log 2>&1 &
