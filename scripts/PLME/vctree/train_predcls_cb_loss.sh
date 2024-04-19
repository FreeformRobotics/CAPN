#!/bin/bash
# source activate sg_benchmark

export OMP_NUM_THREADS=1
export gpu_num=2
export CUDA_VISIBLE_DEVICES="4,5"

python -m torch.distributed.launch --master_port 33085 --nproc_per_node=2 \
      tools/relation_train_net.py \
      --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
      --loss_option CB_LOSS \
      --num_experts 1 \
      MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
      MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
      MODEL.ROI_RELATION_HEAD.USE_RELATION_AWARE_GATING False \
      MODEL.ROI_RELATION_HEAD.USE_PER_CLASS_CONTEXT_AWARE False \
      MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor \
      SOLVER.IMS_PER_BATCH 12 \
      TEST.IMS_PER_BATCH 12 \
      DTYPE "float16" \
      SOLVER.MAX_ITER 20000 \
      SOLVER.VAL_PERIOD 4000 \
      SOLVER.CHECKPOINT_PERIOD 2000 \
      GLOVE_DIR ../Scene-Graph-Benchmark.pytorch/glove \
      MODEL.PRETRAINED_DETECTOR_CKPT ../Scene-Graph-Benchmark.pytorch/checkpoints/pretrained_faster_rcnn/model_final.pth \
      OUTPUT_DIR ./checkpoints/vctree-predcls-CB_LOSS


