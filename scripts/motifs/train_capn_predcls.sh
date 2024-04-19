
export CUDA_VISIBLE_DEVICES=2,3

python -m torch.distributed.launch --master_port 41175 --nproc_per_node=2 tools/relation_train_net.py \
       --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
       --loss_option CAME_LOSS_WO_RW \
       --num_experts 3 \
       MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
       MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
       MODEL.ROI_RELATION_HEAD.USE_RELATION_AWARE_GATING True \
       MODEL.ROI_RELATION_HEAD.USE_PER_CLASS_CONTEXT_AWARE False \
       MODEL.ROI_RELATION_HEAD.PER_CLASS_ALPHA 1.0 \
       MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor_CAME \
       SOLVER.IMS_PER_BATCH 12 \
       TEST.IMS_PER_BATCH 12 \
       DTYPE "float16" \
       SOLVER.MAX_ITER 50000 \
       SOLVER.VAL_PERIOD 2000 \
       SOLVER.CHECKPOINT_PERIOD 2000 \
       GLOVE_DIR ./glove \
       MODEL.PRETRAINED_DETECTOR_CKPT ../Scene-Graph-Benchmark.pytorch/checkpoints/pretrained_faster_rcnn/model_final.pth \
       OUTPUT_DIR ./checkpoints/motif-predcls-CAME_LOSS_WO_RW_3_1.0

