
export CUDA_VISIBLE_DEVICES=4,5
# conda activate usgg

nohup python -m torch.distributed.launch --master_port 43175 --nproc_per_node=2 tools/relation_train_net.py \
       --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
       --loss_option CROSS_ENTROPY_LOSS \
       --num_experts 1 \
       MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
       MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
       MODEL.ROI_RELATION_HEAD.USE_RELATION_AWARE_GATING False \
       MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor \
       MODEL.ROI_RELATION_HEAD.USE_PER_CLASS_CONTEXT_AWARE False \
       SOLVER.IMS_PER_BATCH 12 \
       TEST.IMS_PER_BATCH 12 \
       DTYPE "float16" \
       SOLVER.MAX_ITER 50000 \
       SOLVER.VAL_PERIOD 2000 \
       SOLVER.BASE_LR 0.0001 \
       SOLVER.CHECKPOINT_PERIOD 2000 \
       GLOVE_DIR ./glove \
       MODEL.PRETRAINED_DETECTOR_CKPT ../Scene-Graph-Benchmark.pytorch/checkpoints/pretrained_faster_rcnn/model_final.pth \
       OUTPUT_DIR ./checkpoints/transformer-predcls-CROSS_ENTROPY_LOSS \
       > logs/training_transformer-predcls-CROSS_ENTROPY_LOSS.txt 2>&1 &

