
export CUDA_VISIBLE_DEVICES=0,1

python -m torch.distributed.launch --master_port 31375 --nproc_per_node=2 tools/relation_test_net.py \
       --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
       --loss_option CAME_LOSS \
       --num_experts 4 \
       MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
       MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
       MODEL.ROI_RELATION_HEAD.USE_RELATION_AWARE_GATING True \
       MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor \
       TEST.IMS_PER_BATCH 2 \
       DTYPE "float16" \
       GLOVE_DIR ./glove \
       MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
       OUTPUT_DIR ./checkpoints/vctree-sgdet-CAME-4