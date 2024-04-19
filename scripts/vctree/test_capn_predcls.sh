
export CUDA_VISIBLE_DEVICES=6,7

#python -m torch.distributed.launch --master_port 31175 --nproc_per_node=2 tools/relation_test_net.py \
#       --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
#       --loss_option CROSS_ENTROPY_LOSS \
#       --num_experts 1 \
#       MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
#       MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
#       MODEL.ROI_RELATION_HEAD.USE_RELATION_AWARE_GATING False \
#       MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor \
#       TEST.IMS_PER_BATCH 12 \
#       DTYPE "float16" \
#       GLOVE_DIR ./glove \
#       MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
#       OUTPUT_DIR ./checkpoints/vctree-precls-cross-entropy/

python -m torch.distributed.launch --master_port 31167 --nproc_per_node=2 tools/relation_test_net.py \
       --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
       --loss_option CAME_LOSS \
       --num_experts 3 \
       MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
       MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
       MODEL.ROI_RELATION_HEAD.USE_RELATION_AWARE_GATING True \
       MODEL.ROI_RELATION_HEAD.USE_PER_CLASS_CONTEXT_AWARE True \
       MODEL.ROI_RELATION_HEAD.PER_CLASS_ALPHA 0.5 \
       MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor_CAME \
       TEST.IMS_PER_BATCH 12 \
       DTYPE "float16" \
       GLOVE_DIR ./glove \
       MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
       OUTPUT_DIR ./checkpoints/vctree-precls-CAME_LOSS_0.5/ \
       > logs/CAME/vctree_predcls_output_all_1.txt 2>&1 &