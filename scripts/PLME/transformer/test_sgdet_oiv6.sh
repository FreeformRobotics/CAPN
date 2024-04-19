
export CUDA_VISIBLE_DEVICES=4,5

python -m torch.distributed.launch --master_port 98275 --nproc_per_node=2 tools/relation_test_net.py \
       --config-file "configs/e2e_rel_ovi6.yaml" \
       --loss_option PLME_LOSS \
       --num_experts 3 \
       MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
       MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
       MODEL.ROI_RELATION_HEAD.USE_RELATION_AWARE_GATING False \
       MODEL.ROI_RELATION_HEAD.USE_PER_CLASS_CONTEXT_AWARE False \
       MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor_PL \
       MODEL.ROI_RELATION_HEAD.EXPERT_MODE 'hmt_ht_mt' \
       TEST.IMS_PER_BATCH 2 \
       DTYPE "float16" \
       GLOVE_DIR ./glove \
       MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_detector/oiv6_det.pth \
       OUTPUT_DIR ./checkpoints/transformer-PL-sgdet-PLME_LOSS-oviv6


