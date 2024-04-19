
export CUDA_VISIBLE_DEVICES=6,7

nohup python -m torch.distributed.launch --master_port 98175 --nproc_per_node=2 tools/relation_test_net.py \
       --config-file "configs/e2e_rel_ovi6.yaml" \
       --loss_option PLME_LOSS \
       --num_experts 3 \
       MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
       MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
       MODEL.ROI_RELATION_HEAD.USE_RELATION_AWARE_GATING False \
       MODEL.ROI_RELATION_HEAD.USE_PER_CLASS_CONTEXT_AWARE False \
       MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor_PL \
       MODEL.ROI_RELATION_HEAD.EXPERT_MODE 'hmt_ht_mt' \
       SOLVER.IMS_PER_BATCH 12 \
       TEST.IMS_PER_BATCH 2 \
       DTYPE "float16" \
       GLOVE_DIR ./glove \
       MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_detector/oiv6_det.pth \
       OUTPUT_DIR ./checkpoints/motifs-PL-sgdet-PLME_LOSS-oviv6 > logs/motifs-PL-sgdet-PLME_LOSS-oviv6-hmt_ht_mt.txt 2>&1 &

       # checkpointer.load(cfg.MODEL.PRETRAINED_DETECTOR_CKPT, with_optim=False, load_mapping=load_mapping)



