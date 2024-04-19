
export CUDA_VISIBLE_DEVICES=2

python -m torch.distributed.launch --master_port 98175 --nproc_per_node=1 tools/relation_train_net.py \
       --config-file "configs/e2e_rel_ovi6.yaml" \
       --loss_option CAME_LOSS \
       --num_experts 3 \
       MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
       MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
       MODEL.ROI_RELATION_HEAD.USE_RELATION_AWARE_GATING True \
       MODEL.ROI_RELATION_HEAD.USE_PER_CLASS_CONTEXT_AWARE True \
       MODEL.ROI_RELATION_HEAD.PER_CLASS_ALPHA 0.25 \
       MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor_CAME \
       SOLVER.IMS_PER_BATCH 8 \
       TEST.IMS_PER_BATCH 1 \
       DTYPE "float16" \
       SOLVER.MAX_ITER 50000 \
       SOLVER.VAL_PERIOD 10 \
       SOLVER.CHECKPOINT_PERIOD 2000 \
       GLOVE_DIR ./glove \
       MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_detector/oiv6_det.pth \
       OUTPUT_DIR ./checkpoints/sgdet_plain

       # checkpointer.load(cfg.MODEL.PRETRAINED_DETECTOR_CKPT, with_optim=False, load_mapping=load_mapping)



