# CAME Unbiased Scene Graph in Pytorch

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-1.4.0-%237732a8)

Pytorch implementation of paper:

CAME: Context-aware Mixture-of-Experts for Unbiased Scene Graph Generation 



## Contents

1. [Overview](#Overview)
2. [Install the Requirements](INSTALL.md)
3. [Prepare the Dataset](DATASET.md)
4. [Metrics and Results for our Toolkit](METRICS.md)
    - [Explanation of R@K, mR@K, zR@K, ng-R@K, ng-mR@K, ng-zR@K, A@K, S2G](METRICS.md#explanation-of-our-metrics)
    - [Output Format](METRICS.md#output-format-of-our-code)
    - [Reported Results](METRICS.md#reported-results)
5. [Faster R-CNN Pre-training](#pretrained-models)
6. [Scene Graph Generation as RoI_Head](#scene-graph-generation-as-RoI_Head)
7. [Training on Scene Graph Generation](#perform-training-on-scene-graph-generation)
8. [Evaluation on Scene Graph Generation](#Evaluation)
9. [**Detect Scene Graphs on Your Custom Images** :star2:](#SGDet-on-custom-images)
10. [**Visualize Detected Scene Graphs of Custom Images** :star2:](#Visualize-Detected-SGs-of-Custom-Images)
11. [Other Options that May Improve the SGG](#other-options-that-may-improve-the-SGG)
12. [Tips and Tricks for TDE on any Unbiased Task](#tips-and-Tricks-for-any-unbiased-taskX-from-biased-training)
13. [Frequently Asked Questions](#frequently-asked-questions)
14. [Citations](#Citations)

## Overview

This project aims to build a new CODEBASE of Scene Graph Generation (SGG), and it is also a Pytorch implementation of the paper [Unbiased Scene Graph Generation from Biased Training](https://arxiv.org/abs/2002.11949). The previous widely adopted SGG codebase [neural-motifs](https://github.com/rowanz/neural-motifs) is detached from the recent development of Faster/Mask R-CNN. Therefore, I decided to build a scene graph benchmark on top of the well-known [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) project and define relationship prediction as an additional roi_head. By the way, thanks to their elegant framework, this codebase is much more novice-friendly and easier to read/modify for your own projects than previous neural-motifs framework(at least I hope so). It is a pity that when I was working on this project, the [detectron2](https://github.com/facebookresearch/detectron2) had not been released, but I think we can consider [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) as a more stable version with less bugs, hahahaha. I also introduce all the old and new metrics used in SGG, and clarify two common misunderstandings in SGG metrics in [METRICS.md](METRICS.md), which cause abnormal results in some papers.

### 

Models | SGGen R@20 | SGGen R@50 | SGGen R@100 | SGCls R@20 | SGCls R@50 | SGCls R@100 | PredCls R@20 | PredCls R@50 | PredCls R@100
-- | -- | -- | -- | -- | -- | -- | -- | -- | -- 
VCTree | 24.53 | 31.93 | 36.21 | 42.77 | 46.67 | 47.64 | 59.02 | 65.42 | 67.18

Note that all results of VCTree should be better than what we reported in [Unbiased Scene Graph Generation from Biased Training](https://arxiv.org/abs/2002.11949), because we optimized the tree construction network after the publication.

### The illustration of the Unbiased SGG from 'Unbiased Scene Graph Generation from Biased Training'

![alt text](demo/teaser_figure.png "from 'Unbiased Scene Graph Generation from Biased Training'")

## Installation

Check [INSTALL.md](INSTALL.md) for installation instructions.

## Dataset

Check [DATASET.md](DATASET.md) for instructions of dataset preprocessing.

## Metrics and Results **(IMPORTANT)**
Explanation of metrics in our toolkit and reported results are given in [METRICS.md](METRICS.md)

## Pretrained Models

Since we tested many SGG models in our paper [Unbiased Scene Graph Generation from Biased Training](https://arxiv.org/abs/2002.11949), I won't upload all the pretrained SGG models here. However, you can download the [pretrained Faster R-CNN](https://onedrive.live.com/embed?cid=22376FFAD72C4B64&resid=22376FFAD72C4B64%21779870&authkey=AH5CPVb9g5E67iQ) we used in the paper, which is the most time consuming step in the whole training process (it took 4 2080ti GPUs). As to the SGG model, you can follow the rest instructions to train your own, which only takes 2 GPUs to train each SGG model. The results should be very close to the reported results given in [METRICS.md](METRICS.md)

After you download the [Faster R-CNN model](https://onedrive.live.com/embed?cid=22376FFAD72C4B64&resid=22376FFAD72C4B64%21779870&authkey=AH5CPVb9g5E67iQ), please extract all the files to the directory `/home/username/checkpoints/pretrained_faster_rcnn`. To train your own Faster R-CNN model, please follow the next section.

The above pretrained Faster R-CNN model achives 38.52/26.35/28.14 mAp on VG train/val/test set respectively.

## Faster R-CNN pre-training
The following command can be used to train your own Faster R-CNN model:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 10001 --nproc_per_node=4 tools/detector_pretrain_net.py --config-file "configs/e2e_relation_detector_X_101_32_8_FPN_1x.yaml" SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 4 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.STEPS "(30000, 45000)" SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 MODEL.RELATION_ON False OUTPUT_DIR /home/kaihua/checkpoints/pretrained_faster_rcnn SOLVER.PRE_VAL False
```
where ```CUDA_VISIBLE_DEVICES``` and ```--nproc_per_node``` represent the id of GPUs and number of GPUs you use, ```--config-file``` means the config we use, where you can change other parameters. ```SOLVER.IMS_PER_BATCH``` and ```TEST.IMS_PER_BATCH``` are the training and testing batch size respectively, ```DTYPE "float16"``` enables Automatic Mixed Precision supported by [APEX](https://github.com/NVIDIA/apex), ```SOLVER.MAX_ITER``` is the maximum iteration, ```SOLVER.STEPS``` is the steps where we decay the learning rate, ```SOLVER.VAL_PERIOD``` and ```SOLVER.CHECKPOINT_PERIOD``` are the periods of conducting val and saving checkpoint, ```MODEL.RELATION_ON``` means turning on the relationship head or not (since this is the pretraining phase for Faster R-CNN only, we turn off the relationship head),  ```OUTPUT_DIR``` is the output directory to save checkpoints and log (considering `/home/username/checkpoints/pretrained_faster_rcnn`), ```SOLVER.PRE_VAL``` means whether we conduct validation before training or not.


## Scene Graph Generation as RoI_Head

To standardize the SGG, I define scene graph generation as an RoI_Head. Referring to the design of other roi_heads like box_head, I put most of the SGG codes under ```maskrcnn_benchmark/modeling/roi_heads/relation_head``` and their calling sequence is as follows:

![alt text](demo/relation_head.png "structure of relation_head")


## Perform training on Scene Graph Generation

There are **three standard protocols**: (1) Predicate Classification (PredCls): taking ground truth bounding boxes and labels as inputs, (2) Scene Graph Classification (SGCls) : using ground truth bounding boxes without labels, (3) Scene Graph Detection (SGDet): detecting SGs from scratch. We use two switches ```MODEL.ROI_RELATION_HEAD.USE_GT_BOX``` and ```MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL``` to select the protocols. 

For **Predicate Classification (PredCls)**, we need to set:
``` bash
MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True
```
For **Scene Graph Classification (SGCls)**:
``` bash
MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False
```
For **Scene Graph Detection (SGDet)**:
``` bash
MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False
```

### Predefined Models
We abstract various SGG models to be different ```relation-head predictors``` in the file ```roi_heads/relation_head/roi_relation_predictors.py```, which are independent of the Faster R-CNN backbone and relation-head feature extractor. To select our predefined models, you can use ```MODEL.ROI_RELATION_HEAD.PREDICTOR```.

For [Neural-MOTIFS](https://arxiv.org/abs/1711.06640) Model:
```bash
MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor
```
For [Iterative-Message-Passing(IMP)](https://arxiv.org/abs/1701.02426) Model (Note that SOLVER.BASE_LR should be changed to 0.001 in SGCls, or the model won't converge):
```bash
MODEL.ROI_RELATION_HEAD.PREDICTOR IMPPredictor
```
For [VCTree](https://arxiv.org/abs/1812.01880) Model:
```bash
MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor
```
For our predefined Transformer Model (Note that Transformer Model needs to change SOLVER.BASE_LR to 0.001, SOLVER.SCHEDULE.TYPE to WarmupMultiStepLR, SOLVER.MAX_ITER to 16000, SOLVER.IMS_PER_BATCH to 16, SOLVER.STEPS to (10000, 16000).), which is provided by [Jiaxin Shi](https://github.com/shijx12):
```bash
MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor
```
For [Unbiased-Causal-TDE](https://arxiv.org/abs/2002.11949) Model:
```bash
MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor
```

The default settings are under ```configs/e2e_relation_X_101_32_8_FPN_1x.yaml``` and ```maskrcnn_benchmark/config/defaults.py```. The priority is ```command > yaml > defaults.py```

### Customize Your Own Model
If you want to customize your own model, you can refer ```maskrcnn-benchmark/modeling/roi_heads/relation_head/model_XXXXX.py``` and ```maskrcnn-benchmark/modeling/roi_heads/relation_head/utils_XXXXX.py```. You also need to add corresponding nn.Module in ```maskrcnn-benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py```. Sometimes you may also need to change the inputs & outputs of the module through ```maskrcnn-benchmark/modeling/roi_heads/relation_head/relation_head.py```.



### Examples of the Training Command
Training Example 1 : (PreCls, Motif Model)
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 10025 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR /home/kaihua/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/kaihua/checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR /home/kaihua/checkpoints/motif-precls-exmp
```
Training Example 1 : (PreCls, Motif Model, CAME4)

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 30085 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml"  --loss_option CAME_LOSS --num_experts 4 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.USE_RELATION_AWARE_GATING True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/motif-precls-CAME-4
```

where ```GLOVE_DIR``` is the directory used to save glove initializations, ```MODEL.PRETRAINED_DETECTOR_CKPT``` is the pretrained Faster R-CNN model you want to load, ```OUTPUT_DIR``` is the output directory used to save checkpoints and the log. Since we use the ```WarmupReduceLROnPlateau``` as the learning scheduler for SGG, ```SOLVER.STEPS``` is not required anymore.




## Evaluation

### Examples of the Test Command
Test Example 1 : (PreCls, Motif Model)
```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10027 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR /home/kaihua/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/kaihua/checkpoints/motif-precls-exmp OUTPUT_DIR /home/kaihua/checkpoints/motif-precls-exmp
```

Test Example 2 : (PreCls, Motif Model, CAME4)
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 12027 --nproc_per_node=2 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml"  --loss_option CAME_LOSS --num_experts 4 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.USE_RELATION_AWARE_GATING True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor TEST.IMS_PER_BATCH 12 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/motif-precls-CAME-4
```



## SGDet on Custom Images

Note that evaluation on custum images is only applicable for SGDet model, because PredCls and SGCls model requires additional ground-truth bounding boxes information. To detect scene graphs into a json file on your own images, you need to turn on the switch TEST.CUSTUM_EVAL and give a folder path that contains the custom images to TEST.CUSTUM_PATH. Only JPG files are allowed. The output will be saved as custom_prediction.json in the given DETECTED_SGG_DIR.

Test Example 1 : (SGDet, **Causal TDE**, MOTIFS Model, SUM Fusion) [(checkpoint)](https://onedrive.live.com/embed?cid=22376FFAD72C4B64&resid=22376FFAD72C4B64%21781947&authkey=AF_EM-rkbMyT3gs)
```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10027 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR /home/kaihua/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/kaihua/checkpoints/causal-motifs-sgdet OUTPUT_DIR /home/kaihua/checkpoints/causal-motifs-sgdet TEST.CUSTUM_EVAL True TEST.CUSTUM_PATH /home/kaihua/checkpoints/custom_images DETECTED_SGG_DIR /home/kaihua/checkpoints/your_output_path
```

Test Example 2 : (SGDet, **Original**, MOTIFS Model, SUM Fusion) [(same checkpoint)](https://onedrive.live.com/embed?cid=22376FFAD72C4B64&resid=22376FFAD72C4B64%21781947&authkey=AF_EM-rkbMyT3gs)
```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10027 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE none MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR /home/kaihua/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/kaihua/checkpoints/causal-motifs-sgdet OUTPUT_DIR /home/kaihua/checkpoints/causal-motifs-sgdet TEST.CUSTUM_EVAL True TEST.CUSTUM_PATH /home/kaihua/checkpoints/custom_images DETECTED_SGG_DIR /home/kaihua/checkpoints/your_output_path
```

The output is a json file. For each image, the scene graph information is saved as a dictionary containing bbox(sorted), bbox_labels(sorted), bbox_scores(sorted), rel_pairs(sorted), rel_labels(sorted), rel_scores(sorted), rel_all_scores(sorted), where the last rel_all_scores give all 51 predicates probability for each pair of objects. The dataset information is saved as custom_data_info.json in the same DETECTED_SGG_DIR.

## Visualize Detected SGs of Custom Images
To visualize the detected scene graphs of custom images, you can follow the jupyter note: [visualization/3.visualize_custom_SGDet.jpynb](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/master/visualization/3.visualize_custom_SGDet.ipynb). The inputs of our visualization code are custom_prediction.json and custom_data_info.json in DETECTED_SGG_DIR. They will be automatically generated if you run the above custom SGDet instruction successfully. Note that there may be too much trivial bounding boxes and relationships, so you can select top-k bbox and predicates for better scene graphs by change parameters box_topk and rel_topk. 



## Citations

If you find this project helps your research, please kindly consider citing our project or papers in your publications.

```

```
