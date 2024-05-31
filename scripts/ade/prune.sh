#!/bin/bash/


DATASET='ADE'
MODEL_NAME='deeplabv3'
MODEL_PARA='{}'
BACKBONE='resnet50'
BACKBONE_PARA='{"mg_unit":[1,1,1]}'
ALIGN_CORNER='False'
PRUNE_RATIO=0.6
SNAPSHOT_DIR=ckpt/ADE/dcfp_pretrain_deeplabv3/dcfp_prune_06
RESUME_DIR=ckpt/ADE/dcfp_pretrain_deeplabv3/ADE_scenes_16000.pth
SCORE_DIR=ckpt/ADE/dcfp_pretrain_deeplabv3/score.pth

python prune.py --dataset ${DATASET} --model ${MODEL_NAME} --model-para ${MODEL_PARA} --backbone ${BACKBONE} --backbone-para ${BACKBONE_PARA} --align-corner ${ALIGN_CORNER} --prune-ratio ${PRUNE_RATIO} --save-path ${SNAPSHOT_DIR} --model-path ${RESUME_DIR} --score-path ${SCORE_DIR} 



