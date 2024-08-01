#!/bin/bash

DATA_NAME=IWSLT15
FOLDER_PATH=/work/dat-nt/Machine_Translation/nmt_refactor/nmt_custom
DATA_PATH=${FOLDER_PATH}/${DATA_NAME}
VOCAB_NAME=bert_30_07.pth
VOCAB_PATH=${FOLDER_PATH}/vocab/${VOCAB_NAME}
MODEL_NAME=model_IWSLT15_training_IWSLT_envit5_reverse_v0.pth
MODEL_PATH=${FOLDER_PATH}/model/${MODEL_NAME}
CHECKPOINT_NAME=model_IWSLT15_training_phoMT_envit5_v0.pth
CHECKPOINT_PATH=${FOLDER_PATH}/model/{CHECKPOINT_NAME}
python3 /work/dat-nt/Machine_Translation/nmt_refactor/nmt_custom/main.py \
    --src_language "en" \
    --tgt_language "vi" \
    --data_name $DATA_PATH \
    --emb_size 1024 \
    --nhead 8 \
    --ffn_hid_dim 2048 \
    --max_length 1024 \
    --batch_size 32 \
    --num_encoder_layers 6 \
    --num_decoder_layers 6 \
    --learning_rate 2e-5 \
    --num_epochs 100 \
    --num_phrases 10\
    --dropout 0.1 \
    --activation "gelu" \
    --train_size 2976999 \
    --test_size 1270 \
    --beam_size 3 \
    --warm_up 4000 \
    --number_vi_masked 30000 \
    --number_en_masked 20000 \
    --reverse_translate \
    --train_from_scratch \
    --folder_path $FOLDER_PATH \
    --vocab_path $VOCAB_NAME \
    --save_model_path $MODEL_PATH \
    --checkpoint_path $CHECKPOINT_PATH \
    --self_improve
