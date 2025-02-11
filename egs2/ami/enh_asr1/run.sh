#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


train_set=train
valid_set=dev
test_sets="dev eval"

enh_asr_config=conf/train.yaml
inference_config=conf/tuning/decode_asr_transformer2.yaml
lm_config=conf/tuning/train_lm_transformer2.yaml


use_word_lm=false
word_vocab_size=65000

./enh_asr.sh \
    --feats_normalize utt_mvn \
    --lang en \
    --audio_format wav \
    --nbpe 100 \
    --spk_num 2 \
    --use_speech_ref true \
    --nlsyms_txt data/nlsyms.txt \
    --feats_type raw \
    --feats_normalize utt_mvn \
    --enh_asr_config "${enh_asr_config}" \
    --inference_config "${inference_config}" \
    --use_lm true \
    --lm_config "${lm_config}" \
    --use_word_lm ${use_word_lm} \
    --word_vocab_size ${word_vocab_size} \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --bpe_train_text "../asr1_ihm_1.6.1/data/ihm_train/text" \
    --lm_train_text "../asr1_ihm_1.6.1/data/ihm_train/text" "$@"