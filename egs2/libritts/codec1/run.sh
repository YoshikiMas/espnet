#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

fs=24000

opts=
if [ "${fs}" -eq 24000 ]; then
    # To suppress recreation, specify wav format
    opts="--audio_format wav "
else
    opts="--audio_format flac "
fi


train_set=train-clean-460
valid_set=dev-clean
test_sets="test-clean"


train_config=conf/train_dac.yaml
inference_config=conf/decode.yaml
score_config=conf/score_24k.yaml

./codec.sh \
    --stage 5 \
    --stop_stage 7 \
    --local_data_opts "--trim_all_silence false" \
    --fs ${fs} \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --scoring_config "${score_config}" \
    --gpu_inference false \
    --inference_nj 60 \
    --fs 24000 \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" ${opts} "$@"
