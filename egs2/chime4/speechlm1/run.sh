#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

sample_rate=16000

train_set=tr05_simu_isolated_1ch_track
valid_set=dt05_simu_isolated_1ch_track
test_sets="et05_simu_isolated_1ch_track"

codec_opts="--codec_choice ESPnet --codec_hf_model_tag espnet/owsmdata_soundstream_16k_200epoch"

./speechlm.sh \
    --task "se" \
    --data_name "chime4" \
    --fs $sample_rate \
    --nj 16 \
    --inference_nj 116 \
    --audio_format "flac.ark" \
    --train_config conf/tuning/train_delay.yaml \
    --inference_config conf/decode_asr_debug.yaml \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    "$@"