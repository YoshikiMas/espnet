#!/usr/bin/env bash
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

mic=sdm1

log "$0 $*"
. utils/parse_options.sh

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

if [ ! -e "${AMI}" ]; then
    log "Fill the value of 'AMI' of db.sh"
    exit 1
fi

if [ "$mic" != "sdm1" ]; then
    echo "Error: The variable 'mic' must be 'sdm1'."
    exit 1
fi

python local/ami_preprocessing.py \
    ../asr1/data/local/annotations \
    $AMI \
    ./data

local/prepare_spk2utt.sh