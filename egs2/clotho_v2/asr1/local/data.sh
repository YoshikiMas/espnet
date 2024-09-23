#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=1
stop_stage=100

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ -z "${CLOTHO_V2}" ]; then
    log "Fill the value of 'CLOTHO_V2' of db.sh"
    exit 1
fi

SPLITS=(development validation evaluation)
N_REF=5

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data preparation"
    for split_name in ${SPLITS[@]}; do
        mkdir -p "data/${split_name}"
    done

    if [ ! -d ${CLOTHO_V2} ]; then
        echo Cannot find CLOTHO_V2 root! Exiting...
        exit 1
    fi
    
    # Prepare data in the Kaldi format, including three files:
    # text, wav.scp, utt2spk
    echo "$(which python)"
    python3 local/data_prep.py ${CLOTHO_V2} ${N_REF}

    for split_name in ${SPLITS[@]}; do
        for f in wav.scp utt2spk; do
            sort data/${split_name}/${f} -o data/${split_name}/${f}
        done
        # Sort all text files
        if [ -f data/${split_name}/text ]; then
            sort data/${split_name}/text -o data/${split_name}/text
        fi
        for i in $(seq ${N_REF}); do
            if [ -f data/${split_name}/text_spk${i} ]; then
                sort data/${split_name}/text_spk${i} -o data/${split_name}/text_spk${i}
            fi
        done
        echo "Running spk2utt for ${split_name}"
        utils/utt2spk_to_spk2utt.pl data/${split_name}/utt2spk > "data/${split_name}/spk2utt"
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"


# local/data.sh ${post_process_local_data_opts} --asr_data_dir "${data_feats}/${train_set}"
# TODO(sbharad2): Copy files for own case