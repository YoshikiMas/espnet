#!/usr/bin/env bash

for dset in train dev eval ; do
    utils/utt2spk_to_spk2utt.pl "data/${dset}/utt2spk" > "data/${dset}/spk2utt"
done