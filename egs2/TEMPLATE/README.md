# Recipe Template
Recipe template is used to build recipes easily.
It is designed to support the common functionalities and requirements that each individual tasks often has.

<!-- generated by doctoc https://github.com/thlorenz/doctoc -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Recipe Template](#recipe-template)
  - [Run ESPnet with your own corpus](#run-espnet-with-your-own-corpus)
  - [About Kaldi style data directory](#about-kaldi-style-data-directory)
  - [(For developers) How to make/port new recipe?](#for-developers-how-to-makeport-new-recipe)

## Run ESPnet with your own corpus

1. Copying a template directory
    ```bash
    % task=asr1  # enh1, tts1, mt1, st1
    % egs2/TEMPLATE/${task}/setup.sh egs2/foo/${task}
    ```

1. Create `egs2/foo/${task}/data` directory to put your corpus: See https://github.com/espnet/data_example or next section.
1. Run (e.g. `asr` case)
    ```
    cd egs2/foo/${task}  # We always assume that our scripts are executed at this directory.

    # Assuming Stage1 creating `data`, so you can skip it if you have `data`.
    ./asr.sh \
     --stage 2 \
     --ngpu 1 \
     --train_set train \
     --valid_set valid \
     --test_sets "test" \
     --lm_train_text "data/train/text"

    # Use CUDA_VISIBLE_DEVICES to specify a gpu device id
    # If you meet CUDA out of memory error, change `batch_bins` ( or `batch_size`)
    ```
1. For more detail
    - Read the config files: e.g. https://github.com/espnet/espnet/tree/master/egs2/librispeech/asr1/conf
    - Read the main script: e.g. https://github.com/espnet/espnet/blob/master/egs2/TEMPLATE/asr1/asr.sh
    - Documentation: https://espnet.github.io/espnet/

## About Kaldi style data directory

Each directory of training set, development set, and evaluation set, has same directory structure. See also http://kaldi-asr.org/doc/data_prep.html about Kaldi data structure.
We recommend you running `mini_an4` recipe and checking the contents of `data/` by yourself.

```bash
cd egs2/mini_an4/asr1
./run.sh
```

- Directory structure
    ```
    data/
      train/
        - text     # The transcription
        - wav.scp  # Wave file path
        - utt2spk  # A file mapping utterance-id to speaker-id
        - spk2utt  # A file mapping speaker-id to utterance-id
        - segments # [Option] Specifying start and end time of each utterance
      dev/
        ...
      test/
        ...
    ```

- `text` format
    ```
    uttidA <transcription>
    uttidB <transcription>
    ...
    ```

- `wav.scp` format
    ```
    uttidA /path/to/uttidA.wav
    uttidB /path/to/uttidB.wav
    ...
    ```

- `utt2spk` format
    ```
    uttidA speakerA
    uttidB speakerB
    uttidC speakerA
    uttidD speakerB
    ...
    ```

- `spk2utt` format
    ```
    speakerA uttidA uttidC ...
    speakerB uttidB uttidD ...
    ...
    ```

    Note that `spk2utt` file can be generated by `utt2spk`, and `utt2spk` can be generated by `spk2utt`, so it's enough to create either one of them.

    ```bash
    utils/utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt
    utils/spk2utt_to_utt2spk.pl data/train/spk2utt > data/train/utt2spk
    ```

    If your corpus doesn't include speaker information, give the same speaker id as the utterance id to satisfy the directory format, otherwise give the same speaker id for all utterances (Actually we don't use speaker information for asr recipe now).

    ```bash
    uttidA uttidA
    uttidB uttidB
    ...
    ```

    OR

    ```bash
    uttidA dummy
    uttidB dummy
    ...
    ```

- [Option] `segments` format

    If the audio data is originally long recording, about > ~1 hour, and each audio file includes multiple utterances in each section, you need to create `segments` file to specify the start time and end time of each utterance. The format is `<utterance_id> <wav_id> <start_time> <end_time>`.

    ```
    sw02001-A_000098-001156 sw02001-A 0.98 11.56
    ...
    ```

    Note that if using `segments`, `wav.scp` has `<wav_id>` which corresponds to the `segments` instead of `utterance_id`.

    ```
    sw02001-A /path/to/sw02001-A.wav
    ...
    ```

Once you complete creating the data directory, it's better to check it by `utils/validate_data_dir.sh`.

```bash
utils/validate_data_dir.sh --no-feats data/train
utils/validate_data_dir.sh --no-feats data/dev
utils/validate_data_dir.sh --no-feats data/test
```


## (For developers) How to make/port new recipe?

ESPnet2 doesn't prepare different recipes for each corpus unlike ESPnet1, but we prepare common recipes for each task, which are named as `asr.sh`, `enh.sh`, `tts.sh`, or etc. We carefully designed these common scripts to perform with any types of corpus, so ideally you can train using your own corpus without modifying almost all parts of these recipes. Only you have to do is just creating `local/data.sh`.


1. Create directory in egs/
    ```bash
    % task=asr1  # enh1, tts1, mt1, st1
    % egs2/TEMPLATE/${task}/setup.sh egs2/foo/${task}
    ```

1. Create `run.sh` and `local/data.sh` somehow
    ```bash
    % cd egs2/foo/${task}
    % cp ../../mini_an4/${task}/run.sh .
    % vi run.sh
    ```

    `run.sh` is a thin wrapper of a common recipe for each task as follows,

    ```bash
    # The contents of run.sh
    ./asr.sh \
      --train_set train \
      --valid_set dev \
      --test_sets "dev test1 test2" \
      --lm_train_text "data/train/text" "$@"
    ```

    - We use a common recipe, thus you must absorb the difference of each corpus by the command line options of `asr.sh`.
    - We expect that `local/data.sh` generates training data (e.g., `data/train`), validation data (e.g., `data/dev`), and (multiple) test data (e.g, `data/test1` and `data/test2`), which have Kaldi style (See stage1 of `asr.sh`).
    - Note that some corpora only provide the test data and would not officially prepare the development set. In this case, you can prepare the validation data by extracting the part of the training data and regard the rest of training data as a new training data by yourself (e.g., check `egs2/csj/asr1/local/data.sh`).
    - Also, the validation data used during training must be a single data directory. If you have multiple validation data directories, you must combine them by using `utils/combine_data.sh`.
    - On the other hand, the recipe accepts multiple test data directories during inference. So, you can include the validation data to evaluate the ASR performance of the validation data.
    - If you'll create your recipe from scratch, you have to understand Kaldi data structure. See the next section.
    - If you'll port the recipe from ESPnet1 or Kaldi, you need to embed the data preparation part of the original recipe in `local/data.sh`. Note that the common steps include `Feature extraction`, `Speed Perturbation`, and `Removing long/short utterances`, so you don't need to do them at `local/data.sh`


1. If the recipe uses some corpora and they are not listed in `db.sh`, then write it.
    ```bash
    ...
    YOUR_CORPUS=
    ...
    ```

1. If the recipe depends on some special tools, then write the requirements to `local/path.sh`

    path.sh:
    ```bash
    # e.g. flac command is required
    if ! which flac &> /dev/null; then
        echo "Error: flac is not installed"
        return 1
    fi
    ```
