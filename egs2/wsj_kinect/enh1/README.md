<!-- Generated by ./scripts/utils/show_enh_score.sh -->
# RESULTS
## Environments
- date: `Mon Apr 22 17:21:05 EDT 2024`
- python version: `3.9.18 (main, Sep 11 2023, 13:41:44)  [GCC 11.2.0]`
- espnet version: `espnet 202402`
- pytorch version: `pytorch 2.1.0`
- Git hash: `37828ea9708cd2f541220fdfe180457c7f7d67f1`
  - Commit date: `Thu Mar 21 22:52:57 2024 -0400`


## TF-GridNetV2

- config: conf/tuning/train_enh_tfgridnetv2_tf_lr-patience3_patience5_I_1_J_1_D_128_batch_8.yaml
- pretrained model: https://huggingface.co/atharva253/tfgridnetv2_wsj_kinect

|dataset|STOI|SAR|SDR|SIR|SI_SNR|
|---|---|---|---|---|---|
|enhanced_cv|85.97|10.51|10.07|21.63|9.61|
|enhanced_tt|88.76|11.22|10.69|21.36|10.26|
