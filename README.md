# AS-Net
Code for one-stage adaptive set-based HOI detector AS-Net.

https://arxiv.org/abs/2103.05983

Accepted to CVPR 2021

## Installation
Environment
- python >= 3.6

Install the dependencies.
```shell
 pip install -r requirements.txt
```

## Data preparation
- We first download the [ HICO-DET ](https://drive.google.com/open?id=1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk " HICO-DET ") dataset.
- The data should be prepared in the following structure:
```
data/hico
   |———  images
   |        └——————train
   |        |        └——————anno.json
   |        |        └——————XXX1.jpg
   |        |        └——————XXX2.jpg
   |        └——————test
   |                 └——————anno.json
   |                 └——————XXX1.jpg
   |                 └——————XXX2.jpg
   └——— test_hico.json
   └——— trainval_hico.json
   └——— rel_np.npy
```
Noted:
 - We transformed the original annotation files of HICO-DET to a *.json format, like data/hico/images/train_anno.json and ata/hico/images/test_hico.json.
 - test_hico.json, trainval_hico.json and rel_np.npy are used in the evaluation on HICO-DET. We provided these three files in our data/hico directory.
 - data/hico/train_anno.json and data/hico/images/train/anno.json are the same file.
   `cp data/hico/train_anno.json data/hico/images/train/anno.json`
 - data/hico/test_hico.json and data/hico/images/test/anno.json are the same file.
   `cp data/hico/test_hico.json data/hico/images/test/anno.json`

## Evaluation
To evaluate our model on HICO-DET:
```shell
python3 tools/eval.py --cfg configs/hico.yaml MODEL.RESUME_PATH [checkpoint_path]
```
- The checkpoint is saved on HICO-DET with torch==1.4.0.
