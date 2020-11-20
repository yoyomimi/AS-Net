# AS-Net
Code for one-stage adaptive set-based HOI detector AS-Net.

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
   |                 └——————anno.json
   |                 └——————XXX1.jpg
   |                 └——————XXX2.jpg
   |        └——————test
   |                 └——————anno.json
   |                 └——————XXX1.jpg
   |                 └——————XXX2.jpg
   └——— test_hico.json
   └——— trainval_hico.json
   └——— rel_np.npy
```
Noted:
 - We transformed the original annotation files to a *.json format, like data/hico/images/test/anno.json
 - test_hico.json, trainval_hico.json and rel_np.npy are used in the evaluation on HICO-DET. We provided these three files in our directory.
 - test_hico.json and data/hico/images/test/anno.json are the same.
 
## Checkpoints
Our saved checkpoint on HICO-DET.
- The checkpoint is saved with torch==1.4.0.

## Evaluation
To evaluate our model on HICO-DET:
```shell
python3 tools/eval.py --cfg configs/hico.yaml MODEL.RESUME_PATH [checkpoint_path]
```

