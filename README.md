# Instructions for HW2P2 (Youngin Kim, youngin2)
11-785 HW2P2: Face Recognition & Face Verification

## 0. Prerquisite

**Please modify `configs/path/custom.yaml` for your own path**


## 1. Best Architecture

**You can see Hyperparameters values in `configs/config.yaml`.**

**Or You can see the details and experimental results in [WanDB](https://wandb.ai/normalkim/cmu-hw2p2?workspace=user-normalkim).**

    - Recognition best model: convnext-4772-16:13:41:23
    - Verification best model: resnet50-4662-softtriple-sgd0.01-ls0.1-15:19:40:04

- Model Architecture
    - You can check in `models/convnext.py` and `resnet.py`
    - I change the block number [3,3,9,3] to [4,7,7,2] / [4,6,6,2] because local feature is important in this task.
    - However, in the verification task, custom resnet model is better than convnext.
- Optimizer 
    - AdamW
    - scheduler: CosineAnnealing
- Regularize
    - weight decay
- Augmentation
    - You can check in `datasets/transform.py`
- TTA
  - TTA results in slightly higher scores (+0.001)
  - You can check in `tta.ipynb`

## 2. Run
```
$ python run.py save_name={name_for_submission&weight_file}
```
