# Masked Autoencoder meets GAN for ECG

Pytorch Implementation of [Masked Auto-Encoders Meet Generative Adversarial Networks and Beyond](https://feizc.github.io/resume/ganmae.pdf) for ECG Signals.

To Pretrain run :

```
python main_pretrain.py \
    --batch_size 64 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 500 \
    --warmup_epochs 10 \
    --data_path ${IMAGENET_DIR} \
    --lr 1e-3 \
    --cuda "CUDA"

```

data_path to the physionet -

Eg. if path to the physionet dataset is

/Users/parthagrawal02/Desktop/ECG_CNN/physionet/WFDBRecords

then --datapath '/Users/parthagrawal02/Desktop/ECG_CNN/physionet'

To Finetune :

```
python /kaggle/working/ECG_MAE/main_finetune.py\
    --model vit_1dcnn \
    --finetune '/checkpoint-360.pth' \
    --epochs 70 \
    --lr 5e-3 \
    --data_path /Users/parthagrawal02/Desktop/ECG_CNN/physionet \
    --cuda 'CUDA'\
    --train_start 0 --train_end 46 --data_split 0.85
```

Modify ecg_dataloader according to the dataset
