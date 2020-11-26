# CS470 Project

# Crawling

Install google-images-download library

pip install git+http://github.com/Joeclinton1/google-images-download.git

강아지상 : 워너원 강다니엘, 엑소 백현, 박보검,송중기, 임시완, 박보영, 손예진, 태연, 아이유, 한효주

고양이상 : 워너원 황민현, 엑소 시우민, 강동원, 이종석, 이준기, 한예슬,유인영,김희선,이나영,한채영

곰상 : 마동석, 조진웅, 조세호, 안재홍, 정형돈, 스윙스, 박성웅, 최자, 곽도원, 김구라

공룡상 : 윤두준, 이민기, 김우빈, 육성재, 공유, 송지효, 김아중, 한지민, 천우희, 신민아

토끼상 : 방탄소년단 정국, 아이콘 바비, 워너원 박지훈, 엑소 수호, 안형섭, 수지, 트와이스 나연, 트와이스 다현, 이세영, 백진희

대머리상 : from kaggle

# Dataset download
```
$ wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1W5HKr-M1J98smLiF781jjiLVxl9hmq0i' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1W5HKr-M1J98smLiF781jjiLVxl9hmq0i" -O animal.zip && rm -rf /tmp/cookies.txt

$ wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1yUAutv-VWkfUm1rvbzXcmtOaASvS3dwR' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1yUAutv-VWkfUm1rvbzXcmtOaASvS3dwR" -O bald.zip && rm -rf /tmp/cookies.txt
```

# Directory structure

# Quick Start
```

$ python3 train.py --exp_name resnet18_fulldata_baseline --coeff 0 --learning_rate 4e-3 --batch_size 128 --n_epoch 100 --optim adamw --weight_decay 1e-5 --num_workers 16 --warmup 1 --cutmix_prob 1 --cutmix_beta 1 --label_smoothing 0.05  

$ python train.py --exp_name resnet18_newdata_baseline --cutmix_prob 1 --cutmix_beta 1 --batch_size 128 

```

# Experiments

```
python3 train.py --exp_name resnet50_non --coeff 0 --learning_rate 4e-3 --batch_size 16 --n_epoch 100 --optim adamw --weight_decay 1e-5 --num_workers 16 --warmup 1 --cutmix_prob 0 --cutmix_beta 0 --label_smoothing 0 >> ./logs/resnet50_non.log && python3 train.py --exp_name effb0_non --coeff 0 --learning_rate 4e-3 --batch_size 16 --n_epoch 100 --optim adamw --weight_decay 1e-5 --num_workers 16 --warmup 1 --cutmix_prob 0 --cutmix_beta 0 --label_smoothing 0 >> ./logs/effb0_non.log && python3 train.py --exp_name effb0_cutmix --coeff 0 --learning_rate 4e-3 --batch_size 16 --n_epoch 100 --optim adamw --weight_decay 1e-5 --num_workers 16 --warmup 1 --cutmix_prob 1 --cutmix_beta 1 --label_smoothing 0 >> ./logs/effb0_cutmix.log && python3 train.py --exp_name effb0_cutmix_ls --coeff 0 --learning_rate 4e-3 --batch_size 16 --n_epoch 100 --optim adamw --weight_decay 1e-5 --num_workers 16 --warmup 1 --cutmix_prob 1 --cutmix_beta 1 --label_smoothing 0.05 >> ./logs/effb0_cutmix_ls.log && python3 train.py --exp_name effb4_cutmix_ls --coeff 4 --learning_rate 4e-3 --batch_size 16 --n_epoch 100 --optim adamw --weight_decay 1e-5 --num_workers 16 --warmup 1 --cutmix_prob 1 --cutmix_beta 1 --label_smoothing 0.05 >> ./logs/effb4_cutmix_ls.log
```

``` 
- Default hyperparameters
    - weight_decay
    - optimizer : adamw
    - scheduler : Linear warmup scheduler + cosine annealing
    - epoch : 100
    - batch_size : 32
    - Initialization : Pretrained on Imagenet dataset
    - Preprocessing : Resize 224, Imagenet normalization
    - Strong Augmentations(Color jitter, noise, affineTransformation, ...)
    - Loss : Balanced Cross entropy

- Resnet50(baseline)
python3 train.py --exp_name resnet50_non --coeff 0 --learning_rate 4e-3 --batch_size 16 --n_epoch 100 --optim adamw --weight_decay 1e-5 --num_workers 16 --warmup 1 --cutmix_prob 0 --cutmix_beta 0 --label_smoothing 0 >> logs.txt

- Efficientnet b0
python3 train.py --exp_name effb0_non --coeff 0 --learning_rate 4e-3 --batch_size 16 --n_epoch 100 --optim adamw --weight_decay 1e-5 --num_workers 16 --warmup 1 --cutmix_prob 0 --cutmix_beta 0 --label_smoothing 0 >> logs.txt

- Efficientnet b0 + Cutmix 
python3 train.py --exp_name effb0_cutmix --coeff 0 --learning_rate 4e-3 --batch_size 16 --n_epoch 100 --optim adamw --weight_decay 1e-5 --num_workers 16 --warmup 1 --cutmix_prob 1 --cutmix_beta 1 --label_smoothing 0 >> logs.txt

- Efficientnet b0 + Cutmix + LabelSmoothing(0.05)
python3 train.py --exp_name effb0_cutmix_ls --coeff 0 --learning_rate 4e-3 --batch_size 16 --n_epoch 100 --optim adamw --weight_decay 1e-5 --num_workers 16 --warmup 1 --cutmix_prob 1 --cutmix_beta 1 --label_smoothing 0.05 >> logs.txt

- Efficientnet b4 + Cutmix + LabelSmoothing(0.05)
python3 train.py --exp_name effb0_cutmix_ls --coeff 4 --learning_rate 4e-3 --batch_size 16 --n_epoch 100 --optim adamw --weight_decay 1e-5 --num_workers 16 --warmup 1 --cutmix_prob 1 --cutmix_beta 1 --label_smoothing 0.05 >> logs.txt

- Ensemble
effb0/effb4 ensemble 


```