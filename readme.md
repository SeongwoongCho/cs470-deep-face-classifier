# CS470 Project
- DeepLearning Part1 - human face to animal classification
- Set Requirements,and set data through Download Dataset Section,then see Quick Start Section. 
- All Pretrained models(that we trained) are included in the ./src/logs/${model_name}  

# Dataset
- Class info
    - class 0 : Bald
    - class 1 : Bear
    - class 2 : Rest animals

- Data info
    - 강아지상 : 워너원 강다니엘, 엑소 백현, 박보검,송중기, 임시완, 박보영, 손예진, 태연, 아이유, 한효주
    - 고양이상 : 워너원 황민현, 엑소 시우민, 강동원, 이종석, 이준기, 한예슬,유인영,김희선,이나영,한채영
    - 곰상 : 마동석, 조진웅, 조세호, 안재홍, 정형돈, 스윙스, 박성웅, 최자, 곽도원, 김구라
    - 공룡상 : 윤두준, 이민기, 김우빈, 육성재, 공유, 송지효, 김아중, 한지민, 천우희, 신민아
    - 토끼상 : 방탄소년단 정국, 아이콘 바비, 워너원 박지훈, 엑소 수호, 안형섭, 수지, 트와이스 나연, 트와이스 다현, 이세영, 백진희
    - 대머리상 : from kaggle

# Download Dataset
```
$ pip install gdown
$ gdown [link]
$ unzip data.zip

Then, change the folder name into full_data
```

# Quick Start
```
$ cd src

#################################################
############### Reproduce Training###############
#################################################

/// Resnet50(baseline)
$ python3 train.py --exp_name resnet50_non --coeff 0 --learning_rate 4e-3 --batch_size 16 --n_epoch 100 --optim adamw --weight_decay 1e-5 --num_workers 16 --warmup 1 --cutmix_prob 0 --cutmix_beta 0 --label_smoothing 0 >> ./logs/resnet50.log

/// Efficientnet b0
$ python3 train.py --exp_name effb0_non --coeff 0 --learning_rate 4e-3 --batch_size 16 --n_epoch 100 --optim adamw --weight_decay 1e-5 --num_workers 16 --warmup 1 --cutmix_prob 0 --cutmix_beta 0 --label_smoothing 0 >> ./logs/effb0.log

/// Efficientnet b0 + Cutmix 
$ python3 train.py --exp_name effb0_cutmix --coeff 0 --learning_rate 4e-3 --batch_size 16 --n_epoch 100 --optim adamw --weight_decay 1e-5 --num_workers 16 --warmup 1 --cutmix_prob 1 --cutmix_beta 1 --label_smoothing 0 >> ./logs/effb0_cutmix.log

/// Efficientnet b0 + Cutmix + LabelSmoothing(0.05)
$ python3 train.py --exp_name effb0_cutmix_ls --coeff 0 --learning_rate 4e-3 --batch_size 16 --n_epoch 100 --optim adamw --weight_decay 1e-5 --num_workers 16 --warmup 1 --cutmix_prob 1 --cutmix_beta 1 --label_smoothing 0.05 >> ./logs/effb0_cutmix_ls.log

/// Efficientnet b4 + Cutmix
$ python3 train.py --exp_name effb4_cutmix --coeff 4 --learning_rate 4e-3 --batch_size 16 --n_epoch 100 --optim adamw --weight_decay 1e-5 --num_workers 16 --warmup 1 --cutmix_prob 1 --cutmix_beta 1 --label_smoothing 0 >> ./logs/effb4_cutmix.log

/// Efficientnet b4 + Cutmix + LabelSmoothing(0.05)
$ python3 train.py --exp_name effb4_cutmix_ls --coeff 4 --learning_rate 4e-3 --batch_size 16 --n_epoch 100 --optim adamw --weight_decay 1e-5 --num_workers 16 --warmup 1 --cutmix_prob 1 --cutmix_beta 1 --label_smoothing 0.05 >> ./logs/effb4_cutmix_ls.log

#################################################
############### Test with pretrained model#######
#################################################

$ python3 test.py >> ./logs/benchmark.log

then, see ./logs/benchmark.log

#################################################
############### Export onnx model#######
#################################################

$ python3 export_onnx.py 

```

# Requirements

```


```