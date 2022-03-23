# python run.py save_name=resnet50-4662-softtriple-sgd0.01 \
#               device=cuda:2 \
#               epoch=100 \
#               model=resnet \
#               scheduler=CosineAnnealingLR \
#               lr=0.01

# python run.py save_name=resnetd-adamw0.001-steplr \
#               device=cuda:2 \
#               epoch=80 \
#               model=resnetd \
#               scheduler=StepLR \
#               step_size=50 \
#               optimizer=adamw \
#               lr=0.001

# python run.py save_name=resnetd-adamw0.001-steplr \
#               device=cuda:2 \
#               epoch=80 \
#               model=resnetd \
#               scheduler=CosineAnnealingLR \
#               optimizer=adamw \
#               lr=0.001

# python run.py save_name=resnetd-adamw0.001-steplr-wd0.01 \
#               device=cuda:2 \
#               epoch=80 \
#               model=resnetd \
#               scheduler=StepLR \
#               step_size=50 \
#               optimizer=adamw \
#               lr=0.001 \
#               weight_decay=0.01

# python run.py save_name=convnext-4772 \
#               device=cuda:2 \
#               epoch=80 \
#               model=my_convnext \
#               block_nums=[4,7,7,2] \
#               scheduler=CosineAnnealingLR \
#               optimizer=adamw \
#               lr=0.001 \
#               weight_decay=0.1 \
#               label_smoothing=0.1 \
#               dropout=0

# python run.py save_name=resnet50-triplet \
#               device=cuda:1 \
#               epoch=20 \
#               model=resnet \
#               scheduler=CosineAnnealingLR \
#               optimizer=sgd \
#               lr=0.001 \
#               label_smoothing=0.1 \
#               metric=TripletMarginLoss \
#               resume=True \
#               batch_size=64 

# python run.py save_name=resnet50-triplet \
#               device=cuda:2 \
#               epoch=20 \
#               model=resnet \
#               scheduler=CosineAnnealingLR \
#               optimizer=sgd \
#               lr=0.01 \
#               label_smoothing=0.1 \
#               metric=TripletMarginLoss \
#               batch_size=64

python run.py save_name=convnext-4772-triplet \
              device=cuda:2 \
              epoch=80 \
              batch_size=256 \
              model=my_convnext \
              block_nums=[4,7,7,2] \
              scheduler=CosineAnnealingLR \
              optimizer=adamw \
              lr=0.001 \
              weight_decay=0.1 \
              label_smoothing=0.1 \
              dropout=0