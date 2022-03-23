# python run.py save_name=convnext-emb-do0.1-adamw0.001-wd0.1-ls0.1 \
#               device=cuda:0 \
#               epoch=50 \
#               model=convnext_t \
#               scheduler=StepLR \
#               step_size=50 \
#               optimizer=adamw \
#               lr=0.001 \
#               weight_decay=0.1 \
#               label_smoothing=0.1 \
#               dropout=0.1

# python run.py save_name=convnext-emb-adamw0.001-wd0.1-ls0.1 \
#               device=cuda:0 \
#               epoch=50 \
#               model=convnext_t \
#               scheduler=StepLR \
#               step_size=50 \
#               optimizer=adamw \
#               lr=0.001 \
#               weight_decay=0.1 \
#               label_smoothing=0.1 \
#               dropout=0

# python run.py save_name=convnext-emb-adamw0.001-wd1-ls0.1 \
#               device=cuda:0 \
#               epoch=50 \
#               model=convnext_t \
#               scheduler=StepLR \
#               step_size=50 \
#               optimizer=adamw \
#               lr=0.001 \
#               weight_decay=1 \
#               label_smoothing=0.1 \
#               dropout=0

# python run.py save_name=convnext-adamw0.001-wd0.01-ls0.1 \
#               device=cuda:0 \
#               epoch=80 \
#               model=convnext_t \
#               scheduler=StepLR \
#               step_size=50 \
#               optimizer=adamw \
#               lr=0.001 \
#               weight_decay=0.01 \
#               label_smoothing=0.1

python run.py save_name=convnext-adamw0.001-wd0.1-ls0.1 \
              device=cuda:0 \
              epoch=80 \
              model=convnext_t \
              scheduler=CosineAnnealingLR \
              optimizer=adamw \
              lr=0.001 \
              weight_decay=0.1 \
              label_smoothing=0.1 \
              dropout=0.2