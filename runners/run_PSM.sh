CUDA_VISIBLE_DEVICES=0 python3 -u main.py\
    --n_blocks=2\
    --batch_size=256\
    --window_size=60\
    --train_split=0.6\
    --name=PSM\
    > PSM.log 2>&1 &
