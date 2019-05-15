gpuid='0'
dropout='0.2'
lr='5e0' # 1e-4
lambda='1e-1'
type='cnn'
num_train='100'
loss='default' # default
evaluate=''
checkpoint='none'
save_head='tmp'
epo='500'
save_interval='100'
batch_size='128'
log_file='tmp_mlp'
seed='2'
CUDA_VISIBLE_DEVICES=$gpuid python3.5 main_mlp.py \
    --lr $lr \
    --epochs $epo \
    --num_train $num_train \
    --batch-size $batch_size \
    --log_file $log_file \
    --dropout $dropout \
    --save_interval $save_interval \
    --loss $loss \
    --save_head $save_head \
    $evaluate \
    --checkpoint $checkpoint \
    --type $type \
    --output_s $lambda \
    --seed $seed
