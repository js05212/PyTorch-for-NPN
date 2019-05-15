gpuid='0'
dropout='0.0'
lr='1e0'
output_s='0e1'
type='regress_mlp'
loss='mse'
evaluate=''
checkpoint='none'
save_head='tmp'
epo='1000'
save_interval='100'
batch_size='16'
log_file='tmp_mlp'
seed='1112'
CUDA_VISIBLE_DEVICES=$gpuid python3.5 main_mlp.py \
    --lr $lr \
    --epochs $epo \
    --batch-size $batch_size \
    --log_file $log_file \
    --dropout $dropout \
    --save_interval $save_interval \
    --loss $loss \
    --save_head $save_head \
    $evaluate \
    --checkpoint $checkpoint \
    --type $type \
    --output_s $output_s \
    --seed $seed
