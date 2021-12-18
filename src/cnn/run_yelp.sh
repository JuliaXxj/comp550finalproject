#!/usr/bin/env bash
cd ../../
 
dataset="yelp_polarity"

data_folder="datasets/${dataset}/cnn"
model_folder="models/cnn/${dataset}"
solver='sgd'
config='small'
momentum=0.9
gamma=0.9
lr_halve_interval=15
maxlen=1014
batch_size=128
epochs=10
lr=0.01
snapshot_interval=5
gpuid=0
nthreads=6
log_msg=''
method_name=''



python -m src.cnn.main  --dataset ${dataset} \
                        --model_folder ${model_folder} \
                        --data_folder ${data_folder} \
                        --config ${config} \
                        --maxlen ${maxlen} \
                        --batch_size ${batch_size} \
                        --epochs ${epochs} \
                        --solver ${solver} \
                        --lr ${lr} \
                        --lr_halve_interval ${lr_halve_interval} \
                        --momentum ${momentum} \
                        --snapshot_interval ${snapshot_interval} \
                        --gamma ${gamma} \
                        --gpuid ${gpuid} \
                        --nthreads ${nthreads} \
                        --log_msg='' \
                        --model_name=''
