#!/usr/bin/env bash

dataset="amazon_polarity"

data_folder="datasets/${dataset}/cnn"
model_folder="models/cnn/${dataset}"
solver='sgd'
config='small'
momentum=0.9
gamma=0.9
lr_halve_interval=15
maxlen=1014
batch_size=128
epochs=100
lr=0.01
snapshot_interval=5
gpuid=0
nthreads=4
no_stress=false
add_space=false
special_character=true
spell_check=false
lemma=false
stem=false


python -m main  --dataset ${dataset} \
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
          --no_stress ${no_stress} \
          --add_space ${add_space} \
          --special_character ${special_character} \
          --spell_check ${spell_check} \
          --lemma ${lemma} \
          --stem ${stem}