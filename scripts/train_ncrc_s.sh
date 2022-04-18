###
 # @Author       : Noah
 # @Version      : v1.0.0
 # @Date         : 2020-09-11 20:09:49
 # @LastEditors  : Please set LastEditors
 # @LastEditTime : 2020-11-16 20:46:29
 # @CopyRight (c): 2019 NCRC, SCU. All rights reserved.
 # @Description  : Please add descriptioon
### 
#!/usr/bin/env bash

# ncrc-super
# /home/template/Data

# # For SpkPtn
# export CUDA_VISIBLE_DEVICES=1
# python ../main_snn_v1.py \
#    --model 'MSTO' \
#    --structure 'SNN_2' \
#    --dataSet 'spiking' \
#    --dataDir '/home/template/Data' \
#    --resDir '../results' \
#    --loss_fun 'ce' \
#    --sur_grad 'linear' \
#    --input_size 196 \
#    --output_size 5 \
#    --n_way 5 \
#    --k_shot 1 \
#    --k_query 15 \
#    --rates 1.0 \
#    --v_th 0.2 \
#    --v_decay 0.3 \
#    --T 50 \
#    --step 1 \
#    --lstm_bias 4 6 -5 -4 \
#    --lr 0.001 \
#    --train

# # HMAX-SNN For Omniglot Dataset 
# export CUDA_VISIBLE_DEVICES=2
# python ../main_snn_v1.py \
#    --model 'MSTO' \
#    --structure 'SNN_3_HMAX_v1' \
#    --dataSet 'omniglot' \
#    --dataDir '/home/template/Data' \
#    --resDir '../results' \
#    --preprocess 'hmax' \
#    --loss_fun 'ce' \
#    --sur_grad 'linear' \
#    --input_size 360 \
#    --output_size 5 \
#    --n_way 5 \
#    --k_shot 1 \
#    --k_query 15 \
#    --rates 1.0 \
#    --v_th 1.0 \
#    --v_decay 0.9 \
#    --T 20 \
#    --step 1 \
#    --lstm_bias 4 6 -5 -4 \
#    --lr 0.001 \
#    --train

# # For Omniglot + CSNN
# export CUDA_VISIBLE_DEVICES=1
# python ../main_snn_v1.py \
#    --model 'MSTOv1' \
#    --structure 'SCN_4_32_v1' \
#    --dataSet 'omniglot' \
#    --dataDir '/home/template/Data' \
#    --resDir '../results' \
#    --loss_fun 'ce' \
#    --sur_grad 'linear' \
#    --output_size 5 \
#    --n_way 5 \
#    --k_shot 5 \
#    --k_query 15 \
#    --rates 1.0 \
#    --v_th 0.2 \
#    --v_decay 0.3 \
#    --T 20 \
#    --lstm_bias 4 6 -5 -4 \
#    --lr 0.001 \
#    --train

# # For Omniglot + CNN
# export CUDA_VISIBLE_DEVICES=1
# python ../main_ann.py \
#    --model 'MSTO' \
#    --structure 'CNN_4_32' \
#    --dataSet 'omniglot' \
#    --dataDir '/home/template/Data' \
#    --resDir '../results' \
#    --loss_fun 'ce' \
#    --output_size 5 \
#    --n_way 5 \
#    --k_shot 5 \
#    --k_query 15 \
#    --lstm_bias 4 6 -5 -4 \
#    --lr 0.001 \
#    --train

# CSNN For Gesture_DVS Dataset 
export CUDA_VISIBLE_DEVICES=2
python ../main_snn_dvs.py \
   --model 'MSTO' \
   --structure 'SCN_4_32_v3' \
   --dataSet 'gesture_dvs' \
   --dataDir '/home/template/Data' \
   --resDir '../results' \
   --loss_fun 'ce' \
   --sur_grad 'linear' \
   --img_size 128 \
   --output_size 5 \
   --n_way 5 \
   --k_shot 5 \
   --k_query 15 \
   --rates 1.0 \
   --v_th 0.2 \
   --v_decay 0.3 \
   --T 12 \
   --step 1 \
   --resolution 25000 \
   --lstm_bias 4 6 -5 -4 \
   --lr 0.0001 \
   --train