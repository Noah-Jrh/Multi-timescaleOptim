###
 # @Author       : Noah
 # @Version      : v1.0.0
 # @Date         : 2019-12-28 20:42:52
 # @LastEditors  : Please set LastEditors
 # @LastEditTime : 2020-10-08 20:12:13
 # @FilePath     : /workspace/Meta_Synaptic/scripts/train.sh
 # @CopyRight (c): 2019 NCRC, SCU. All rights reserved.
 # @Description  :
 ###
#!/usr/bin/env bash

# swjtu-20
# /DATACENTER2/noah/workspace/DATA
export CUDA_VISIBLE_DEVICES=3
python ../main_snn_v1.py \
   --model 'MSTOv5' \
   --structure 'SCN_4_32' \
   --dataSet 'omniglot' \
   --dataDir '/DATACENTER2/noah/workspace/DATA' \
   --resDir '../results' \
   --loss_fun 'ce' \
   --sur_grad 'linear' \
   --img_size 28 \
   --output_size 5 \
   --n_way 5 \
   --k_shot 1 \
   --k_query 15 \
   --rates 1.0 \
   --v_th 0.2 \
   --v_decay 0.3 \
   --T 20 \
   --step 1 \
   --lstm_bias 4 6 -5 -4 \
   --lr 0.001 \
   --train