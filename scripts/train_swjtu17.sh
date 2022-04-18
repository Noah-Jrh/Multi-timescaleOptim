###
 # @Author       : Noah
 # @Version      : v1.0.0
 # @Date         : 2019-12-28 20:42:52
 # @LastEditors  : Please set LastEditors
 # @LastEditTime : 2020-10-08 17:03:53
 # @FilePath     : /workspace/Meta_Synaptic/scripts/train.sh
 # @CopyRight (c): 2019 NCRC, SCU. All rights reserved.
 # @Description  :
 ###
#!/usr/bin/env bash

# ncrc-white
# /opt/data/durian
# 无batchNorm, + adaptive v_th v_decay 从第二个time step 开始反传
export CUDA_VISIBLE_DEVICES=0
python ../main_snn_v1.py \
   --model 'MSTO' \
   --structure 'SCN_5_64_bn' \
   --dataSet 'omniglot' \
   --dataDir '/opt/data/durian' \
   --resDir '../results' \
   --loss_fun 'ce' \
   --sur_grad 'linear' \
   --img_size 28 \
   --output_size 5 \
   --n_way 5 \
   --k_shot 5 \
   --k_query 15 \
   --rates 1.0 \
   --v_th 0.2 \
   --v_decay 0.3 \
   --T 8 \
   --lstm_bias 4 6 -5 -4 \
   --lr 0.0001 \
   --train