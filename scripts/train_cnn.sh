###
 # @Author       : Noah
 # @Version      : v1.0.0
 # @Date         : 2020-09-11 20:08:37
 # @LastEditors  : Please set LastEditors
 # @LastEditTime : 2020-10-06 17:26:55
 # @CopyRight (c): 2019 NCRC, SCU. All rights reserved.
 # @Description  : Please add descriptioon
### 
#!/usr/bin/env bash
# export CUDA_VISIBLE_DEVICES=0

# ncrc-black
# /home/noah/Data
# export CUDA_VISIBLE_DEVICES=0
# python ../main_ann.py \
#    --model 'MSTO' \
#    --structure 'CNN_4_32' \
#    --dataSet 'omniglot' \
#    --dataDir '/home/noah/Data' \
#    --resDir '../results' \
#    --loss_fun 'ce' \
#    --output_size 5 \
#    --n_way 5 \
#    --k_shot 5 \
#    --k_query 15 \
#    --lstm_bias 4 6 -5 -4 \
#    --lr 0.001 \
#    --train

# ncrc-white
# /opt/data/durian
export CUDA_VISIBLE_DEVICES=1
python ../main_ann.py \
   --model 'MSTO' \
   --structure 'CNN_4_32' \
   --dataSet 'omniglot' \
   --dataDir '/opt/data/durian' \
   --resDir '../results' \
   --loss_fun 'ce' \
   --output_size 5 \
   --n_way 5 \
   --k_shot 5 \
   --k_query 15 \
   --lstm_bias 4 6 -5 -4 \
   --lr 0.001 \
   --train