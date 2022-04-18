###
 # @Author       : Noah
 # @Version      : v1.0.0
 # @Date         : 2020-09-11 11:37:47
 # @LastEditors  : Please set LastEditors
 # @LastEditTime : 2020-11-19 16:22:37
 # @CopyRight (c): 2019 NCRC, SCU. All rights reserved.
 # @Description  : Please add descriptioon
### 

# # For Omniglot
# # 5w5s T=20 step=1 2020-10-02-17_47_57
# # 5w5s T=15 step=1 2020-10-02-17_47_33
# # 5w5s T=10 step=1 2020-10-02-17_47_16
# # 5w5s T=5 step=1 2020-10-02-17_46_49
# # 5w5s T=8 step=2 2020-10-01-22_54_15
# export CUDA_VISIBLE_DEVICES=1
# python ../main_snn_v1.py \
#    --model 'MSTOv4' \
#    --structure 'SCN_4_32' \
#    --checkpoints '2020-10-01-22_54_15' \
#    --dataSet 'omniglot' \
#    --dataDir '/opt/data/durian' \
#    --resDir '../results' \
#    --loss_fun 'ce' \
#    --sur_grad 'linear' \
#    --img_size 28 \
#    --output_size 5 \
#    --n_way 5 \
#    --k_shot 5 \
#    --k_query 15 \
#    --rates 1.0 \
#    --v_th 0.2 \
#    --v_decay 0.3 \
#    --T 8 \
#    --step 2 \
#    --lstm_bias 4 6 -5 -4 \
#    --lr 0.001

# # For Spiking
# # 5w1s 2020-11-06-20_31_44 
# # 5w5s 2020-11-06-20_24_16
# export CUDA_VISIBLE_DEVICES=0
# python ../main_snn_v1.py \
#    --model 'MSTO' \
#    --structure 'SNN_2' \
#    --checkpoints '2020-11-06-20_24_16' \
#    --dataSet 'spiking' \
#    --dataDir '/opt/data/durian' \
#    --resDir '../results' \
#    --loss_fun 'ce' \
#    --sur_grad 'linear' \
#    --input_size 200 \
#    --output_size 5 \
#    --n_way 5 \
#    --k_shot 5 \
#    --k_query 15 \
#    --rates 1.0 \
#    --v_th 0.2 \
#    --v_decay 0.3 \
#    --T 50 \
#    --step 1 \
#    --lstm_bias 4 6 -5 -4

# CSNN For Gesture_DVS Dataset
# 15ms 5w1s 2020-11-12-23_07_16 85%
# 15ms 5w1s 2020-11-12-23_07_51 97%

# NPO 10ms 5w5s 2020-11-10-22_58_21 93%
# NPO 15ms 5w5s 2020-11-10-23_00_49 88%
# NPO 20ms 5w5s 2020-11-10-23_02_44 85%
# NPO 25ms 5w5s 2020-11-10-23_04_39 80%

export CUDA_VISIBLE_DEVICES=0
python ../main_snn_dvs.py \
   --model 'MSTO' \
   --structure 'SCN_4_32_v2' \
   --checkpoints '2020-11-10-23_04_39' \
   --dataSet 'gesture_dvs' \
   --dataDir '/opt/data/durian' \
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
   --lr 0.0001