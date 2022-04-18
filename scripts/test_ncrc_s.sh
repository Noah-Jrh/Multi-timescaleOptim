###
 # @Author       : Noah
 # @Version      : v1.0.0
 # @Date         : 2020-09-11 11:37:47
 # @LastEditors  : Please set LastEditors
 # @LastEditTime : 2020-11-18 22:05:00
 # @CopyRight (c): 2019 NCRC, SCU. All rights reserved.
 # @Description  : Please add descriptioon
### 

# For Omniglot
# LTS 5w1s T=20 step=1 2020-11-09-06_04_13 (0.4, 0.2)
# LTS 5w5s T=20 step=1 2020-10-02-17_47_57
# LTS 5w5s T=15 step=1 2020-10-02-17_47_33
# LTS 5w5s T=10 step=1 2020-10-02-17_47_16
# LTS 5w5s T=5 step=1 2020-10-02-17_46_49
# LTS 5w5s T=8 step=2 2020-10-01-22_54_15
# noLTS 5w1s T=20 step=1 2020-11-11-07_32_42 2020-11-11-07_37_34
# noLTS 5w5s T=20 step=1 2020-11-11-07_31_12 2020-11-11-07_38_12
# export CUDA_VISIBLE_DEVICES=0
# python ../main_snn_v1.py \
#    --model 'MSTOv1' \
#    --structure 'SCN_4_32' \
#    --checkpoints '2020-11-09-06_04_13' \
#    --dataSet 'omniglot' \
#    --dataDir '/home/template/Data' \
#    --resDir '../results' \
#    --loss_fun 'ce' \
#    --sur_grad 'linear' \
#    --img_size 28 \
#    --output_size 5 \
#    --n_way 5 \
#    --k_shot 1 \
#    --k_query 15 \
#    --rates 1.0 \
#    --v_decay 0.4 \
#    --v_th 0.2 \
#    --T 20 \
#    --step 1 \
#    --lstm_bias 4 6 -5 -4 \
#    --lr 0.001

# # For Omniglot HMAX-SNN
# # LTS 5w1s 2020-11-10-14_25_17
# # LTS 5w5s 2020-11-10-16_28_26
# # noLTS 5w1s 2020-11-10-14_05_33(0.3 0.3) 2020-11-10-14_18_36(0.9 1.0) 2020-11-10-15_07_57(0.9 1.0)
# # noLTS 5w5s 2020-11-10-12_58_15
# export CUDA_VISIBLE_DEVICES=0
# python ../main_snn_v1.py \
#    --model 'MSTO' \
#    --structure 'SNN_3_HMAX_v1' \
#    --checkpoints '2020-11-10-12_58_15' \
#    --dataSet 'omniglot' \
#    --dataDir '/home/template/Data' \
#    --preprocess 'hmax' \
#    --resDir '../results' \
#    --loss_fun 'ce' \
#    --sur_grad 'linear' \
#    --input_size 360 \
#    --output_size 5 \
#    --n_way 5 \
#    --k_shot 5 \
#    --k_query 15 \
#    --rates 1.0 \
#    --v_decay 0.9 \
#    --v_th 1.0 \
#    --T 20 \
#    --step 1 \
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
#    --dataDir '/home/template/Data' \
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
# NPO 25ms 5w1s 2020-11-12-09_32_22 83%
# NPO 15ms 5w1s 2020-11-12-23_07_16 85%
# NPO 20ms 5w1s 2020-11-12-09_30_19 86%
# NPO 10ms 5w1s 2020-11-12-15_03_48 91%

# NPO 25ms 5w1s 2020-11-12-09_33_29 88%
# NPO 20ms 5w1s 2020-11-12-09_31_26 92%
# NPO 15ms 5w1s 2020-11-12-23_07_51 97%
# NPO 10ms 5w1s 2020-11-12-14_55_34 72%

# noNPO 10ms 5w1s 2020-11-15-11_49_38
# noNPO 15ms 5w1s 2020-11-15-11_58_38
# noNPO 20ms 5w1s 2020-11-15-12_00_13
# noNPO 25ms 5w1s 2020-11-15-12_45_54

# noNPO 10ms 5w5s 2020-11-16-12_12_30
# noNPO 15ms 5w5s 2020-11-16-12_26_04
# noNPO 20ms 5w5s 2020-11-16-12_46_16
# noNPO 25ms 5w5s 2020-11-16-12_47_26
# export CUDA_VISIBLE_DEVICES=3
# python ../main_snn_dvs.py \
#    --model 'MSTO' \
#    --structure 'SCN_4_32_v3' \
#    --checkpoints '2020-11-16-12_47_26' \
#    --dataSet 'gesture_dvs' \
#    --dataDir '/home/template/Data' \
#    --resDir '../results' \
#    --loss_fun 'ce' \
#    --sur_grad 'linear' \
#    --img_size 128 \
#    --output_size 5 \
#    --n_way 5 \
#    --k_shot 5 \
#    --k_query 15 \
#    --rates 1.0 \
#    --v_th 0.2 \
#    --v_decay 0.3 \
#    --T 12 \
#    --step 1 \
#    --resolution 25000 \
#    --lstm_bias 4 6 -5 -4 \
#    --lr 0.0001

# # For Omniglot + CNN
# # 5w1s 2020-11-12-01_33_58
# # 5w5s 2020-11-12-01_34_35
# export CUDA_VISIBLE_DEVICES=2
# python ../main_ann.py \
#    --model 'MSTO' \
#    --structure 'CNN_4_32' \
#    --checkpoints '2020-11-12-01_34_35' \
#    --dataSet 'omniglot' \
#    --dataDir '/home/template/Data' \
#    --resDir '../results' \
#    --loss_fun 'ce' \
#    --output_size 5 \
#    --n_way 5 \
#    --k_shot 5 \
#    --k_query 15 \
#    --lstm_bias 4 6 -5 -4 \
#    --lr 0.001