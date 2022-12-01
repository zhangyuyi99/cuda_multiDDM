#!/bin/bash
# path = '/u/homes/yz655/net/cicutagroup/yz655/'
# ./main -o '/u/homes/yz655/net/cicutagroup/yz655/cuda_run_result/0036/' -v -f '/u/homes/yz655/net/cicutagroup/yz655/testing/compressed_wheat_DJI_0036_al_19.0_fl_161_scale_8.47x.mp4' -N 500 -Z -C 150 -T '/u/homes/yz655/net/cicutagroup/yz655/cuda_run/tau.txt' -S '/u/homes/yz655/net/cicutagroup/yz655/cuda_run/scale.txt' -Q '/u/homes/yz655/net/cicutagroup/yz655/cuda_run/lambda.txt'

# ./main -o '/u/homes/yz655/net/cicutagroup/yz655/cuda_run_result/0050/' -v -f '/u/homes/yz655/net/cicutagroup/yz655/testing/wheat_DJI_0050_al_20.8_fl_161_scale_7.74x.avi' -N 500 -T '/u/homes/yz655/cuda_multiDDM/resources/example2/tau_vector.txt' -S '/u/homes/yz655/cuda_multiDDM/resources/example2/scale_vector.txt' -Q '/u/homes/yz655/cuda_multiDDM/resources/example2/lamda_vector.txt'

# ./main -o '/cicutagroup/yz655/cuda_run_result/compressed_nottingham_sample_new/' -v -f '/cicutagroup/yz655/testing/compressed_nottingham_sample.avi' -N 180 -C 180 -T '/cicutagroup/yz655/cuda_run/tau.txt' -S '/cicutagroup/yz655/cuda_run/scale.txt' -Q '/cicutagroup/yz655/cuda_run/lambda.txt'

# ./main -o '/cicutagroup/yz655/cuda_run_result/trimmed_14s_WinterWheat1_2022-07-25-140826-0000/' -v -f '/cicutagroup/yz655/testing/trimmed_14s_WinterWheat1_2022-07-25-140826-0000.avi' -Z -N 300 -C 300 -T '/cicutagroup/yz655/cuda_run/tau.txt' -S '/cicutagroup/yz655/cuda_run/scale.txt' -Q '/cicutagroup/yz655/cuda_run/lambda.txt'

# ./main -o '/cicutagroup/yz655/cuda_run_result/singlestream/fps_10_trimmed_14s_WinterWheat1_2022-07-25-140826-0000/' -v -f '/cicutagroup/yz655/testing/fps_10_trimmed_14s_WinterWheat1_2022-07-25-140826-0000.avi' -N 140 -C 140 -Z -T '/cicutagroup/yz655/cuda_run/tau.txt' -S '/cicutagroup/yz655/cuda_run/scale.txt' -Q '/cicutagroup/yz655/cuda_run/lambda.txt'

# ./main -o '/cicutagroup/yz655/cuda_run_result/multistream/fps_10_trimmed_14s_WinterWheat1_2022-07-25-140826-0000_new/' -v -f '/cicutagroup/yz655/testing/fps_10_trimmed_14s_WinterWheat1_2022-07-25-140826-0000.avi' -N 140 -C 140 -T '/cicutagroup/yz655/cuda_run/tau.txt' -S '/cicutagroup/yz655/cuda_run/scale.txt' -Q '/cicutagroup/yz655/cuda_run/lambda.txt'

#################
# ./main -o '/cicutagroup/yz655/cuda_run_result/multistream/fps_10_compressed_wheat_DJI_0050_al_20.8_fl_161_scale_7.74x/' -v -f '/cicutagroup/yz655/testing/fps_10_compressed_wheat_DJI_0050_al_20.8_fl_161_scale_7.74x.avi' -N 260 -C 100 -T '/cicutagroup/yz655/cuda_run/tau.txt' -S '/cicutagroup/yz655/cuda_run/scale.txt' -Q '/cicutagroup/yz655/cuda_run/lambda.txt'

# ./main -o '/cicutagroup/yz655/cuda_run_result/test_nonAve_ISF_DJI_0052/' -v -K -f '/cicutagroup/yz655/testing/fps_10_compressed_gray_DJI_0052.MP4' -N 260 -C 100 -T '/cicutagroup/yz655/cuda_run/tau.txt' -S '/cicutagroup/yz655/cuda_run/scale.txt' -Q '/cicutagroup/yz655/cuda_run/lambda.txt'  

# ./main -o '/cicutagroup/yz655/cuda_run_result/testing/test_nonAve_ISF_DJI_0052/' -v -K -f '/cicutagroup/yz655/testing/fps_10_compressed_gray_DJI_0052.MP4' -N 260 -C 100 -T '/cicutagroup/yz655/cuda_run/tau.txt' -S '/cicutagroup/yz655/cuda_run/scale.txt' -Q '/cicutagroup/yz655/cuda_run/lambda.txt'  

# ./main -o '/cicutagroup/yz655/cuda_run_result/testing/test_nonAve_ISF_compressed_fps10_WinterWheat1_2022-07-25-140826-0000/' -v -K -f '/cicutagroup/yz655/nott_videos/compressed_fps10_nott_videos/compressed_fps10_WinterWheat1_2022-07-25-140826-0000.avi' -N 300 -C 100 -T '/cicutagroup/yz655/cuda_run/tau.txt' -S '/cicutagroup/yz655/cuda_run/scale.txt' -Q '/cicutagroup/yz655/cuda_run/lambda.txt'  

# ./main -o '/cicutagroup/yz655/cuda_run_result/testing/test_nonAve_ISF_DJI_0050/' -v -K -f '/cicutagroup/yz655/drone_videos/compressed_fps10_gray_drone_videos/compressed_fps10_gray_DJI_0050.MP4' -N 300 -C 100 -T '/cicutagroup/yz655/cuda_run/tau.txt' -S '/cicutagroup/yz655/cuda_run/scale.txt' -Q '/cicutagroup/yz655/cuda_run/lambda.txt'  

./main -o '/cicutagroup/yz655/cuda_run_result/testing/test_nonAve_ISF_DJI_0052/' -v -K -f '/cicutagroup/yz655/drone_videos/compressed_fps10_gray_drone_videos/compressed_fps10_gray_DJI_0052.MP4' -N 300 -C 100 -T '/cicutagroup/yz655/cuda_run/tau.txt' -S '/cicutagroup/yz655/cuda_run/scale.txt' -Q '/cicutagroup/yz655/cuda_run/lambda.txt'  



# ./main -o '/cicutagroup/yz655/cuda_run_result/multistream/fps_10_compressed_gray_DJI_0051/' -v -f '/cicutagroup/yz655/testing/fps_10_compressed_gray_DJI_0051.MP4' -N 210 -C 210 -T '/cicutagroup/yz655/cuda_run/tau.txt' -S '/cicutagroup/yz655/cuda_run/scale.txt' -Q '/cicutagroup/yz655/cuda_run/lambda.txt'

# ./main -o '/cicutagroup/yz655/cuda_run_result/multistream/fps_10_compressed_gray_DJI_0052/' -v -f '/cicutagroup/yz655/testing/fps_10_compressed_gray_DJI_0052.MP4' -N 210 -C 210 -T '/cicutagroup/yz655/cuda_run/tau.txt' -S '/cicutagroup/yz655/cuda_run/scale.txt' -Q '/cicutagroup/yz655/cuda_run/lambda.txt'

# ./main -o '/cicutagroup/yz655/cuda_run_result/nott_sample_new/' -v -f '/cicutagroup/yz655/testing/nottingham_sample.avi' -N 180 -C 180 -T '/cicutagroup/yz655/cuda_run/tau.txt' -S '/cicutagroup/yz655/cuda_run/scale.txt' -Q '/cicutagroup/yz655/cuda_run/lambda.txt'

# ./main -o '/u/homes/yz655/net/cicutagroup/yz655/cuda_run_result/0050/' -v -f '/u/homes/yz655/net/cicutagroup/yz655/testing/stable_compressed_gray_DJI_0050.MP4' -N 300 -C 150 -Z -T '/u/homes/yz655/net/cicutagroup/yz655/cuda_run/tau.txt' -S '/u/homes/yz655/net/cicutagroup/yz655/cuda_run/scale.txt' -Q '/u/homes/yz655/net/cicutagroup/yz655/cuda_run/lambda.txt'

# ./main -o '/cicutagroup/yz655/cuda_run_result/compressed_WinterWheat1_2022-07-25-140826-0000/' -v -f '/cicutagroup/yz655/testing/compressed_WinterWheat1_2022-07-25-140826-0000.avi' -N 1000 -C 500 -Z -T '/cicutagroup/yz655/cuda_run/tau.txt' -S '/cicutagroup/yz655/cuda_run/scale.txt' -Q '/cicutagroup/yz655/cuda_run/lambda.txt'

# 8bit_compressed_gray_DJI_0050.MP4

# without multistream
# ./main -o '/cicutagroup/yz655/cuda_run_result/single_channel_8bit_compressed_gray_DJI_0050/' -v -f '/cicutagroup/yz655/testing/single_channel_8bit_compressed_gray_DJI_0050.MP4' -N 799 -C 300 -Z -T '/cicutagroup/yz655/cuda_run/tau.txt' -S '/cicutagroup/yz655/cuda_run/scale.txt' -Q '/cicutagroup/yz655/cuda_run/lambda.txt'
# with multistream
# ./main -o '/cicutagroup/yz655/cuda_run_result/single_channel_8bit_compressed_gray_DJI_0050/' -v -f '/cicutagroup/yz655/testing/single_channel_8bit_compressed_gray_DJI_0050.MP4' -N 600 -C 300 -T '/cicutagroup/yz655/cuda_run/tau.txt' -S '/cicutagroup/yz655/cuda_run/scale.txt' -Q '/cicutagroup/yz655/cuda_run/lambda.txt'


# ffmpeg -i input.mp4 -vf scale=480:320 output_320.mp4
# ffmpeg -i '/u/homes/yz655/net/cicutagroup/yz655/testing/wheat_DJI_0036_al_19.0_fl_161_scale_8.47x.mp4' -vf "scale=1920:1080" -b:v 10M '/u/homes/yz655/net/cicutagroup/yz655/testing/compressed_wheat_DJI_0036_al_19.0_fl_161_scale_8.47x.mp4'
# ffmpeg -i '/u/homes/yz655/net/cicutagroup/yz655/testing/wheat_DJI_0050_al_20.8_fl_161_scale_7.74x.avi' -vf "scale=1920:1080" -b:v 10M '/u/homes/yz655/net/cicutagroup/yz655/testing/compressed_wheat_DJI_0050_al_20.8_fl_161_scale_7.74x.avi'
# ffmpeg -i '/u/homes/yz655/net/cicutagroup/yz655/testing/nottingham_sample.avi' -vf "scale=1024:1024" -b:v 10M '/u/homes/yz655/net/cicutagroup/yz655/testing/compressed_nottingham_sample.avi'
# DJI_0050.MP4

# ffmpeg -i '/u/homes/yz655/net/cicutagroup/yz655/testing/WinterWheat1_2022-07-25-140826-0000.avi' -vf "scale=1024:1024" -b:v 10M '/u/homes/yz655/net/cicutagroup/yz655/testing/compressed_WinterWheat1_2022-07-25-140826-0000.avi'

# ffmpeg -i '/u/homes/yz655/net/cicutagroup/yz655/testing/DJI_0050.MP4' -vf format=gray '/u/homes/yz655/net/cicutagroup/yz655/testing/gray_DJI_0050.MP4'
# ffmpeg -i '/u/homes/yz655/net/cicutagroup/yz655/testing/gray_DJI_0050.MP4' -vf scale=1920:1080 -b:v 10M '/u/homes/yz655/net/cicutagroup/yz655/testing/compressed_gray_DJI_0050.MP4'

# ffmpeg -ss 00:00:16 -to 00:00:26 -i '/u/homes/yz655/net/cicutagroup/yz655/testing/compressed_gray_DJI_0050.MP4' -c copy '/u/homes/yz655/net/cicutagroup/yz655/testing/trimmed_compressed_gray_DJI_0050.MP4'

# ffmpeg -ss 00:00:01 -to 00:00:13 -i '/cicutagroup/yz655/testing/WinterWheat1_2022-07-25-140826-0000.avi' -c copy '/cicutagroup/yz655/testing/trimmed_12s_WinterWheat1_2022-07-25-140826-0000.avi'

# ffmpeg -i '/u/homes/yz655/net/cicutagroup/yz655/testing/compressed_gray_DJI_0050.MP4' -vf deshake '/u/homes/yz655/net/cicutagroup/yz655/testing/stable_compressed_gray_DJI_0050.MP4'

# python plot_averaged.py --root '/u/homes/yz655/net/cicutagroup/yz655/cuda_run_result/0050/' --scales 1024 512 256 128 --tiles 1 4 4 16 --umpx 0.09700

# ffmpeg -i '/u/homes/yz655/net/cicutagroup/yz655/testing/compressed_gray_DJI_0050.MP4' -c:a copy -sample_fmt u8 '/u/homes/yz655/net/cicutagroup/yz655/testing/8bit_compressed_gray_DJI_0050.MP4'


# convert to single channel, 8 bits/pixel gray video
# ffmpeg -i '/u/homes/yz655/net/cicutagroup/yz655/testing/8bit_compressed_gray_DJI_0050.MP4' -filter:v "format=gray" '/u/homes/yz655/net/cicutagroup/yz655/testing/single_channel_8bit_compressed_gray_DJI_0050.MP4'
# ffmpeg -i '/cicutagroup/yz655/testing/8bit_compressed_gray_DJI_0050.MP4' -c:v libx265 -pix_fmt gray '/cicutagroup/yz655/testing/single_channel_8bit_compressed_gray_DJI_0050.MP4'

# check video pixel format
# ffprobe -loglevel error -show_entries stream=pix_fmt -of csv=p=0 '/cicutagroup/yz655/testing/single_channel_8bit_compressed_gray_DJI_0050.MP4'

# check video corruptions
# ffmpeg –v error –i '/cicutagroup/yz655/testing/single_channel_8bit_compressed_gray_DJI_0050.MP4' -f null – &> '/cicutagroup/yz655/testing/corruptions.log'
# ffmpeg -v error -i '/cicutagroup/yz655/testing/compressed_nottingham_sample.avi' -f null - 2>'/cicutagroup/yz655/testing/corruptions.log'
# ffprobe '/cicutagroup/yz655/testing/compressed_nottingham_sample.avi'; echo $?

# reduce video fps
# ffmpeg -i '/cicutagroup/yz655/testing/trimmed_14s_WinterWheat1_2022-07-25-140826-0000.avi' -filter:v fps=fps=10 '/cicutagroup/yz655/testing/fps_10_trimmed_14s_WinterWheat1_2022-07-25-140826-0000.avi'\

# ffmpeg -i '/cicutagroup/yz655/testing/compressed_wheat_DJI_0050_al_20.8_fl_161_scale_7.74x.avi' -filter:v fps=fps=10 '/cicutagroup/yz655/testing/fps_10_compressed_wheat_DJI_0050_al_20.8_fl_161_scale_7.74x.avi'\

# Python plot
# python /u/homes/yz655/cuda_multiDDM/resources/python_tools/plot_averaged.py --root /cicutagroup/yz655/cuda_run_result/multistream/fps_10_trimmed_14s_WinterWheat1_2022-07-25-140826-0000/ --scales 1024 512 --tiles 1 4 --umpx 0.09700

#########################
# ffmpeg -i '/cicutagroup/yz655/testing/DJI_0051.MP4' -vf format=gray '/cicutagroup/yz655/testing/gray_DJI_0051.MP4'
# ffmpeg -i '/cicutagroup/yz655/testing/gray_DJI_0051.MP4' -vf scale=1920:1080 -b:v 10M '/cicutagroup/yz655/testing/compressed_gray_DJI_0051.MP4'
# ffmpeg -i '/cicutagroup/yz655/testing/compressed_gray_DJI_0051.MP4' -filter:v fps=fps=10 '/cicutagroup/yz655/testing/fps_10_compressed_gray_DJI_0051.MP4'\

# ffmpeg -i '/cicutagroup/yz655/testing/DJI_0052.MP4' -vf format=gray '/cicutagroup/yz655/testing/gray_DJI_0052.MP4'
# ffmpeg -i '/cicutagroup/yz655/testing/gray_DJI_0052.MP4' -vf scale=1920:1080 -b:v 10M '/cicutagroup/yz655/testing/compressed_gray_DJI_0052.MP4'
# ffmpeg -i '/cicutagroup/yz655/testing/compressed_gray_DJI_0052.MP4' -filter:v fps=fps=10 '/cicutagroup/yz655/testing/fps_10_compressed_gray_DJI_0052.MP4'\

######################### deshake
# The first pass ('detect') generates stabilization data and saves to `transforms.trf`
# The `-f null -` tells ffmpeg there's no output video file
# ffmpeg -i '/cicutagroup/yz655/testing/compressed_gray_DJI_0050.MP4' -vf vidstabdetect -f null -

# The second pass ('transform') uses the .trf and creates the new stabilized video.
# ffmpeg -i '/cicutagroup/yz655/testing/compressed_gray_DJI_0050.MP4' -vf vidstabtransform '/cicutagroup/yz655/testing/new_stable_compressed_gray_DJI_0050.MP4'