#!/bin/bash

original_video_path='/cicutagroup/yz655/drone/100MEDIA/'
gray_drone_videos_path='/cicutagroup/yz655/drone_videos/gray_drone_videos/'
fps10_gray_drone_videos_path='/cicutagroup/yz655/drone_videos/fps10_gray_drone_videos/'


compressed_fps10_gray_drone_videos_path='/cicutagroup/yz655/drone_videos/compressed_fps10_gray_drone_videos/'
stable_compressed_fps10_gray_drone_videos_path='/cicutagroup/yz655/drone_videos/stable_compressed_fps10_gray_drone_videos/'

result_path='/cicutagroup/yz655/cuda_run_result/drone_even_lambda/'          
for filename in ${compressed_fps10_gray_drone_videos_path}*.MP4; do
    # name=${filename:34:12}

    arr_filename=(${filename//// })
    video_name=${arr_filename[-1]}
    arr_video_name=(${video_name//./ })
    folder_name=${arr_video_name[0]}
    # echo "${folder_name}"

    video_frame_number=$(ffprobe -select_streams v -show_streams ${filename} 2>/dev/null | grep nb_frames | sed -e 's/nb_frames=//')
    chunk_frame_number=$((${video_frame_number}/3))
    # echo "${chunk_frame_number}"

    /bin/python /u/homes/yz655/cuda_multiDDM/generate_param/generate_param.py ${chunk_frame_number}

    output_path=${result_path}${folder_name}/
    mkdir -p ${output_path}

    ./main -o ${output_path} -v -f ${filename} -N ${video_frame_number} -C ${chunk_frame_number} -T '/cicutagroup/yz655/cuda_run/tau.txt' -S '/cicutagroup/yz655/cuda_run/scale.txt' -Q '/cicutagroup/yz655/cuda_run/lambda.txt'

done


# nott_video_path='/cicutagroup/yz655/Field_videos_25-7-22/Field_videos_25-7-22/'
# fps10_nott_videos_path='/cicutagroup/yz655/nott_videos/fps10_nott_videos/'
# compressed_fps10_nott_videos_path='/cicutagroup/yz655/nott_videos/compressed_fps10_nott_videos/'

# result_path='/cicutagroup/yz655/cuda_run_result/nott/'          
# for filename in ${compressed_fps10_nott_videos_path}*.avi; do
#     # name=${filename:34:12}

#     arr_filename=(${filename//// })
#     video_name=${arr_filename[-1]}
#     arr_video_name=(${video_name//./ })
#     folder_name=${arr_video_name[0]}
#     # echo "${folder_name}"

#     video_frame_number=$(ffprobe -select_streams v -show_streams ${filename} 2>/dev/null | grep nb_frames | sed -e 's/nb_frames=//')
#     chunk_frame_number=$((${video_frame_number}/3))
#     echo "${chunk_frame_number}"

#     /bin/python /u/homes/yz655/cuda_multiDDM/generate_param/generate_param.py ${chunk_frame_number}

#     output_path=${result_path}${folder_name}/
#     mkdir -p ${output_path}

#     ./main -o ${output_path} -v -f ${filename} -N ${video_frame_number} -C ${chunk_frame_number} -T '/cicutagroup/yz655/cuda_run/tau.txt' -S '/cicutagroup/yz655/cuda_run/scale.txt' -Q '/cicutagroup/yz655/cuda_run/lambda.txt'

# done