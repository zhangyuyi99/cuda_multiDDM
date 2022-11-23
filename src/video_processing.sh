#!/bin/bash

original_video_path='/cicutagroup/yz655/drone/100MEDIA/'
gray_drone_videos_path='/cicutagroup/yz655/drone_videos/gray_drone_videos/'
fps10_gray_drone_videos_path='/cicutagroup/yz655/drone_videos/fps10_gray_drone_videos/'
compressed_fps10_gray_drone_videos_path='/cicutagroup/yz655/drone_videos/compressed_fps10_gray_drone_videos/'
stable_compressed_fps10_gray_drone_videos_path='/cicutagroup/yz655/drone_videos/stable_compressed_fps10_gray_drone_videos/' 

# for filename in ${original_video_path}*.MP4; do
#     name=${filename:34:12}
#     # echo "${name}"
    
#     ffprobe -select_streams v -show_streams input.avi 2>/dev/null | grep nb_frames | sed -e 's/nb_frames=//'

#     # echo "${frame_number}"

#     # ffmpeg -i ${filename} -vf format=gray ${gray_drone_videos_path}gray_${name}
#     # ffmpeg -i ${gray_drone_videos_path}gray_${name} -filter:v fps=fps=10 ${fps10_gray_drone_videos_path}fps10_gray_${name}
#     # ffmpeg -i ${fps10_gray_drone_videos_path}fps10_gray_${name} -vf scale=1920:1080 -b:v 10M ${compressed_fps10_gray_drone_videos_path}compressed_fps10_gray_${name}
#     # ffmpeg -i ${compressed_fps10_gray_drone_videos_path}compressed_fps10_gray_${name} -vf deshake ${stable_compressed_fps10_gray_drone_videos_path}stable_compressed_fps10_gray_${name}

# done

nott_video_path='/cicutagroup/yz655/Field_videos_25-7-22/Field_videos_25-7-22/'
fps10_nott_videos_path='/cicutagroup/yz655/nott_videos/fps10_nott_videos/'
compressed_fps10_nott_videos_path='/cicutagroup/yz655/nott_videos/compressed_fps10_nott_videos/'
for filename in ${nott_video_path}*.avi; do
    name=${filename:61:80}
    # echo "${name}"

    ffmpeg -i ${filename} -filter:v fps=fps=10 ${fps10_nott_videos_path}fps10_${name}
    ffmpeg -i ${fps10_nott_videos_path}fps10_${name} -vf scale=1024:1024 -b:v 10M ${compressed_fps10_nott_videos_path}compressed_fps10_${name}
    # ffmpeg -i ${compressed_fps10_gray_drone_videos_path}compressed_fps10_gray_${name} -vf deshake ${stable_compressed_fps10_gray_drone_videos_path}stable_compressed_fps10_gray_${name}

done