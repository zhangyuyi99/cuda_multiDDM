#!/bin/bash

set -euo pipefail
set -x
input_file="/cicutagroup/yz655/drone_videos/compressed_fps10_gray_drone_videos/compressed_fps10_gray_DJI_0051.MP4"
output_file="/cicutagroup/yz655/drone_videos/stable_compressed_fps10_gray_drone_videos/stable_compressed_fps10_gray_DJI_0051_new.MP4"
transform_file="${output_file}.trf"
# ffmpeg -i "$input_file" -vf "vidstabdetect=shakiness=10:accuracy=15:result=$transform_file" -f null -
# ffmpeg -i "$input_file" -filter:v vidstabtransform=zoom=25:input="$transform_file":interpol=bicubic -crf 23 -preset slow "$output_file"

ffmpeg -i "$input_file" -vf vidstabdetect=stepsize=32:shakiness=10:accuracy=10:result=$transform_file -f null -

ffmpeg -i "$input_file" -vf vidstabtransform=input=$transform_file:zoom=0:smoothing=10,unsharp=5:5:0.8:3:3:0.4 -vcodec libx264 -tune film -acodec copy -preset slow "$output_file"