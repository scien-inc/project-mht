#!/bin/bash



VIDEO_PATH="./gsam2/notebooks/videos/images"
OUTPUT_DIR="./data/frames"
OUTPUT_DIR_GSAM2="./data/gsam2_output"
DURATION=150 #秒数ではなく枚数
INTERVAL=50 #秒数ではなく枚数

# OUTPUT_DIR内のフォルダを削除
rm -rf $OUTPUT_DIR/*
rm -rf $OUTPUT_DIR_GSAM2/*

python module/split.py --frames_folder "$VIDEO_PATH" --output_base_dir "$OUTPUT_DIR" --duration $DURATION --interval $INTERVAL

# OUTPUT_DIR内のすべてのディレクトリに対して処理を行う  
for dir in `ls $OUTPUT_DIR`
do
    python gsam2/gsam2_c-idv2.py --input_folder $OUTPUT_DIR/$dir --output_dir $OUTPUT_DIR_GSAM2/$dir --device_id 1
done


# $OUTPUT_DIR_GSAM2内のすべてのディレクトリに対して処理を行う
for dir in `ls $OUTPUT_DIR_GSAM2`
do
    echo $dir
    MASK_DATA_DIR="$OUTPUT_DIR_GSAM2/$dir/mask_data"
    JSON_DATA_DIR="$OUTPUT_DIR_GSAM2/$dir/json_data"
    CSV_FILE_PATH="./data/results.csv"
    CORRECTED_MASK_DIR="$OUTPUT_DIR_GSAM2/$dir/corrected_masks"
    CORRECTED_JSON_DIR="$OUTPUT_DIR_GSAM2/$dir/corrected_jsons"
    DEVICE="cuda"

    python module/correct_id.py \
        --mask_data_dir "$MASK_DATA_DIR" \
        --json_data_dir "$JSON_DATA_DIR" \
        --csv_file_path "$CSV_FILE_PATH" \
        --corrected_mask_dir "$CORRECTED_MASK_DIR" \
        --corrected_json_dir "$CORRECTED_JSON_DIR" \
        --device $DEVICE

    # python gsam2/create_correct_id_video.py \
    #     --input_folder ./data/frames/$dir \
    #     --mask_data_dir $OUTPUT_DIR_GSAM2/$dir/mask_data \
    #     --json_data_dir $OUTPUT_DIR_GSAM2/$dir/corrected_jsons \
    #     --result_dir $OUTPUT_DIR_GSAM2/$dir/result2 \
    #     --output_video_path $OUTPUT_DIR_GSAM2/$dir/output2.mp4 \
    #     --frame_rate 5
    

done
python module/merge_segment.py \
    --base_dir $OUTPUT_DIR_GSAM2/ \
    --merge_dir ./data/merged_jsons