#!/bin/bash

# Enhanced run script for GDN with detailed analysis capabilities
# Usage: bash run_enhanced.sh <gpu_id> <dataset>

gpu_n=$1
DATASET=$2

seed=5
BATCH_SIZE=32
SLIDE_WIN=5
dim=64
out_layer_num=1
SLIDE_STRIDE=1
topk=5
out_layer_inter_dim=128
val_ratio=0.2
decay=0

# Enhanced analysis options
enable_detailed_analysis=True
enable_visualizations=True
log_level="INFO"
save_intermediate_results=True
visualization_interval=10

path_pattern="${DATASET}_enhanced"
COMMENT="${DATASET}_with_enhanced_analysis"

EPOCH=30
report='best'

echo "Starting GDN with Enhanced Anomaly Detection Analysis"
echo "Dataset: $DATASET"
echo "Enhanced Analysis: $enable_detailed_analysis"
echo "Visualizations: $enable_visualizations"
echo "================================================================"

if [[ "$gpu_n" == "cpu" ]]; then
    python main.py \
        -dataset $DATASET \
        -save_path_pattern $path_pattern \
        -slide_stride $SLIDE_STRIDE \
        -slide_win $SLIDE_WIN \
        -batch $BATCH_SIZE \
        -epoch $EPOCH \
        -comment $COMMENT \
        -random_seed $seed \
        -decay $decay \
        -dim $dim \
        -out_layer_num $out_layer_num \
        -out_layer_inter_dim $out_layer_inter_dim \
        -decay $decay \
        -val_ratio $val_ratio \
        -report $report \
        -topk $topk \
        -device 'cpu' \
        -enable_detailed_analysis $enable_detailed_analysis \
        -enable_visualizations $enable_visualizations \
        -log_level $log_level \
        -save_intermediate_results $save_intermediate_results \
        -visualization_interval $visualization_interval
else
    CUDA_VISIBLE_DEVICES=$gpu_n python main.py \
        -dataset $DATASET \
        -save_path_pattern $path_pattern \
        -slide_stride $SLIDE_STRIDE \
        -slide_win $SLIDE_WIN \
        -batch $BATCH_SIZE \
        -epoch $EPOCH \
        -comment $COMMENT \
        -random_seed $seed \
        -decay $decay \
        -dim $dim \
        -out_layer_num $out_layer_num \
        -out_layer_inter_dim $out_layer_inter_dim \
        -decay $decay \
        -val_ratio $val_ratio \
        -report $report \
        -topk $topk \
        -enable_detailed_analysis $enable_detailed_analysis \
        -enable_visualizations $enable_visualizations \
        -log_level $log_level \
        -save_intermediate_results $save_intermediate_results \
        -visualization_interval $visualization_interval
fi

echo "================================================================"
echo "GDN Enhanced Analysis Complete!"
echo "Check the following directories for results:"
echo "  - logs/: Training and testing logs"
echo "  - visualizations/: Generated plots and charts"
echo "  - results/: Detailed analysis results"
echo "================================================================"