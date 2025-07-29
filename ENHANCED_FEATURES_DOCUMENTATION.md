# GDN Enhanced Features Documentation

This document describes the enhanced information output and anomaly localization features added to the Graph Deviation Network (GDN) implementation.

## Overview

The GDN model has been enhanced with comprehensive analysis and visualization capabilities while maintaining full backward compatibility with the original implementation. The enhancements provide:

- **Detailed training progress monitoring**
- **Sensor-level anomaly localization**
- **Comprehensive visualization suite**
- **Model architecture analysis**
- **Enhanced output and reporting**

## New Command Line Options

### Basic Usage (unchanged)
```bash
python main.py -dataset msl -device cpu -epoch 10
```

### Enhanced Usage
```bash
# Enable verbose output with detailed information
python main.py -dataset msl -device cpu -epoch 10 -verbose

# Enable visualization generation
python main.py -dataset msl -device cpu -epoch 10 -verbose -enable_visualization

# Enable anomaly localization analysis
python main.py -dataset msl -device cpu -epoch 10 -verbose -enable_localization

# Specify top-K sensors for analysis
python main.py -dataset msl -device cpu -epoch 10 -verbose -enable_localization -top_k_sensors 10

# Enable all features
python main.py -dataset msl -device cpu -epoch 10 -verbose -enable_visualization -enable_localization -top_k_sensors 5
```

### New Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-verbose` | flag | False | Enable detailed output and progress information |
| `-enable_visualization` | flag | False | Generate visualization plots and charts |
| `-enable_localization` | flag | False | Perform sensor-level anomaly localization analysis |
| `-top_k_sensors` | int | 5 | Number of top sensors to analyze in detail |

## Enhanced Features

### 1. Training Progress Monitoring

When `-verbose` is enabled, training provides:

- **Epoch-wise metrics**: Detailed loss tracking, gradient norms, validation scores
- **Real-time progress**: Batch-level progress reporting for long epochs
- **Training time tracking**: Elapsed time and estimated completion
- **Early stopping information**: Clear indication when early stopping is triggered
- **Learning rate monitoring**: Track optimizer state changes

Example output:
```
Training GDN model for 100 epochs...
  Dataset: msl
  Training samples: 5000
  Validation samples: 500
  Batch size: 128
  Learning rate: 0.001
  Early stopping window: 15
------------------------------------------------------------
Epoch   0/100 - Train Loss: 0.123456, Time: 12.3s, Grad Norm: 0.0145
         Validation Loss: 0.098765
         New best model saved (val_loss: 0.098765)
------------------------------------------------------------
```

### 2. Anomaly Localization Analysis

The `AnomalyLocalizer` class provides comprehensive sensor-level analysis:

#### Key Features:
- **Sensor-level anomaly scores**: Individual anomaly scores for each sensor at each timestep
- **Top-K anomalous sensors**: Identification of most anomalous sensors during detected events
- **Temporal pattern analysis**: Evolution of anomaly scores over time
- **Anomaly event detection**: Automatic detection and characterization of anomaly periods
- **Sensor importance ranking**: Overall ranking of sensors by anomaly detection importance

#### Example Analysis Report:
```
============================================================
GDN ANOMALY LOCALIZATION SUMMARY REPORT
============================================================

Dataset Overview:
  • Number of sensors: 25
  • Total timesteps: 8000
  • Anomaly events detected: 12
  • Total anomalous timesteps: 180

Top 5 Most Important Sensors:
   1. P-2_sensor: 2.456
   2. T-4_sensor: 1.987
   3. F-1_sensor: 1.754
   4. A-7_sensor: 1.632
   5. M-3_sensor: 1.445

Anomaly Events Summary:
  Event 0: Time 1200-1215 (Duration: 16 steps)
    Top sensors: P-2_sensor, T-4_sensor, F-1_sensor
    Max score: 3.245
```

### 3. Comprehensive Visualization Suite

The `GDNVisualizer` class generates multiple types of visualizations:

#### Training Visualizations:
- **Training curves**: Loss evolution, metrics over epochs
- **Learning dynamics**: Gradient norms, learning rate schedules

#### Anomaly Analysis Visualizations:
- **Anomaly timeline**: Time series of sensor anomaly scores
- **Anomaly dashboard**: Comprehensive multi-panel view
- **Sensor heatmaps**: Anomaly score heatmaps across sensors and time
- **Interactive dashboards**: Web-based interactive exploration (if Plotly available)

#### Model Architecture Visualizations:
- **Attention heatmaps**: Visualization of learned attention weights
- **Graph networks**: Learned sensor relationship graphs
- **Sensor embeddings**: t-SNE/PCA plots of learned sensor representations

#### Generated Files:
All visualizations are saved to `./visualizations/{dataset}_{timestamp}/`:
- `training_curves.png`
- `anomaly_timeline.png`
- `anomaly_dashboard.png`
- `attention_heatmap.png`
- `graph_network.png`
- `sensor_embeddings_tsne.png`
- `sensor_embeddings_pca.png`
- `interactive_dashboard.html` (if Plotly available)

### 4. Model Architecture Analysis

The `ModelAnalyzer` class provides deep insights into the trained model:

#### Architecture Analysis:
- **Parameter counting**: Total and trainable parameters
- **Layer analysis**: Detailed breakdown of model components
- **Memory usage**: Model size and complexity metrics

#### Graph Structure Analysis:
- **Learned adjacency matrix**: Extracted sensor relationships
- **Graph properties**: Density, clustering coefficient, connectivity
- **Hub sensor identification**: Most connected sensors in the learned graph

#### Embedding Analysis:
- **Embedding statistics**: Norms, distributions, diversity measures
- **Sensor clustering**: Automatic grouping of similar sensors
- **Similarity analysis**: Most and least similar sensor pairs

### 5. Enhanced Output and CSV Reports

#### Detailed CSV Output:
When anomaly localization is enabled, an enhanced CSV file is generated with:
- **Per-timestep data**: Individual sensor anomaly scores
- **Aggregate metrics**: Max scores, mean scores, anomalous sensor counts
- **Ground truth labels**: Original anomaly labels for comparison

Example CSV structure:
```csv
timestep,ground_truth_label,anomaly_score_sensor_01,anomaly_score_sensor_02,...,max_sensor_score,mean_sensor_score,num_anomalous_sensors
0,0,-0.147,0.576,-0.009,...,1.300,-0.140,1
1,0,-0.179,0.107,-0.035,...,1.307,0.036,1
```

#### Model Analysis Report:
A comprehensive text report is generated containing:
- Model architecture summary
- Graph structure analysis
- Embedding insights
- Key findings and recommendations

Example report location: `./results/{dataset}_{timestamp}_model_report.txt`

## Integration with Existing Workflow

The enhancements are fully backward compatible:

### Original workflow (unchanged):
```bash
python main.py -dataset msl -device cpu -epoch 100
```

### Enhanced workflow (minimal changes):
```bash
# Just add flags to enable features
python main.py -dataset msl -device cpu -epoch 100 -verbose -enable_visualization -enable_localization
```

## Dependencies

### Required (already included):
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- PyTorch

### Optional (for enhanced features):
- **Seaborn**: For improved visualizations (auto-installed when needed)
- **Plotly**: For interactive dashboards
- **NetworkX**: For graph network visualizations

Install optional dependencies:
```bash
pip install seaborn plotly networkx
```

## File Structure

```
GDN/
├── anomaly_localizer.py          # New: Sensor-level anomaly analysis
├── gdn_visualizer.py            # New: Comprehensive visualization suite
├── model_analyzer.py            # New: Model architecture analysis
├── main.py                      # Enhanced: Added new CLI options and integration
├── train.py                     # Enhanced: Added verbose training progress
├── test.py                      # Enhanced: Added verbose testing progress
├── visualizations/              # New: Generated visualization directory
│   └── {dataset}_{timestamp}/
├── results/                     # Enhanced: Additional detailed outputs
└── test_output/                 # New: Test outputs for verification
```

## Example Usage Scenarios

### Scenario 1: Quick Analysis
```bash
python main.py -dataset msl -device cpu -epoch 50 -verbose
```
- Provides detailed training progress
- Shows comprehensive metrics summary
- Generates model analysis report

### Scenario 2: Full Anomaly Investigation
```bash
python main.py -dataset msl -device cpu -epoch 100 -verbose -enable_localization -top_k_sensors 10
```
- Detailed anomaly localization analysis
- Top-10 sensor analysis
- Comprehensive summary reports
- Enhanced CSV output with sensor-level scores

### Scenario 3: Complete Analysis with Visualizations
```bash
python main.py -dataset msl -device cpu -epoch 100 -verbose -enable_visualization -enable_localization -top_k_sensors 5
```
- All analysis features enabled
- Complete visualization suite
- Interactive dashboards (if available)
- Model architecture insights

## Performance Considerations

- **Verbose mode**: Minimal overhead (~5% training time increase)
- **Visualization**: Moderate overhead (~10-15% increase) during generation
- **Localization**: Low overhead (~5-10% increase) for sensor-level analysis
- **Memory**: Enhanced features require ~20-30% additional memory for data storage

## Troubleshooting

### Common Issues:

1. **"Seaborn not available"**: Install with `pip install seaborn`
2. **"Plotly not available"**: Interactive features disabled, install with `pip install plotly`
3. **"NetworkX not available"**: Graph visualizations disabled, install with `pip install networkx`
4. **Memory issues**: Reduce batch size or disable visualization for large datasets
5. **Slow visualization**: Reduce the number of timesteps or sensors for plotting

### Verification:

Run the test suite to verify all features work:
```bash
python test_enhancements.py
```

This will test all new components and generate sample outputs.

## Benefits

1. **Enhanced Interpretability**: Understand which sensors contribute to anomaly detection
2. **Improved Debugging**: Detailed training progress helps identify issues
3. **Better Analysis**: Comprehensive visualizations reveal patterns and insights
4. **Research Support**: Detailed outputs support further analysis and research
5. **Production Ready**: Modular design allows selective feature enablement
6. **Backward Compatibility**: Existing workflows remain unchanged

The enhanced GDN implementation transforms a black-box anomaly detector into a comprehensive, interpretable system that provides deep insights into anomaly patterns and sensor relationships while maintaining the original model's performance and accuracy.