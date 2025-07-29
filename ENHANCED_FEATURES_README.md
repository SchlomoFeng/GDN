# GDN Enhanced Anomaly Detection Features

This enhanced version of GDN (Graph Neural Network-Based Anomaly Detection) includes comprehensive logging, visualization, and anomaly analysis capabilities.

## 🆕 New Features

### 1. Enhanced Training and Testing Information Output
- **Detailed epoch logging**: Training loss, validation loss, learning rate, epoch time
- **Batch progress tracking**: Real-time progress during training and testing
- **Comprehensive metrics**: F1, precision, recall, AUC scores with detailed reporting
- **Training history persistence**: All metrics saved to JSON files for later analysis
- **Early stopping with detailed tracking**: Monitor improvement counts and best model saving

### 2. Advanced Anomaly Localization
- **Time-step anomaly scores**: Detailed scores for each time step and sensor
- **Anomaly period detection**: Automatic identification of anomalous time periods
- **Sensor ranking**: Top-k most anomalous sensors with detailed scores and statistics
- **Pattern analysis**: Detection of common patterns in anomalous periods
- **Correlation analysis**: Understanding relationships between sensor anomalies

### 3. Comprehensive Visualization Suite
- **Training progress plots**: Loss curves, learning rate schedules, training times
- **Anomaly time series**: Interactive plots with threshold lines and anomaly highlights
- **Sensor heatmaps**: Visual representation of anomaly scores across sensors and time
- **Top sensor analysis**: Bar charts of most anomalous sensors
- **Performance visualization**: ROC curves, confusion matrices
- **Graph structure plots**: Visualization of learned sensor relationships

### 4. Detailed Analysis and Reporting
- **Anomaly reports**: Comprehensive JSON reports with all analysis results
- **Statistical summaries**: Percentiles, distributions, and summary statistics
- **Performance metrics**: Detailed breakdown of model performance
- **Export capabilities**: CSV exports of scores, rankings, and analysis results

### 5. Configuration and Control
- **Flexible logging levels**: Control output verbosity (INFO/DEBUG)
- **Visualization control**: Enable/disable specific visualizations
- **Analysis options**: Configure threshold methods and anomaly detection parameters
- **Batch processing**: Control batch progress reporting frequency

## 📁 New Directory Structure

```
GDN/
├── logs/                           # Training and testing logs
│   ├── {experiment}_training.log   # Detailed training logs
│   ├── {experiment}_testing.log    # Testing progress and anomaly logs
│   ├── {experiment}_metrics.json   # Training metrics history
│   └── {experiment}_final_results.json  # Comprehensive results
├── visualizations/                 # Generated plots and charts
│   ├── {experiment}_training_history.png
│   ├── {experiment}_anomaly_timeseries.png
│   ├── {experiment}_sensor_heatmap.png
│   ├── {experiment}_top_sensors.png
│   ├── {experiment}_confusion_matrix.png
│   └── {experiment}_roc_curve.png
├── results/                       # Detailed analysis results
│   ├── {experiment}_detailed_report.json
│   ├── {experiment}_scores.csv
│   ├── {experiment}_periods.csv
│   └── {experiment}_sensor_rankings.csv
└── util/                         # Enhanced utility modules
    ├── logger.py                 # Comprehensive logging system
    ├── visualizer.py            # Visualization toolkit
    └── anomaly_analyzer.py      # Anomaly analysis engine
```

## 🚀 Quick Start with Enhanced Features

### Using the Enhanced Run Script
```bash
# Run with enhanced analysis on MSL dataset
bash run_enhanced.sh cpu msl

# Run with GPU
bash run_enhanced.sh 0 msl
```

### Configuration Options
The enhanced version adds several new command-line options:

```bash
python main.py \
    -dataset msl \
    -enable_detailed_analysis True \     # Enable detailed anomaly analysis
    -enable_visualizations True \        # Enable result visualizations
    -log_level INFO \                    # Logging level (INFO/DEBUG)
    -save_intermediate_results True \    # Save intermediate training results
    -visualization_interval 10          # Epochs between visualization updates
```

### Demo Script
Run the demo to see all enhanced features:
```bash
python demo_enhanced_features.py
```

## 📊 Enhanced Output Examples

### Training Progress
```
[12:34:56] Epoch 5/30 Summary:
  Train Loss: 0.045623
  Val Loss: 0.038901
  Time: 45.32s, LR: 0.000850
  Val F1: 0.8234, Precision: 0.8456, Recall: 0.8015
  Best Val Loss: 0.038901, Stop Count: 0/15
------------------------------------------------------------
```

### Anomaly Analysis Results
```
Anomaly Analysis Summary:
  Total anomaly periods detected: 3
  Average anomaly score: 0.234567
  Max anomaly score: 2.456789
  Anomaly threshold: 0.567890
  Performance - F1: 0.8523, Precision: 0.8734, Recall: 0.8321

Top 5 Most Anomalous Sensors:
  1. P-10: 0.456789
  2. M-6: 0.345678
  3. T-4: 0.234567
  4. F-7: 0.198765
  5. M-1: 0.187654

Detected Anomaly Periods:
  1. Steps 150-175 (Length: 25, Max Score: 1.234567)
  2. Steps 430-445 (Length: 15, Max Score: 0.987654)
  3. Steps 780-805 (Length: 25, Max Score: 1.567890)
```

## 🛠️ Enhanced API Usage

### Using the Logger
```python
from util.logger import GDNLogger

# Initialize logger
logger = GDNLogger(experiment_name="my_experiment", config=train_config)

# Log training progress
logger.log_epoch_start(epoch, total_epochs, learning_rate)
logger.log_epoch_end(epoch, train_loss, val_loss, epoch_time, learning_rate)

# Log performance metrics
logger.log_performance_metrics({
    'f1_score': 0.85,
    'precision': 0.87,
    'recall': 0.83
}, 'test')

# Save final results
logger.save_final_results(test_results, val_results, best_metrics)
```

### Using the Visualizer
```python
from util.visualizer import GDNVisualizer

# Initialize visualizer
visualizer = GDNVisualizer(experiment_name="my_experiment")

# Create training history plot
visualizer.plot_training_history(training_history, save=True)

# Create anomaly analysis plots
visualizer.plot_anomaly_scores_timeseries(scores, ground_truth=labels, save=True)
visualizer.plot_sensor_anomaly_heatmap(sensor_scores, feature_names, save=True)
visualizer.plot_top_anomalous_sensors(sensor_rankings, save=True)
```

### Using the Anomaly Analyzer
```python
from util.anomaly_analyzer import AnomalyAnalyzer

# Initialize analyzer
analyzer = AnomalyAnalyzer(feature_names=feature_names)

# Calculate anomaly scores
anomaly_scores = analyzer.calculate_anomaly_scores(predictions, ground_truth)

# Detect anomaly periods
anomaly_periods = analyzer.detect_anomaly_periods(anomaly_scores, labels)

# Rank sensors
sensor_rankings = analyzer.rank_sensors_by_anomaly(anomaly_scores, top_k=10)

# Generate comprehensive report
report = analyzer.generate_anomaly_report(anomaly_scores, labels, 
                                        anomaly_periods, sensor_rankings)
```

## 📈 Benefits of Enhanced Features

1. **Better Understanding**: Detailed insights into model behavior and anomaly patterns
2. **Improved Debugging**: Comprehensive logs help identify training issues
3. **Visual Analysis**: Charts and plots make results more interpretable
4. **Reproducibility**: All configurations and results are logged for reproducibility
5. **Production Ready**: Enhanced monitoring suitable for production deployments
6. **Research Support**: Detailed analysis supports research and experimentation

## 🔧 Dependencies

The enhanced features require additional dependencies:
```bash
pip install matplotlib seaborn pandas scikit-learn
```

For graph visualization (optional):
```bash
pip install networkx
```

## 📝 Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable_detailed_analysis` | bool | True | Enable detailed anomaly analysis |
| `enable_visualizations` | bool | True | Enable result visualizations |
| `log_level` | str | INFO | Logging level (INFO/DEBUG) |
| `save_intermediate_results` | bool | True | Save intermediate training results |
| `visualization_interval` | int | 10 | Epochs between visualization updates |

## 🎯 MSL Dataset Focus

The enhanced features are optimized for the MSL (Mars Science Laboratory) dataset:
- **27 sensors**: Tailored visualizations for MSL's 27-sensor configuration
- **Real data focus**: Uses actual MSL data from `data/msl/` directory
- **Sensor naming**: Proper handling of MSL sensor names (M-6, P-10, T-4, etc.)
- **Time series patterns**: Optimized for spacecraft telemetry patterns

## 📋 Best Practices

1. **Use descriptive experiment names**: Helps organize results across multiple runs
2. **Enable detailed analysis for final runs**: Provides comprehensive insights
3. **Monitor training logs**: Use logs to identify overfitting or convergence issues
4. **Review anomaly periods**: Manually validate detected anomalies
5. **Compare sensor rankings**: Use rankings to understand which sensors are most informative
6. **Save intermediate results**: Helpful for long training runs

## 🔍 Troubleshooting

**Common Issues:**

1. **Missing dependencies**: Install required packages with pip
2. **Memory issues**: Reduce batch size or disable some visualizations
3. **Slow visualization**: Set `enable_visualizations=False` for faster runs
4. **Large log files**: Adjust `log_level` to INFO to reduce verbosity

**Performance Tips:**

1. Use `visualization_interval` to control visualization frequency
2. Disable detailed analysis for quick experiments
3. Use appropriate batch sizes for your hardware
4. Monitor disk space for logs and visualizations

## 📄 License

Same as original GDN project. Enhanced features maintain compatibility with existing license.