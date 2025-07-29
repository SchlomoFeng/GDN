# -*- coding: utf-8 -*-
"""
Demo script to showcase GDN enhanced anomaly detection features
This script demonstrates the new logging, visualization, and analysis capabilities
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add the project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from util.logger import GDNLogger
from util.visualizer import GDNVisualizer
from util.anomaly_analyzer import AnomalyAnalyzer

def create_demo_data():
    """Create synthetic demo data for showcasing features"""
    print("Creating synthetic demo data...")
    
    # Create synthetic time series data
    time_steps = 1000
    num_features = 27  # Same as MSL dataset
    
    # Generate normal data with some patterns
    np.random.seed(42)
    normal_data = np.random.randn(time_steps, num_features) * 0.1
    
    # Add some periodic patterns
    for i in range(num_features):
        normal_data[:, i] += 0.5 * np.sin(2 * np.pi * np.arange(time_steps) / (50 + i * 5))
    
    # Add anomalies at specific periods
    anomaly_periods = [(200, 250), (500, 530), (750, 780)]
    ground_truth = np.zeros(time_steps)
    
    for start, end in anomaly_periods:
        # Inject anomalies
        normal_data[start:end, :5] += np.random.randn(end-start, 5) * 2.0  # Higher anomaly in first 5 sensors
        ground_truth[start:end] = 1
    
    # Generate predictions (ground truth + some noise for demo)
    predictions = normal_data + np.random.randn(time_steps, num_features) * 0.05
    
    # Feature names (same as MSL)
    feature_names = ['M-6', 'M-1', 'M-2', 'S-2', 'P-10', 'T-4', 'T-5', 'F-7', 'M-3', 'M-4',
                     'M-5', 'P-15', 'C-1', 'C-2', 'T-12', 'T-13', 'F-4', 'F-5', 'D-14', 'T-9',
                     'P-14', 'T-8', 'P-11', 'D-15', 'D-16', 'M-7', 'F-8']
    
    return normal_data, predictions, ground_truth, feature_names

def demo_logger():
    """Demonstrate the enhanced logging capabilities"""
    print("\n" + "="*60)
    print("DEMO: Enhanced Logging Capabilities")
    print("="*60)
    
    # Initialize logger
    logger = GDNLogger(experiment_name="demo_enhanced_logging")
    
    # Simulate training progress
    print("Simulating training progress...")
    epochs = 10
    for epoch in range(epochs):
        logger.log_epoch_start(epoch + 1, epochs, learning_rate=0.001 * (0.9 ** epoch))
        
        # Simulate training
        train_loss = 1.0 * np.exp(-epoch * 0.1) + np.random.normal(0, 0.05)
        val_loss = 0.8 * np.exp(-epoch * 0.12) + np.random.normal(0, 0.03)
        epoch_time = 15.0 + np.random.normal(0, 2)
        
        logger.log_epoch_end(epoch + 1, train_loss, val_loss, epoch_time, 0.001 * (0.9 ** epoch))
    
    # Log final performance metrics
    logger.log_performance_metrics({
        'f1_score': 0.85,
        'precision': 0.87,
        'recall': 0.83,
        'auc_score': 0.92
    }, 'test')
    
    # Save final results
    logger.save_final_results()
    
    print("Logging demo completed. Check logs/ directory for output files.")

def demo_visualizer():
    """Demonstrate the visualization capabilities"""
    print("\n" + "="*60)
    print("DEMO: Visualization Capabilities")
    print("="*60)
    
    # Initialize visualizer
    visualizer = GDNVisualizer(experiment_name="demo_visualizations")
    
    # Create demo training history
    epochs = list(range(1, 21))
    train_losses = [1.0 * np.exp(-i * 0.1) + np.random.normal(0, 0.05) for i in epochs]
    val_losses = [0.8 * np.exp(-i * 0.12) + np.random.normal(0, 0.03) for i in epochs]
    learning_rates = [0.001 * (0.95 ** i) for i in epochs]
    train_times = [15.0 + np.random.normal(0, 2) for _ in epochs]
    
    training_history = {
        'epochs': epochs,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'learning_rates': learning_rates,
        'train_times': train_times
    }
    
    print("Creating training history visualization...")
    visualizer.plot_training_history(training_history, save=True)
    
    # Create demo anomaly data
    normal_data, predictions, ground_truth, feature_names = create_demo_data()
    
    # Create anomaly scores
    analyzer = AnomalyAnalyzer(feature_names=feature_names)
    anomaly_scores = analyzer.calculate_anomaly_scores(predictions, normal_data)
    
    print("Creating anomaly analysis visualizations...")
    
    # Time series plot
    time_scores = np.mean(anomaly_scores, axis=1)
    threshold = analyzer.determine_threshold(anomaly_scores)
    visualizer.plot_anomaly_scores_timeseries(time_scores, ground_truth=ground_truth, 
                                            threshold=threshold, save=True)
    
    # Sensor heatmap
    visualizer.plot_sensor_anomaly_heatmap(anomaly_scores, feature_names, save=True)
    
    # Top sensors
    sensor_rankings = analyzer.rank_sensors_by_anomaly(anomaly_scores)
    visualizer.plot_top_anomalous_sensors(
        {name: info['score'] for name, info in sensor_rankings.items()}, save=True
    )
    
    # ROC curve and confusion matrix
    predictions_binary = (time_scores > threshold).astype(int)
    visualizer.plot_confusion_matrix(ground_truth, predictions_binary, save=True)
    visualizer.plot_roc_curve(ground_truth, time_scores, save=True)
    
    # Sensor time series
    visualizer.plot_sensor_timeseries(normal_data, feature_names, 
                                    anomaly_labels=ground_truth, save=True)
    
    print("Visualization demo completed. Check visualizations/ directory for plots.")

def demo_anomaly_analyzer():
    """Demonstrate the anomaly analysis capabilities"""
    print("\n" + "="*60)
    print("DEMO: Anomaly Analysis Capabilities")
    print("="*60)
    
    # Create demo data
    normal_data, predictions, ground_truth, feature_names = create_demo_data()
    
    # Initialize analyzer
    analyzer = AnomalyAnalyzer(feature_names=feature_names)
    
    print("Calculating anomaly scores...")
    anomaly_scores = analyzer.calculate_anomaly_scores(predictions, normal_data)
    
    print("Detecting anomaly periods...")
    anomaly_periods = analyzer.detect_anomaly_periods(anomaly_scores, ground_truth)
    
    print("Ranking sensors by anomaly contribution...")
    sensor_rankings = analyzer.rank_sensors_by_anomaly(anomaly_scores, top_k=10)
    
    print("Generating comprehensive anomaly report...")
    report = analyzer.generate_anomaly_report(anomaly_scores, ground_truth, 
                                            anomaly_periods, sensor_rankings)
    
    # Display results
    print(f"\nAnomaly Analysis Results:")
    print(f"  Detected {len(anomaly_periods)} anomaly periods")
    print(f"  Average anomaly score: {report['summary_statistics']['avg_anomaly_score']:.6f}")
    print(f"  Anomaly threshold: {report['summary_statistics']['anomaly_threshold']:.6f}")
    
    if report['performance_metrics']:
        metrics = report['performance_metrics']
        print(f"  F1 Score: {metrics.get('f1_score', 0):.4f}")
        print(f"  Precision: {metrics.get('precision', 0):.4f}")
        print(f"  Recall: {metrics.get('recall', 0):.4f}")
        print(f"  AUC Score: {metrics.get('auc_score', 0):.4f}")
    
    print(f"\nTop 5 Most Anomalous Sensors:")
    for i, (sensor_name, info) in enumerate(list(sensor_rankings.items())[:5], 1):
        print(f"  {i}. {sensor_name}: Score {info['score']:.6f} (Rank {info['rank']})")
    
    print(f"\nDetected Anomaly Periods:")
    for i, period in enumerate(anomaly_periods, 1):
        print(f"  {i}. Steps {period['start']}-{period['end']} "
              f"(Length: {period['length']}, Max Score: {period['max_score']:.6f})")
        if 'true_anomaly' in period:
            print(f"      True Anomaly: {period['true_anomaly']}, "
                  f"Overlap: {period['overlap_ratio']:.2%}")
    
    # Export detailed results
    print("Exporting detailed analysis results...")
    saved_files = analyzer.export_detailed_results(anomaly_scores, ground_truth, 
                                                  save_path="demo_analysis_results")
    
    print(f"Results exported to: {list(saved_files.values())}")
    print("Anomaly analysis demo completed.")

def main():
    """Main demo function"""
    print("GDN Enhanced Anomaly Detection Features Demo")
    print("This demo showcases the new capabilities added to GDN")
    print("="*60)
    
    # Create directories
    for directory in ['logs', 'visualizations', 'results']:
        Path(directory).mkdir(exist_ok=True)
    
    try:
        # Run demos
        demo_logger()
        demo_visualizer()
        demo_anomaly_analyzer()
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("All enhanced features have been demonstrated.")
        print("Check the following directories for outputs:")
        print("  - logs/: Training and testing logs")
        print("  - visualizations/: Generated plots and charts")
        print("  - results/: Detailed analysis results and exported data")
        print("\nYou can now run the enhanced GDN on real data using:")
        print("  bash run_enhanced.sh cpu msl")
        print("="*60)
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
        print("\nNote: Some dependencies may be missing. Install them with:")
        print("  pip install torch pandas numpy scikit-learn matplotlib seaborn")

if __name__ == "__main__":
    main()