import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from util.time import *
from util.env import *

import argparse
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn.functional as F

from util.data import *
from util.preprocess import *
from util.logger import GDNLogger
from util.visualizer import GDNVisualizer
from util.anomaly_analyzer import AnomalyAnalyzer



def test(model, dataloader, feature_names=None, enable_detailed_analysis=True, experiment_name=None):
    """
    Enhanced test function with detailed anomaly analysis and logging
    
    Args:
        model: The GDN model
        dataloader: Test data loader
        feature_names: List of feature/sensor names
        enable_detailed_analysis: Whether to perform detailed anomaly analysis
        experiment_name: Name for logging and visualization
    """
    # test
    loss_func = nn.MSELoss(reduction='mean')
    device = get_device()

    test_loss_list = []
    now = time.time()

    test_predicted_list = []
    test_ground_list = []
    test_labels_list = []

    t_test_predicted_list = []
    t_test_ground_list = []
    t_test_labels_list = []

    test_len = len(dataloader)
    
    # Initialize enhanced logging and analysis if enabled
    logger = None
    visualizer = None
    analyzer = None
    
    if enable_detailed_analysis:
        if experiment_name is None:
            experiment_name = f"test_{time.strftime('%Y%m%d_%H%M%S')}"
        
        logger = GDNLogger(experiment_name=f"{experiment_name}_test")
        visualizer = GDNVisualizer(experiment_name=experiment_name)
        analyzer = AnomalyAnalyzer(feature_names=feature_names)
        
        print(f"Starting enhanced testing with detailed analysis...")
        print(f"Test batches: {test_len}, Features: {len(feature_names) if feature_names else 'Unknown'}")
        print("="*60)

    model.eval()

    i = 0
    acu_loss = 0
    batch_predictions = []
    batch_ground_truth = []
    batch_labels = []
    
    for batch_idx, (x, y, labels, edge_index) in enumerate(dataloader):
        batch_start_time = time.time()
        x, y, labels, edge_index = [item.to(device).float() for item in [x, y, labels, edge_index]]
        
        with torch.no_grad():
            predicted = model(x, edge_index).float().to(device)
            loss = loss_func(predicted, y)
            
            labels = labels.unsqueeze(1).repeat(1, predicted.shape[1])

            if len(t_test_predicted_list) <= 0:
                t_test_predicted_list = predicted
                t_test_ground_list = y
                t_test_labels_list = labels
            else:
                t_test_predicted_list = torch.cat((t_test_predicted_list, predicted), dim=0)
                t_test_ground_list = torch.cat((t_test_ground_list, y), dim=0)
                t_test_labels_list = torch.cat((t_test_labels_list, labels), dim=0)
        
        test_loss_list.append(loss.item())
        acu_loss += loss.item()
        
        # Store batch data for detailed analysis
        if enable_detailed_analysis:
            batch_pred = predicted.cpu().numpy()
            batch_gt = y.cpu().numpy()
            batch_lbl = labels.cpu().numpy()
            
            batch_predictions.append(batch_pred)
            batch_ground_truth.append(batch_gt)
            batch_labels.append(batch_lbl)
            
            # Calculate anomaly scores for this batch
            anomaly_scores = analyzer.calculate_anomaly_scores(batch_pred, batch_gt)
            
            # Log detailed batch information
            logger.log_test_batch_detailed(
                batch_idx, batch_pred, batch_gt, batch_lbl, 
                feature_names, anomaly_scores
            )
            
            # Log batch progress
            batch_time = time.time() - batch_start_time
            logger.log_batch_progress(batch_idx, test_len, loss.item(), 'test')
        
        i += 1

        if i % 10000 == 1 and i > 1:
            print(timeSincePlus(now, i / test_len))
        
        # Print progress every 100 batches or at significant intervals
        if batch_idx % 100 == 0 or batch_idx == test_len - 1:
            progress_pct = (batch_idx + 1) / test_len * 100
            print(f"[{time.strftime('%H:%M:%S')}] Test Batch {batch_idx + 1}/{test_len} ({progress_pct:.1f}%) - Loss: {loss.item():.6f}")

    test_predicted_list = t_test_predicted_list.tolist()        
    test_ground_list = t_test_ground_list.tolist()        
    test_labels_list = t_test_labels_list.tolist()      
    
    avg_loss = sum(test_loss_list)/len(test_loss_list)
    
    # Perform detailed anomaly analysis if enabled
    if enable_detailed_analysis and len(batch_predictions) > 0:
        print("\nPerforming detailed anomaly analysis...")
        
        try:
            # Combine all batch data
            all_predictions = np.concatenate(batch_predictions, axis=0)
            all_ground_truth = np.concatenate(batch_ground_truth, axis=0)
            all_labels = np.concatenate(batch_labels, axis=0)
            
            # Calculate comprehensive anomaly scores
            anomaly_scores = analyzer.calculate_anomaly_scores(all_predictions, all_ground_truth)
            
            # Detect anomaly periods
            true_labels = all_labels[:, 0] if len(all_labels.shape) > 1 else all_labels
            anomaly_periods = analyzer.detect_anomaly_periods(anomaly_scores, true_labels)
            
            # Rank sensors by anomaly contribution
            sensor_rankings = analyzer.rank_sensors_by_anomaly(anomaly_scores, top_k=10)
            
            # Log anomaly detection results
            logger.log_anomaly_detection(anomaly_periods, sensor_rankings)
            
            # Generate comprehensive anomaly report
            anomaly_report = analyzer.generate_anomaly_report(
                anomaly_scores, true_labels, anomaly_periods, sensor_rankings
            )
            
            print(f"\nAnomaly Analysis Summary:")
            print(f"  Total anomaly periods detected: {len(anomaly_periods)}")
            print(f"  Average anomaly score: {anomaly_report['summary_statistics']['avg_anomaly_score']:.6f}")
            print(f"  Max anomaly score: {anomaly_report['summary_statistics']['max_anomaly_score']:.6f}")
            print(f"  Anomaly threshold: {anomaly_report['summary_statistics']['anomaly_threshold']:.6f}")
            
            if anomaly_report.get('performance_metrics'):
                metrics = anomaly_report['performance_metrics']
                print(f"  Performance - F1: {metrics.get('f1_score', 0):.4f}, "
                      f"Precision: {metrics.get('precision', 0):.4f}, "
                      f"Recall: {metrics.get('recall', 0):.4f}")
            
            # Display top anomalous sensors
            print(f"\nTop 5 Most Anomalous Sensors:")
            for i, (sensor_name, info) in enumerate(list(sensor_rankings.items())[:5], 1):
                print(f"  {i}. {sensor_name}: {info['score']:.6f}")
            
            # Display anomaly periods details
            if anomaly_periods:
                print(f"\nDetected Anomaly Periods:")
                for i, period in enumerate(anomaly_periods[:5], 1):  # Show first 5 periods
                    print(f"  {i}. Steps {period['start']}-{period['end']} "
                          f"(Length: {period['length']}, Max Score: {period['max_score']:.6f})")
                if len(anomaly_periods) > 5:
                    print(f"  ... and {len(anomaly_periods) - 5} more periods")
            
            # Create visualizations
            print("\nCreating visualizations...")
            try:
                # Anomaly scores time series
                time_scores = np.mean(anomaly_scores, axis=1) if len(anomaly_scores.shape) > 1 else anomaly_scores
                visualizer.plot_anomaly_scores_timeseries(
                    time_scores, ground_truth=true_labels, 
                    threshold=analyzer.anomaly_threshold, save=True
                )
                
                # Sensor anomaly heatmap
                if len(anomaly_scores.shape) > 1 and feature_names:
                    visualizer.plot_sensor_anomaly_heatmap(
                        anomaly_scores, feature_names, save=True
                    )
                
                # Top anomalous sensors
                visualizer.plot_top_anomalous_sensors(
                    {name: info['score'] for name, info in sensor_rankings.items()}, 
                    save=True
                )
                
                # ROC curve and confusion matrix if we have binary labels
                if len(np.unique(true_labels)) == 2:
                    threshold = analyzer.anomaly_threshold or analyzer.determine_threshold(anomaly_scores)
                    predictions = (time_scores > threshold).astype(int)
                    
                    visualizer.plot_confusion_matrix(true_labels, predictions, save=True)
                    visualizer.plot_roc_curve(true_labels, time_scores, save=True)
                
                print("Visualizations completed successfully!")
                
            except Exception as e:
                print(f"Warning: Some visualizations failed: {e}")
            
            # Save comprehensive results
            logger.save_final_results(
                test_results={'avg_loss': avg_loss, 'test_loss_list': test_loss_list},
                best_metrics=anomaly_report.get('performance_metrics', {})
            )
            
            # Export detailed analysis results
            try:
                saved_files = analyzer.export_detailed_results(
                    anomaly_scores, true_labels, 
                    save_path=f"results/{experiment_name}_detailed"
                )
                print(f"Detailed analysis exported to: {list(saved_files.values())}")
            except Exception as e:
                print(f"Warning: Could not export detailed results: {e}")
                
        except Exception as e:
            print(f"Warning: Detailed anomaly analysis failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nTesting completed. Average loss: {avg_loss:.6f}")
    print("="*60)
    
    return avg_loss, [test_predicted_list, test_ground_list, test_labels_list]




