# -*- coding: utf-8 -*-
"""
Anomaly Localizer for Graph Deviation Network (GDN)
Provides sensor-level anomaly analysis and localization capabilities
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Union
from scipy.stats import rankdata


class AnomalyLocalizer:
    """
    Provides sensor-level anomaly analysis and localization for GDN model.
    
    Features:
    - Sensor-level anomaly score computation
    - Top-K anomalous sensor identification
    - Temporal anomaly pattern analysis
    - Attention weight analysis for anomaly localization
    """
    
    def __init__(self, feature_map: Dict, verbose: bool = False):
        """
        Initialize the AnomalyLocalizer.
        
        Args:
            feature_map: Dictionary mapping sensor indices to sensor names
            verbose: Whether to print detailed information
        """
        self.feature_map = feature_map
        self.sensor_names = list(feature_map.values())
        self.num_sensors = len(feature_map)
        self.verbose = verbose
        
        # Storage for analysis results
        self.sensor_anomaly_scores = []
        self.temporal_patterns = {}
        self.attention_weights = []
        self.anomaly_events = []
        
    def compute_sensor_anomaly_scores(self, 
                                    test_result: List, 
                                    val_result: List) -> np.ndarray:
        """
        Compute anomaly scores for each sensor at each time step.
        
        Args:
            test_result: Test results from GDN model [predictions, ground_truth, labels]
            val_result: Validation results for normalization
            
        Returns:
            Array of shape (num_sensors, num_timesteps) with anomaly scores
        """
        np_test_result = np.array(test_result)
        np_val_result = np.array(val_result)
        
        num_timesteps = np_test_result.shape[1]
        sensor_scores = np.zeros((self.num_sensors, num_timesteps))
        
        if self.verbose:
            print(f"Computing sensor-level anomaly scores for {self.num_sensors} sensors across {num_timesteps} timesteps...")
        
        for sensor_idx in range(self.num_sensors):
            # Extract predictions and ground truth for this sensor
            test_pred = np_test_result[0, :, sensor_idx]
            test_gt = np_test_result[1, :, sensor_idx]
            val_pred = np_val_result[0, :, sensor_idx]
            val_gt = np_val_result[1, :, sensor_idx]
            
            # Compute reconstruction error
            test_error = np.abs(test_pred - test_gt)
            val_error = np.abs(val_pred - val_gt)
            
            # Normalize using validation error statistics
            val_median = np.median(val_error)
            val_iqr = np.percentile(val_error, 75) - np.percentile(val_error, 25)
            
            # Compute anomaly scores (standardized reconstruction error)
            epsilon = 1e-6
            sensor_scores[sensor_idx, :] = (test_error - val_median) / (val_iqr + epsilon)
            
            # Apply smoothing
            sensor_scores[sensor_idx, :] = self._smooth_scores(sensor_scores[sensor_idx, :])
        
        self.sensor_anomaly_scores = sensor_scores
        
        if self.verbose:
            print(f"Sensor anomaly scores computed. Shape: {sensor_scores.shape}")
            print(f"Score range: [{np.min(sensor_scores):.3f}, {np.max(sensor_scores):.3f}]")
        
        return sensor_scores
    
    def identify_top_k_anomalous_sensors(self, 
                                       scores: np.ndarray, 
                                       labels: List[int], 
                                       k: int = 5) -> Dict:
        """
        Identify top-K anomalous sensors during detected anomaly periods.
        
        Args:
            scores: Sensor anomaly scores array (num_sensors, num_timesteps)
            labels: Ground truth labels (0=normal, 1=anomaly)
            k: Number of top sensors to identify
            
        Returns:
            Dictionary with anomaly events and their top-K sensors
        """
        anomaly_events = []
        current_event = None
        
        for t, label in enumerate(labels):
            if label == 1 and (current_event is None or current_event['end'] < t - 1):
                # Start of new anomaly event
                current_event = {'start': t, 'end': t}
                anomaly_events.append(current_event)
            elif label == 1 and current_event is not None:
                # Continue current anomaly event
                current_event['end'] = t
        
        # Analyze each anomaly event
        event_analysis = []
        for i, event in enumerate(anomaly_events):
            start, end = event['start'], event['end']
            event_scores = scores[:, start:end+1]
            
            # Compute mean scores for this event
            mean_scores = np.mean(event_scores, axis=1)
            
            # Get top-K sensors
            top_k_indices = np.argsort(mean_scores)[-k:][::-1]
            top_k_sensors = [(self.sensor_names[idx], mean_scores[idx]) for idx in top_k_indices]
            
            event_info = {
                'event_id': i,
                'start_time': start,
                'end_time': end,
                'duration': end - start + 1,
                'top_k_sensors': top_k_sensors,
                'max_score': np.max(mean_scores),
                'affected_sensors': np.sum(mean_scores > np.percentile(mean_scores, 80))
            }
            event_analysis.append(event_info)
        
        self.anomaly_events = event_analysis
        
        if self.verbose:
            print(f"Identified {len(anomaly_events)} anomaly events:")
            for event in event_analysis:
                print(f"  Event {event['event_id']}: Time {event['start_time']}-{event['end_time']} "
                      f"(duration: {event['duration']}, max_score: {event['max_score']:.3f})")
                print(f"    Top sensors: {[name for name, _ in event['top_k_sensors'][:3]]}")
        
        return {
            'events': event_analysis,
            'total_events': len(anomaly_events),
            'total_anomaly_timesteps': sum(labels)
        }
    
    def analyze_temporal_patterns(self, scores: np.ndarray, window_size: int = 10) -> Dict:
        """
        Analyze temporal patterns in anomaly scores for each sensor.
        
        Args:
            scores: Sensor anomaly scores array
            window_size: Size of sliding window for pattern analysis
            
        Returns:
            Dictionary with temporal pattern analysis
        """
        patterns = {}
        
        for sensor_idx, sensor_name in enumerate(self.sensor_names):
            sensor_scores = scores[sensor_idx, :]
            
            # Compute rolling statistics
            rolling_mean = np.convolve(sensor_scores, np.ones(window_size)/window_size, mode='valid')
            rolling_std = np.array([np.std(sensor_scores[i:i+window_size]) 
                                  for i in range(len(sensor_scores) - window_size + 1)])
            
            # Detect anomaly periods for this sensor
            threshold = np.percentile(sensor_scores, 95)
            anomaly_periods = self._find_anomaly_periods(sensor_scores, threshold)
            
            patterns[sensor_name] = {
                'mean_score': np.mean(sensor_scores),
                'std_score': np.std(sensor_scores),
                'max_score': np.max(sensor_scores),
                'anomaly_periods': anomaly_periods,
                'volatility': np.mean(rolling_std),
                'trend': np.corrcoef(np.arange(len(sensor_scores)), sensor_scores)[0, 1]
            }
        
        self.temporal_patterns = patterns
        
        if self.verbose:
            print(f"Temporal pattern analysis completed for {len(patterns)} sensors")
            high_volatility_sensors = [name for name, data in patterns.items() 
                                     if data['volatility'] > np.percentile([d['volatility'] for d in patterns.values()], 80)]
            print(f"High volatility sensors: {high_volatility_sensors}")
        
        return patterns
    
    def extract_attention_insights(self, model, dataloader, device, max_batches: int = 10) -> Dict:
        """
        Extract attention weights from GDN model for anomaly interpretation.
        
        Args:
            model: Trained GDN model
            dataloader: Data loader for processing
            device: Computing device
            max_batches: Maximum number of batches to process
            
        Returns:
            Dictionary with attention weight analysis
        """
        model.eval()
        attention_data = []
        
        with torch.no_grad():
            for batch_idx, (x, y, labels, edge_index) in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break
                
                x, y, edge_index = [item.to(device).float() for item in [x, y, edge_index]]
                
                # Forward pass to get attention weights
                predictions = model(x, edge_index)
                
                # Extract attention weights from GNN layers
                for layer_idx, gnn_layer in enumerate(model.gnn_layers):
                    if hasattr(gnn_layer, 'att_weight_1'):
                        att_weights = gnn_layer.att_weight_1.cpu().numpy()
                        attention_data.append({
                            'batch': batch_idx,
                            'layer': layer_idx,
                            'weights': att_weights,
                            'labels': labels.cpu().numpy()
                        })
        
        # Analyze attention patterns
        analysis = self._analyze_attention_patterns(attention_data)
        self.attention_weights = attention_data
        
        if self.verbose:
            print(f"Extracted attention weights from {len(attention_data)} batches")
            print(f"Average attention entropy: {analysis.get('avg_entropy', 0):.3f}")
        
        return analysis
    
    def get_sensor_importance_ranking(self, scores: np.ndarray) -> List[Tuple[str, float]]:
        """
        Rank sensors by their overall importance in anomaly detection.
        
        Args:
            scores: Sensor anomaly scores array
            
        Returns:
            List of (sensor_name, importance_score) tuples, sorted by importance
        """
        # Compute various importance metrics
        importance_scores = []
        
        for sensor_idx, sensor_name in enumerate(self.sensor_names):
            sensor_scores = scores[sensor_idx, :]
            
            # Combine multiple importance factors
            variance = np.var(sensor_scores)
            max_score = np.max(sensor_scores)
            frequency_high = np.sum(sensor_scores > np.percentile(sensor_scores, 90)) / len(sensor_scores)
            
            # Weighted importance score
            importance = 0.4 * variance + 0.4 * max_score + 0.2 * frequency_high
            importance_scores.append((sensor_name, importance))
        
        # Sort by importance
        importance_scores.sort(key=lambda x: x[1], reverse=True)
        
        if self.verbose:
            print("Sensor Importance Ranking:")
            for i, (name, score) in enumerate(importance_scores[:10]):
                print(f"  {i+1:2d}. {name}: {score:.3f}")
        
        return importance_scores
    
    def generate_summary_report(self, scores: np.ndarray, labels: List[int], k: int = 5) -> str:
        """
        Generate a comprehensive summary report of anomaly localization analysis.
        
        Args:
            scores: Sensor anomaly scores array
            labels: Ground truth labels
            k: Number of top sensors to report
            
        Returns:
            Formatted summary report as string
        """
        # Perform all analyses
        top_k_analysis = self.identify_top_k_anomalous_sensors(scores, labels, k)
        temporal_analysis = self.analyze_temporal_patterns(scores)
        importance_ranking = self.get_sensor_importance_ranking(scores)
        
        # Generate report
        report = []
        report.append("=" * 60)
        report.append("GDN ANOMALY LOCALIZATION SUMMARY REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Dataset summary
        report.append(f"Dataset Overview:")
        report.append(f"  • Number of sensors: {self.num_sensors}")
        report.append(f"  • Total timesteps: {scores.shape[1]}")
        report.append(f"  • Anomaly events detected: {top_k_analysis['total_events']}")
        report.append(f"  • Total anomalous timesteps: {top_k_analysis['total_anomaly_timesteps']}")
        report.append("")
        
        # Top anomalous sensors
        report.append(f"Top {k} Most Important Sensors:")
        for i, (name, score) in enumerate(importance_ranking[:k]):
            report.append(f"  {i+1:2d}. {name}: {score:.3f}")
        report.append("")
        
        # Anomaly events summary
        if top_k_analysis['events']:
            report.append("Anomaly Events Summary:")
            for event in top_k_analysis['events'][:5]:  # Show first 5 events
                report.append(f"  Event {event['event_id']}: "
                            f"Time {event['start_time']}-{event['end_time']} "
                            f"(Duration: {event['duration']} steps)")
                report.append(f"    Top sensors: {', '.join([name for name, _ in event['top_k_sensors'][:3]])}")
                report.append(f"    Max score: {event['max_score']:.3f}")
            report.append("")
        
        # Temporal patterns
        high_volatility_sensors = sorted([(name, data['volatility']) for name, data in temporal_analysis.items()], 
                                       key=lambda x: x[1], reverse=True)[:3]
        report.append("High Volatility Sensors (Top 3):")
        for name, volatility in high_volatility_sensors:
            report.append(f"  • {name}: {volatility:.3f}")
        report.append("")
        
        # Statistics
        all_scores = scores.flatten()
        report.append("Score Statistics:")
        report.append(f"  • Mean: {np.mean(all_scores):.3f}")
        report.append(f"  • Std: {np.std(all_scores):.3f}")
        report.append(f"  • Min: {np.min(all_scores):.3f}")
        report.append(f"  • Max: {np.max(all_scores):.3f}")
        report.append(f"  • 95th percentile: {np.percentile(all_scores, 95):.3f}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def save_detailed_results(self, filepath: str, scores: np.ndarray, labels: List[int]) -> None:
        """
        Save detailed anomaly localization results to CSV file.
        
        Args:
            filepath: Path to save the CSV file
            scores: Sensor anomaly scores array
            labels: Ground truth labels
        """
        # Prepare data for CSV
        data = {}
        data['timestep'] = list(range(scores.shape[1]))
        data['ground_truth_label'] = labels
        
        # Add sensor scores
        for sensor_idx, sensor_name in enumerate(self.sensor_names):
            data[f'anomaly_score_{sensor_name}'] = scores[sensor_idx, :].tolist()
        
        # Add aggregate scores
        data['max_sensor_score'] = np.max(scores, axis=0).tolist()
        data['mean_sensor_score'] = np.mean(scores, axis=0).tolist()
        data['num_anomalous_sensors'] = np.sum(scores > np.percentile(scores, 90), axis=0).tolist()
        
        # Create DataFrame and save
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        
        if self.verbose:
            print(f"Detailed results saved to: {filepath}")
            print(f"CSV contains {len(df)} rows and {len(df.columns)} columns")
    
    def _smooth_scores(self, scores: np.ndarray, window_size: int = 3) -> np.ndarray:
        """Apply smoothing to anomaly scores to reduce noise."""
        if window_size <= 1:
            return scores
        
        smoothed = np.convolve(scores, np.ones(window_size)/window_size, mode='same')
        
        # Handle edges
        for i in range(min(window_size//2, len(scores))):
            smoothed[i] = np.mean(scores[:i+window_size//2+1])
            smoothed[-(i+1)] = np.mean(scores[-(i+window_size//2+1):])
        
        return smoothed
    
    def _find_anomaly_periods(self, scores: np.ndarray, threshold: float) -> List[Tuple[int, int]]:
        """Find continuous periods where scores exceed threshold."""
        above_threshold = scores > threshold
        periods = []
        start = None
        
        for i, is_above in enumerate(above_threshold):
            if is_above and start is None:
                start = i
            elif not is_above and start is not None:
                periods.append((start, i-1))
                start = None
        
        # Handle case where anomaly period extends to end
        if start is not None:
            periods.append((start, len(scores)-1))
        
        return periods
    
    def _analyze_attention_patterns(self, attention_data: List[Dict]) -> Dict:
        """Analyze patterns in attention weights."""
        if not attention_data:
            return {}
        
        # Compute attention statistics
        all_weights = []
        entropy_values = []
        
        for data in attention_data:
            weights = data['weights']
            all_weights.extend(weights.flatten())
            
            # Compute entropy for each attention distribution
            for w in weights:
                if len(w) > 0:
                    # Normalize to get probabilities
                    probs = np.abs(w) / (np.sum(np.abs(w)) + 1e-8)
                    entropy = -np.sum(probs * np.log(probs + 1e-8))
                    entropy_values.append(entropy)
        
        return {
            'avg_attention': np.mean(all_weights),
            'std_attention': np.std(all_weights),
            'avg_entropy': np.mean(entropy_values),
            'num_samples': len(attention_data)
        }