# -*- coding: utf-8 -*-
"""
Anomaly analysis utilities for GDN
Provides detailed anomaly localization, sensor ranking, and analysis capabilities
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime
import json

class AnomalyAnalyzer:
    """Comprehensive anomaly analysis toolkit for GDN"""
    
    def __init__(self, feature_names=None, threshold_method='percentile', threshold_value=95):
        """
        Initialize anomaly analyzer
        
        Args:
            feature_names (list): Names of sensors/features
            threshold_method (str): Method to determine anomaly threshold ('percentile', 'std', 'fixed')
            threshold_value (float): Value for threshold calculation
        """
        self.feature_names = feature_names or []
        self.threshold_method = threshold_method
        self.threshold_value = threshold_value
        self.anomaly_threshold = None
        
    def calculate_anomaly_scores(self, predictions, ground_truth, method='mse'):
        """
        Calculate anomaly scores from predictions and ground truth
        
        Args:
            predictions (array): Model predictions
            ground_truth (array): Ground truth values
            method (str): Method to calculate scores ('mse', 'mae', 'combined')
            
        Returns:
            array: Anomaly scores for each time step and sensor
        """
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        
        if method == 'mse':
            scores = np.square(predictions - ground_truth)
        elif method == 'mae':
            scores = np.abs(predictions - ground_truth)
        elif method == 'combined':
            mse_scores = np.square(predictions - ground_truth)
            mae_scores = np.abs(predictions - ground_truth)
            scores = 0.7 * mse_scores + 0.3 * mae_scores
        else:
            raise ValueError(f"Unknown scoring method: {method}")
        
        return scores
    
    def determine_threshold(self, scores):
        """
        Determine anomaly threshold based on scores
        
        Args:
            scores (array): Anomaly scores
            
        Returns:
            float: Anomaly threshold
        """
        scores_flat = scores.flatten()
        
        if self.threshold_method == 'percentile':
            threshold = np.percentile(scores_flat, self.threshold_value)
        elif self.threshold_method == 'std':
            mean_score = np.mean(scores_flat)
            std_score = np.std(scores_flat)
            threshold = mean_score + self.threshold_value * std_score
        elif self.threshold_method == 'fixed':
            threshold = self.threshold_value
        else:
            # Default to 95th percentile
            threshold = np.percentile(scores_flat, 95)
        
        self.anomaly_threshold = threshold
        return threshold
    
    def detect_anomaly_periods(self, scores, labels=None, min_period_length=3):
        """
        Detect anomalous time periods
        
        Args:
            scores (array): Anomaly scores (time_steps, features) or (time_steps,)
            labels (array): Ground truth labels (optional)
            min_period_length (int): Minimum length of an anomaly period
            
        Returns:
            list: List of anomaly periods with details
        """
        # Calculate aggregate scores per time step
        if len(scores.shape) > 1:
            time_scores = np.mean(scores, axis=1)
        else:
            time_scores = scores
        
        # Determine threshold if not set
        if self.anomaly_threshold is None:
            threshold = self.determine_threshold(scores)
        else:
            threshold = self.anomaly_threshold
        
        # Find anomalous time steps
        anomaly_mask = time_scores > threshold
        
        # Group consecutive anomalous time steps into periods
        periods = []
        start_idx = None
        
        for i, is_anomaly in enumerate(anomaly_mask):
            if is_anomaly and start_idx is None:
                start_idx = i
            elif not is_anomaly and start_idx is not None:
                # End of anomaly period
                period_length = i - start_idx
                if period_length >= min_period_length:
                    period = {
                        'start': start_idx,
                        'end': i - 1,
                        'length': period_length,
                        'max_score': float(np.max(time_scores[start_idx:i])),
                        'avg_score': float(np.mean(time_scores[start_idx:i])),
                        'score_std': float(np.std(time_scores[start_idx:i]))
                    }
                    
                    # Add ground truth information if available
                    if labels is not None:
                        period['true_anomaly'] = bool(np.any(labels[start_idx:i]))
                        period['overlap_ratio'] = float(np.mean(labels[start_idx:i]))
                    
                    periods.append(period)
                
                start_idx = None
        
        # Handle case where anomaly extends to end
        if start_idx is not None:
            period_length = len(anomaly_mask) - start_idx
            if period_length >= min_period_length:
                period = {
                    'start': start_idx,
                    'end': len(anomaly_mask) - 1,
                    'length': period_length,
                    'max_score': float(np.max(time_scores[start_idx:])),
                    'avg_score': float(np.mean(time_scores[start_idx:])),
                    'score_std': float(np.std(time_scores[start_idx:]))
                }
                
                if labels is not None:
                    period['true_anomaly'] = bool(np.any(labels[start_idx:]))
                    period['overlap_ratio'] = float(np.mean(labels[start_idx:]))
                
                periods.append(period)
        
        return periods
    
    def rank_sensors_by_anomaly(self, scores, top_k=None):
        """
        Rank sensors by their anomaly contribution
        
        Args:
            scores (array): Anomaly scores (time_steps, features)
            top_k (int): Number of top sensors to return (None for all)
            
        Returns:
            dict: Sensor rankings with scores
        """
        if len(scores.shape) == 1:
            scores = scores.reshape(-1, 1)
        
        # Calculate aggregate anomaly score per sensor
        sensor_scores = np.mean(scores, axis=0)
        
        # Create rankings
        rankings = {}
        feature_names = self.feature_names or [f'Sensor_{i}' for i in range(len(sensor_scores))]
        
        for i, score in enumerate(sensor_scores):
            sensor_name = feature_names[i] if i < len(feature_names) else f'Sensor_{i}'
            rankings[sensor_name] = {
                'score': float(score),
                'rank': 0,  # Will be filled below
                'percentile': float(np.percentile(scores[:, i], 95)),
                'std': float(np.std(scores[:, i])),
                'max': float(np.max(scores[:, i]))
            }
        
        # Sort by score and assign ranks
        sorted_sensors = sorted(rankings.items(), key=lambda x: x[1]['score'], reverse=True)
        for rank, (sensor_name, info) in enumerate(sorted_sensors, 1):
            rankings[sensor_name]['rank'] = rank
        
        # Return top-k if specified
        if top_k is not None:
            top_sensors = dict(sorted_sensors[:top_k])
            return top_sensors
        
        return rankings
    
    def analyze_sensor_correlations(self, scores, method='pearson'):
        """
        Analyze correlations between sensor anomaly scores
        
        Args:
            scores (array): Anomaly scores (time_steps, features)
            method (str): Correlation method ('pearson', 'spearman')
            
        Returns:
            array: Correlation matrix
        """
        if len(scores.shape) == 1:
            return np.array([[1.0]])
        
        if method == 'pearson':
            correlation_matrix = np.corrcoef(scores.T)
        elif method == 'spearman':
            from scipy.stats import spearmanr
            correlation_matrix, _ = spearmanr(scores, axis=0)
        else:
            raise ValueError(f"Unknown correlation method: {method}")
        
        return correlation_matrix
    
    def find_anomaly_patterns(self, scores, periods, pattern_length=5):
        """
        Find common patterns in anomalous periods
        
        Args:
            scores (array): Anomaly scores
            periods (list): List of anomaly periods
            pattern_length (int): Length of patterns to analyze
            
        Returns:
            dict: Analysis of anomaly patterns
        """
        patterns = []
        
        for period in periods:
            start, end = period['start'], period['end']
            
            if end - start + 1 >= pattern_length:
                # Extract pattern around the peak
                period_scores = scores[start:end+1] if len(scores.shape) == 1 else np.mean(scores[start:end+1], axis=1)
                peak_idx = np.argmax(period_scores)
                
                # Extract pattern around peak
                pattern_start = max(0, start + peak_idx - pattern_length // 2)
                pattern_end = min(len(scores), pattern_start + pattern_length)
                
                if len(scores.shape) == 1:
                    pattern = scores[pattern_start:pattern_end]
                else:
                    pattern = np.mean(scores[pattern_start:pattern_end], axis=1)
                
                patterns.append({
                    'pattern': pattern.tolist(),
                    'period_info': period,
                    'pattern_start': pattern_start,
                    'pattern_end': pattern_end
                })
        
        # Analyze pattern similarities
        pattern_analysis = {
            'num_patterns': len(patterns),
            'avg_pattern_length': np.mean([len(p['pattern']) for p in patterns]) if patterns else 0,
            'patterns': patterns
        }
        
        return pattern_analysis
    
    def generate_anomaly_report(self, scores, labels=None, periods=None, sensor_rankings=None):
        """
        Generate comprehensive anomaly analysis report
        
        Args:
            scores (array): Anomaly scores
            labels (array): Ground truth labels
            periods (list): Anomaly periods
            sensor_rankings (dict): Sensor rankings
            
        Returns:
            dict: Comprehensive analysis report
        """
        if periods is None:
            periods = self.detect_anomaly_periods(scores, labels)
        
        if sensor_rankings is None:
            sensor_rankings = self.rank_sensors_by_anomaly(scores)
        
        # Calculate summary statistics
        scores_flat = scores.flatten()
        
        summary_stats = {
            'total_time_steps': len(scores) if len(scores.shape) == 1 else scores.shape[0],
            'num_sensors': 1 if len(scores.shape) == 1 else scores.shape[1],
            'anomaly_threshold': float(self.anomaly_threshold) if self.anomaly_threshold else None,
            'total_anomaly_periods': len(periods),
            'avg_anomaly_score': float(np.mean(scores_flat)),
            'max_anomaly_score': float(np.max(scores_flat)),
            'score_std': float(np.std(scores_flat)),
            'score_percentiles': {
                '50': float(np.percentile(scores_flat, 50)),
                '90': float(np.percentile(scores_flat, 90)),
                '95': float(np.percentile(scores_flat, 95)),
                '99': float(np.percentile(scores_flat, 99))
            }
        }
        
        # Performance metrics if ground truth available
        performance_metrics = {}
        if labels is not None:
            # Calculate aggregate predictions
            if len(scores.shape) > 1:
                time_scores = np.mean(scores, axis=1)
            else:
                time_scores = scores
            
            threshold = self.anomaly_threshold or self.determine_threshold(scores)
            predictions = (time_scores > threshold).astype(int)
            
            from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
            
            try:
                performance_metrics = {
                    'precision': float(precision_score(labels, predictions, zero_division=0)),
                    'recall': float(recall_score(labels, predictions, zero_division=0)),
                    'f1_score': float(f1_score(labels, predictions, zero_division=0)),
                    'auc_score': float(roc_auc_score(labels, time_scores)) if len(np.unique(labels)) > 1 else 0.0
                }
            except Exception as e:
                performance_metrics = {'error': str(e)}
        
        # Compile comprehensive report
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary_statistics': summary_stats,
            'performance_metrics': performance_metrics,
            'anomaly_periods': periods,
            'sensor_rankings': sensor_rankings,
            'top_5_sensors': dict(list(sensor_rankings.items())[:5]) if sensor_rankings else {},
            'analysis_config': {
                'threshold_method': self.threshold_method,
                'threshold_value': self.threshold_value,
                'feature_names': self.feature_names
            }
        }
        
        return report
    
    def export_detailed_results(self, scores, labels=None, timestamps=None, save_path=None):
        """
        Export detailed anomaly analysis results to files
        
        Args:
            scores (array): Anomaly scores
            labels (array): Ground truth labels
            timestamps (array): Timestamps
            save_path (str): Base path for saving files
            
        Returns:
            dict: Paths of saved files
        """
        if save_path is None:
            save_path = f"anomaly_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Generate comprehensive analysis
        periods = self.detect_anomaly_periods(scores, labels)
        sensor_rankings = self.rank_sensors_by_anomaly(scores)
        report = self.generate_anomaly_report(scores, labels, periods, sensor_rankings)
        
        saved_files = {}
        
        # Save main report
        report_file = f"{save_path}_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        saved_files['report'] = report_file
        
        # Save detailed scores if multidimensional
        if len(scores.shape) > 1:
            scores_df = pd.DataFrame(scores)
            if self.feature_names:
                scores_df.columns = self.feature_names[:scores.shape[1]]
            if timestamps is not None:
                scores_df.index = timestamps[:len(scores)]
            
            scores_file = f"{save_path}_scores.csv"
            scores_df.to_csv(scores_file)
            saved_files['scores'] = scores_file
        
        # Save anomaly periods details
        if periods:
            periods_df = pd.DataFrame(periods)
            periods_file = f"{save_path}_periods.csv"
            periods_df.to_csv(periods_file, index=False)
            saved_files['periods'] = periods_file
        
        # Save sensor rankings
        if sensor_rankings:
            rankings_df = pd.DataFrame.from_dict(sensor_rankings, orient='index')
            rankings_file = f"{save_path}_sensor_rankings.csv"
            rankings_df.to_csv(rankings_file)
            saved_files['rankings'] = rankings_file
        
        return saved_files
    
    def compare_anomaly_methods(self, predictions, ground_truth, methods=['mse', 'mae', 'combined']):
        """
        Compare different anomaly scoring methods
        
        Args:
            predictions (array): Model predictions
            ground_truth (array): Ground truth values
            methods (list): List of scoring methods to compare
            
        Returns:
            dict: Comparison results
        """
        comparison = {}
        
        for method in methods:
            scores = self.calculate_anomaly_scores(predictions, ground_truth, method)
            periods = self.detect_anomaly_periods(scores)
            
            comparison[method] = {
                'num_periods': len(periods),
                'avg_score': float(np.mean(scores)),
                'max_score': float(np.max(scores)),
                'threshold': float(self.determine_threshold(scores))
            }
        
        return comparison