# -*- coding: utf-8 -*-
"""
Enhanced logging module for GDN anomaly detection
Provides comprehensive logging for training progress, testing progress, and anomaly analysis
"""

import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
import numpy as np

class GDNLogger:
    """Enhanced logger for GDN training and testing processes"""
    
    def __init__(self, log_dir='./logs', experiment_name=None, config=None):
        """
        Initialize logger with specified directory and experiment name
        
        Args:
            log_dir (str): Directory to save log files
            experiment_name (str): Name of the experiment, defaults to timestamp
            config (dict): Configuration dictionary to save
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if experiment_name is None:
            experiment_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.experiment_name = experiment_name
        self.config = config or {}
        
        # Create log files
        self.train_log_file = self.log_dir / f'{experiment_name}_training.log'
        self.test_log_file = self.log_dir / f'{experiment_name}_testing.log'
        self.metrics_file = self.log_dir / f'{experiment_name}_metrics.json'
        self.anomaly_file = self.log_dir / f'{experiment_name}_anomalies.json'
        
        # Setup Python logging
        self._setup_loggers()
        
        # Initialize storage for metrics and results
        self.training_history = {
            'epochs': [],
            'train_losses': [],
            'val_losses': [],
            'train_times': [],
            'learning_rates': []
        }
        
        self.test_results = {
            'batch_progress': [],
            'anomaly_scores': [],
            'predictions': [],
            'ground_truth': [],
            'timestamps': []
        }
        
        self.anomaly_analysis = {
            'anomaly_periods': [],
            'top_sensors': [],
            'sensor_rankings': {},
            'attention_weights': []
        }
        
        # Save initial configuration
        self._save_config()
    
    def _setup_loggers(self):
        """Setup Python loggers for training and testing"""
        # Training logger
        self.train_logger = logging.getLogger(f'gdn_train_{self.experiment_name}')
        self.train_logger.setLevel(logging.INFO)
        train_handler = logging.FileHandler(self.train_log_file)
        train_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        train_handler.setFormatter(train_formatter)
        self.train_logger.addHandler(train_handler)
        
        # Testing logger
        self.test_logger = logging.getLogger(f'gdn_test_{self.experiment_name}')
        self.test_logger.setLevel(logging.INFO)
        test_handler = logging.FileHandler(self.test_log_file)
        test_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        test_handler.setFormatter(test_formatter)
        self.test_logger.addHandler(test_handler)
    
    def _save_config(self):
        """Save experiment configuration"""
        config_file = self.log_dir / f'{self.experiment_name}_config.json'
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
    
    def log_epoch_start(self, epoch, total_epochs, learning_rate=None):
        """Log the start of an epoch"""
        message = f"Epoch {epoch}/{total_epochs} started"
        if learning_rate is not None:
            message += f" - Learning Rate: {learning_rate:.6f}"
        
        self.train_logger.info(message)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
    
    def log_epoch_end(self, epoch, train_loss, val_loss=None, epoch_time=None, learning_rate=None):
        """Log the end of an epoch with metrics"""
        # Store in history
        self.training_history['epochs'].append(epoch)
        self.training_history['train_losses'].append(train_loss)
        self.training_history['val_losses'].append(val_loss)
        self.training_history['train_times'].append(epoch_time)
        self.training_history['learning_rates'].append(learning_rate)
        
        # Create log message
        message = f"Epoch {epoch} completed - Train Loss: {train_loss:.6f}"
        if val_loss is not None:
            message += f" - Val Loss: {val_loss:.6f}"
        if epoch_time is not None:
            message += f" - Time: {epoch_time:.2f}s"
        if learning_rate is not None:
            message += f" - LR: {learning_rate:.6f}"
        
        self.train_logger.info(message)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        
        # Save training history periodically
        if epoch % 5 == 0:
            self._save_training_history()
    
    def log_batch_progress(self, batch_idx, total_batches, batch_loss=None, stage='train'):
        """Log batch processing progress"""
        progress_pct = (batch_idx + 1) / total_batches * 100
        message = f"{stage.capitalize()} Batch {batch_idx + 1}/{total_batches} ({progress_pct:.1f}%)"
        
        if batch_loss is not None:
            message += f" - Loss: {batch_loss:.6f}"
        
        if stage == 'train':
            self.train_logger.debug(message)
        else:
            self.test_logger.info(message)
        
        # Print progress every 10% or every 100 batches
        if (batch_idx + 1) % max(1, total_batches // 10) == 0 or (batch_idx + 1) % 100 == 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
    
    def log_test_batch_detailed(self, batch_idx, predictions, ground_truth, labels, 
                               feature_names=None, anomaly_scores=None):
        """Log detailed test batch information for anomaly analysis"""
        batch_info = {
            'batch_idx': batch_idx,
            'timestamp': datetime.now().isoformat(),
            'num_samples': len(predictions),
            'anomaly_detected': bool(np.any(labels)) if labels is not None else False
        }
        
        if anomaly_scores is not None:
            batch_info['avg_anomaly_score'] = float(np.mean(anomaly_scores))
            batch_info['max_anomaly_score'] = float(np.max(anomaly_scores))
            
            # Find top anomalous sensors if feature names provided
            if feature_names is not None and len(anomaly_scores.shape) > 1:
                top_sensor_indices = np.argsort(np.mean(anomaly_scores, axis=0))[-5:]
                batch_info['top_anomalous_sensors'] = [
                    {
                        'sensor': feature_names[idx],
                        'score': float(np.mean(anomaly_scores[:, idx]))
                    } for idx in reversed(top_sensor_indices)
                ]
        
        self.test_results['batch_progress'].append(batch_info)
        
        # Log to file
        self.test_logger.info(f"Batch {batch_idx}: {json.dumps(batch_info)}")
    
    def log_anomaly_detection(self, anomaly_periods, sensor_rankings, attention_weights=None):
        """Log detailed anomaly detection results"""
        self.anomaly_analysis['anomaly_periods'] = anomaly_periods
        self.anomaly_analysis['sensor_rankings'] = sensor_rankings
        
        if attention_weights is not None:
            self.anomaly_analysis['attention_weights'] = attention_weights
        
        # Log summary
        num_anomalies = len(anomaly_periods)
        self.test_logger.info(f"Detected {num_anomalies} anomalous periods")
        
        for i, period in enumerate(anomaly_periods):
            self.test_logger.info(f"Anomaly {i+1}: {period}")
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Detected {num_anomalies} anomalous periods")
    
    def log_performance_metrics(self, metrics, stage='test'):
        """Log performance metrics (F1, precision, recall, etc.)"""
        message = f"{stage.capitalize()} Performance:"
        for metric_name, value in metrics.items():
            message += f" {metric_name}: {value:.4f}"
        
        if stage == 'test':
            self.test_logger.info(message)
        else:
            self.train_logger.info(message)
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
    
    def log_model_info(self, model, dataset_info=None):
        """Log model architecture and dataset information"""
        try:
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            model_info = {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_type': model.__class__.__name__
            }
            
            if dataset_info:
                model_info.update(dataset_info)
            
            self.train_logger.info(f"Model Info: {json.dumps(model_info)}")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Model: {trainable_params:,} trainable parameters")
            
        except Exception as e:
            self.train_logger.warning(f"Could not log model info: {e}")
    
    def _save_training_history(self):
        """Save training history to JSON file"""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump({
                    'training_history': self.training_history,
                    'config': self.config
                }, f, indent=2, default=str)
        except Exception as e:
            self.train_logger.error(f"Error saving training history: {e}")
    
    def save_final_results(self, test_results=None, val_results=None, best_metrics=None):
        """Save final test results and anomaly analysis"""
        final_results = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'training_history': self.training_history,
            'test_batch_progress': self.test_results['batch_progress'],
            'anomaly_analysis': self.anomaly_analysis
        }
        
        if test_results:
            final_results['test_results'] = test_results
        if val_results:
            final_results['val_results'] = val_results  
        if best_metrics:
            final_results['best_metrics'] = best_metrics
        
        # Save comprehensive results
        results_file = self.log_dir / f'{self.experiment_name}_final_results.json'
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        # Save anomaly analysis separately
        with open(self.anomaly_file, 'w') as f:
            json.dump(self.anomaly_analysis, f, indent=2, default=str)
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Results saved to {results_file}")
        
    def get_training_summary(self):
        """Get summary of training progress"""
        if not self.training_history['epochs']:
            return "No training data available"
        
        summary = {
            'total_epochs': len(self.training_history['epochs']),
            'final_train_loss': self.training_history['train_losses'][-1] if self.training_history['train_losses'] else None,
            'final_val_loss': self.training_history['val_losses'][-1] if self.training_history['val_losses'] and self.training_history['val_losses'][-1] is not None else None,
            'best_val_loss': min([loss for loss in self.training_history['val_losses'] if loss is not None]) if any(loss is not None for loss in self.training_history['val_losses']) else None,
            'total_training_time': sum([t for t in self.training_history['train_times'] if t is not None]) if any(t is not None for t in self.training_history['train_times']) else None
        }
        
        return summary