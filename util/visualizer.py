# -*- coding: utf-8 -*-
"""
Visualization utilities for GDN anomaly detection results
Provides comprehensive visualization capabilities for training progress, anomaly analysis, and model insights
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import pandas as pd

# Set style for better looking plots
plt.style.use('default')
sns.set_palette("husl")

class GDNVisualizer:
    """Comprehensive visualization toolkit for GDN anomaly detection"""
    
    def __init__(self, save_dir='./visualizations', experiment_name=None, figsize=(12, 8)):
        """
        Initialize visualizer
        
        Args:
            save_dir (str): Directory to save visualization files
            experiment_name (str): Name of the experiment
            figsize (tuple): Default figure size
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        if experiment_name is None:
            experiment_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.experiment_name = experiment_name
        self.figsize = figsize
        
        # Configure matplotlib for better plots
        plt.rcParams.update({
            'figure.figsize': figsize,
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9
        })
    
    def plot_training_history(self, training_history, save=True):
        """
        Plot training and validation loss curves
        
        Args:
            training_history (dict): Dictionary containing training metrics
            save (bool): Whether to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training History - {self.experiment_name}', fontsize=16)
        
        epochs = training_history.get('epochs', [])
        train_losses = training_history.get('train_losses', [])
        val_losses = training_history.get('val_losses', [])
        learning_rates = training_history.get('learning_rates', [])
        train_times = training_history.get('train_times', [])
        
        # Plot 1: Training and Validation Loss
        axes[0, 0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        if val_losses and any(loss is not None for loss in val_losses):
            val_losses_clean = [loss for loss in val_losses if loss is not None]
            epochs_clean = [epochs[i] for i, loss in enumerate(val_losses) if loss is not None]
            axes[0, 0].plot(epochs_clean, val_losses_clean, 'r-', label='Validation Loss', linewidth=2)
        
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Learning Rate Schedule
        if learning_rates and any(lr is not None for lr in learning_rates):
            lr_clean = [lr for lr in learning_rates if lr is not None]
            epochs_lr = [epochs[i] for i, lr in enumerate(learning_rates) if lr is not None]
            axes[0, 1].plot(epochs_lr, lr_clean, 'g-', linewidth=2)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].set_title('Learning Rate Schedule')
            axes[0, 1].set_yscale('log')
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'No Learning Rate Data', 
                           horizontalalignment='center', verticalalignment='center',
                           transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Learning Rate Schedule')
        
        # Plot 3: Training Time per Epoch
        if train_times and any(t is not None for t in train_times):
            times_clean = [t for t in train_times if t is not None]
            epochs_time = [epochs[i] for i, t in enumerate(train_times) if t is not None]
            axes[1, 0].bar(epochs_time, times_clean, alpha=0.7)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Time (seconds)')
            axes[1, 0].set_title('Training Time per Epoch')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No Timing Data', 
                           horizontalalignment='center', verticalalignment='center',
                           transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Training Time per Epoch')
        
        # Plot 4: Loss Distribution (Histogram)
        if train_losses:
            axes[1, 1].hist(train_losses, bins=30, alpha=0.7, label='Train Loss', color='blue')
            if val_losses and any(loss is not None for loss in val_losses):
                val_losses_clean = [loss for loss in val_losses if loss is not None]
                axes[1, 1].hist(val_losses_clean, bins=30, alpha=0.7, label='Val Loss', color='red')
            axes[1, 1].set_xlabel('Loss Value')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Loss Distribution')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / f'{self.experiment_name}_training_history.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()
        return fig
    
    def plot_anomaly_scores_timeseries(self, anomaly_scores, timestamps=None, ground_truth=None, 
                                     threshold=None, save=True):
        """
        Plot anomaly scores as time series
        
        Args:
            anomaly_scores (array): Anomaly scores over time
            timestamps (array): Timestamps for the scores
            ground_truth (array): Ground truth labels
            threshold (float): Anomaly threshold
            save (bool): Whether to save the plot
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if timestamps is None:
            timestamps = np.arange(len(anomaly_scores))
        
        # Plot anomaly scores
        ax.plot(timestamps, anomaly_scores, 'b-', linewidth=1, alpha=0.8, label='Anomaly Score')
        
        # Plot ground truth if available
        if ground_truth is not None:
            # Highlight anomalous regions
            anomaly_mask = np.array(ground_truth) > 0
            if np.any(anomaly_mask):
                ax.fill_between(timestamps, 0, np.max(anomaly_scores), 
                               where=anomaly_mask, alpha=0.3, color='red', 
                               label='True Anomalies')
        
        # Plot threshold if provided
        if threshold is not None:
            ax.axhline(y=threshold, color='r', linestyle='--', linewidth=2, 
                      label=f'Threshold ({threshold:.3f})')
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Anomaly Score')
        ax.set_title(f'Anomaly Scores Time Series - {self.experiment_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / f'{self.experiment_name}_anomaly_timeseries.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Anomaly time series plot saved to {save_path}")
        
        plt.show()
        return fig
    
    def plot_sensor_anomaly_heatmap(self, sensor_scores, feature_names=None, timestamps=None, save=True):
        """
        Create heatmap showing anomaly scores for each sensor over time
        
        Args:
            sensor_scores (array): 2D array of shape (time_steps, num_sensors)
            feature_names (list): Names of sensors/features
            timestamps (array): Timestamps
            save (bool): Whether to save the plot
        """
        fig, ax = plt.subplots(figsize=(15, 8))
        
        if feature_names is None:
            feature_names = [f'Sensor_{i}' for i in range(sensor_scores.shape[1])]
        
        if timestamps is None:
            timestamps = np.arange(sensor_scores.shape[0])
        
        # Create heatmap
        im = ax.imshow(sensor_scores.T, cmap='YlOrRd', aspect='auto', interpolation='nearest')
        
        # Set labels
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Sensors')
        ax.set_title(f'Sensor Anomaly Heatmap - {self.experiment_name}')
        
        # Set ticks
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(feature_names)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Anomaly Score')
        
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / f'{self.experiment_name}_sensor_heatmap.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Sensor heatmap saved to {save_path}")
        
        plt.show()
        return fig
    
    def plot_top_anomalous_sensors(self, sensor_rankings, top_k=10, save=True):
        """
        Plot top-k most anomalous sensors
        
        Args:
            sensor_rankings (dict): Dictionary with sensor names as keys and scores as values
            top_k (int): Number of top sensors to show
            save (bool): Whether to save the plot
        """
        # Sort sensors by score
        sorted_sensors = sorted(sensor_rankings.items(), key=lambda x: x[1], reverse=True)
        top_sensors = sorted_sensors[:top_k]
        
        sensors = [item[0] for item in top_sensors]
        scores = [item[1] for item in top_sensors]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.barh(sensors, scores, color=plt.cm.Reds(np.linspace(0.4, 0.9, len(sensors))))
        
        ax.set_xlabel('Anomaly Score')
        ax.set_title(f'Top {top_k} Most Anomalous Sensors - {self.experiment_name}')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add score labels on bars
        for i, (sensor, score) in enumerate(top_sensors):
            ax.text(score + 0.01 * max(scores), i, f'{score:.3f}', 
                   va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / f'{self.experiment_name}_top_sensors.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Top sensors plot saved to {save_path}")
        
        plt.show()
        return fig
    
    def plot_confusion_matrix(self, y_true, y_pred, save=True):
        """
        Plot confusion matrix for anomaly detection
        
        Args:
            y_true (array): True labels
            y_pred (array): Predicted labels
            save (bool): Whether to save the plot
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Anomaly'],
                   yticklabels=['Normal', 'Anomaly'], ax=ax)
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix - {self.experiment_name}')
        
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / f'{self.experiment_name}_confusion_matrix.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
        return fig
    
    def plot_roc_curve(self, y_true, y_scores, save=True):
        """
        Plot ROC curve for anomaly detection
        
        Args:
            y_true (array): True labels
            y_scores (array): Anomaly scores
            save (bool): Whether to save the plot
        """
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
               label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
               label='Random classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - {self.experiment_name}')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / f'{self.experiment_name}_roc_curve.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")
        
        plt.show()
        return fig
    
    def plot_sensor_timeseries(self, sensor_data, feature_names=None, timestamps=None, 
                              anomaly_labels=None, top_k=5, save=True):
        """
        Plot time series for top-k most anomalous sensors
        
        Args:
            sensor_data (array): 2D array of sensor readings
            feature_names (list): Names of sensors
            timestamps (array): Timestamps
            anomaly_labels (array): Binary anomaly labels
            top_k (int): Number of sensors to plot
            save (bool): Whether to save the plot
        """
        if feature_names is None:
            feature_names = [f'Sensor_{i}' for i in range(sensor_data.shape[1])]
        
        if timestamps is None:
            timestamps = np.arange(sensor_data.shape[0])
        
        # Calculate sensor importance (std or variance as proxy)
        sensor_importance = np.std(sensor_data, axis=0)
        top_indices = np.argsort(sensor_importance)[-top_k:][::-1]
        
        fig, axes = plt.subplots(top_k, 1, figsize=(15, 3*top_k), sharex=True)
        if top_k == 1:
            axes = [axes]
        
        for i, sensor_idx in enumerate(top_indices):
            ax = axes[i]
            sensor_name = feature_names[sensor_idx]
            
            # Plot sensor data
            ax.plot(timestamps, sensor_data[:, sensor_idx], 'b-', linewidth=1, alpha=0.8)
            
            # Highlight anomalous regions if labels available
            if anomaly_labels is not None:
                anomaly_mask = np.array(anomaly_labels) > 0
                if np.any(anomaly_mask):
                    y_min, y_max = ax.get_ylim()
                    ax.fill_between(timestamps, y_min, y_max, 
                                   where=anomaly_mask, alpha=0.3, color='red')
            
            ax.set_ylabel(f'{sensor_name}\nValue')
            ax.set_title(f'Top {i+1} Anomalous Sensor: {sensor_name}')
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time Step')
        fig.suptitle(f'Top {top_k} Anomalous Sensors Time Series - {self.experiment_name}', 
                    fontsize=16)
        
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / f'{self.experiment_name}_sensor_timeseries.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Sensor time series plot saved to {save_path}")
        
        plt.show()
        return fig
    
    def plot_graph_structure(self, adjacency_matrix, feature_names=None, pos=None, save=True):
        """
        Visualize graph structure/attention weights between sensors
        
        Args:
            adjacency_matrix (array): Adjacency or attention matrix
            feature_names (list): Names of sensors/nodes
            pos (dict): Position dictionary for nodes
            save (bool): Whether to save the plot
        """
        try:
            import networkx as nx
        except ImportError:
            print("NetworkX not available. Skipping graph visualization.")
            return None
        
        # Create graph
        G = nx.from_numpy_array(adjacency_matrix)
        
        if feature_names is not None:
            mapping = {i: name for i, name in enumerate(feature_names)}
            G = nx.relabel_nodes(G, mapping)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Calculate node positions
        if pos is None:
            pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Draw edges with weights
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        
        if weights:
            # Normalize weights for visualization
            weights = np.array(weights)
            weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
            
            nx.draw_networkx_edges(G, pos, width=weights*5, alpha=0.6, 
                                  edge_color=weights, edge_cmap=plt.cm.Reds, ax=ax)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=500, alpha=0.8, ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
        
        ax.set_title(f'Graph Structure - {self.experiment_name}')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / f'{self.experiment_name}_graph_structure.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Graph structure plot saved to {save_path}")
        
        plt.show()
        return fig
    
    def create_comprehensive_report(self, training_history=None, test_results=None, 
                                   anomaly_analysis=None, save=True):
        """
        Create a comprehensive visualization report with all plots
        
        Args:
            training_history (dict): Training metrics
            test_results (dict): Test results
            anomaly_analysis (dict): Anomaly analysis results
            save (bool): Whether to save plots
        """
        print(f"Creating comprehensive visualization report for {self.experiment_name}")
        
        figures = []
        
        # 1. Training history
        if training_history:
            print("Creating training history plots...")
            fig = self.plot_training_history(training_history, save=save)
            figures.append(('training_history', fig))
        
        # 2. Anomaly scores time series
        if anomaly_analysis and 'anomaly_scores' in anomaly_analysis:
            print("Creating anomaly scores time series...")
            fig = self.plot_anomaly_scores_timeseries(
                anomaly_analysis['anomaly_scores'],
                save=save
            )
            figures.append(('anomaly_timeseries', fig))
        
        # 3. Top anomalous sensors
        if anomaly_analysis and 'sensor_rankings' in anomaly_analysis:
            print("Creating top sensors plot...")
            fig = self.plot_top_anomalous_sensors(
                anomaly_analysis['sensor_rankings'], 
                save=save
            )
            figures.append(('top_sensors', fig))
        
        print(f"Visualization report completed. {len(figures)} plots created.")
        
        if save:
            # Create summary file
            summary_file = self.save_dir / f'{self.experiment_name}_visualization_summary.txt'
            with open(summary_file, 'w') as f:
                f.write(f"Visualization Report for {self.experiment_name}\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"Created {len(figures)} visualizations:\n")
                for name, _ in figures:
                    f.write(f"- {name}\n")
        
        return figures