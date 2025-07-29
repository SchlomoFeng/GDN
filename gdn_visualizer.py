# -*- coding: utf-8 -*-
"""
GDN Visualizer for Graph Deviation Network
Provides comprehensive visualization capabilities for training, testing, and anomaly analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import os
from pathlib import Path
import warnings

# Optional imports for advanced visualizations
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Some interactive visualizations will be disabled.")

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available. Embedding visualizations will be disabled.")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    warnings.warn("NetworkX not available. Graph visualizations will be disabled.")


class GDNVisualizer:
    """
    Comprehensive visualization suite for GDN model analysis.
    
    Features:
    - Training curves and metrics evolution
    - Attention weight heatmaps
    - Anomaly score timelines
    - Sensor embedding visualizations
    - Graph network plots
    - Anomaly localization dashboards
    """
    
    def __init__(self, 
                 output_dir: str = './visualizations',
                 feature_map: Optional[Dict] = None,
                 verbose: bool = False):
        """
        Initialize the GDNVisualizer.
        
        Args:
            output_dir: Directory to save visualizations
            feature_map: Dictionary mapping sensor indices to names
            verbose: Whether to print detailed information
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.feature_map = feature_map or {}
        self.sensor_names = list(feature_map.values()) if feature_map else []
        self.verbose = verbose
        
        # Set visualization style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        sns.set_palette("husl")
        
        if self.verbose:
            print(f"GDNVisualizer initialized. Output directory: {self.output_dir}")
            print(f"Available sensors: {len(self.sensor_names)}")
            print(f"Plotly available: {PLOTLY_AVAILABLE}")
            print(f"NetworkX available: {NETWORKX_AVAILABLE}")
    
    def plot_training_curves(self, 
                           train_losses: List[float],
                           val_losses: Optional[List[float]] = None,
                           metrics: Optional[Dict[str, List[float]]] = None,
                           save_path: Optional[str] = None) -> None:
        """
        Plot training curves including loss and metrics evolution.
        
        Args:
            train_losses: Training loss values per epoch
            val_losses: Validation loss values per epoch
            metrics: Dictionary of metric names to values over epochs
            save_path: Custom save path for the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('GDN Training Progress', fontsize=16, fontweight='bold')
        
        # Plot 1: Training Loss
        axes[0, 0].plot(train_losses, label='Training Loss', color='blue', linewidth=2)
        if val_losses:
            axes[0, 0].plot(val_losses, label='Validation Loss', color='red', linewidth=2)
        axes[0, 0].set_title('Loss Evolution')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Learning Rate / Loss Smoothed
        if len(train_losses) > 10:
            smoothed_loss = self._smooth_curve(train_losses, window=5)
            axes[0, 1].plot(smoothed_loss, label='Smoothed Training Loss', color='green', linewidth=2)
            axes[0, 1].set_title('Smoothed Loss Trend')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Smoothed Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'Insufficient data\nfor smoothing', 
                          ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Smoothed Loss Trend')
        
        # Plot 3: Metrics
        if metrics:
            for metric_name, values in metrics.items():
                axes[1, 0].plot(values, label=metric_name, linewidth=2)
            axes[1, 0].set_title('Metrics Evolution')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Metric Value')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No metrics\navailable', 
                          ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Metrics Evolution')
        
        # Plot 4: Loss Distribution
        if len(train_losses) > 5:
            axes[1, 1].hist(train_losses, bins=min(20, len(train_losses)//2), 
                          alpha=0.7, color='skyblue', edgecolor='black')
            axes[1, 1].set_title('Training Loss Distribution')
            axes[1, 1].set_xlabel('Loss Value')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Insufficient data\nfor distribution', 
                          ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Training Loss Distribution')
        
        plt.tight_layout()
        
        save_path = save_path or str(self.output_dir / 'training_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if self.verbose:
            print(f"Training curves saved to: {save_path}")
        
        plt.close()
    
    def plot_attention_heatmap(self, 
                             attention_weights: np.ndarray,
                             sensor_names: Optional[List[str]] = None,
                             title: str = "Attention Weights Heatmap",
                             save_path: Optional[str] = None) -> None:
        """
        Plot attention weights as a heatmap.
        
        Args:
            attention_weights: 2D array of attention weights
            sensor_names: List of sensor names for labels
            title: Title for the plot
            save_path: Custom save path for the plot
        """
        sensor_names = sensor_names or self.sensor_names
        if len(sensor_names) != attention_weights.shape[0]:
            sensor_names = [f'Sensor_{i}' for i in range(attention_weights.shape[0])]
        
        plt.figure(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(attention_weights, 
                   xticklabels=sensor_names[:attention_weights.shape[1]], 
                   yticklabels=sensor_names[:attention_weights.shape[0]],
                   cmap='viridis', 
                   annot=False,
                   cbar_kws={'label': 'Attention Weight'})
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Target Sensors')
        plt.ylabel('Source Sensors')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        save_path = save_path or str(self.output_dir / 'attention_heatmap.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if self.verbose:
            print(f"Attention heatmap saved to: {save_path}")
        
        plt.close()
    
    def plot_anomaly_timeline(self, 
                            sensor_scores: np.ndarray,
                            labels: List[int],
                            sensor_names: Optional[List[str]] = None,
                            top_k: int = 5,
                            save_path: Optional[str] = None) -> None:
        """
        Plot anomaly score timeline for top-K sensors.
        
        Args:
            sensor_scores: Array of shape (num_sensors, num_timesteps)
            labels: Ground truth labels
            sensor_names: List of sensor names
            top_k: Number of top sensors to plot
            save_path: Custom save path for the plot
        """
        sensor_names = sensor_names or self.sensor_names
        if len(sensor_names) != sensor_scores.shape[0]:
            sensor_names = [f'Sensor_{i}' for i in range(sensor_scores.shape[0])]
        
        # Find top-K sensors by maximum score
        max_scores = np.max(sensor_scores, axis=1)
        top_indices = np.argsort(max_scores)[-top_k:][::-1]
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: Top-K sensor scores
        timesteps = np.arange(sensor_scores.shape[1])
        
        for i, sensor_idx in enumerate(top_indices):
            axes[0].plot(timesteps, sensor_scores[sensor_idx, :], 
                        label=f'{sensor_names[sensor_idx]} (max: {max_scores[sensor_idx]:.2f})',
                        linewidth=2, alpha=0.8)
        
        # Highlight anomaly periods
        anomaly_mask = np.array(labels) == 1
        if np.any(anomaly_mask):
            axes[0].fill_between(timesteps, 
                               axes[0].get_ylim()[0], 
                               axes[0].get_ylim()[1],
                               where=anomaly_mask, 
                               alpha=0.2, 
                               color='red', 
                               label='Anomaly Periods')
        
        axes[0].set_title(f'Top-{top_k} Sensors Anomaly Scores Timeline')
        axes[0].set_xlabel('Timestep')
        axes[0].set_ylabel('Anomaly Score')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Aggregate anomaly score
        aggregate_score = np.max(sensor_scores, axis=0)
        axes[1].plot(timesteps, aggregate_score, color='black', linewidth=2, label='Max Sensor Score')
        axes[1].fill_between(timesteps, aggregate_score, alpha=0.3, color='gray')
        
        if np.any(anomaly_mask):
            axes[1].fill_between(timesteps, 
                               axes[1].get_ylim()[0], 
                               axes[1].get_ylim()[1],
                               where=anomaly_mask, 
                               alpha=0.2, 
                               color='red', 
                               label='Anomaly Periods')
        
        axes[1].set_title('Aggregate Anomaly Score')
        axes[1].set_xlabel('Timestep')
        axes[1].set_ylabel('Max Anomaly Score')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = save_path or str(self.output_dir / 'anomaly_timeline.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if self.verbose:
            print(f"Anomaly timeline saved to: {save_path}")
        
        plt.close()
    
    def plot_sensor_embeddings(self, 
                             embeddings: np.ndarray,
                             sensor_names: Optional[List[str]] = None,
                             method: str = 'tsne',
                             save_path: Optional[str] = None) -> None:
        """
        Plot sensor embeddings using dimensionality reduction.
        
        Args:
            embeddings: 2D array of embeddings (num_sensors, embedding_dim)
            sensor_names: List of sensor names
            method: Reduction method ('tsne' or 'pca')
            save_path: Custom save path for the plot
        """
        if not SKLEARN_AVAILABLE:
            if self.verbose:
                print("Scikit-learn not available. Skipping embedding visualization.")
            return
        
        sensor_names = sensor_names or self.sensor_names
        if len(sensor_names) != embeddings.shape[0]:
            sensor_names = [f'Sensor_{i}' for i in range(embeddings.shape[0])]
        
        # Apply dimensionality reduction
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, embeddings.shape[0]-1))
        else:  # PCA
            reducer = PCA(n_components=2, random_state=42)
        
        reduced_embeddings = reducer.fit_transform(embeddings)
        
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot
        scatter = plt.scatter(reduced_embeddings[:, 0], 
                            reduced_embeddings[:, 1], 
                            s=100, 
                            alpha=0.7,
                            c=range(len(sensor_names)),
                            cmap='tab20')
        
        # Add labels
        for i, name in enumerate(sensor_names):
            plt.annotate(name, 
                        (reduced_embeddings[i, 0], reduced_embeddings[i, 1]),
                        xytext=(5, 5), 
                        textcoords='offset points',
                        fontsize=8,
                        alpha=0.8)
        
        plt.title(f'Sensor Embeddings Visualization ({method.upper()})', fontsize=14, fontweight='bold')
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        plt.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Sensor Index')
        
        save_path = save_path or str(self.output_dir / f'sensor_embeddings_{method}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if self.verbose:
            print(f"Sensor embeddings plot saved to: {save_path}")
        
        plt.close()
    
    def plot_graph_network(self, 
                          adjacency_matrix: np.ndarray,
                          sensor_names: Optional[List[str]] = None,
                          threshold: float = 0.1,
                          save_path: Optional[str] = None) -> None:
        """
        Plot the learned graph network structure.
        
        Args:
            adjacency_matrix: Adjacency matrix of the graph
            sensor_names: List of sensor names
            threshold: Threshold for edge display
            save_path: Custom save path for the plot
        """
        if not NETWORKX_AVAILABLE:
            if self.verbose:
                print("NetworkX not available. Skipping graph network visualization.")
            return
        
        sensor_names = sensor_names or self.sensor_names
        if len(sensor_names) != adjacency_matrix.shape[0]:
            sensor_names = [f'Sensor_{i}' for i in range(adjacency_matrix.shape[0])]
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes
        for i, name in enumerate(sensor_names):
            G.add_node(i, name=name)
        
        # Add edges above threshold
        for i in range(adjacency_matrix.shape[0]):
            for j in range(i+1, adjacency_matrix.shape[1]):
                weight = abs(adjacency_matrix[i, j])
                if weight > threshold:
                    G.add_edge(i, j, weight=weight)
        
        plt.figure(figsize=(14, 10))
        
        # Layout
        if len(G.nodes()) > 50:
            pos = nx.spring_layout(G, k=1, iterations=50)
        else:
            pos = nx.spring_layout(G, k=2, iterations=100)
        
        # Draw nodes
        node_sizes = [G.degree(node) * 100 + 200 for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, 
                              node_size=node_sizes,
                              node_color='lightblue',
                              alpha=0.7)
        
        # Draw edges with varying thickness
        edges = G.edges()
        if edges:
            weights = [G[u][v]['weight'] for u, v in edges]
            max_weight = max(weights) if weights else 1
            edge_widths = [w / max_weight * 5 for w in weights]
            
            nx.draw_networkx_edges(G, pos,
                                  width=edge_widths,
                                  alpha=0.6,
                                  edge_color='gray')
        
        # Draw labels
        labels = {i: name[:8] + '...' if len(name) > 8 else name 
                 for i, name in enumerate(sensor_names)}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        plt.title(f'Learned Graph Network Structure (threshold={threshold})', 
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        
        # Add statistics
        stats_text = f"Nodes: {G.number_of_nodes()}\nEdges: {G.number_of_edges()}\n"
        stats_text += f"Avg Degree: {np.mean([G.degree(n) for n in G.nodes()]):.2f}"
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        save_path = save_path or str(self.output_dir / 'graph_network.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if self.verbose:
            print(f"Graph network plot saved to: {save_path}")
            print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        plt.close()
    
    def create_anomaly_dashboard(self, 
                               sensor_scores: np.ndarray,
                               labels: List[int],
                               attention_weights: Optional[np.ndarray] = None,
                               sensor_names: Optional[List[str]] = None,
                               save_path: Optional[str] = None) -> None:
        """
        Create a comprehensive anomaly localization dashboard.
        
        Args:
            sensor_scores: Array of shape (num_sensors, num_timesteps)
            labels: Ground truth labels
            attention_weights: Optional attention weights matrix
            sensor_names: List of sensor names
            save_path: Custom save path for the plot
        """
        sensor_names = sensor_names or self.sensor_names
        if len(sensor_names) != sensor_scores.shape[0]:
            sensor_names = [f'Sensor_{i}' for i in range(sensor_scores.shape[0])]
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Anomaly Score Heatmap
        ax1 = fig.add_subplot(gs[0, :2])
        im1 = ax1.imshow(sensor_scores, aspect='auto', cmap='viridis', interpolation='nearest')
        ax1.set_title('Sensor Anomaly Scores Heatmap', fontweight='bold')
        ax1.set_xlabel('Timestep')
        ax1.set_ylabel('Sensors')
        ax1.set_yticks(range(0, len(sensor_names), max(1, len(sensor_names)//10)))
        ax1.set_yticklabels(sensor_names[::max(1, len(sensor_names)//10)])
        plt.colorbar(im1, ax=ax1, label='Anomaly Score')
        
        # 2. Top Sensors Timeline
        ax2 = fig.add_subplot(gs[1, :2])
        top_5_indices = np.argsort(np.max(sensor_scores, axis=1))[-5:][::-1]
        timesteps = np.arange(sensor_scores.shape[1])
        
        for i, sensor_idx in enumerate(top_5_indices):
            ax2.plot(timesteps, sensor_scores[sensor_idx, :], 
                    label=sensor_names[sensor_idx], linewidth=2, alpha=0.8)
        
        # Highlight anomalies
        anomaly_mask = np.array(labels) == 1
        if np.any(anomaly_mask):
            ax2.fill_between(timesteps, ax2.get_ylim()[0], ax2.get_ylim()[1],
                           where=anomaly_mask, alpha=0.2, color='red', label='Anomaly')
        
        ax2.set_title('Top 5 Sensors Timeline', fontweight='bold')
        ax2.set_xlabel('Timestep')
        ax2.set_ylabel('Anomaly Score')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # 3. Score Distribution
        ax3 = fig.add_subplot(gs[0, 2])
        all_scores = sensor_scores.flatten()
        ax3.hist(all_scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(np.percentile(all_scores, 95), color='red', linestyle='--', 
                   label='95th percentile')
        ax3.set_title('Score Distribution', fontweight='bold')
        ax3.set_xlabel('Anomaly Score')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Sensor Importance
        ax4 = fig.add_subplot(gs[1, 2])
        importance_scores = np.max(sensor_scores, axis=1)
        top_10_indices = np.argsort(importance_scores)[-10:]
        
        ax4.barh(range(len(top_10_indices)), importance_scores[top_10_indices])
        ax4.set_yticks(range(len(top_10_indices)))
        ax4.set_yticklabels([sensor_names[i] for i in top_10_indices])
        ax4.set_title('Top 10 Sensor Importance', fontweight='bold')
        ax4.set_xlabel('Max Anomaly Score')
        
        # 5. Temporal Patterns
        ax5 = fig.add_subplot(gs[2, :])
        
        # Calculate rolling statistics
        window_size = min(50, sensor_scores.shape[1] // 10)
        if window_size > 1:
            rolling_mean = np.array([np.mean(sensor_scores[:, max(0, i-window_size):i+1]) 
                                   for i in range(sensor_scores.shape[1])])
            rolling_std = np.array([np.std(sensor_scores[:, max(0, i-window_size):i+1]) 
                                  for i in range(sensor_scores.shape[1])])
            
            ax5.plot(timesteps, rolling_mean, label='Rolling Mean', linewidth=2)
            ax5.fill_between(timesteps, rolling_mean - rolling_std, rolling_mean + rolling_std,
                           alpha=0.3, label='±1 Std')
            
            if np.any(anomaly_mask):
                ax5.fill_between(timesteps, ax5.get_ylim()[0], ax5.get_ylim()[1],
                               where=anomaly_mask, alpha=0.2, color='red', label='Anomaly')
        
        ax5.set_title('Temporal Anomaly Patterns', fontweight='bold')
        ax5.set_xlabel('Timestep')
        ax5.set_ylabel('Anomaly Score')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        plt.suptitle('GDN Anomaly Localization Dashboard', fontsize=16, fontweight='bold')
        
        save_path = save_path or str(self.output_dir / 'anomaly_dashboard.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if self.verbose:
            print(f"Anomaly dashboard saved to: {save_path}")
        
        plt.close()
    
    def create_interactive_dashboard(self, 
                                   sensor_scores: np.ndarray,
                                   labels: List[int],
                                   sensor_names: Optional[List[str]] = None,
                                   save_path: Optional[str] = None) -> None:
        """
        Create an interactive dashboard using Plotly (if available).
        
        Args:
            sensor_scores: Array of shape (num_sensors, num_timesteps)
            labels: Ground truth labels
            sensor_names: List of sensor names
            save_path: Custom save path for the HTML file
        """
        if not PLOTLY_AVAILABLE:
            if self.verbose:
                print("Plotly not available. Skipping interactive dashboard.")
            return
        
        sensor_names = sensor_names or self.sensor_names
        if len(sensor_names) != sensor_scores.shape[0]:
            sensor_names = [f'Sensor_{i}' for i in range(sensor_scores.shape[0])]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Anomaly Scores Heatmap', 'Top Sensors Timeline', 
                          'Score Distribution', 'Sensor Importance'),
            specs=[[{"type": "heatmap"}, {"type": "scatter"}],
                   [{"type": "histogram"}, {"type": "bar"}]]
        )
        
        # 1. Heatmap
        fig.add_trace(
            go.Heatmap(z=sensor_scores, 
                      x=list(range(sensor_scores.shape[1])),
                      y=sensor_names,
                      colorscale='Viridis',
                      name='Anomaly Scores'),
            row=1, col=1
        )
        
        # 2. Timeline for top sensors
        top_5_indices = np.argsort(np.max(sensor_scores, axis=1))[-5:][::-1]
        timesteps = list(range(sensor_scores.shape[1]))
        
        for i, sensor_idx in enumerate(top_5_indices):
            fig.add_trace(
                go.Scatter(x=timesteps, 
                          y=sensor_scores[sensor_idx, :],
                          mode='lines',
                          name=sensor_names[sensor_idx],
                          line=dict(width=2)),
                row=1, col=2
            )
        
        # 3. Score distribution
        fig.add_trace(
            go.Histogram(x=sensor_scores.flatten(),
                        nbinsx=50,
                        name='Score Distribution'),
            row=2, col=1
        )
        
        # 4. Sensor importance
        importance_scores = np.max(sensor_scores, axis=1)
        top_10_indices = np.argsort(importance_scores)[-10:]
        
        fig.add_trace(
            go.Bar(x=importance_scores[top_10_indices],
                  y=[sensor_names[i] for i in top_10_indices],
                  orientation='h',
                  name='Importance'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='GDN Interactive Anomaly Dashboard',
            height=800,
            showlegend=True
        )
        
        save_path = save_path or str(self.output_dir / 'interactive_dashboard.html')
        fig.write_html(save_path)
        
        if self.verbose:
            print(f"Interactive dashboard saved to: {save_path}")
    
    def _smooth_curve(self, values: List[float], window: int = 5) -> List[float]:
        """Apply smoothing to a curve using moving average."""
        if len(values) < window:
            return values
        
        smoothed = []
        for i in range(len(values)):
            start = max(0, i - window // 2)
            end = min(len(values), i + window // 2 + 1)
            smoothed.append(np.mean(values[start:end]))
        
        return smoothed
    
    def save_all_plots(self, 
                      train_data: Dict,
                      test_data: Dict,
                      model_data: Optional[Dict] = None) -> None:
        """
        Generate and save all visualization plots.
        
        Args:
            train_data: Dictionary with training data (losses, metrics, etc.)
            test_data: Dictionary with test data (scores, labels, etc.)
            model_data: Optional dictionary with model data (embeddings, attention, etc.)
        """
        if self.verbose:
            print("Generating all visualization plots...")
        
        # Training curves
        if 'train_losses' in train_data:
            self.plot_training_curves(
                train_data['train_losses'],
                train_data.get('val_losses'),
                train_data.get('metrics')
            )
        
        # Anomaly timeline
        if 'sensor_scores' in test_data and 'labels' in test_data:
            self.plot_anomaly_timeline(
                test_data['sensor_scores'],
                test_data['labels']
            )
        
        # Anomaly dashboard
        if 'sensor_scores' in test_data and 'labels' in test_data:
            self.create_anomaly_dashboard(
                test_data['sensor_scores'],
                test_data['labels'],
                test_data.get('attention_weights')
            )
        
        # Interactive dashboard
        if 'sensor_scores' in test_data and 'labels' in test_data:
            self.create_interactive_dashboard(
                test_data['sensor_scores'],
                test_data['labels']
            )
        
        # Model-specific plots
        if model_data:
            if 'embeddings' in model_data:
                self.plot_sensor_embeddings(model_data['embeddings'], method='tsne')
                self.plot_sensor_embeddings(model_data['embeddings'], method='pca')
            
            if 'attention_weights' in model_data:
                self.plot_attention_heatmap(model_data['attention_weights'])
            
            if 'adjacency_matrix' in model_data:
                self.plot_graph_network(model_data['adjacency_matrix'])
        
        if self.verbose:
            print(f"All plots saved to: {self.output_dir}")