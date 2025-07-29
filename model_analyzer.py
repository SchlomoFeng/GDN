# -*- coding: utf-8 -*-
"""
Model Analyzer for Graph Deviation Network (GDN)
Provides comprehensive analysis of trained GDN models including graph structure,
embeddings, and attention patterns.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
from pathlib import Path


class ModelAnalyzer:
    """
    Comprehensive analysis suite for GDN model components.
    
    Features:
    - Graph structure analysis and interpretation
    - Sensor embedding analysis and clustering
    - Attention pattern extraction and interpretation
    - Model architecture insights
    - Performance attribution analysis
    """
    
    def __init__(self, model, feature_map: Dict, device: str = 'cpu', verbose: bool = False):
        """
        Initialize the ModelAnalyzer.
        
        Args:
            model: Trained GDN model
            feature_map: Dictionary mapping sensor indices to sensor names
            device: Computing device ('cpu' or 'cuda')
            verbose: Whether to print detailed information
        """
        self.model = model
        self.feature_map = feature_map
        self.sensor_names = list(feature_map.values())
        self.num_sensors = len(feature_map)
        self.device = device
        self.verbose = verbose
        
        # Analysis storage
        self.graph_analysis = {}
        self.embedding_analysis = {}
        self.attention_analysis = {}
        self.architecture_info = {}
        
        if self.verbose:
            print(f"ModelAnalyzer initialized for {self.num_sensors} sensors")
            print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def analyze_graph_structure(self) -> Dict:
        """
        Analyze the learned graph structure from the GDN model.
        
        Returns:
            Dictionary containing graph structure analysis
        """
        self.model.eval()
        
        with torch.no_grad():
            # Extract learned adjacency matrix from model
            if hasattr(self.model, 'learned_graph') and self.model.learned_graph is not None:
                learned_graph = self.model.learned_graph.cpu().numpy()
                
                # Create adjacency matrix
                adjacency_matrix = np.zeros((self.num_sensors, self.num_sensors))
                
                for i in range(learned_graph.shape[0]):
                    connections = learned_graph[i]
                    for j in connections:
                        if 0 <= j < self.num_sensors:
                            adjacency_matrix[i, j] = 1
                
                # Analyze graph properties
                analysis = self._analyze_adjacency_matrix(adjacency_matrix)
                
            else:
                # Use embeddings to infer graph structure
                embeddings = self.model.embedding.weight.data.cpu().numpy()
                adjacency_matrix = self._compute_similarity_matrix(embeddings)
                analysis = self._analyze_adjacency_matrix(adjacency_matrix)
                analysis['note'] = 'Graph structure inferred from embeddings'
        
        analysis['adjacency_matrix'] = adjacency_matrix
        self.graph_analysis = analysis
        
        if self.verbose:
            print(f"Graph Analysis Summary:")
            print(f"  Density: {analysis.get('density', 0):.3f}")
            print(f"  Average degree: {analysis.get('avg_degree', 0):.2f}")
            print(f"  Number of components: {analysis.get('num_components', 0)}")
            print(f"  Clustering coefficient: {analysis.get('clustering_coefficient', 0):.3f}")
        
        return analysis
    
    def analyze_sensor_embeddings(self) -> Dict:
        """
        Analyze sensor embeddings learned by the model.
        
        Returns:
            Dictionary containing embedding analysis
        """
        # Extract embeddings
        embeddings = self.model.embedding.weight.data.cpu().numpy()
        
        # Compute embedding statistics
        analysis = {
            'embeddings': embeddings,
            'embedding_dim': embeddings.shape[1],
            'norm_stats': {
                'mean_norm': np.mean(np.linalg.norm(embeddings, axis=1)),
                'std_norm': np.std(np.linalg.norm(embeddings, axis=1)),
                'min_norm': np.min(np.linalg.norm(embeddings, axis=1)),
                'max_norm': np.max(np.linalg.norm(embeddings, axis=1))
            }
        }
        
        # Compute pairwise similarities
        similarity_matrix = self._compute_similarity_matrix(embeddings)
        analysis['similarity_matrix'] = similarity_matrix
        
        # Find most similar sensor pairs
        similar_pairs = self._find_similar_pairs(similarity_matrix, top_k=10)
        analysis['most_similar_pairs'] = similar_pairs
        
        # Find sensor clusters
        clusters = self._cluster_sensors(embeddings)
        analysis['clusters'] = clusters
        
        # Compute embedding diversity
        analysis['diversity_score'] = self._compute_embedding_diversity(embeddings)
        
        self.embedding_analysis = analysis
        
        if self.verbose:
            print(f"Embedding Analysis Summary:")
            print(f"  Embedding dimension: {analysis['embedding_dim']}")
            print(f"  Average embedding norm: {analysis['norm_stats']['mean_norm']:.3f}")
            print(f"  Number of clusters: {len(clusters)}")
            print(f"  Diversity score: {analysis['diversity_score']:.3f}")
            print(f"  Most similar pair: {similar_pairs[0] if similar_pairs else 'None'}")
        
        return analysis
    
    def extract_attention_patterns(self, dataloader, max_batches: int = 5) -> Dict:
        """
        Extract and analyze attention patterns from the model.
        
        Args:
            dataloader: Data loader for processing
            max_batches: Maximum number of batches to process
            
        Returns:
            Dictionary containing attention pattern analysis
        """
        self.model.eval()
        attention_data = []
        
        with torch.no_grad():
            for batch_idx, (x, y, labels, edge_index) in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break
                
                x, y, edge_index = [item.to(self.device).float() for item in [x, y, edge_index]]
                
                # Forward pass
                _ = self.model(x, edge_index)
                
                # Extract attention weights from each GNN layer
                batch_attention = {}
                for layer_idx, gnn_layer in enumerate(self.model.gnn_layers):
                    if hasattr(gnn_layer, 'att_weight_1') and gnn_layer.att_weight_1 is not None:
                        att_weights = gnn_layer.att_weight_1.cpu().numpy()
                        batch_attention[f'layer_{layer_idx}'] = att_weights
                
                attention_data.append({
                    'batch': batch_idx,
                    'attention': batch_attention,
                    'labels': labels.cpu().numpy()
                })
        
        # Analyze attention patterns
        analysis = self._analyze_attention_data(attention_data)
        self.attention_analysis = analysis
        
        if self.verbose:
            print(f"Attention Pattern Analysis:")
            print(f"  Processed {len(attention_data)} batches")
            print(f"  Layers with attention: {analysis.get('num_layers', 0)}")
            print(f"  Average attention entropy: {analysis.get('avg_entropy', 0):.3f}")
        
        return analysis
    
    def get_model_architecture_info(self) -> Dict:
        """
        Extract detailed information about the model architecture.
        
        Returns:
            Dictionary containing architecture information
        """
        info = {
            'model_type': type(self.model).__name__,
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'num_sensors': self.num_sensors,
            'embedding_dim': self.model.embedding.embedding_dim,
            'num_embeddings': self.model.embedding.num_embeddings,
            'topk': getattr(self.model, 'topk', None),
            'edge_sets': len(self.model.edge_index_sets) if hasattr(self.model, 'edge_index_sets') else 0
        }
        
        # Analyze GNN layers
        gnn_info = []
        for i, layer in enumerate(self.model.gnn_layers):
            layer_info = {
                'layer_id': i,
                'type': type(layer).__name__,
                'parameters': sum(p.numel() for p in layer.parameters())
            }
            
            # Extract GNN-specific info
            if hasattr(layer, 'gnn'):
                gnn = layer.gnn
                layer_info.update({
                    'in_channels': getattr(gnn, 'in_channels', None),
                    'out_channels': getattr(gnn, 'out_channels', None),
                    'heads': getattr(gnn, 'heads', None)
                })
            
            gnn_info.append(layer_info)
        
        info['gnn_layers'] = gnn_info
        
        # Analyze output layers
        if hasattr(self.model, 'out_layer'):
            out_layer = self.model.out_layer
            info['output_layer'] = {
                'type': type(out_layer).__name__,
                'parameters': sum(p.numel() for p in out_layer.parameters()),
                'num_layers': len(out_layer.mlp) if hasattr(out_layer, 'mlp') else 0
            }
        
        self.architecture_info = info
        
        if self.verbose:
            print(f"Model Architecture:")
            print(f"  Total parameters: {info['total_parameters']:,}")
            print(f"  Trainable parameters: {info['trainable_parameters']:,}")
            print(f"  Embedding dimension: {info['embedding_dim']}")
            print(f"  Number of GNN layers: {len(gnn_info)}")
            print(f"  Top-K connections: {info['topk']}")
        
        return info
    
    def analyze_feature_importance(self, 
                                 dataloader, 
                                 perturbation_std: float = 0.1,
                                 max_batches: int = 3) -> Dict:
        """
        Analyze feature importance using perturbation-based method.
        
        Args:
            dataloader: Data loader for processing
            perturbation_std: Standard deviation for noise perturbation
            max_batches: Maximum number of batches to process
            
        Returns:
            Dictionary containing feature importance analysis
        """
        self.model.eval()
        importance_scores = np.zeros(self.num_sensors)
        
        with torch.no_grad():
            for batch_idx, (x, y, labels, edge_index) in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break
                
                x, y, edge_index = [item.to(self.device).float() for item in [x, y, edge_index]]
                
                # Get baseline prediction
                baseline_pred = self.model(x, edge_index)
                baseline_loss = F.mse_loss(baseline_pred, y).item()
                
                # Perturb each sensor and measure impact
                for sensor_idx in range(self.num_sensors):
                    x_perturbed = x.clone()
                    noise = torch.randn_like(x_perturbed[:, sensor_idx, :]) * perturbation_std
                    x_perturbed[:, sensor_idx, :] += noise
                    
                    perturbed_pred = self.model(x_perturbed, edge_index)
                    perturbed_loss = F.mse_loss(perturbed_pred, y).item()
                    
                    # Importance is the increase in loss due to perturbation
                    importance_scores[sensor_idx] += (perturbed_loss - baseline_loss)
        
        # Normalize by number of batches
        importance_scores /= max_batches
        
        # Create importance ranking
        importance_ranking = [(self.sensor_names[i], importance_scores[i]) 
                            for i in np.argsort(importance_scores)[::-1]]
        
        analysis = {
            'importance_scores': importance_scores,
            'importance_ranking': importance_ranking,
            'perturbation_std': perturbation_std,
            'num_batches': max_batches
        }
        
        if self.verbose:
            print(f"Feature Importance Analysis:")
            print(f"  Top 5 important sensors:")
            for i, (name, score) in enumerate(importance_ranking[:5]):
                print(f"    {i+1}. {name}: {score:.4f}")
        
        return analysis
    
    def generate_model_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive model analysis report.
        
        Args:
            save_path: Optional path to save the report
            
        Returns:
            Formatted report as string
        """
        # Ensure all analyses are performed
        if not self.architecture_info:
            self.get_model_architecture_info()
        
        if not self.graph_analysis:
            self.analyze_graph_structure()
        
        if not self.embedding_analysis:
            self.analyze_sensor_embeddings()
        
        # Generate report
        report = []
        report.append("=" * 70)
        report.append("GDN MODEL ANALYSIS REPORT")
        report.append("=" * 70)
        report.append("")
        
        # Architecture Section
        report.append("MODEL ARCHITECTURE")
        report.append("-" * 50)
        arch = self.architecture_info
        report.append(f"Model Type: {arch['model_type']}")
        report.append(f"Total Parameters: {arch['total_parameters']:,}")
        report.append(f"Trainable Parameters: {arch['trainable_parameters']:,}")
        report.append(f"Number of Sensors: {arch['num_sensors']}")
        report.append(f"Embedding Dimension: {arch['embedding_dim']}")
        report.append(f"Top-K Connections: {arch['topk']}")
        report.append(f"Number of GNN Layers: {len(arch['gnn_layers'])}")
        report.append("")
        
        # Graph Structure Section
        report.append("LEARNED GRAPH STRUCTURE")
        report.append("-" * 50)
        graph = self.graph_analysis
        report.append(f"Graph Density: {graph.get('density', 0):.4f}")
        report.append(f"Average Degree: {graph.get('avg_degree', 0):.2f}")
        report.append(f"Number of Components: {graph.get('num_components', 0)}")
        report.append(f"Clustering Coefficient: {graph.get('clustering_coefficient', 0):.4f}")
        
        if 'hub_sensors' in graph:
            report.append(f"Hub Sensors (top 3): {', '.join(graph['hub_sensors'][:3])}")
        report.append("")
        
        # Embedding Section
        report.append("SENSOR EMBEDDINGS")
        report.append("-" * 50)
        embed = self.embedding_analysis
        report.append(f"Embedding Dimension: {embed['embedding_dim']}")
        report.append(f"Average Norm: {embed['norm_stats']['mean_norm']:.3f}")
        report.append(f"Norm Range: [{embed['norm_stats']['min_norm']:.3f}, {embed['norm_stats']['max_norm']:.3f}]")
        report.append(f"Diversity Score: {embed['diversity_score']:.3f}")
        report.append(f"Number of Clusters: {len(embed['clusters'])}")
        
        if embed['most_similar_pairs']:
            pair = embed['most_similar_pairs'][0]
            report.append(f"Most Similar Pair: {pair[0]} - {pair[1]} (similarity: {pair[2]:.3f})")
        report.append("")
        
        # Attention Section (if available)
        if self.attention_analysis:
            report.append("ATTENTION PATTERNS")
            report.append("-" * 50)
            att = self.attention_analysis
            report.append(f"Layers with Attention: {att.get('num_layers', 0)}")
            report.append(f"Average Entropy: {att.get('avg_entropy', 0):.3f}")
            report.append(f"Attention Sparsity: {att.get('sparsity', 0):.3f}")
            report.append("")
        
        # Model Insights
        report.append("KEY INSIGHTS")
        report.append("-" * 50)
        insights = self._generate_insights()
        for insight in insights:
            report.append(f"• {insight}")
        
        report.append("")
        report.append("=" * 70)
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            if self.verbose:
                print(f"Model report saved to: {save_path}")
        
        return report_text
    
    def _analyze_adjacency_matrix(self, adj_matrix: np.ndarray) -> Dict:
        """Analyze properties of adjacency matrix."""
        analysis = {}
        
        # Basic properties
        analysis['density'] = np.sum(adj_matrix) / (adj_matrix.shape[0] * adj_matrix.shape[1])
        analysis['avg_degree'] = np.mean(np.sum(adj_matrix, axis=1))
        
        # Symmetry
        analysis['is_symmetric'] = np.allclose(adj_matrix, adj_matrix.T)
        
        # Find hub sensors (high degree)
        degrees = np.sum(adj_matrix, axis=1)
        hub_indices = np.argsort(degrees)[-5:][::-1]
        analysis['hub_sensors'] = [self.sensor_names[i] for i in hub_indices]
        analysis['hub_degrees'] = degrees[hub_indices].tolist()
        
        # Connected components (simplified)
        analysis['num_components'] = self._count_components(adj_matrix)
        
        # Clustering coefficient (approximate)
        analysis['clustering_coefficient'] = self._compute_clustering_coefficient(adj_matrix)
        
        return analysis
    
    def _compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity matrix from embeddings."""
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / (norms + 1e-8)
        
        # Compute cosine similarity
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        
        return similarity_matrix
    
    def _find_similar_pairs(self, similarity_matrix: np.ndarray, top_k: int = 10) -> List[Tuple]:
        """Find most similar sensor pairs."""
        pairs = []
        
        for i in range(similarity_matrix.shape[0]):
            for j in range(i + 1, similarity_matrix.shape[1]):
                similarity = similarity_matrix[i, j]
                pairs.append((self.sensor_names[i], self.sensor_names[j], similarity))
        
        # Sort by similarity and return top-k
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[:top_k]
    
    def _cluster_sensors(self, embeddings: np.ndarray, n_clusters: int = 5) -> List[List[str]]:
        """Simple clustering of sensors based on embeddings."""
        try:
            from sklearn.cluster import KMeans
            
            kmeans = KMeans(n_clusters=min(n_clusters, len(embeddings)), random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            clusters = [[] for _ in range(n_clusters)]
            for i, label in enumerate(cluster_labels):
                if label < len(clusters):
                    clusters[label].append(self.sensor_names[i])
            
            # Remove empty clusters
            clusters = [cluster for cluster in clusters if cluster]
            
        except ImportError:
            # Fallback: simple distance-based clustering
            clusters = self._simple_clustering(embeddings)
        
        return clusters
    
    def _simple_clustering(self, embeddings: np.ndarray) -> List[List[str]]:
        """Simple distance-based clustering fallback."""
        # Compute pairwise distances
        distances = np.linalg.norm(embeddings[:, np.newaxis] - embeddings, axis=2)
        
        # Simple clustering based on distance threshold
        threshold = np.percentile(distances[distances > 0], 20)
        
        clusters = []
        used = set()
        
        for i in range(len(embeddings)):
            if i in used:
                continue
            
            cluster = [self.sensor_names[i]]
            used.add(i)
            
            for j in range(i + 1, len(embeddings)):
                if j not in used and distances[i, j] < threshold:
                    cluster.append(self.sensor_names[j])
                    used.add(j)
            
            if len(cluster) > 1:
                clusters.append(cluster)
        
        return clusters
    
    def _compute_embedding_diversity(self, embeddings: np.ndarray) -> float:
        """Compute diversity score of embeddings."""
        # Average pairwise distance
        distances = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                dist = np.linalg.norm(embeddings[i] - embeddings[j])
                distances.append(dist)
        
        return np.mean(distances) if distances else 0.0
    
    def _count_components(self, adj_matrix: np.ndarray) -> int:
        """Count connected components in graph (simplified DFS)."""
        visited = set()
        components = 0
        
        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            for neighbor in range(len(adj_matrix)):
                if adj_matrix[node, neighbor] > 0:
                    dfs(neighbor)
        
        for i in range(len(adj_matrix)):
            if i not in visited:
                dfs(i)
                components += 1
        
        return components
    
    def _compute_clustering_coefficient(self, adj_matrix: np.ndarray) -> float:
        """Compute average clustering coefficient."""
        coefficients = []
        
        for i in range(len(adj_matrix)):
            neighbors = np.where(adj_matrix[i] > 0)[0]
            if len(neighbors) < 2:
                coefficients.append(0.0)
                continue
            
            # Count triangles
            triangles = 0
            for j in range(len(neighbors)):
                for k in range(j + 1, len(neighbors)):
                    if adj_matrix[neighbors[j], neighbors[k]] > 0:
                        triangles += 1
            
            # Clustering coefficient for node i
            possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
            coefficients.append(triangles / possible_edges if possible_edges > 0 else 0.0)
        
        return np.mean(coefficients)
    
    def _analyze_attention_data(self, attention_data: List[Dict]) -> Dict:
        """Analyze extracted attention data."""
        if not attention_data:
            return {}
        
        analysis = {
            'num_batches': len(attention_data),
            'num_layers': 0,
            'avg_entropy': 0.0,
            'sparsity': 0.0,
            'attention_patterns': {}
        }
        
        all_entropies = []
        all_sparsities = []
        
        for batch_data in attention_data:
            attention = batch_data['attention']
            analysis['num_layers'] = max(analysis['num_layers'], len(attention))
            
            for layer_name, weights in attention.items():
                if weights is not None and len(weights) > 0:
                    # Compute entropy
                    flat_weights = weights.flatten()
                    if len(flat_weights) > 0:
                        # Normalize to probabilities
                        probs = np.abs(flat_weights) / (np.sum(np.abs(flat_weights)) + 1e-8)
                        entropy = -np.sum(probs * np.log(probs + 1e-8))
                        all_entropies.append(entropy)
                        
                        # Compute sparsity (fraction of near-zero weights)
                        sparsity = np.sum(np.abs(flat_weights) < 1e-6) / len(flat_weights)
                        all_sparsities.append(sparsity)
        
        if all_entropies:
            analysis['avg_entropy'] = np.mean(all_entropies)
        if all_sparsities:
            analysis['sparsity'] = np.mean(all_sparsities)
        
        return analysis
    
    def _generate_insights(self) -> List[str]:
        """Generate key insights from the model analysis."""
        insights = []
        
        # Architecture insights
        arch = self.architecture_info
        if arch['total_parameters'] > 100000:
            insights.append("Model has high complexity with >100K parameters")
        elif arch['total_parameters'] < 10000:
            insights.append("Model is relatively lightweight with <10K parameters")
        
        # Graph insights
        if self.graph_analysis:
            graph = self.graph_analysis
            if graph.get('density', 0) > 0.1:
                insights.append("Graph is relatively dense, suggesting strong sensor interconnections")
            elif graph.get('density', 0) < 0.05:
                insights.append("Graph is sparse, with selective sensor connections")
            
            if graph.get('clustering_coefficient', 0) > 0.3:
                insights.append("High clustering coefficient indicates modular sensor groups")
        
        # Embedding insights
        if self.embedding_analysis:
            embed = self.embedding_analysis
            if embed['diversity_score'] > 1.0:
                insights.append("High embedding diversity suggests well-separated sensor representations")
            elif embed['diversity_score'] < 0.5:
                insights.append("Low embedding diversity may indicate similar sensor patterns")
            
            if len(embed['clusters']) > self.num_sensors // 3:
                insights.append("Many clusters detected, indicating diverse sensor behaviors")
        
        # Attention insights
        if self.attention_analysis:
            att = self.attention_analysis
            if att.get('sparsity', 0) > 0.8:
                insights.append("High attention sparsity suggests focused, selective attention patterns")
            elif att.get('avg_entropy', 0) > 2.0:
                insights.append("High attention entropy indicates distributed attention across sensors")
        
        if not insights:
            insights.append("Model shows standard characteristics within expected ranges")
        
        return insights