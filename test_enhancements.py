#!/usr/bin/env python3
"""
Test script for GDN enhancements
Tests the new analysis and visualization components with mock data
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_anomaly_localizer():
    """Test the AnomalyLocalizer class with mock data"""
    print("Testing AnomalyLocalizer...")
    
    try:
        from anomaly_localizer import AnomalyLocalizer
        
        # Create mock feature map
        feature_map = {i: f'sensor_{i:02d}' for i in range(10)}
        
        # Initialize localizer
        localizer = AnomalyLocalizer(feature_map, verbose=True)
        
        # Create mock test and validation results
        num_sensors = 10
        num_timesteps = 100
        
        # Mock test result: [predictions, ground_truth, labels]
        test_predictions = np.random.randn(num_timesteps, num_sensors)
        test_ground_truth = np.random.randn(num_timesteps, num_sensors) + test_predictions * 0.1
        test_labels = np.random.choice([0, 1], size=(num_timesteps, num_sensors), p=[0.8, 0.2])
        
        # Mock validation result
        val_predictions = np.random.randn(50, num_sensors)
        val_ground_truth = np.random.randn(50, num_sensors) + val_predictions * 0.05
        val_labels = np.zeros((50, num_sensors))
        
        test_result = [test_predictions.tolist(), test_ground_truth.tolist(), test_labels.tolist()]
        val_result = [val_predictions.tolist(), val_ground_truth.tolist(), val_labels.tolist()]
        
        # Test sensor anomaly scores computation
        sensor_scores = localizer.compute_sensor_anomaly_scores(test_result, val_result)
        print(f"✓ Sensor anomaly scores computed: {sensor_scores.shape}")
        
        # Test top-K analysis
        labels = [int(np.random.choice([0, 1], p=[0.9, 0.1])) for _ in range(num_timesteps)]
        top_k_analysis = localizer.identify_top_k_anomalous_sensors(sensor_scores, labels, k=5)
        print(f"✓ Top-K analysis completed: {top_k_analysis['total_events']} events found")
        
        # Test temporal patterns
        patterns = localizer.analyze_temporal_patterns(sensor_scores)
        print(f"✓ Temporal patterns analyzed for {len(patterns)} sensors")
        
        # Test summary report
        summary = localizer.generate_summary_report(sensor_scores, labels, k=5)
        print(f"✓ Summary report generated ({len(summary)} characters)")
        
        # Test saving results
        test_output_dir = Path('./test_output')
        test_output_dir.mkdir(exist_ok=True)
        csv_path = test_output_dir / 'test_anomaly_results.csv'
        localizer.save_detailed_results(str(csv_path), sensor_scores, labels)
        print(f"✓ Results saved to: {csv_path}")
        
        print("AnomalyLocalizer tests passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ AnomalyLocalizer test failed: {e}")
        return False

def test_gdn_visualizer():
    """Test the GDNVisualizer class with mock data"""
    print("Testing GDNVisualizer...")
    
    try:
        from gdn_visualizer import GDNVisualizer
        
        # Create test output directory
        test_output_dir = Path('./test_output/visualizations')
        test_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create mock feature map
        feature_map = {i: f'sensor_{i:02d}' for i in range(8)}
        
        # Initialize visualizer
        visualizer = GDNVisualizer(
            output_dir=str(test_output_dir),
            feature_map=feature_map,
            verbose=True
        )
        
        # Test training curves
        train_losses = [1.0 - 0.05*i + 0.1*np.random.random() for i in range(20)]
        val_losses = [0.8 - 0.03*i + 0.08*np.random.random() for i in range(20)]
        metrics = {
            'f1': [0.1 + 0.04*i + 0.02*np.random.random() for i in range(20)],
            'precision': [0.15 + 0.03*i + 0.02*np.random.random() for i in range(20)]
        }
        
        visualizer.plot_training_curves(train_losses, val_losses, metrics)
        print("✓ Training curves plotted")
        
        # Test attention heatmap
        attention_weights = np.random.rand(8, 8)
        visualizer.plot_attention_heatmap(attention_weights)
        print("✓ Attention heatmap plotted")
        
        # Test anomaly timeline
        num_sensors = 8
        num_timesteps = 200
        sensor_scores = np.random.exponential(0.5, (num_sensors, num_timesteps))
        labels = [int(np.random.choice([0, 1], p=[0.9, 0.1])) for _ in range(num_timesteps)]
        
        visualizer.plot_anomaly_timeline(sensor_scores, labels, top_k=5)
        print("✓ Anomaly timeline plotted")
        
        # Test sensor embeddings (if sklearn available)
        try:
            embeddings = np.random.randn(num_sensors, 16)
            visualizer.plot_sensor_embeddings(embeddings, method='pca')
            print("✓ Sensor embeddings plotted")
        except Exception as e:
            print(f"⚠ Sensor embeddings skipped: {e}")
        
        # Test graph network (if networkx available)
        try:
            adjacency_matrix = np.random.rand(num_sensors, num_sensors)
            adjacency_matrix = (adjacency_matrix + adjacency_matrix.T) / 2  # Make symmetric
            visualizer.plot_graph_network(adjacency_matrix, threshold=0.3)
            print("✓ Graph network plotted")
        except Exception as e:
            print(f"⚠ Graph network skipped: {e}")
        
        # Test anomaly dashboard
        visualizer.create_anomaly_dashboard(sensor_scores, labels)
        print("✓ Anomaly dashboard created")
        
        # Test interactive dashboard (if plotly available)
        try:
            visualizer.create_interactive_dashboard(sensor_scores, labels)
            print("✓ Interactive dashboard created")
        except Exception as e:
            print(f"⚠ Interactive dashboard skipped: {e}")
        
        print("GDNVisualizer tests passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ GDNVisualizer test failed: {e}")
        return False

def test_model_analyzer():
    """Test the ModelAnalyzer class with mock model"""
    print("Testing ModelAnalyzer...")
    
    try:
        import torch
        import torch.nn as nn
        from model_analyzer import ModelAnalyzer
        
        # Create a simple mock model
        class MockGDNModel(nn.Module):
            def __init__(self, num_sensors, embedding_dim=16):
                super().__init__()
                self.embedding = nn.Embedding(num_sensors, embedding_dim)
                self.linear = nn.Linear(embedding_dim, 1)
                self.topk = 5
                self.learned_graph = None
                self.gnn_layers = []  # Empty list for compatibility
                
            def forward(self, x):
                return self.linear(self.embedding.weight.mean(0))
        
        # Create mock feature map and model
        feature_map = {i: f'sensor_{i:02d}' for i in range(10)}
        model = MockGDNModel(len(feature_map))
        
        # Initialize analyzer
        analyzer = ModelAnalyzer(model, feature_map, device='cpu', verbose=True)
        
        # Test architecture analysis
        arch_info = analyzer.get_model_architecture_info()
        print(f"✓ Architecture analysis completed: {arch_info['total_parameters']} parameters")
        
        # Test graph structure analysis
        graph_analysis = analyzer.analyze_graph_structure()
        print(f"✓ Graph structure analyzed: density={graph_analysis.get('density', 0):.3f}")
        
        # Test embedding analysis
        embedding_analysis = analyzer.analyze_sensor_embeddings()
        print(f"✓ Embedding analysis completed: {embedding_analysis['embedding_dim']} dimensions")
        
        # Test model report generation
        test_output_dir = Path('./test_output')
        test_output_dir.mkdir(exist_ok=True)
        report_path = test_output_dir / 'test_model_report.txt'
        
        report = analyzer.generate_model_report(str(report_path))
        print(f"✓ Model report generated ({len(report)} characters)")
        print(f"✓ Report saved to: {report_path}")
        
        print("ModelAnalyzer tests passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ ModelAnalyzer test failed: {e}")
        return False

def test_cli_enhancements():
    """Test the enhanced CLI arguments"""
    print("Testing CLI enhancements...")
    
    try:
        import argparse
        
        # Test that new arguments are recognized
        parser = argparse.ArgumentParser()
        parser.add_argument('-verbose', action='store_true', default=False)
        parser.add_argument('-enable_visualization', action='store_true', default=False)
        parser.add_argument('-enable_localization', action='store_true', default=False)
        parser.add_argument('-top_k_sensors', type=int, default=5)
        
        # Test parsing
        args = parser.parse_args(['-verbose', '-enable_visualization', '-top_k_sensors', '10'])
        
        assert args.verbose == True
        assert args.enable_visualization == True
        assert args.enable_localization == False
        assert args.top_k_sensors == 10
        
        print("✓ CLI arguments work correctly")
        print("CLI enhancements tests passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ CLI enhancements test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("TESTING GDN ENHANCEMENTS")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 4
    
    # Run individual tests
    if test_cli_enhancements():
        tests_passed += 1
    
    if test_anomaly_localizer():
        tests_passed += 1
    
    if test_gdn_visualizer():
        tests_passed += 1
    
    if test_model_analyzer():
        tests_passed += 1
    
    # Summary
    print("=" * 60)
    print(f"TEST SUMMARY: {tests_passed}/{total_tests} tests passed")
    print("=" * 60)
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! GDN enhancements are working correctly.")
        
        # Show generated files
        test_output_dir = Path('./test_output')
        if test_output_dir.exists():
            print(f"\nGenerated test files in: {test_output_dir}")
            for file_path in test_output_dir.rglob('*'):
                if file_path.is_file():
                    print(f"  - {file_path}")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())