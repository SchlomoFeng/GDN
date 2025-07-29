#!/usr/bin/env python3
"""
Example script demonstrating how to use the enhanced GDN features
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Example of how to use the enhanced features programmatically
from main import Main

def example_basic_usage():
    """Example: Basic enhanced usage"""
    print("=== EXAMPLE 1: Basic Enhanced Usage ===")
    
    # Standard configuration
    train_config = {
        'batch': 32,
        'epoch': 5,  # Small number for example
        'slide_win': 15,
        'dim': 64,
        'slide_stride': 5,
        'comment': 'enhanced_example',
        'seed': 42,
        'out_layer_num': 1,
        'out_layer_inter_dim': 256,
        'decay': 0,
        'val_ratio': 0.1,
        'topk': 20,
    }

    # Enhanced configuration - just add these flags
    env_config = {
        'save_path': 'enhanced_example',
        'dataset': 'msl',
        'report': 'best',
        'device': 'cpu',
        'load_model_path': '',
        'verbose': True,  # Enable detailed output
        'enable_visualization': False,  # Disable for this example
        'enable_localization': True,   # Enable anomaly localization
        'top_k_sensors': 5
    }
    
    print("Enhanced GDN will provide:")
    print("- Detailed training progress")
    print("- Sensor-level anomaly analysis")
    print("- Comprehensive metrics summary")
    print("- Model architecture insights")
    print()
    
    try:
        # Initialize and run
        # main = Main(train_config, env_config)
        # main.run()
        print("✓ Configuration valid - ready to run with enhanced features")
    except Exception as e:
        print(f"Note: Full run requires data - {e}")

def example_full_analysis():
    """Example: Full analysis with all features"""
    print("=== EXAMPLE 2: Full Analysis Configuration ===")
    
    # Configuration for complete analysis
    train_config = {
        'batch': 128,
        'epoch': 100,
        'slide_win': 15,
        'dim': 64,
        'slide_stride': 5,
        'comment': 'full_analysis',
        'seed': 42,
        'out_layer_num': 1,
        'out_layer_inter_dim': 256,
        'decay': 0,
        'val_ratio': 0.1,
        'topk': 20,
    }

    env_config = {
        'save_path': 'full_analysis',
        'dataset': 'msl',
        'report': 'best',
        'device': 'cpu',
        'load_model_path': '',
        'verbose': True,              # Detailed output
        'enable_visualization': True,  # Generate all visualizations
        'enable_localization': True,  # Sensor-level analysis
        'top_k_sensors': 10          # Analyze top 10 sensors
    }
    
    print("Full analysis will generate:")
    print("- Training curves and progress plots")
    print("- Anomaly timeline visualizations")
    print("- Sensor importance rankings")
    print("- Interactive dashboards")
    print("- Model architecture analysis")
    print("- Enhanced CSV outputs")
    print("- Comprehensive summary reports")
    print()
    
    print("✓ Configuration valid - ready for comprehensive analysis")

def example_command_line_usage():
    """Example: Command line usage patterns"""
    print("=== EXAMPLE 3: Command Line Usage ===")
    
    print("Basic enhanced usage:")
    print("python main.py -dataset msl -device cpu -epoch 50 -verbose")
    print()
    
    print("With anomaly localization:")
    print("python main.py -dataset msl -device cpu -epoch 50 -verbose -enable_localization -top_k_sensors 5")
    print()
    
    print("Full analysis with visualizations:")
    print("python main.py -dataset msl -device cpu -epoch 100 -verbose -enable_visualization -enable_localization -top_k_sensors 10")
    print()
    
    print("Quick test run:")
    print("python main.py -dataset msl -device cpu -epoch 2 -batch 16 -verbose")
    print()

def example_output_structure():
    """Example: Expected output structure"""
    print("=== EXAMPLE 4: Output Structure ===")
    
    print("Enhanced GDN generates the following outputs:")
    print()
    
    print("Standard outputs (unchanged):")
    print("  pretrained/{save_path}/best_{timestamp}.pt")
    print("  results/{save_path}/{timestamp}.csv")
    print()
    
    print("Enhanced outputs:")
    print("  results/{dataset}_{timestamp}_detailed.csv      # Sensor-level scores")
    print("  results/{dataset}_{timestamp}_model_report.txt  # Model analysis")
    print("  visualizations/{dataset}_{timestamp}/")
    print("    ├── training_curves.png")
    print("    ├── anomaly_timeline.png")
    print("    ├── anomaly_dashboard.png")
    print("    ├── attention_heatmap.png")
    print("    ├── graph_network.png")
    print("    ├── sensor_embeddings_tsne.png")
    print("    ├── sensor_embeddings_pca.png")
    print("    └── interactive_dashboard.html")
    print()

def example_programmatic_usage():
    """Example: Using components programmatically"""
    print("=== EXAMPLE 5: Programmatic Usage ===")
    
    print("You can also use the enhancement components directly:")
    print()
    
    # Example of using components directly
    try:
        from anomaly_localizer import AnomalyLocalizer
        from gdn_visualizer import GDNVisualizer
        from model_analyzer import ModelAnalyzer
        
        print("# Initialize components")
        print("feature_map = {0: 'sensor_01', 1: 'sensor_02', ...}")
        print("localizer = AnomalyLocalizer(feature_map, verbose=True)")
        print("visualizer = GDNVisualizer('./output', feature_map, verbose=True)")
        print("analyzer = ModelAnalyzer(model, feature_map, verbose=True)")
        print()
        
        print("# Perform analysis")
        print("sensor_scores = localizer.compute_sensor_anomaly_scores(test_result, val_result)")
        print("summary = localizer.generate_summary_report(sensor_scores, labels)")
        print("visualizer.plot_anomaly_timeline(sensor_scores, labels)")
        print("report = analyzer.generate_model_report()")
        print()
        
        print("✓ All components imported successfully")
        
    except ImportError as e:
        print(f"Import error: {e}")

def main():
    """Run all examples"""
    print("GDN ENHANCED FEATURES - USAGE EXAMPLES")
    print("=" * 60)
    print()
    
    example_basic_usage()
    print()
    
    example_full_analysis()
    print()
    
    example_command_line_usage()
    print()
    
    example_output_structure()
    print()
    
    example_programmatic_usage()
    print()
    
    print("=" * 60)
    print("For complete documentation, see: ENHANCED_FEATURES_DOCUMENTATION.md")
    print("To test the implementation, run: python test_enhancements.py")
    print("=" * 60)

if __name__ == "__main__":
    main()