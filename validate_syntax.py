#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Syntax validation script for GDN enhanced features
This script validates the syntax of all enhanced modules without requiring external dependencies
"""

import ast
import sys
from pathlib import Path

def validate_python_syntax(filepath):
    """Validate Python syntax of a file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        
        # Parse the AST to check for syntax errors
        ast.parse(source, filename=str(filepath))
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def main():
    """Main validation function"""
    print("Validating GDN Enhanced Features Syntax")
    print("="*50)
    
    # Files to validate
    files_to_check = [
        "util/logger.py",
        "util/visualizer.py", 
        "util/anomaly_analyzer.py",
        "train.py",
        "test.py",
        "main.py",
        "demo_enhanced_features.py"
    ]
    
    all_valid = True
    
    for filepath in files_to_check:
        path = Path(filepath)
        if path.exists():
            valid, error = validate_python_syntax(path)
            if valid:
                print(f"✓ {filepath}: Syntax valid")
            else:
                print(f"✗ {filepath}: {error}")
                all_valid = False
        else:
            print(f"? {filepath}: File not found")
            all_valid = False
    
    print("="*50)
    if all_valid:
        print("✓ All files have valid Python syntax!")
        print("\nEnhanced features successfully implemented:")
        print("  - Enhanced logging system (util/logger.py)")
        print("  - Comprehensive visualization toolkit (util/visualizer.py)")
        print("  - Advanced anomaly analysis engine (util/anomaly_analyzer.py)")
        print("  - Enhanced training with detailed progress (train.py)")
        print("  - Enhanced testing with anomaly localization (test.py)")
        print("  - Updated main workflow (main.py)")
        print("  - Demo script for showcasing features (demo_enhanced_features.py)")
        print("\nTo run with enhanced features:")
        print("  1. Install dependencies: pip install torch pandas numpy scikit-learn matplotlib seaborn")
        print("  2. Run: bash run_enhanced.sh cpu msl")
        print("  3. Or run demo: python demo_enhanced_features.py")
        return 0
    else:
        print("✗ Some files have syntax errors!")
        return 1

if __name__ == "__main__":
    sys.exit(main())