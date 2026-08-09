"""Training entry point for CACR-SP VCR."""
import argparse
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    parser = argparse.ArgumentParser(description='Train CACR-SP on VCR')
    parser.add_argument('--config', type=str, default='configs/base.yaml')
    parser.add_argument('--override', type=str, default=None)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    
    # Placeholder for configuration loading
    config = {'experiment': {'name': 'base'}}
    
    print("Training not yet implemented. Use scripts/sanity_check.py first.")
    print(f"Config loaded: experiment={config.get('experiment', {}).get('name', 'unknown')}")

if __name__ == '__main__':
    main()
