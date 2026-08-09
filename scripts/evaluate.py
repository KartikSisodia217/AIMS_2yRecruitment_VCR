"""Evaluation entry point for CACR-SP VCR."""
import argparse
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    parser = argparse.ArgumentParser(description='Evaluate CACR-SP on VCR')
    parser.add_argument('--config', type=str, default='configs/base.yaml')
    parser.add_argument('--ckpt', type=str, required=False)
    args = parser.parse_args()
    
    print("Evaluation not yet implemented. Use scripts/sanity_check.py first.")

if __name__ == '__main__':
    main()
