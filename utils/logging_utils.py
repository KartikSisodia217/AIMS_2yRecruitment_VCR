"""Logging utilities for tracking experiments and metrics."""

import os
import logging
from typing import Dict, Optional

def setup_logging(experiment_name: str, output_dir: str, level: str = "INFO") -> logging.Logger:
    """
    Setup logging to both console and a file in the output directory.
    
    Args:
        experiment_name (str): Name of the experiment.
        output_dir (str): Directory to save logs.
        level (str): Logging level (e.g., 'INFO', 'DEBUG').
        
    Returns:
        logging.Logger: The configured logger.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"{experiment_name}.log")
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # File handler
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(getattr(logging, level.upper(), logging.INFO))
    fh.setFormatter(formatter)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))
    ch.setFormatter(formatter)
    
    # Add handlers (avoid duplicate handlers if setup is called multiple times)
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)
        
    return logging.getLogger(__name__)


class MetricTracker:
    """Tracks running averages of named metrics."""
    
    def __init__(self):
        self.reset()
        
    def reset(self) -> None:
        """Reset all tracked metrics."""
        self._sums = {}
        self._counts = {}
        
    def update(self, name: str, value: float, n: int = 1) -> None:
        """
        Update a metric with a new value.
        
        Args:
            name (str): The name of the metric.
            value (float): The value to add.
            n (int): The number of samples this value represents.
        """
        if name not in self._sums:
            self._sums[name] = 0.0
            self._counts[name] = 0
            
        self._sums[name] += value * n
        self._counts[name] += n
        
    def avg(self, name: str) -> Optional[float]:
        """
        Get the average value of a metric.
        
        Args:
            name (str): The name of the metric.
            
        Returns:
            Optional[float]: The average value, or None if the metric hasn't been tracked.
        """
        if name not in self._sums or self._counts[name] == 0:
            return None
        return self._sums[name] / self._counts[name]
        
    def summary(self) -> Dict[str, float]:
        """
        Get a summary dictionary of all averages.
        
        Returns:
            Dict[str, float]: Dictionary mapping metric names to their averages.
        """
        return {name: self.avg(name) for name in self._sums}
