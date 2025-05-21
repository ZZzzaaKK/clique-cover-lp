import time
import logging
from functools import wraps

# Logging setup
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Global stats tracker
reduction_stats = {
    "isolated_vertex": 0,
    "degree_2_folding": 0,
    "twin_removal": 0,
    "twin_folding": 0,
    "domination": 0,
    "crown": 0,
}

def reset_stats():
    for key in reduction_stats:
        reduction_stats[key] = 0

def log_reduction(name, items):
    count = len(items)
    if count > 0:
        reduction_stats[name] += count
        logging.info(f"[{name}] Applied to {count} item(s)")

def timed_step(name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            logging.info(f"[{name}] took {elapsed:.4f} seconds")
            return result
        return wrapper
    return decorator

def print_final_stats():
    logging.info("\nReduction Summary:")
    for name, count in reduction_stats.items():
        logging.info(f"  {name.replace('_', ' ').title()}: {count}")
