"""
Utility functions for calculating performance metrics for wireless networks.
"""
import numpy as np
from typing import List

def calculate_jains_fairness(throughputs: List[float]) -> float:
    """
    Calculates Jain's Fairness Index for a list of user throughputs.

    A value of 1 indicates perfect fairness.
    A value of 1/N indicates the worst-case scenario (N=number of users).
    """
    if not throughputs:
        return 0.0

    th_array = np.array(throughputs)
    if np.any(th_array < 0):
        raise ValueError("Throughputs cannot be negative.")
        
    sum_sq = np.sum(th_array) ** 2
    sq_sum = np.sum(th_array ** 2)

    if sq_sum == 0:
        return 1.0 # Perfect fairness if all throughputs are zero
        
    return sum_sq / (len(th_array) * sq_sum)

def calculate_energy_efficiency(total_throughput_bps: float, total_power_watts: float) -> float:
    """
    Calculates energy efficiency in bits per Joule.
    """
    if total_power_watts <= 0:
        return 0.0
    return total_throughput_bps / total_power_watts