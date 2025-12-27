# src/envs/physics.py
"""
Core physics models for wireless communication, based on reference papers.
"""
import numpy as np
from typing import Tuple

# Constants
K_BOLTZMANN = 1.38e-23
T_NOISE = 290  # Kelvin

def db_to_linear(db_val: float) -> float:
    """Converts a value from dB or dBm to linear scale."""
    return 10 ** (db_val / 10.0)

def watts_to_dbm(watts: float) -> float:
    """Converts a value from Watts to dBm."""
    return 10 * np.log10(watts * 1000)

def path_loss_noman2024(d_km: float, is_d2d: bool) -> float:
    """
    Calculates path loss in dB based on Noman et al., 2024 (Table 3).
    Args:
        d_km: Distance in kilometers.
        is_d2d: True for D2D link, False for cellular (MBS) link.
    Returns:
        Path loss in dB.
    """
    if d_km == 0:
        return 0.0
    if is_d2d:
        # D2D path loss from Noman et al., 2024
        return 148.0 + 40.0 * np.log10(d_km)
    else:
        # Cellular (MBS) path loss from Noman et al., 2024
        return 128.1 + 37.6 * np.log10(d_km)

def calculate_noise_watts(noise_dbm_per_hz: float, bandwidth_hz: float) -> float:
    """Calculates total noise power in Watts."""
    print(f"Type of bandwidth_hz: {type(bandwidth_hz)}")
    print(f"Value of bandwidth_hz: {bandwidth_hz}")
    # Convert bandwidth_hz to a float
    bandwidth_hz = float(bandwidth_hz)
    noise_db_per_hz = noise_dbm_per_hz - 30
    total_noise_db = noise_db_per_hz + 10 * np.log10(bandwidth_hz)
    return db_to_linear(total_noise_db)

def calculate_sinr(
    tx_power_watts: float,
    channel_gain_linear: float,
    interference_watts: float,
    noise_watts: float,
) -> float:
    """
    Calculates Signal-to-Interference-plus-Noise Ratio (SINR).
    All inputs must be in linear scale (Watts).
    """
    signal_power = tx_power_watts * channel_gain_linear
    return signal_power / (interference_watts + noise_watts)

def shannon_throughput(sinr_linear: float, bandwidth_hz: float) -> float:
    """
    Calculates throughput using the Shannon-Hartley theorem.
    Returns:
        Throughput in bits per second (bps).
    """
    if sinr_linear < 0:
        return 0.0
    return bandwidth_hz * np.log2(1 + sinr_linear)