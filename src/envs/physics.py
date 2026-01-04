# src/envs/physics.py
"""
Core physics models for wireless communication, based on reference papers.
Optimized for Vectorized Operations (NumPy Broadcasting).
"""
import numpy as np

def db_to_linear(db_val):
    """Converts a value from dB or dBm to linear scale. Supports arrays."""
    return 10.0 ** (np.asanyarray(db_val) / 10.0)

def watts_to_dbm(watts):
    """Converts a value from Watts to dBm. Supports arrays."""
    return 10.0 * np.log10(np.maximum(watts, 1e-12) * 1000.0) # Protect against log(0)

def path_loss_noman2024(d_km, is_d2d: bool):
    """
    Calculates path loss in dB. Supports NumPy arrays.
    Handles d=0 safely without crushing the whole array.
    """
    d_km = np.asanyarray(d_km)
    
    # Avoid log10(0) by clipping strict zeros to a tiny distance (e.g., 1 meter)
    # 1e-3 km = 1 meter. This is physically realistic minimum distance.
    d_safe = np.maximum(d_km, 1e-4) 

    if is_d2d:
        pl = 148.0 + 40.0 * np.log10(d_safe)
    else:
        pl = 128.1 + 37.6 * np.log10(d_safe)
        
    # If original distance was exactly 0, theoretical path loss is 0 (no loss)
    # apply mask to set PL=0 where d=0 strictly
    if np.ndim(d_km) > 0:
        pl[d_km == 0] = 0.0
    elif d_km == 0:
        return 0.0
        
    return pl

def calculate_noise_watts(noise_dbm_per_hz: float, bandwidth_hz: float) -> float:
    """Calculates total noise power in Watts."""
    bandwidth_hz = float(bandwidth_hz)
    noise_db_per_hz = noise_dbm_per_hz - 30
    total_noise_db = noise_db_per_hz + 10 * np.log10(bandwidth_hz)
    return float(db_to_linear(total_noise_db)) # Return float, noise is usually scalar global

def calculate_sinr(
    tx_power_watts,
    channel_gain_linear,
    interference_watts,
    noise_watts,
):
    """
    Calculates SINR. Inputs can be arrays.
    """
    signal_power = tx_power_watts * channel_gain_linear
    return signal_power / (interference_watts + noise_watts)

def shannon_throughput(sinr_linear, bandwidth_hz: float):
    """
    Calculates throughput using Shannon-Hartley. Supports arrays.
    """
    sinr_linear = np.asanyarray(sinr_linear)
    # Ensure SINR is non-negative (physics dictates this, but good for safety)
    sinr_safe = np.maximum(sinr_linear, 0.0)
    return bandwidth_hz * np.log2(1 + sinr_safe)