"""
Estimate scan temperature from a temperature log and the corresponding water diffusivity.
"""

import numpy as np
from datetime import datetime, timedelta


def estimate_temp(scan_dt: datetime, temp_log, time_window_minutes=80, f_past = 1, n_min=None):
    """
    Estimate the temperature of the phantom by taking the mean of temperature readings within a window of time of 
    the scan time. The window is defined by the time_window_minutes and f_past.

    :param scan_dt: datatime object of the scan time of interest
    :param temp_log: list of tuples of (datatime object, temperature reading)
    :param time_window_minutes: duration of time window from which to average temperature readings (minutes)
    :param f_past: fraction of the time window to place with respect to the past of the scan time. """

    if not isinstance(scan_dt, datetime):
        raise ValueError("scan_dt must be a datetime object")

    if not all(isinstance(t_dt, datetime) for t_dt, _ in temp_log) or not all(isinstance(temp, float) for _, temp in temp_log):
        raise ValueError("temp_log must be a list of entries with type (datetime, float)")
    
    MEASUREMENT_INTERVAL = 5 # minutes

    if n_min is None:
        n_min = (time_window_minutes // MEASUREMENT_INTERVAL) - 1

    past_delta = timedelta(minutes=time_window_minutes * f_past)
    future_delta = timedelta(minutes=time_window_minutes * (1 - f_past))
    window_temp_data = [temp for t_dt, temp in temp_log if scan_dt - past_delta <= t_dt <= scan_dt + future_delta]

    if not window_temp_data:
        return None
    elif len(window_temp_data) < n_min:
        return None
    else:
        return float(np.mean(window_temp_data))


def water_diffusivity(temp, polynomial=True):
    
    """
    Calculate the diffusivity of water for the temperature provided to the function.
    :param temp: temperature in degrees Celsius
    :param polynomial: If True, use a second-order polynomial fit to the diffusivity data. If False, use a linear fit.
    
    Returns:
        diffusivity in units: mm2/s
    """

    if polynomial:
        D_um2_ms = 0.0004 * temp **2 + 0.0374 * temp + 1.1222
    else:
        D_um2_ms = 0.0587 * temp + 0.9084
    D_mm2_s = D_um2_ms * 1e-3
    
    return D_mm2_s
