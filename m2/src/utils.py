import numpy as np
from scipy.stats import chi2

chi2_95_1d = chi2.ppf(0.95, df=1)
chi2_95_2d = chi2.ppf(0.95, df=2)
chi2_95_4d = chi2.ppf(0.95, df=4)


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]

    Args:
        angle: angle in radians
    Returns:
        normalized angle in radians
    """
    new_angle = (angle + np.pi) % (2 * np.pi) - np.pi
    return new_angle
