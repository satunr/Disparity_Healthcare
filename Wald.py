import numpy as np
from scipy.stats import norm


def wald_test_or(OR_pre, OR_post, SE_pre, SE_post):
    """
    Wald test comparing odds ratios across two timepoints.

    Parameters
    ----------
    OR_pre : float
        Odds ratio before timepoint.
    OR_post : float
        Odds ratio after timepoint.
    SE_pre : float
        Standard error of beta_pre (NOT OR_pre).
    SE_post : float
        Standard error of beta_post (NOT OR_post).

    Returns
    -------
    Z : float
        Wald statistic.
    p_value : float
        Two-sided p-value.
    """

    beta_pre = np.log(OR_pre)
    beta_post = np.log(OR_post)

    SE_diff = np.sqrt(SE_pre**2 + SE_post**2)

    Z = (beta_pre - beta_post) / SE_diff

    p_value = 2 * (1 - norm.cdf(abs(Z)))

    return Z, p_value


OR_pre = 1.5
OR_post = 4.0

SE_pre = 0.15
SE_post = 0.15

Z, p_value = wald_test_or(OR_pre, OR_post, SE_pre, SE_post)
print(Z, p_value)

