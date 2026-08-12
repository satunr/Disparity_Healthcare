import pandas as pd
import numpy as np
import statsmodels.api as sm


def categorical_logistic_betas(df, feature_col, baseline_category):
    """
    Logistic regression for a categorical feature with a fixed baseline.

    Model:

    logit(P(Alive)) =
        beta0 +
        beta1*Category1 +
        beta2*Category2 + ...

    where all coefficients are interpreted relative
    to the baseline_category.

    Parameters
    ----------
    df : pandas.DataFrame
        Clinical dataframe

    feature_col : str
        Name of categorical feature

    baseline_category : str
        Reference category

    Returns
    -------
    results : pandas.DataFrame
        Beta, SE, z-score, p-value, N, odds ratio
    """

    # Remove weight columns
    df = df.loc[
         :,
         ~df.columns.str.contains(
             "weight",
             case=False
         )
         ]

    # Select required columns
    data = df[
        [
            feature_col,
            "VitalStatus"
        ]
    ].dropna()

    # Encode outcome
    y = (
        data["VitalStatus"]
        .astype(str)
        .str.lower()
        .map(
            {
                "alive": 1,
                "dead": 0
            }
        )
    )

    valid = y.notna()

    data = data.loc[valid]
    y = y.loc[valid]

    # Check baseline exists
    if baseline_category not in data[feature_col].unique():
        raise ValueError(
            f"{baseline_category} is not present in {feature_col}"
        )

    # Force baseline ordering
    categories = list(
        data[feature_col].unique()
    )

    categories.remove(
        baseline_category
    )

    categories = [
                     baseline_category
                 ] + categories

    data[feature_col] = pd.Categorical(
        data[feature_col],
        categories=categories
    )

    # One-hot encode
    # baseline is automatically dropped
    X = pd.get_dummies(
        data[feature_col],
        drop_first=True
    )

    # Convert to numeric
    X = X.astype(float)

    y = y.astype(float)

    # Add intercept
    X = sm.add_constant(X)

    # Fit logistic regression
    model = sm.Logit(
        y,
        X
    ).fit(
        disp=False
    )

    # Extract coefficients
    results = pd.DataFrame(
        {
            "Beta": model.params,
            "SE": model.bse,
            "z_score": model.tvalues,
            "p_value": model.pvalues
        }
    )

    # Remove intercept
    results = results.drop(
        "const"
    )

    # Clean category names
    results.index = [
        x.replace(
            feature_col + "_",
            ""
        )
        for x in results.index
    ]

    # Add category counts
    counts = (
        data[feature_col]
        .value_counts()
    )

    results["N"] = counts[
        results.index
    ]

    # Odds ratios
    results["Odds_Ratio"] = np.exp(
        results["Beta"]
    )

    # Add baseline information
    results.attrs["baseline"] = baseline_category

    return results


# Load the imputed dataframe
df = pd.read_excel('./datasets/patients_imputed_renamed.xlsx')
df = df.head(10000).copy()
print(df.head())

results = categorical_logistic_betas(
    df,
    feature_col="SmokingHistory",
    baseline_category="Current"
)
print(results)
