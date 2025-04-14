
import os
# os.chdir('c:\\Users\\Yizhi\\OneDrive\\Research\\Health-Econ\\01_External_Local_RCT')
os.chdir('/schhome/users/yizhilia/projects/local_RCT_VOI')


import pandas as pd
import numpy as np
import math
# !pip install -q lifelines
from lifelines import ExponentialFitter, WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter

hr_pfs = 0.49
hr_pfs_low = 0.38
hr_pfs_high = 0.63
hr_os = 0.60
hr_os_low = 0.45
hr_os_high = 0.79

surv_data_path = "01_data/surv_curve_data"

def read_and_preprocess(filepath):
    try:
        df = pd.read_csv(filepath)
        # Select columns, excluding V7, V8, V9
        cols_to_keep = ["start_time_event", "start_time_censor", "end_time_event", "end_time_censor", "n_events", "n_censors"]
        df = df[cols_to_keep]

        # Exclude rows with NAs
        df = df.dropna()

        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

# Read and preprocess each file
pfs_chemo_df = read_and_preprocess(f"{surv_data_path}/01_pfs_control.csv")
# pfs_combo_df = read_and_preprocess(f"{surv_data_path}01_pfs_combo.csv")
os_chemo_df = read_and_preprocess(f"{surv_data_path}/02_os_control.csv")
# os_combo_df = read_and_preprocess(f"{surv_data_path}02_os_combo.csv")

def reshape_time_data(df, add_extra=False):
    # Cast repeats to int
    df["n_censors"] = df["n_censors"].astype(int)
    df["n_events"]  = df["n_events"].astype(int)

    times_start_censor = np.repeat(df["start_time_censor"].values, df["n_censors"].values)
    times_end_censor   = np.repeat(df["end_time_censor"].values,   df["n_censors"].values)

    times_start_event = np.repeat(df["start_time_event"].values, df["n_events"].values)
    times_end_event   = np.repeat(df["end_time_event"].values,   df["n_events"].values)

    # Concatenate censor + event arrays
    times_start_all = np.concatenate([times_start_censor, times_start_event])
    times_end_all   = np.concatenate([times_end_censor,   times_end_event])

    # If needed, append rows with (18, 10000) repeated 3 times
    if add_extra:
        times_start_extra = np.repeat([18], 3)
        times_end_extra   = np.repeat([10000], 3)
        times_start_all   = np.concatenate([times_start_all, times_start_extra])
        times_end_all     = np.concatenate([times_end_all,   times_end_extra])

    return pd.DataFrame({
        "times_start": times_start_all,
        "times_end":   times_end_all
    })

# For PFS (chemo): add_extra = True
pfs_chemo_for_model = reshape_time_data(pfs_chemo_df, add_extra=True)

# For PFS (combo): add_extra = True
# pfs_combo_for_model = reshape_time_data(pfs_combo_df, add_extra=True)

# For OS (chemo): add_extra = False
os_chemo_for_model  = reshape_time_data(os_chemo_df, add_extra=False)

# For OS (combo): add_extra = False
# os_combo_for_model  = reshape_time_data(os_combo_df, add_extra=False)

def fit_interval_models(df, lower_col="times_start", upper_col="times_end"):
    """
    Fits interval-censored models for:
      - WeibullAFTFitter
      - LogNormalAFTFitter
      - LogLogisticAFTFitter
      - ExponentialFitter   (may not always handle interval-censoring)

    Returns:
      (results_df, fitted_models)
        where results_df has columns [distribution, AIC, BIC, log_likelihood],
        and fitted_models is a dict {distribution_name: fitter_object}.
    """
    results = []
    fitted_models = {}
    n = len(df)  # number of subjects

    def compute_bic(loglik, k, n_obs):
        return k * math.log(n_obs) - 2.0 * loglik

    candidate_models = {
        "weibull":     WeibullAFTFitter,
        "lognormal":   LogNormalAFTFitter,
        "loglogistic": LogLogisticAFTFitter,
        "exponential": ExponentialFitter
    }

    for dist_name, FitterClass in candidate_models.items():
        try:
            fitter = FitterClass()

            # If the fitter supports .fit_interval_censoring(), use it:
            if hasattr(fitter, "fit_interval_censoring"):
                fitter.fit_interval_censoring(
                    df,
                    lower_bound_col=lower_col,
                    upper_bound_col=upper_col
                )
            else:
                # If no interval-censoring method is found, this likely fails for real interval data
                fitter.fit(
                    df[upper_col],
                    event_observed=np.ones(len(df), dtype=int)  # treat as exact
                )

            # Gather metrics
            loglik = fitter.log_likelihood_
            aic_ = fitter.AIC_

            # Determine number of parameters
            if hasattr(fitter, "summary") and fitter.summary is not None:
                k = fitter.summary.shape[0]  # row count in summary
            else:
                k = 1  # minimal fallback if summary not found

            bic_ = compute_bic(loglik, k, n)

            results.append({
                "distribution":   dist_name,
                "AIC":            aic_,
                "BIC":            bic_,
                "log_likelihood": loglik
            })

            fitted_models[dist_name] = fitter

        except Exception as e:
            # Fit failed or not implemented
            results.append({
                "distribution":   dist_name,
                "AIC":            np.nan,
                "BIC":            np.nan,
                "log_likelihood": np.nan
            })
            fitted_models[dist_name] = None

    results_df = pd.DataFrame(results).sort_values("AIC")

    return results_df, fitted_models

def pick_best_distribution_by_aic_bic(results_df):
    df_valid = results_df.dropna(subset=["AIC", "BIC"])
    if df_valid.empty:
        return None
    # We'll pick the distribution that minimizes (AIC + BIC)
    df_valid["aic_plus_bic"] = df_valid["AIC"] + df_valid["BIC"]
    best_idx = df_valid["aic_plus_bic"].idxmin()
    return df_valid.loc[best_idx, "distribution"]

def get_intercept_and_log_scale(fitter):
    """
    Extracts the intercept (mu_), log_scale (sigma_), their variances, and their covariance
    from the fitted model.

    Returns:
        intercept_val: Coefficient for param='mu_' & covariate='Intercept'
        log_scale_val: Coefficient for param='sigma_' & covariate='Intercept'
        intercept_var_val: Variance of intercept (from the covariance matrix)
        log_scale_var_val: Variance of log_scale (from the covariance matrix)
        cov_val: Covariance between intercept and log_scale (from the covariance matrix)
    """
    import numpy as np

    if not hasattr(fitter, "summary") or fitter.summary is None:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # Extract intercept and log_scale from the summary
    df_summary = fitter.summary.reset_index()
    mu_row = df_summary[
        (df_summary["param"] == "mu_") &
        (df_summary["covariate"] == "Intercept")
    ]
    sigma_row = df_summary[
        (df_summary["param"] == "sigma_") &
        (df_summary["covariate"] == "Intercept")
    ]

    # Extract the coefficients
    if len(mu_row) > 0:
        intercept_val = mu_row["coef"].iloc[0]
    else:
        intercept_val = np.nan

    if len(sigma_row) > 0:
        log_scale_val = sigma_row["coef"].iloc[0]
    else:
        log_scale_val = np.nan

    # # Extract variances and covariance from the variance matrix
    # if hasattr(fitter, "variance_matrix_") and fitter.variance_matrix_ is not None:
    #     variance_matrix = fitter.variance_matrix_
    #     intercept_var_val = variance_matrix.iloc[0, 0]  # Variance of intercept
    #     log_scale_var_val = variance_matrix.iloc[1, 1]  # Variance of log_scale
    #     cov_val = variance_matrix.iloc[0, 1]  # Covariance between intercept and log_scale
    # else:
    #     intercept_var_val = np.nan
    #     log_scale_var_val = np.nan
    #     cov_val = np.nan

    return intercept_val, log_scale_val

cases = {
    "pfs_chemo": pfs_chemo_for_model,
    "os_chemo":  os_chemo_for_model
}

# cases = {
#     "pfs_chemo": pfs_chemo_for_model,
#     "pfs_combo": pfs_combo_for_model,
#     "os_chemo":  os_chemo_for_model,
#     "os_combo":  os_combo_for_model
# }

rows_for_df = []

for case_name, df_data in cases.items():
    # Fit your interval-censored distributions
    results_df, fitted_models = fit_interval_models(df_data)

    # Force the best distribution to be 'lognormal'
    best_dist = "lognormal"

    if best_dist in fitted_models and fitted_models[best_dist] is not None:
        best_model = fitted_models[best_dist]
        # Get intercept, log_scale, and covariance matrix details from the best model
        intercept_val, log_scale_val = get_intercept_and_log_scale(best_model)
    else:
        # If something goes wrong, just store None
        intercept_val, log_scale_val = None, None

    # Append a dict row with the results for this case
    rows_for_df.append({
        "case_name": case_name,
        "intercept": intercept_val,
        "log_scale": log_scale_val
    })

# Convert the list of rows to a DataFrame
surv_params = pd.DataFrame(rows_for_df)

surv_params

surv_params_path = "03_output/01_surv_params"
surv_params.to_csv(f"{surv_params_path}/surv_params.csv", index=False)

surv_raw_path = "01_data/surv_curve_data/raw"

def read_raw_points(filepath):
    try:
        # Read the CSV file
        df = pd.read_csv(filepath)

        # Rename columns
        df = df.rename(columns={"x": "month", "y": "surv_prob"})

        # Add the 'cycle' column
        df["cycle"] = df["month"] * (4 / 3)

        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

# Read and preprocess each file
pfs_chemo_raw = read_raw_points(f"{surv_raw_path}/01_PFS_control.csv")
pfs_combo_raw = read_raw_points(f"{surv_raw_path}/01_PFS_experimental.csv")
os_chemo_raw = read_raw_points(f"{surv_raw_path}/02_OS_control.csv")
os_combo_raw = read_raw_points(f"{surv_raw_path}/02_OS_experimental.csv")

from scipy.stats import norm

def simulate_survival(cycle_range, intercept, log_scale, hr=None):
    """
    Simulates survival probabilities for a given range of cycles using the log-normal distribution.
    Optionally adjusts for a hazard ratio (HR) to compute survival probabilities for the experimental group.

    Args:
        cycle_range (np.ndarray): Array of cycle values (e.g., np.arange(0, 201)).
        intercept (float): The intercept (mu) for the log-normal distribution.
        log_scale (float): The log-scale (sigma) for the log-normal distribution.
        hr (float, optional): Hazard ratio for the experimental group. Default is None.

    Returns:
        pd.DataFrame: A DataFrame with columns 'cycle' and 'surv_prob'.
    """
    # Convert cycle to the adjusted scale (cycle * 3 / 4)
    adjusted_cycle = cycle_range * (3 / 4)

    # Compute survival probabilities for the control group
    surv_ctrl = 1 - norm.cdf((np.log(adjusted_cycle + 1e-6) - intercept) / np.exp(log_scale))  # Add small value to avoid log(0)

    # Set the last survival probability to 0
    surv_ctrl[-1] = 0

    # If a hazard ratio is provided, compute survival probabilities for the experimental group
    if hr is not None:
        surv_prob = surv_ctrl ** hr
    else:
        surv_prob = surv_ctrl

    # Set the last survival probability to 0
    surv_prob[-1] = 0

    # Create a DataFrame
    return pd.DataFrame({
        "cycle": cycle_range,
        "surv_prob": surv_prob
    })

case_params = {
    row["case_name"]: {
        "intercept": row["intercept"],
        "log_scale": row["log_scale"]
    }
    for _, row in surv_params.iterrows()
}

# Define the cycle range
max_range = 200
cycle_range = np.arange(0, max_range + 1)  # 0 to 200 inclusive

# Simulate survival probabilities for each case
pfs_chemo_simu = simulate_survival(
    cycle_range,
    intercept=case_params["pfs_chemo"]["intercept"],
    log_scale=case_params["pfs_chemo"]["log_scale"]
)

pfs_combo_simu = simulate_survival(
    cycle_range,
    intercept=case_params["pfs_chemo"]["intercept"],
    log_scale=case_params["pfs_chemo"]["log_scale"],
    hr=hr_pfs
)

os_chemo_simu = simulate_survival(
    cycle_range,
    intercept=case_params["os_chemo"]["intercept"],
    log_scale=case_params["os_chemo"]["log_scale"]
)

os_combo_simu = simulate_survival(
    cycle_range,
    intercept=case_params["os_chemo"]["intercept"],
    log_scale=case_params["os_chemo"]["log_scale"],
    hr=hr_os
)

case_params

# !pip install -q lets-plot
from lets_plot import *
LetsPlot.setup_html()

# Colors for lines
colors = {
    "combo": "#434343",
    "chemo": "#8c8c8c"
}

# Relabeled legend names
legend_labels = {
    "chemo": "Chemotherapy",
    "combo": "Sintilimab + Chemotherapy"
}

# Add font family configuration
times_new_roman_theme = theme(
    axis_text_x=element_text(size=10, family="Times New Roman"),
    axis_text_y=element_text(size=10, family="Times New Roman"),
    legend_text=element_text(size=12, family="Times New Roman"),
    legend_position="bottom",  # Move legend to the bottom
    legend_title=element_blank(),  # Remove color legend title
    plot_title=element_text(size=14, family="Times New Roman"),
    plot_subtitle=element_text(size=12, family="Times New Roman"),
    axis_title_x=element_text(size=12, family="Times New Roman"),
    axis_title_y=element_text(size=12, family="Times New Roman")
)

# PFS plot
pfs_chemo_raw["source"] = "raw"
pfs_chemo_simu["source"] = "simu"
pfs_chemo_raw["group"] = "chemo"
pfs_chemo_simu["group"] = "chemo"

pfs_combo_raw["source"] = "raw"
pfs_combo_simu["source"] = "simu"
pfs_combo_raw["group"] = "combo"
pfs_combo_simu["group"] = "combo"

pfs_data = pd.concat([pfs_chemo_raw, pfs_chemo_simu, pfs_combo_raw, pfs_combo_simu])

pfs_plot = (ggplot(pfs_data, aes("cycle", "surv_prob", color="group")) +
    geom_line(data=pfs_data[pfs_data["source"] == "simu"], size=1.2) +
    geom_point(data=pfs_data[pfs_data["source"] == "raw"], size=2) +
    scale_color_manual(values=colors, breaks=list(legend_labels.keys()), labels=list(legend_labels.values())) +
    labs(
        x = "Treatment Cycle",
        y = "Survival Probability",
        title = "Progression-Free Survival Curves (PFS)",
        subtitle="Simulated curves with log-normal distribution vs. True points"
    ) +
    theme_classic() +
    times_new_roman_theme +
    labs(color="")
)

# OS plot
os_chemo_raw["source"] = "raw"
os_chemo_simu["source"] = "simu"
os_chemo_raw["group"] = "chemo"
os_chemo_simu["group"] = "chemo"

os_combo_raw["source"] = "raw"
os_combo_simu["source"] = "simu"
os_combo_raw["group"] = "combo"
os_combo_simu["group"] = "combo"

os_data = pd.concat([os_chemo_raw, os_chemo_simu, os_combo_raw, os_combo_simu])

os_plot = (ggplot(os_data, aes("cycle", "surv_prob", color="group")) +
    geom_line(data=os_data[os_data["source"] == "simu"], size=1.2) +
    geom_point(data=os_data[os_data["source"] == "raw"], size=2) +
    scale_color_manual(values=colors, breaks=list(legend_labels.keys()), labels=list(legend_labels.values())) +
    labs(
        x = "Treatment Cycle",
        y = "Survival Probability",
        title = "Overall Survival Curves (OS)",
        subtitle="Simulated curves with log-normal distribution vs. True points"
    ) +
    theme_classic() +
    times_new_roman_theme +
    labs(color="")
)

combined_plot = gggrid([os_plot, pfs_plot], ncol=2)
combined_plot

# !pip install -q CairoSVG
ggsave(plot=os_plot, filename=f"{surv_params_path}/01_OS_plot.pdf", dpi=1500, w=4*4, h=3*4, unit='in')
ggsave(plot=pfs_plot, filename=f"{surv_params_path}/02_PFS_plot.pdf", dpi=1500, w=4*4, h=3*4, unit='in')

ggsave(plot=os_plot, filename=f"{surv_params_path}/01_OS_plot.svg", dpi=1500, w=4*4, h=3*4, unit='in')
ggsave(plot=pfs_plot, filename=f"{surv_params_path}/02_PFS_plot.svg", dpi=1500, w=4*4, h=3*4, unit='in')