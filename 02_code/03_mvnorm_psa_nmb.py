
import os
os.chdir('c:\\Users\\Yizhi\\OneDrive\\Research\\Health-Econ\\01_External_Local_RCT')

import pandas as pd
import numpy as np
from scipy.stats import norm
# import pprint

# A set of fixed input
a_value = 1.0
n_psa = 20000
seed = 2025

sample_size_now = 397
sample_size_new = 616

wtp = 150000

max_cycle = 200
# Define the cycle range
cycle_range = np.arange(0, max_cycle + 1)  # 0 to 200 inclusive

def get_discount_factor(cycle_range, dr=0.03, cycles_per_year=16):
    """
    Calculates the discount factor for each cycle based on the discount rate.

    Args:
        cycle_range (array-like): The range of cycles (e.g., np.arange(0, 201)).
        dr (float): The yearly discount rate (default is 0.03, or 3%).
        cycles_per_year (int): The number of cycles in a year (default is 16).

    Returns:
        np.ndarray: An array of discount factors corresponding to the cycle range.
    """
    # Convert yearly discount rate to per-cycle discount rate
    discount_rate_cycle = (1 + dr) ** (1 / cycles_per_year) - 1

    # Calculate the discount factor for each cycle
    discount_factor = 1 / (1 + discount_rate_cycle) ** cycle_range

    return discount_factor

discount_factor = get_discount_factor(cycle_range)

# Hazard ratio for Combo versus Chemo
hr_params = {
    "pfs": {"hr": 0.49, "low": 0.38, "high": 0.63},
    "os": {"hr": 0.60, "low": 0.45, "high": 0.79}
}

# Define the utility dictionary
utility = {
    "stable": 0.75,  # Utility for stable state
    "prog": 0.59     # Utility for progression state
}

# Define the cost dictionary
cost = {
    # Monitoring costs
    "monitoring_stable": 464.85 * 3,
    "monitoring_prog": 1075.49 * 3,

    # Parameters for drug costs
    "surface": 1.86,
    "admin_first": 158.7,
    "admin_sub": 33.6,
    "terminal": 16441.83,
    "price_premetrxed": 7.49,
    "price_cisplatin": 0.18,
    "price_carboplatin": 0.05,
    "dose_premetrxed": 500,
    "dose_cisplatin": 75,
    "dose_carboplatin": 550,
    "dose_drug": 200,

    # Calculated costs
    "intro_cost": (
        1.86 * 500 * 7.49 +
        0.277 * 1.86 * 75 * 0.18 +
        (1 - 0.277) * 550 * 0.05
    ),
    "maintain_cost_chemo": 1.86 * 500 * 7.49
}

value_based_price_loc = "03_output/02_value_price"
value_based_price = pd.read_csv(f"{value_based_price_loc}/value_price.csv").iloc[0,0]

surv_params_path = "03_output/01_surv_params"
surv_params = pd.read_csv(f"{surv_params_path}/surv_params.csv")
case_params = {
    row["case_name"]: {
        "intercept": row["intercept"],
        "log_scale": row["log_scale"]
    }
    for _, row in surv_params.iterrows()
}

def hr_log_transform(hr: float,
                     hr_lower: float,
                     hr_upper: float,
                     a_value: float=a_value) -> dict:
    """
    Compute the mean and standard deviation of the log-transformed hazard ratio (HR).

    Parameters:
    hr (float): Hazard ratio estimate.
    ci_lower (float): Lower bound of the confidence interval.
    ci_upper (float): Upper bound of the confidence interval.

    Returns:
    tuple: (mean of log(HR), standard deviation of log(HR))
    """
    log_hr = np.log(hr)
    log_ci_lower = np.log(hr_lower)
    log_ci_upper = np.log(hr_upper)

    # Approximate standard deviation using CI width
    sigma_log_hr_old = (log_ci_upper - log_ci_lower) / (2 * norm.ppf(1 - 0.05 / 2))
    sigma_log_hr_updated = np.sqrt(sigma_log_hr_old**2 / a_value)

    return {"mu_log_hr": log_hr,
            "sigma_log_hr": sigma_log_hr_old,
            "sigma_log_hr_updated": sigma_log_hr_updated}

def get_simparams(
    a_value: float = a_value,
    sample_size_now: int = sample_size_now,
    sample_size_new: int = sample_size_new,
    hr_params: dict = hr_params,
    n_psa: int = n_psa,
    seed_for_prior: int = seed
) -> dict:
    """
    Generate prior and posterior hazard ratio (HR) samples for PFS and OS.

    Returns:
        dict: A nested dictionary with keys ["prior"]["pfs"], ["prior"]["os"],
              ["posterior"]["pfs"], ["posterior"]["os"] each containing arrays of HR samples.
    """
    from numpy.random import default_rng

    # RNG seeded for reproducibility of the "prior" samples
    rng_prior = default_rng(seed_for_prior)

    # Dictionary to hold final results
    res = {
        "external": {},
        "prior": {},
        "posterior": {}
    }

    # Compute log-HR parameters (mu and sigma) for PFS and OS
    log_hr_dict = {
        "pfs": hr_log_transform(
            hr_params["pfs"]["hr"],
            hr_params["pfs"]["low"],
            hr_params["pfs"]["high"],
            a_value=a_value
        ),
        "os": hr_log_transform(
            hr_params["os"]["hr"],
            hr_params["os"]["low"],
            hr_params["os"]["high"],
            a_value=a_value
        )
    }

    # Iterate over PFS and OS
    for outcome, params in log_hr_dict.items():
        prior_mean       = params["mu_log_hr"]
        prior_sd         = params["sigma_log_hr"]
        prior_sd_updated = params["sigma_log_hr_updated"]

        # Precision of the *original* prior distribution
        prior_precision = 1.0 / (prior_sd**2)

        # Draw from the prior distribution
        prior_mean_samples_0         = rng_prior.normal(prior_mean, prior_sd,         size=n_psa)
        prior_mean_updated_samples = rng_prior.normal(prior_mean, prior_sd_updated, size=n_psa)

        # Exponentiate to get HR samples (prior)
        prior_hr_arr_0 = np.exp(prior_mean_samples_0)
        prior_hr_arr = np.exp(prior_mean_updated_samples)

        # Population variance of the log HR scaled by sample_size_now
        pop_var = (prior_sd**2) * sample_size_now
        # The sample variance for the sample mean in new data
        sample_var = pop_var / sample_size_new
        sample_precision = 1.0 / sample_var

        # For each PSA replicate, simulate new data
        # Generate a (n_psa*sample_size_new) array
        new_data_samples = rng_prior.normal(
            loc = prior_mean_updated_samples.reshape(n_psa, 1),
            scale = np.sqrt(sample_var),
            size = (n_psa, sample_size_new)
        )

        # Compute the sample mean for each replicate
        X_bar = new_data_samples.mean(axis=1)

        # Normal-Normal Bayesian update for each replicate
        post_mean = (prior_precision * prior_mean + sample_precision * X_bar) / (prior_precision + sample_precision)
        post_sd = prior_sd * np.sqrt(sample_var / (sample_var + prior_sd**2))

        # Draw posterior samples for each replicate
        # Generate an (n_psa*n_psa) array: each row i contains n_psa samples from N(post_mean[i], post_sd)
        post_samples = rng_prior.normal(
            loc=post_mean.reshape(n_psa, 1),
            scale=post_sd,
            size=(n_psa, n_psa)
        )

        post_hr_matrix = np.exp(post_samples)  # Convert to HR values
        post_hr_dict = {i: post_hr_matrix[i, :] for i in range(n_psa)}  # Store as dictionary entry

        # post_hr_dict = {}  # Initialize dictionary for posterior samples

        # # Sample each "PSA replicate" of the posterior
        # for i in range(n_psa):

        #     # Draw one "true" mean from updated prior
        #     prior_sample_mean_i = prior_mean_updated_samples[i]

        #     # Suppose the population variance of the log(HR) is scaled by sample_size_now
        #     # pop_var = (prior_sd_updated**2) * sample_size_now
        #     pop_var = (prior_sd**2) * sample_size_now
        #     # pop_sd  = np.sqrt(pop_var)

        #     # The sample variance of the (sample mean) around that "true" mean
        #     sample_var = pop_var / sample_size_new

        #     # Simulate a sample mean X̄ for the new data
        #     X_bar_i = rng_prior.normal(prior_sample_mean_i, np.sqrt(sample_var), size=sample_size_new).mean()

        #     # Combine prior and sample by normal-normal update
        #     sample_precision = 1.0 / sample_var

        #     ## mean
        #     post_mean_1 = prior_precision * prior_mean + sample_precision * X_bar_i
        #     post_mean_2 = prior_precision + sample_precision
        #     post_mean_i = post_mean_1 / post_mean_2

        #     ## variance
        #     post_sd_i = prior_sd * np.sqrt(sample_var / (sample_var + prior_sd**2))

        #     ## posterior log hr
        #     post_mean_arr = rng_prior.normal(post_mean_i, post_sd_i, size=n_psa)

        #     # Store exponentiated posterior mean
        #     post_mean_arr = rng_prior.normal(post_mean_i, post_sd_i, size=n_psa)
        #     post_hr_dict[i] = np.exp(post_mean_arr)  # Store as dictionary entry

        # Place results into the final dictionary
        res["external"][outcome] = prior_hr_arr_0
        res["prior"][outcome] = prior_hr_arr
        res["posterior"][outcome] = post_hr_dict

    return res

# --- Vectorized survival simulation -----------------------------------------
def simulate_survival_vectorized(cycle_range, intercept, log_scale, hr=None):
    """
    Computes survival probabilities for each cycle.

    If hr is None, computes the control survival curve (1D array over cycles).
    If hr is provided and is a 1D array (shape = (n_draws,)), returns an array of
    survival curves with shape (n_draws, len(cycle_range)).
    """
    # Each cycle is assumed to be 3/4 month
    adjusted_cycle = cycle_range * (3/4)  # shape (n_cycles,)
    # Compute the control survival curve
    surv_ctrl = 1 - norm.cdf((np.log(adjusted_cycle + 1e-10) - intercept) / np.exp(log_scale))
    surv_ctrl[-1] = 0  # enforce last cycle = 0

    if hr is None:
        return surv_ctrl  # shape (n_cycles,)
    else:
        # hr is assumed to be a 1D array; we use broadcasting to raise surv_ctrl to each hr value.
        surv = surv_ctrl ** hr[:, None]  # shape: (n_draws, n_cycles)
        surv[:, -1] = 0  # enforce boundary condition for each simulation
        return surv

# --- Vectorized net monetary benefit (NMB) computation -----------------------
def get_nmb_vectorized(hr_pfs_vec, hr_os_vec,
                       case_params, price_drug, wtp,
                       utility, cost,
                       cycle_range, discount_factor):
    """
    Vectorized version of get_nmb.

    Parameters:
      - hr_pfs_vec and hr_os_vec: 1D arrays of hazard ratios for the combo arm
        (e.g., shape (n_inner,)). If a one-element array is passed, the returned
        outputs for the combo arm will be one-element arrays.
      - The other parameters are as before.

    Returns a tuple:
      (Avg_cost_per_month_chemo, Avg_cost_per_month_combo,
       MB_chemo, MB_combo, NMB, ICER)

      * For the chemo arm, the values are scalars (computed once).
      * For the combo arm, the values are computed for each draw (vectorized over hr_pfs_vec/hr_os_vec).
    """
    n_cycles = len(cycle_range)

    # --- Compute chemo (control) arm outcomes once ---
    pfs_chemo = simulate_survival_vectorized(cycle_range,
                                             intercept=case_params["pfs_chemo"]["intercept"],
                                             log_scale=case_params["pfs_chemo"]["log_scale"],
                                             hr=None)  # shape (n_cycles,)
    os_chemo = simulate_survival_vectorized(cycle_range,
                                            intercept=case_params["os_chemo"]["intercept"],
                                            log_scale=case_params["os_chemo"]["log_scale"],
                                            hr=None)
    prog_chemo = np.maximum(os_chemo - pfs_chemo, 0)
    stable_chemo = pfs_chemo.copy()
    dead_chemo = 1 - os_chemo
    stable_chemo[-1] = 0
    prog_chemo[-1] = 0
    dead_chemo[-1] = 1

    cycle_factor = 3/(4*12)
    qalys_chemo = np.sum(
        stable_chemo * (utility["stable"] * cycle_factor) * discount_factor +
        prog_chemo   * (utility["prog"]   * cycle_factor) * discount_factor
    )
    monitoring_chemo = cost["monitoring_stable"] * stable_chemo + cost["monitoring_prog"] * prog_chemo
    admin_chemo = np.zeros(n_cycles)
    admin_chemo[:37] = stable_chemo[:37] * (cost["admin_first"] + cost["admin_sub"])
    final_chemo = np.concatenate(([dead_chemo[0]], np.diff(dead_chemo))) * cost["terminal"]
    treatment_chemo = np.zeros(n_cycles)
    treatment_chemo[:4] = cost["intro_cost"] * stable_chemo[:4]
    treatment_chemo[4:36] = cost["maintain_cost_chemo"] * stable_chemo[4:36]
    costs_chemo = np.sum((monitoring_chemo + admin_chemo + final_chemo + treatment_chemo) * discount_factor)
    MB_chemo = wtp * qalys_chemo - costs_chemo
    Avg_cost_per_month_chemo = (costs_chemo / n_cycles) * (4/3)

    # --- Compute combo outcomes in a vectorized manner ---
    # The survival curves for the combo arm are computed for each element of hr_pfs_vec/hr_os_vec.
    pfs_combo = simulate_survival_vectorized(cycle_range,
                                             intercept=case_params["pfs_chemo"]["intercept"],
                                             log_scale=case_params["pfs_chemo"]["log_scale"],
                                             hr=hr_pfs_vec)  # shape: (n_inner, n_cycles)
    os_combo = simulate_survival_vectorized(cycle_range,
                                            intercept=case_params["os_chemo"]["intercept"],
                                            log_scale=case_params["os_chemo"]["log_scale"],
                                            hr=hr_os_vec)  # shape: (n_inner, n_cycles)
    pfs_combo[:, -1] = 0
    os_combo[:, -1] = 0
    prog_combo = np.maximum(os_combo - pfs_combo, 0)
    stable_combo = pfs_combo
    dead_combo = 1 - os_combo
    dead_combo[:, -1] = 1

    qalys_combo = np.sum(
        stable_combo * (utility["stable"] * cycle_factor) * discount_factor +
        prog_combo   * (utility["prog"]   * cycle_factor) * discount_factor,
        axis=1
    )
    monitoring_combo = cost["monitoring_stable"] * stable_combo + cost["monitoring_prog"] * prog_combo
    admin_combo = np.zeros_like(stable_combo)
    admin_combo[:, :37] = stable_combo[:, :37] * (cost["admin_first"] + cost["admin_sub"])
    final_combo = np.hstack([dead_combo[:, :1], np.diff(dead_combo, axis=1)]) * cost["terminal"]
    treatment_combo = np.zeros_like(stable_combo)
    drug_cost_intro = cost["dose_drug"] * price_drug + cost["intro_cost"]
    drug_cost_maintain = cost["dose_drug"] * price_drug + cost["maintain_cost_chemo"]
    treatment_combo[:, :4] = stable_combo[:, :4] * drug_cost_intro
    treatment_combo[:, 4:36] = stable_combo[:, 4:36] * drug_cost_maintain
    costs_combo = np.sum(
        (monitoring_combo + admin_combo + final_combo + treatment_combo) * discount_factor,
        axis=1
    )
    MB_combo = wtp * qalys_combo - costs_combo
    Avg_cost_per_month_combo = (costs_combo / n_cycles) * (4/3)

    NMB = MB_combo - MB_chemo
    delta_cost = costs_combo - costs_chemo
    delta_qaly = qalys_combo - qalys_chemo
    ICER = np.where(delta_qaly == 0,
                    np.where(delta_cost > 0, np.inf, 0),
                    delta_cost / delta_qaly)

    return (Avg_cost_per_month_chemo, Avg_cost_per_month_combo, MB_chemo, MB_combo, NMB, ICER)

def compute_psa_nmb_vectorized(psa_params, case_params, price_drug, wtp,
                               utility, cost, cycle_range, discount_factor):
    """
    Compute PSA results for both prior and posterior using the vectorized net benefit function.

    psa_params is assumed to have the structure produced by get_simparams:
      - psa_params["prior"]["pfs"] and psa_params["prior"]["os"] are 1D arrays of length n_psa.
      - psa_params["posterior"]["pfs"] and psa_params["posterior"]["os"] are dictionaries
        mapping an outer index to a 1D array of inner draws (each of length n_psa).

    Returns a dictionary with arrays for:
      - "Avg_cost_per_month_chemo_prior", "Avg_cost_per_month_combo_prior"
      - "MB_chemo_prior", "MB_combo_prior"
      - "Avg_cost_per_month_chemo_post", "Avg_cost_per_month_combo_post"
      - "MB_chemo_post", "MB_combo_post"
    """
    n_psa = len(psa_params["prior"]["pfs"])

    # Preallocate arrays for external results (each element will be a scalar)
    Avg_cost_per_month_chemo_0 = np.empty(n_psa)
    Avg_cost_per_month_combo_0 = np.empty(n_psa)
    MB_chemo_0 = np.empty(n_psa)
    MB_combo_0 = np.empty(n_psa)

    # --- Prior-based PSA (outer loop) ---
    for i in range(n_psa):
        # Pass one-element arrays for the hazard ratios
        (avg_cost_chemo_arr,
         avg_cost_combo_arr,
         mb_chemo_arr,
         mb_combo_arr,
         _,
         _) = get_nmb_vectorized(
             np.array([psa_params["external"]["pfs"][i]]),
             np.array([psa_params["external"]["os"][i]]),
             case_params, price_drug, wtp,
             utility, cost, cycle_range, discount_factor
        )
        # Explicitly extract scalar values from one-element arrays
        Avg_cost_per_month_chemo_0[i] = avg_cost_chemo_arr.item()
        Avg_cost_per_month_combo_0[i]  = avg_cost_combo_arr.item()
        MB_chemo_0[i] = mb_chemo_arr.item()
        MB_combo_0[i] = mb_combo_arr.item()

    # Preallocate arrays for prior-based results (each element will be a scalar)
    Avg_cost_per_month_chemo_prior = np.empty(n_psa)
    Avg_cost_per_month_combo_prior = np.empty(n_psa)
    MB_chemo_prior = np.empty(n_psa)
    MB_combo_prior = np.empty(n_psa)

    # --- Prior-based PSA (outer loop) ---
    for i in range(n_psa):
        # Pass one-element arrays for the hazard ratios
        (avg_cost_chemo_arr,
         avg_cost_combo_arr,
         mb_chemo_arr,
         mb_combo_arr,
         _,
         _) = get_nmb_vectorized(
             np.array([psa_params["prior"]["pfs"][i]]),
             np.array([psa_params["prior"]["os"][i]]),
             case_params, price_drug, wtp,
             utility, cost, cycle_range, discount_factor
        )
        # Explicitly extract scalar values from one-element arrays
        Avg_cost_per_month_chemo_prior[i] = avg_cost_chemo_arr.item()
        Avg_cost_per_month_combo_prior[i]  = avg_cost_combo_arr.item()
        MB_chemo_prior[i] = mb_chemo_arr.item()
        MB_combo_prior[i] = mb_combo_arr.item()

    # Preallocate arrays for posterior-based results
    Avg_cost_per_month_chemo_post = np.empty(n_psa)
    Avg_cost_per_month_combo_post = np.empty(n_psa)
    MB_chemo_post = np.empty(n_psa)
    MB_combo_post = np.empty(n_psa)

    # --- Posterior-based PSA ---
    for i in range(n_psa):
        # For each outer PSA draw, get the inner draws (each is a 1D array of length n_psa)
        post_pfs_arr = psa_params["posterior"]["pfs"][i]  # shape (n_psa,)
        post_os_arr  = psa_params["posterior"]["os"][i]   # shape (n_psa,)

        # Call the vectorized function with all inner draws at once
        (avg_cost_chemo_vec,
         avg_cost_combo_vec,
         mb_chemo_vec,
         mb_combo_vec,
         _,
         _) = get_nmb_vectorized(
             post_pfs_arr, post_os_arr,
             case_params, price_drug, wtp,
             utility, cost, cycle_range, discount_factor
        )
        # Average the inner PSA draws to obtain a single value per outer simulation
        Avg_cost_per_month_chemo_post[i] = np.mean(avg_cost_chemo_vec)
        Avg_cost_per_month_combo_post[i]  = np.mean(avg_cost_combo_vec)
        MB_chemo_post[i] = np.mean(mb_chemo_vec)
        MB_combo_post[i] = np.mean(mb_combo_vec)

    return {
        "Avg_cost_per_month_chemo_0": Avg_cost_per_month_chemo_0,
        "Avg_cost_per_month_combo_0": Avg_cost_per_month_combo_0,
        "MB_chemo_0": MB_chemo_0,
        "MB_combo_0": MB_combo_0,
        "Avg_cost_per_month_chemo_prior": Avg_cost_per_month_chemo_prior,
        "Avg_cost_per_month_combo_prior": Avg_cost_per_month_combo_prior,
        "MB_chemo_prior": MB_chemo_prior,
        "MB_combo_prior": MB_combo_prior,
        "Avg_cost_per_month_chemo_post": Avg_cost_per_month_chemo_post,
        "Avg_cost_per_month_combo_post": Avg_cost_per_month_combo_post,
        "MB_chemo_post": MB_chemo_post,
        "MB_combo_post": MB_combo_post
    }

def get_ev(
    a_value: float = a_value,
    sample_size_now: int = sample_size_now,
    sample_size_new: int = sample_size_new,
    n_psa: int = n_psa,
    seed_for_prior: int = seed,
    price_drug: float = value_based_price,
    wtp: float = wtp,
    case_params: dict = case_params,
    hr_params: dict = hr_params,
    utility: dict = utility,
    cost: dict = cost,
    cycle_range: np.ndarray = cycle_range,
    discount_factor: np.ndarray = discount_factor
) -> dict:
    # # 0) Some fixed values
    # incidence = 71794
    # prevalence = 640488
    # cost_per_sample = 48324.48
    # uptake_rate = 0.4
    # d_r = 0.03

    # 1) Get PSA results using the vectorized version
    psa_params = get_simparams(
        a_value=a_value,
        sample_size_now=sample_size_now,
        sample_size_new=sample_size_new,
        hr_params=hr_params,
        n_psa=n_psa,
        seed_for_prior=seed_for_prior
    )

    # Call the vectorized PSA computation function
    psa_results = compute_psa_nmb_vectorized(psa_params, case_params, price_drug, wtp,
                                               utility, cost, cycle_range, discount_factor)

    # 2) Extract relevant MB arrays
    MB_chemo_0 = psa_results["MB_chemo_0"]
    MB_combo_0 = psa_results["MB_combo_0"]
    MB_chemo_prior = psa_results["MB_chemo_prior"]
    MB_combo_prior = psa_results["MB_combo_prior"]
    MB_chemo_post  = psa_results["MB_chemo_post"]
    MB_combo_post  = psa_results["MB_combo_post"]

    # 3) Expected Values
    ## Expected benefits from current treatment
    EV_ct_pp = np.mean(MB_chemo_prior)
    ## Expected benefits from new treatment
    EV_nt_pp = np.mean(MB_combo_prior)
    ## Expected benefits from new treatment with new evidence
    max_MB_post = np.maximum(MB_chemo_post, MB_combo_post)
    EV_update_pp = np.mean(max_MB_post)

    return {
        "EV_ct_pp": EV_ct_pp,
        "EV_nt_pp": EV_nt_pp,
        "EV_update_pp": EV_update_pp
    }

# a value from 0.5 to 1.0 by 0.1
a_value_arr = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

# price range
point_price =  value_based_price * (1-0.1)
price_drug_arr = np.array([value_based_price * 0.7, value_based_price * 0.75, value_based_price * 0.8, value_based_price * 0.85, point_price, value_based_price * 0.95, value_based_price])

# !pip install tqdm joblib tqdm_joblib
import multiprocessing
import itertools
from joblib import Parallel, delayed
from tqdm.notebook import tqdm
from tqdm_joblib import tqdm_joblib

# Number of parallel jobs: using one less than the number of CPUs
n_jobs = multiprocessing.cpu_count() - 12 # 16 in windows total

# The run_combination function remains the same:
def run_combination(a_val, drug_price):
    """Runs a single combination and returns the computed EV values."""
    out = get_ev(
        a_value=a_val,
        price_drug=drug_price
    )
    return {
        "a_value": a_val,
        "price_drug": drug_price,
        "EV_ct_pp": out['EV_ct_pp'],
        "EV_nt_pp": out['EV_nt_pp'],
        "EV_update_pp": out['EV_update_pp']
    }

# Create all combinations of a_value and drug_price
combinations = list(itertools.product(a_value_arr, price_drug_arr))

# Use tqdm_joblib for progress bar with parallel processing
with tqdm_joblib(tqdm(desc="Running combinations", total=len(combinations))):
    results = Parallel(n_jobs=n_jobs, pre_dispatch="2*n_jobs")(
        delayed(run_combination)(a_val, drug_price)
        for a_val, drug_price in combinations
    )

# Create a DataFrame from the results
df_a_price = pd.DataFrame(results)

enbs_paths = "03_output/03_enbs"

df_a_price.to_csv(f"{enbs_paths}/01_combination_a_unit_price.csv", index=False)