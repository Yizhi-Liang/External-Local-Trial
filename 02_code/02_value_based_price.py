
import os
os.chdir('c:\\Users\\Yizhi\\OneDrive\\Research\\Health-Econ\\01_External_Local_RCT')

import pandas as pd
import numpy as np
from scipy.stats import norm

# A set of fixed input
sample_size_now = 397
sample_size_new = 619

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
hr_pfs = 0.49
hr_pfs_low = 0.38
hr_pfs_high = 0.63
hr_os = 0.60
hr_os_low = 0.45
hr_os_high = 0.79

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

surv_params_path = "03_output/01_surv_params"
surv_params = pd.read_csv(f"{surv_params_path}/surv_params.csv")
case_params = {
    row["case_name"]: {
        "intercept": row["intercept"],
        "log_scale": row["log_scale"]
    }
    for _, row in surv_params.iterrows()
}

case_params

def get_icer_with_price(price_drug,
                        case_params,
                        wtp = 150000,
                        hr_pfs = hr_pfs,
                        hr_os = hr_os,
                        utility = utility,
                        cost = cost,
                        cycle_range = cycle_range,
                        discount_factor = discount_factor):
    """
    Calculate the ICER (Incremental Cost-Effectiveness Ratio) for combination therapy
    vs. standard chemotherapy, given the price of the drug and other inputs.

    Parameters
    ----------
    price_drug : float
        The price of the new drug per dose.

    case_params : dict
        A dictionary containing intercept and log_scale for each case.
        Example:
        {
            "pfs_chemo": {"intercept": ..., "log_scale": ...},
            "pfs_combo": {"intercept": ..., "log_scale": ...},
            "os_chemo": {"intercept": ..., "log_scale": ...},
            "os_combo": {"intercept": ..., "log_scale": ...}
        }

    utility : dict
        A dictionary containing utility values for "stable" and "progression" states.
        Example: {"stable": 0.75, "prog": 0.59}

    cost : dict
        A dictionary containing the relevant cost parameters.
        Example:
        {
            "monitoring_stable": ...,
            "monitoring_prog": ...,
            "admin_first": ...,
            "admin_sub": ...,
            "terminal": ...,
            "intro_cost": ...,
            "maintain_cost_chemo": ...,
            "dose_drug": ...
        }

    cycle_range : np.ndarray
        The range of cycles for the simulation (e.g., np.arange(0, max_cycle + 1)).

    discount_factor : np.ndarray
        An array of discount factors corresponding to the cycle range.

    Returns
    -------
    ICER : float
        The incremental cost-effectiveness ratio for combination therapy vs. chemotherapy.
    """

    # Survival simulation function
    def simulate_survival(cycle_range, intercept, log_scale, hr=None):

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

      return surv_prob

    # Simulate PFS and OS for chemotherapy
    pfs_chemo = simulate_survival(
        cycle_range,
        intercept=case_params["pfs_chemo"]["intercept"],
        log_scale=case_params["pfs_chemo"]["log_scale"]
    )
    os_chemo = simulate_survival(
        cycle_range,
        intercept=case_params["os_chemo"]["intercept"],
        log_scale=case_params["os_chemo"]["log_scale"]
    )

    # Simulate PFS and OS for combination therapy
    pfs_combo = simulate_survival(
        cycle_range,
        intercept=case_params["pfs_chemo"]["intercept"],
        log_scale=case_params["pfs_chemo"]["log_scale"],
        hr = hr_pfs
    )
    os_combo = simulate_survival(
        cycle_range,
        intercept=case_params["os_chemo"]["intercept"],
        log_scale=case_params["os_chemo"]["log_scale"],
        hr = hr_os
    )


    # pfs_combo = simulate_survival(
    #     cycle_range,
    #     intercept=case_params["pfs_combo"]["intercept"],
    #     log_scale=case_params["pfs_combo"]["log_scale"]
    # )
    # os_combo = simulate_survival(
    #     cycle_range,
    #     intercept=case_params["os_combo"]["intercept"],
    #     log_scale=case_params["os_combo"]["log_scale"]
    # )

    # Define states for chemotherapy
    prog_chemo = np.maximum(os_chemo - pfs_chemo, 0)
    stable_chemo = pfs_chemo
    dead_chemo = 1 - os_chemo

    # Ensure proper endpoint
    stable_chemo[-1] = 0
    prog_chemo[-1] = 0
    dead_chemo[-1] = 1

    # QALYs for chemotherapy
    qalys_chemo = np.sum(
        stable_chemo * (utility["stable"] * 3 / (4 * 12)) * discount_factor +
        prog_chemo * (utility["prog"] * 3 / (4 * 12)) * discount_factor
    )

    # Costs for chemotherapy
    monitoring_chemo = cost["monitoring_stable"] * stable_chemo + cost["monitoring_prog"] * prog_chemo
    admin_chemo = np.zeros(len(cycle_range))
    admin_chemo[:37] = stable_chemo[:37] * (cost["admin_first"] + cost["admin_sub"])
    final_chemo = np.concatenate(([dead_chemo[0]], np.diff(dead_chemo))) * cost["terminal"]
    treatment_chemo = np.zeros(len(cycle_range))
    treatment_chemo[:4] = cost["intro_cost"] * stable_chemo[:4]
    treatment_chemo[4:36] = stable_chemo[4:36] * cost["maintain_cost_chemo"]

    costs_chemo = np.sum((monitoring_chemo + admin_chemo + final_chemo + treatment_chemo) * discount_factor)

    # Define states for combination therapy
    prog_combo = np.maximum(os_combo - pfs_combo, 0)
    stable_combo = pfs_combo
    dead_combo = 1 - os_combo

    # Ensure proper endpoint
    stable_combo[-1] = 0
    prog_combo[-1] = 0
    dead_combo[-1] = 1

    # QALYs for combination therapy
    qalys_combo = np.sum(
        stable_combo * (utility["stable"] * 3 / (4 * 12)) * discount_factor +
        prog_combo * (utility["prog"] * 3 / (4 * 12)) * discount_factor
    )

    # Costs for combination therapy
    drug_cost_intro = cost["dose_drug"] * price_drug + cost["intro_cost"]
    drug_cost_maintain = cost["dose_drug"] * price_drug + cost["maintain_cost_chemo"]
    monitoring_combo = cost["monitoring_stable"] * stable_combo + cost["monitoring_prog"] * prog_combo
    admin_combo = np.zeros(len(cycle_range))
    admin_combo[:37] = stable_combo[:37] * (cost["admin_first"] + cost["admin_sub"])
    final_combo = np.concatenate(([dead_combo[0]], np.diff(dead_combo))) * cost["terminal"]
    treatment_combo = np.zeros(len(cycle_range))
    treatment_combo[:4] = drug_cost_intro * stable_combo[:4]
    treatment_combo[4:36] = stable_combo[4:36] * drug_cost_maintain

    costs_combo = np.sum((monitoring_combo + admin_combo + final_combo + treatment_combo) * discount_factor)

    # Calculate and return NMB
    EIB_chemo = wtp * qalys_chemo - costs_chemo
    EIB_combo = wtp * qalys_combo - costs_combo
    NMB = EIB_combo - EIB_chemo
    ICER = (costs_combo - costs_chemo) / (qalys_combo - qalys_chemo)
    return EIB_chemo, EIB_combo, NMB, ICER

from scipy.optimize import brentq

def find_price_drug(target_icer,
                    case_params = case_params,
                    wtp = 150000,
                    utility = utility,
                    cost = cost,
                    cycle_range = cycle_range,
                    discount_factor = discount_factor,
                    price_drug_min=10,
                    price_drug_max=20):
    """
    Find the price of the drug such that the ICER equals the target value.

    Parameters
    ----------
    target_icer : float
        The target ICER value (e.g., 150000).

    case_params : dict
        Dictionary containing intercept and log scale for each survival case.

    utility : dict
        Dictionary with utility values for "stable" and "progression" states.

    cost : dict
        Dictionary with cost parameters.

    cycle_range : np.ndarray
        Range of cycles for the simulation (e.g., np.arange(0, max_cycle + 1)).

    discount_factor : np.ndarray
        Discount factors for each cycle.

    wtp : float
        Willingness-to-pay threshold.

    price_drug_min : float, optional
        Minimum price of the drug to consider in the search.

    price_drug_max : float, optional
        Maximum price of the drug to consider in the search.

    Returns
    -------
    float
        The price of the drug that results in the target ICER.
    """

    def icer_difference(price_drug):
        # Calculate the ICER for the given price_drug
        _, _, _, icer = get_icer_with_price(price_drug, case_params = case_params)
        return icer - target_icer  # The difference from the target ICER

    # Use Brent's method to find the price_drug that makes ICER = target_icer
    price_drug_solution = brentq(icer_difference, price_drug_min, price_drug_max)
    return price_drug_solution

value_price = find_price_drug(target_icer = 150000)
value_price

# price from 10 to 20 with 0.5 change and add the "value_price"
price_range = np.arange(10, 20.5, 0.5)
price_range = np.append(price_range, value_price)
price_range = np.sort(price_range)

# get the corresponding icer
icer_values = [get_icer_with_price(price_drug = price, case_params = case_params)[3] for price in price_range]

# create a dataframe
price_icer = pd.DataFrame({'price': price_range, 'icer': icer_values})
price_icer["intro_drug"] = 200 * price_icer["price"] + 6992.538
price_icer["maintain_drug"] = 200 * price_icer["price"] + 6965.7
price_icer['drug_cost_per_cycle'] = (price_icer['intro_drug']*4 + price_icer['maintain_drug']*32)/36

# show df with "price" and "icer" with two digits
price_icer.round(2)

# !pip install -q lets-plot
from lets_plot import *
LetsPlot.setup_html()

scaler = 1.2

# Add font family configuration
times_new_roman_theme = theme(
    title=element_text(size=12*scaler, family="Times New Roman"),
    axis_text_x=element_text(size=10*scaler, family="Times New Roman", angle=0),
    axis_text_y=element_text(size=10*scaler, family="Times New Roman"),
    legend_text=element_text(size=12*scaler, family="Times New Roman"),
    legend_position="bottom",  # Move legend to the bottom
    # legend_title=element_blank(),  # Remove color legend title
    legend_title=element_text(size=12*scaler, family="Times New Roman"),
    plot_title=element_text(size=14*scaler, family="Times New Roman"),
    plot_subtitle=element_text(size=12*scaler, family="Times New Roman"),
    axis_title_x=element_text(size=12*scaler, family="Times New Roman"),
    axis_title_y=element_text(size=12*scaler, family="Times New Roman")
)

price_icer_plot = (
    ggplot() +
    geom_line(data=price_icer, mapping=aes(x='price', y='icer'), size=1) +
    # Add a vertical dashed line at the price where ICER = 150,000
    geom_vline(xintercept=value_price, linetype='dashed', color='#434343') +
    # Add a horizontal dashed line at ICER = 150,000
    geom_hline(yintercept=150000, linetype='dashed', color='#434343') +
    geom_label(x = value_price, y=150000,
               label = f"{value_price:.2f}",
               color = "#434343",
               size = 8,
               family = "Times New Roman",
               fontface="bold") +
    labs(
        x="Price of Sintilimab per Dose",
        y="Incremental Cost-Effectiveness Ratio",
        title="Incremental Cost-Effectiveness Ratio vs. Price of Sintilimab per Dose"
    ) +
    theme_classic() +
    times_new_roman_theme
)

price_icer_plot

value_price_cycle = price_icer.loc[price_icer['price'] == value_price, 'drug_cost_per_cycle'].values[0]
price_cycle_icer_plot = (
    ggplot() +
    geom_line(data=price_icer, mapping=aes(x='drug_cost_per_cycle', y='icer'), size=1) +
    # Add a vertical dashed line at the price where ICER = 150,000
    geom_vline(xintercept=value_price_cycle, linetype='dashed', color='#434343') +
    # Add a horizontal dashed line at ICER = 150,000
    geom_hline(yintercept=150000, linetype='dashed', color='#434343') +
    scale_x_continuous(format="$,.2~f") +
    scale_y_continuous(format="$,.2~f") +
    geom_label(x = value_price_cycle, y=150000,
               label = f"{value_price_cycle:.2f}",
               color = "#434343",
               size = 8,
               family = "Times New Roman",
               fontface="bold") +
    labs(
        x="Average Treatment Cost per Cycle",
        y="Incremental Cost-Effectiveness Ratio",
        title="Incremental Cost-Effectiveness Ratio vs. Average Treatment Cost per Cycle"
    ) +
    theme_classic() +
    times_new_roman_theme
)

price_cycle_icer_plot

export_price = "03_output/02_value_price"

# Create a DataFrame to store the value_price
value_price_df = pd.DataFrame({'value_price': [value_price]})

# export it to a CSV file
value_price_df.to_csv(f"{export_price}/value_price.csv", index=False)

# !pip install -q CairoSVG
ggsave(plot=price_icer_plot, filename=f"{export_price}/price_with_icer.pdf", dpi=1500, w=4*4, h=3*4, unit='in')
ggsave(plot=price_cycle_icer_plot, filename=f"{export_price}/price_cycle_with_icer.pdf", dpi=1500, w=4*4, h=3*4, unit='in')

ggsave(plot=price_icer_plot, filename=f"{export_price}/price_with_icer.svg", dpi=1500, w=4*4, h=3*4, unit='in')
ggsave(plot=price_cycle_icer_plot, filename=f"{export_price}/price_cycle_with_icer.svg", dpi=1500, w=4*4, h=3*4, unit='in')