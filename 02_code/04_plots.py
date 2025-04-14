
import os
os.chdir('c:\\Users\\Yizhi\\OneDrive\\Research\\Health-Econ\\01_External_Local_RCT')

import pandas as pd
import numpy as np

d_r = 0.03
sample_size_new = 616
incidence = 71794
prevalence = 640488
# market share (0.01 2024, 0.04 2025, 0.05 2025): https://doi.org/10.1080/13696998.2024.2411877
uptake_rate = 0.04

trial_years = 3
delayed_trial_years = 3 # 10.1001/jamanetworkopen.2025.2026
total_years = 10

enbs_paths = "03_output/03_enbs"

# main results
df_a_val_unit_price = pd.read_csv(f"{enbs_paths}/01_combination_a_unit_price.csv")

# add two columns:
## "intro_drug": 200 * price_drug + 6992.538
## "maintain_drug": 200 * price_drug + 6965.7
df_a_val_unit_price["intro_drug"] = 200 * df_a_val_unit_price["price_drug"] + 6992.538
df_a_val_unit_price["maintain_drug"] = 200 * df_a_val_unit_price["price_drug"] + 6965.7
df_a_val_unit_price['drug_cost_per_cycle_K'] = ((df_a_val_unit_price['intro_drug']*4 + df_a_val_unit_price['maintain_drug']*32)/36)/1000

def calculate_policy_comparison(df_input=None,
                                trial_years=trial_years,
                                prevalence=prevalence,
                                sample_size_new=sample_size_new,
                                incidence=incidence,
                                d_r=d_r,
                                uptake_rate=uptake_rate,
                                total_years=total_years,
                                delayed_trial_years=delayed_trial_years):
    """
    Calculate policy comparisons for different healthcare scenarios.

    Parameters:
    -----------
    df_input : pandas DataFrame, optional
        Input dataframe with drug price data. Default is None (should be provided when calling).
    trial_years : int, optional
        Number of years for the trial period. Default is 5.
    prevalence : int, optional
        Initial disease prevalence. Default is 1,000,000.
    sample_size_new : int, optional
        Sample size for new treatments. Default is 1,000.
    incidence : int, optional
        Annual incidence of new cases. Default is 50,000.
    d_r : float, optional
        Discount rate (as a decimal). Default is 0.03 (3%).
    uptake_rate : float, optional
        Rate of treatment uptake. Default is 0.5 (50%).
    total_years : int, optional
        Total time horizon for analysis in years. Default is 10.
    delayed_trial_years : int, optional
        Additional delay years for Policy 3. Default is 2.

    Returns:
    --------
    pandas DataFrame
        Combined policy comparison dataframe with EV calculations.
    """
    if df_input is None:
        raise ValueError("df_input must be provided")

    # Create a copy to avoid modifying the original
    df_a_val_unit_price = df_input.copy()

    # Policy 1
    P1_df = df_a_val_unit_price.copy()
    P1_df['part1'] = (prevalence - sample_size_new) * P1_df['EV_ct_pp']
    P1_df['part2'] = (sample_size_new/2) * P1_df['EV_ct_pp']
    P1_df['part3'] = (sample_size_new/2) * P1_df['EV_nt_pp']
    P1_df['part4'] = np.sum(incidence * (1 / (1+d_r) ** np.arange(0, trial_years, 1))) * P1_df['EV_ct_pp']
    P1_df['part5'] = (
        P1_df['EV_update_pp'] * np.sum(incidence * uptake_rate * (1 / (1 + d_r) ** np.arange(trial_years, total_years, 1)))
        + P1_df['EV_ct_pp'] * np.sum(incidence * (1 - uptake_rate) * (1 / (1 + d_r) ** np.arange(trial_years, total_years, 1)))
    )
    P1_df['EV_total'] = P1_df['part1'] + P1_df['part2'] + P1_df['part3'] + P1_df['part4'] + P1_df['part5']
    P1_df = P1_df[['a_value', 'price_drug', 'drug_cost_per_cycle_K','EV_total']]

    # Policy 2
    P2_df = df_a_val_unit_price.copy()
    P2_df['part1'] = (
        (prevalence + np.sum(incidence * (1 / (1 + d_r) ** np.arange(0, total_years, 1)))) * P2_df['EV_nt_pp'] * uptake_rate
        + (prevalence + np.sum(incidence * (1 / (1 + d_r) ** np.arange(0, total_years, 1)))) * P2_df['EV_ct_pp'] * (1 - uptake_rate)
    )
    P2_df['EV_total'] = P2_df['part1']
    P2_df = P2_df[['a_value', 'price_drug', 'drug_cost_per_cycle_K','EV_total']]

    # Policy 3
    P3_df = df_a_val_unit_price.copy()
    P3_df['part1'] = (
        (prevalence - sample_size_new) * P3_df['EV_nt_pp'] * uptake_rate
        + (prevalence - sample_size_new) * P3_df['EV_ct_pp'] * (1 - uptake_rate)
    )
    P3_df['part2'] = (sample_size_new/2) * P3_df['EV_ct_pp']
    P3_df['part3'] = (sample_size_new/2) * P3_df['EV_nt_pp']
    P3_df['part4'] = (
        np.sum(incidence * (1 / (1+d_r) ** np.arange(0, trial_years+delayed_trial_years, 1))) * P3_df['EV_nt_pp'] * uptake_rate
        + np.sum(incidence * (1 / (1+d_r) ** np.arange(0, trial_years+delayed_trial_years, 1))) * P3_df['EV_ct_pp'] * (1 - uptake_rate)
    )
    P3_df['part5'] = (
        P3_df['EV_update_pp'] * np.sum(incidence * uptake_rate * (1 / (1 + d_r) ** np.arange(trial_years+delayed_trial_years, total_years, 1)))
        + P3_df['EV_ct_pp'] * np.sum(incidence * (1 - uptake_rate) * (1 / (1 + d_r) ** np.arange(trial_years+delayed_trial_years, total_years, 1)))
    )
    P3_df['EV_total'] = P3_df['part1'] + P3_df['part2'] + P3_df['part3'] + P3_df['part4'] + P3_df['part5']
    P3_df = P3_df[['a_value', 'price_drug', 'drug_cost_per_cycle_K','EV_total']]

    # Policy 4
    P4_df = df_a_val_unit_price.copy()
    P4_df['part1'] = (prevalence + np.sum(incidence * (1 / (1 + d_r) ** np.arange(0, total_years, 1)))) * P4_df['EV_ct_pp']
    P4_df['EV_total'] = P4_df['part1']
    P4_df = P4_df[['a_value', 'price_drug', 'drug_cost_per_cycle_K','EV_total']]

    # Combine all policies
    dfs = [P1_df, P2_df, P3_df, P4_df]

    # Add policy indicator to each DataFrame
    for i, df in enumerate(dfs):
        df['policy_indicator'] = i + 1

    # Concatenate the DataFrames
    df_policy = pd.concat(dfs)
    df_policy['policy_indicator'] = df_policy['policy_indicator'].map({
        1: 'Policy 1', 2: 'Policy 2', 3: 'Policy 3', 4: 'Policy 4'
    })

    # Reset the index
    df_policy = df_policy.reset_index(drop=True)

    # Convert EV to billions
    df_policy['EV_total'] = df_policy['EV_total'] / 1000000000

    return df_policy

df_policy_3yrs_trial = calculate_policy_comparison(df_input=df_a_val_unit_price)

# !pip install -q lets-plot
from lets_plot import *
LetsPlot.setup_html()

scaler = 1.5

# Add font family configuration
times_new_roman_theme = theme(
    # title=element_text(size=12*scaler, family="Times New Roman"),
    axis_text_x=element_text(size=10*scaler, family="Times New Roman", angle=0),
    axis_text_y=element_text(size=10*scaler, family="Times New Roman"),
    legend_text=element_text(size=12*scaler, family="Times New Roman"),
    legend_position="bottom",  # Move legend to the bottom
    # legend_title=element_blank(),  # Remove color legend title
    legend_title=element_text(size=12*scaler, family="Times New Roman"),
    plot_title=element_text(size=16, family="Times New Roman", face = "bold"),
    plot_subtitle=element_text(size=14, family="Times New Roman"),
    axis_title_x=element_text(size=12*scaler, family="Times New Roman"),
    axis_title_y=element_text(size=12*scaler, family="Times New Roman"),
    text=element_text(size=8, family="Times New Roman")
)

"""# Figure 2: Expected value of four policies over the power prior"""

# Figure 2: Expected value of four policies over the power prior

df_point_price = df_policy_3yrs_trial[(df_policy_3yrs_trial['price_drug'] > 13) & (df_policy_3yrs_trial['price_drug'] < 14)]

figure2_plot = (
    ggplot(df_point_price, aes(x="a_value", y="EV_total", shape="policy_indicator", group="policy_indicator")) +
    geom_point(size = 3.2) +
    geom_line() +

    scale_x_continuous(breaks=df_a_val_unit_price["a_value"].unique(), format="{.1f}") +
    scale_y_continuous(format="${.1f}") +

    labs(
        x = "Power Parameter",
        y = "Expected Benefits ($billion)",
        shape =""
    ) +
    theme_bw() +
    times_new_roman_theme
)

figure2_plot

"""# Figure 3: The optimal policy as a function of power prior and trial length

"""

# Create datasets for trial years 1 to 6
policy_dfs = {}
for year in range(1, 7):
    policy_df = calculate_policy_comparison(df_input=df_a_val_unit_price, trial_years=year)
    # Subset for price_drug > 13 and < 14
    policy_df_filtered = policy_df[(policy_df['price_drug'] > 13) & (policy_df['price_drug'] < 14)]
    # Store in dictionary with appropriate name
    policy_dfs[f"df_policy_{year}yrs_trial"] = policy_df_filtered

# Create the final summary dataframe
df_rows = []

# For each trial year dataset
for trial_yr, df_name in enumerate(policy_dfs.keys(), 1):
    df = policy_dfs[df_name]

    # Group by a_value and find the policy with the maximum EV_total
    best_policies = df.loc[df.groupby('a_value')['EV_total'].idxmax()]

    # Create rows for the summary dataframe
    for _, row in best_policies.iterrows():
        df_rows.append({
            'a_value': row['a_value'],
            'trial_yr': trial_yr,
            'best_policy': row['policy_indicator'],
            'EV_total': row['EV_total']
        })

# Create the final dataframe
df_a_trial_yr = pd.DataFrame(df_rows)

# Sort by a_value and trial_yr for better organization
df_a_trial_yr = df_a_trial_yr.sort_values(by=['a_value', 'trial_yr']).reset_index(drop=True)

df_a_trial_yr['label'] = df_a_trial_yr.apply(
    lambda row: f"{row['best_policy']}\n${row['EV_total']:.1f} B", axis=1
)

figure3_plot = (
    ggplot(df_a_trial_yr, aes(x='a_value',y='trial_yr')) +
    geom_tile(aes(fill='EV_total')) +
    geom_text(aes(label='label')) +
    scale_fill_gradient(low='#F5F5F5', high='#A9A9A9') +
    scale_x_continuous(breaks=df_a_trial_yr["a_value"].unique(), format="{.1f}") +
    labs(
        x = "Power Parameter",
        y = "Trial Length (Years)",
        fill = ""
    ) +
    theme_minimal() +
    times_new_roman_theme +
    theme(
        legend_position="none"
    )
)

figure3_plot

"""# Figure 4: The optimal policy as a function of power prior and price"""

# First, identify the best policy for each a_value and drug_cost_per_cycle_K combination
best_policies_p4 = df_policy_3yrs_trial.loc[
    df_policy_3yrs_trial.groupby(['a_value', 'drug_cost_per_cycle_K'])['EV_total'].idxmax()
]

# Format the label with dollar sign and "B" suffix
best_policies_p4['label'] = best_policies_p4.apply(
    lambda row: f"{row['policy_indicator']}\n${row['EV_total']:.1f} B", axis=1
)

figure4_plot = (
    ggplot(best_policies_p4, aes(x='a_value', y='drug_cost_per_cycle_K')) +
    geom_tile(aes(fill='EV_total')) +
    geom_text(aes(label='label')) +
    scale_fill_gradient(low='#F5F5F5', high='#A9A9A9') +
    scale_x_continuous(breaks=best_policies_p4["a_value"].unique(), format="{.1f}") +
    scale_y_continuous(breaks=best_policies_p4["drug_cost_per_cycle_K"].unique(), format="{.1f}") +
    labs(
        x = "Power Parameter",
        y = "Average Treatment Cost per Cycle ($thousand)",
        fill = ""
    ) +
    theme_minimal() +
    times_new_roman_theme +
    theme(
        legend_position="none"
    )
)

figure4_plot

# !pip install -q CairoSVG
ggsave(plot=figure2_plot, filename=f"{enbs_paths}/02_over_a_value.pdf", dpi=1000, w=4*4, h=3*4, unit='in')
ggsave(plot=figure3_plot, filename=f"{enbs_paths}/03_best_a_trial_length.pdf", dpi=1000, w=4*4, h=3*4, unit='in')
ggsave(plot=figure4_plot, filename=f"{enbs_paths}/04_best_a_price.pdf", dpi=1000, w=4*4, h=3*4, unit='in')

# save as svg
ggsave(plot=figure2_plot, filename=f"{enbs_paths}/02_over_a_value.svg", dpi=1000, w=4*4, h=3*4, unit='in')
ggsave(plot=figure3_plot, filename=f"{enbs_paths}/03_best_a_trial_length.svg", dpi=1000, w=4*4, h=3*4, unit='in')
ggsave(plot=figure4_plot, filename=f"{enbs_paths}/04_best_a_price.svg", dpi=1000, w=4*4, h=3*4, unit='in')