import warnings
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but StandardScaler was fitted with feature names"
)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import KFold, RepeatedKFold, RepeatedStratifiedKFold
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed

from pytabkit import XGB_TD_Regressor, CatBoost_TD_Regressor,\
    LGBM_TD_Regressor, RealMLP_TD_Regressor
from model_utils  import load_and_preprocess_data
from model_config import model_feature_sets

# ─────────────────────────────────────────────────────────────────────────────
# Compute mean nRMSE, sd, and percent-change for one feature-group & one estimator
# ─────────────────────────────────────────────────────────────────────────────
def pct_for_group(gid, Est, baseline_mean, baseline_nrmse_vals, cleaned_df, y, y_binned, cv):
    # select, align, and scale features for this group
    Xg = cleaned_df[model_feature_sets[gid]].dropna()
    yg = y.loc[Xg.index]
    Xg = Xg.loc[yg.index]

    scaler_g = StandardScaler().fit(Xg)
    Xg_scaled = pd.DataFrame(
        scaler_g.transform(Xg),
        index=Xg.index, columns=Xg.columns
    )
    
    # Align y_binned to this group's index
    y_stratify_g = y_binned.loc[Xg_scaled.index].values

    mean_nrmse, sd_nrmse, nrmse_vals = cv_nrmse(Est, Xg_scaled, yg, cv, y_stratify=y_stratify_g, cv_n_jobs=-1, device="cpu")
    pct = (mean_nrmse - baseline_mean) / baseline_mean * 100
    
    # Paired t-test on fold-level nRMSE values
    t_stat, p_val = ttest_rel(nrmse_vals, baseline_nrmse_vals)
    
    return gid, pct, mean_nrmse, sd_nrmse, p_val
    
# ─────────────────────────────────────────────────────────────────────────────
# Compute nRMSE on one fold
# ─────────────────────────────────────────────────────────────────────────────
def compute_fold_nrmse(train_idx, test_idx, X, y, model_cls, model_kwargs):
    model = model_cls(**model_kwargs)
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    preds = model.predict(X.iloc[test_idx])
    rmse = np.sqrt(mean_squared_error(y.iloc[test_idx], preds))
    return rmse / y.iloc[test_idx].mean()

# ─────────────────────────────────────────────────────────────────────────────
# Cross-validated nRMSE (with parallelization)
# ─────────────────────────────────────────────────────────────────────────────
def cv_nrmse(model_cls, X, y, cv, y_stratify=None, cv_n_jobs=-1, **model_kwargs):
    if y_stratify is not None:
        splits = cv.split(X, y_stratify)
    else:
        splits = cv.split(X)
        
    tasks = (delayed(compute_fold_nrmse)(train_idx, test_idx, X, y, model_cls, model_kwargs)
             for train_idx, test_idx in splits)
    nrmse_vals = Parallel(n_jobs=cv_n_jobs)(tasks)
    return np.mean(nrmse_vals), np.std(nrmse_vals), nrmse_vals

# ─────────────────────────────────────────────────────────────────────────────
# Plot function
# ─────────────────────────────────────────────────────────────────────────────
def plot_pct_changes(name, groups, pct_changes, pct_sds, labels, p_values, ax=None, colors=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,4))
    else:
        fig = ax.figure
    
    x = np.arange(len(groups))
    if colors is None:
        colors = ['#3E7CB1', '#66A182', '#F5A623', '#D65A31', '#8C564B']

    bars = ax.bar(x, pct_changes,
        yerr=pct_sds,
        capsize=4,
        color=colors, edgecolor='black', linewidth=0.6)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{labels[gid]}" for gid in groups],
                       fontsize=9, rotation=15, ha='center')
    ax.set_ylabel("Percent change in nRMSE\n(relative to BASE_RS)", fontsize=11)
    ax.set_title(f"Stage C – {name}: Benefit of canopy‐height info", pad=20, fontsize=12)
    # ax.text(0.01, 0.97, rf"$\bf{{({lab})}}$ {labName}", transform=ax.transAxes,
    #                 ha='left', va='top', multialignment='center',  # every line right‑justified
    #                 fontsize=11)
    ax.margins(y=0.15)
    
    # Add labels with significance stars
    for bar, pct, p_val in zip(bars, pct_changes, p_values):
        stars = '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_y() + bar.get_height() + (1 if pct > 0 else -1) * 3,
                f"{pct:.1f}% {stars}", ha='center', va='bottom' if pct > 0 else 'top',
                fontsize=9, weight='bold')
                
    if ax is None:
        plt.tight_layout()
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Plot function
# ─────────────────────────────────────────────────────────────────────────────
def plot_pct_changes_without_title(lab, labName, groups, pct_changes, pct_sds, labels, p_values, ax=None, colors=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,4))
    else:
        fig = ax.figure
    
    x = np.arange(len(groups))
    if colors is None:
        colors = ['#3E7CB1', '#66A182', '#F5A623', '#D65A31', '#8C564B']

    # bars = ax.bar(x, pct_changes,
    #     yerr=pct_sds,
    #     capsize=4,
    #     color=colors, edgecolor='black', linewidth=0.6,
    #     error_kw={'linestyle': '--', 'linewidth': 1, 'capthick': 1.5})
    
    # Bars (no error bars here)
    bars = ax.bar(
        x,
        pct_changes,
        color=colors,
        edgecolor="black",
        linewidth=0.6
    )
    
    # Add error bars separately
    line, caplines, barlinecols = ax.errorbar(
        x, pct_changes,
        yerr=pct_sds,
        fmt="none",        # no markers/line
        ecolor="black",
        elinewidth=1,
        capsize=4
    )
    
    # Make stems dashed
    for blc in barlinecols:
        blc.set_linestyle((0, (3, 3)))  # dashed pattern
    
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    
    # Hide the x-axis labels on panels A and B by setting them to an empty list
    if lab in ["c","d"]:
        ax.set_xticklabels([f"{labels[gid]}" for gid in groups],
                       fontsize=14, rotation=15, ha='center')
    else:
        ax.set_xticklabels([])  # Remove x-axis labels for panels A and B
                       
    if lab in ["a","c"]:
        ax.set_ylabel("Percent change in nRMSE\n(relative to BASE_RS)", fontsize=14)
    # ax.set_title(f"Stage C – {name}: Benefit of canopy‐height info", pad=20, fontsize=12)
    
    ax.text(0.05, 0.97, f"({lab}) {labName}", transform=ax.transAxes,
                    ha='left', va='top', multialignment='left', 
                    fontsize=14)

    # Set y-axis tick label font size
    ax.tick_params(axis='y', labelsize=14)
    
    ax.margins(y=0.15)
    
    # Add labels with significance stars
    for bar, pct, p_val in zip(bars, pct_changes, p_values):
        stars = '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
        if stars:
            label_text = f"{pct:.1f}% $\\mathbf{{{stars}}}$"  # Bold stars only
        else:
            label_text = f"{pct:.1f}%"  # No stars, regular text
        
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_y() + bar.get_height() + (1 if pct > 0 else -1) * 3,
                label_text, ha='center', va='bottom' if pct > 0 else 'top',
                fontsize=14)
                
    if ax is None:
        plt.tight_layout()
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# MAIN: compute and plot percent change in nRMSE for Stage C
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    FILE_NAME  = "unc_chao_fia_data.xlsx"
    NA_VALUES  = ['1.#QNB','1.#INF','-1.#INF','nan','NaN','inf','-inf']
    output_dir = "/work/users/w/a/wayne128/Biomass_ML/Dataset/OutBiomassRaster/Fig4StageC_CHM_nocut"
    os.makedirs(output_dir, exist_ok=True)

    # --- load data ---
    cleaned_df, _ = load_and_preprocess_data(FILE_NAME, na_values=NA_VALUES)

    # --- Exclude cut plots (TRTCD = 10 in any condition) ---
    fia_cond  = pd.read_excel(FILE_NAME, sheet_name="fia_cond", na_values=NA_VALUES)
    trt_cols  = [c for c in ['TRTCD1', 'TRTCD2', 'TRTCD3'] if c in fia_cond.columns]
    cut_plt_cn = set(fia_cond.loc[fia_cond[trt_cols].isin([10]).any(axis=1), 'PLT_CN'])
    n_before  = len(cleaned_df)
    cleaned_df = cleaned_df[~cleaned_df['PLT_CN'].isin(cut_plt_cn)].copy()
    print(f"Cut-plot filter: removed {n_before - len(cleaned_df)} plots "
          f"(TRTCD=10). Remaining: {len(cleaned_df)}")

    y = cleaned_df["total_biomass_tons_ha"]

    # --- Bin y for stratification ---
    n_bins = 10  # Recommended starting point; adjust as needed
    binner = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    y_binned_array = binner.fit_transform(y.values.reshape(-1, 1)).flatten()
    y_binned = pd.Series(y_binned_array, index=y.index)
    
    # --- CV scheme ---
    # cv = KFold(n_splits=5, shuffle=True, random_state=42)
    # cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    

    # --- feature groups for Stage C ---
    groups = ["G12",
        "G17",
        "G14",
        # "G15",
        "G18",
        # "G19"
        ]   # see Table 2
    # labels = {
    #     "G-CHM25":"GEDI-derived CHM25",
    #     "G-Base+CHM25":"BASE_RS + CHM25",
    #     "G-Profile":"NAIP-DAP Profiles ",
    #     # "G15":"Profile + CHM",
    #     "G-Base+Profile":"BASE_RS + Profiles",
    #     # "G19":"BASE_RS + CHM + profile"
    # }
    labels = {
        "G12":"G-CHM25",
        "G17":"G-Base+CHM25",
        "G14":"G-Profile",
        # "G15":"Profile + CHM",
        "G18":"G-Base+Profile",
        # "G19":"BASE_RS + CHM + profile"
    }
    # baseline = BASE_RS = G16
    base_id = "G16"

    # --- scale BASE_RS predictors once ---
    X_base = cleaned_df[model_feature_sets[base_id]].dropna()
    y_base = pd.Series(y.loc[X_base.index])
    X_base = X_base.loc[y_base.index]
    scaler = StandardScaler().fit(X_base)
    Xb_scaled = pd.DataFrame(
        scaler.transform(X_base),
        index=X_base.index,
        columns=X_base.columns
    )
    

    ###############################################
    # --- define your four regressors in a dict ---
    estimators = {
        "XGB"     : XGB_TD_Regressor,
        "CatBoost": CatBoost_TD_Regressor,
        "LGBM"    : LGBM_TD_Regressor,
        "RealMLP" : RealMLP_TD_Regressor
    }
    
    # collect all results
    all_results = []
    combined_data = {}  # To store per-estimator data for combined plot
    
    # Check if results file already exists
    results_file = f"{output_dir}/stageC_results_nocut.csv"
    run_analysis = not os.path.exists(results_file)  # or use your preferred condition

    if run_analysis:
        print("Running analysis and saving results...")
    
        for name, Est in estimators.items():
            # 1) compute baseline for this model
            # baseline_nrmse = cv_nrmse(Est, Xb_scaled, y_base, cv, device="cpu")
            
            # correct: pull out only the mean (and optionally sd if you need it)
            # Align y_binned for baseline
            y_stratify_base = y_binned.loc[Xb_scaled.index].values
            baseline_mean, baseline_sd, baseline_nrmse_vals = cv_nrmse(
                Est, Xb_scaled, y_base, cv,
                y_stratify=y_stratify_base,
                cv_n_jobs=-1, device="cpu"
            )
            print(f"{name} baseline nRMSE = {baseline_mean:.4f} ± {baseline_sd:.4f}")
    
            
            # 2) compute pct_changes in parallel over groups
            #    here n_jobs controls # of concurrent group‐jobs
            results = Parallel(n_jobs=len(groups))(
                delayed(pct_for_group)(
                    gid, Est, baseline_mean, baseline_nrmse_vals,
                    cleaned_df, y, y_binned, cv
                )
                for gid in groups
            )
            
            #save the result and export to csv
            # all_results.extend(results)
            for gid, pct, mean_nrmse, sd_nrmse, p_val in results:
                all_results.append({
                  "estimator": name,
                  "baseline_mean": baseline_mean,
                  "group": gid,
                  "pct_change": pct,
                  "mean_nrmse": mean_nrmse,
                  "sd_nrmse": sd_nrmse,
                  "p_value": p_val
                })
            
            
            # convert list of tuples back into ordered lists
            results_dict = {gid: (pct, m, s, p_val) for gid, pct, m, s, p_val in results}
        
            pct_changes     = [results_dict[gid][0] for gid in groups]
            raw_nrmse_means = [results_dict[gid][1] for gid in groups]
            raw_nrmse_sds   = [results_dict[gid][2] for gid in groups]
            p_values        = [results_dict[gid][3] for gid in groups]
            
            pct_sds = [sd / baseline_mean * 100 for sd in raw_nrmse_sds]
    
            # 3) plot individual
            fig, ax = plot_pct_changes(name, groups, pct_changes, pct_sds, labels, p_values)
            fig_out = f"{output_dir}/Fig4StageC_{name}_5cv_benchmark_nocut.png"
            fig.savefig(fig_out, dpi=400)
            plt.close(fig)
            print(f"Saved Figure to: {fig_out}\n")
            
            
            # Store for combined plot
            combined_data[name] = (pct_changes, pct_sds, p_values)
            
        # 4) turn saved result into DataFrame 
        # if I need to change the figure and not necessary rerun the code again
        df = pd.DataFrame(all_results)
        # save as CSV
        df.to_csv(results_file, index=False)
        print(f"Saved results to: {results_file}")
    
    else:
        print("Loading existing results...")
        # Load the saved results
        df = pd.read_csv(results_file)
        
        # Recreate combined_data from saved results
        for estimator in df['estimator'].unique():
            estimator_data = df[df['estimator'] == estimator]
            
            # Create ordered lists matching the original groups order
            results_dict = {}
            for _, row in estimator_data.iterrows():
                gid = row['group']
                pct = row['pct_change']
                sd_nrmse = row['sd_nrmse']
                baseline_mean = row['baseline_mean']
                p_val = row['p_value']
                pct_sd = sd_nrmse / baseline_mean * 100
                results_dict[gid] = (pct, pct_sd, p_val)
            
            # Extract ordered lists
            pct_changes = [results_dict[gid][0] for gid in groups]
            pct_sds = [results_dict[gid][1] for gid in groups]
            p_values = [results_dict[gid][2] for gid in groups]
            
            combined_data[estimator] = (pct_changes, pct_sds, p_values)

    
    # 5) Combined 2x2 subplot figure at the end
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    axes = axes.flatten()
    estimator_names = list(estimators.keys())
    colors = ['#3E7CB1', '#66A182', '#F5A623', '#D65A31', '#8C564B']  # Shared colors

    # Re-use data from all_results
    for i, labname in enumerate(estimator_names):
        pct_changes, pct_sds, p_values = combined_data[labname]
        lab = chr(ord('a') + i)  # Convert number 0, 1, 2, 3 to A, B, C, D
        plot_pct_changes_without_title(lab, labname, groups, pct_changes, pct_sds, labels, p_values, ax=axes[i])

    plt.tight_layout()
    combined_fig_out = f"{output_dir}/Fig4StageC_combined_2x2_nocut.png"
    plt.savefig(combined_fig_out, dpi=400)
    plt.close()
    print(f"Saved Combined Figure to: {combined_fig_out}")
        