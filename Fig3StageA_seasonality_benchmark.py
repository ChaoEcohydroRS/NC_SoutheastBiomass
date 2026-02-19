# stageA_seasonality_benchmark.py
import warnings
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but StandardScaler was fitted with feature names"
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error

from pytabkit import CatBoost_TD_Regressor,XGB_TD_Regressor

from scipy.stats import wilcoxon, ttest_rel            # NEW

from model_config import model_feature_sets   # make sure G1a/G1sw/G1c are in this dict
from model_utils  import load_and_preprocess_data

import torch
torch.set_float32_matmul_precision('high')

# ------------------------------------------------------------------
# Cross-validated nRMSE (explicit KFold loop)
# ------------------------------------------------------------------

# def cv_nrmse_catboost(X, y, n_splits=10, device="cpu"):
#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
#     nrmse_folds = []

#     for train_idx, test_idx in kf.split(X):
#         model = CatBoost_TD_Regressor(device=device)     # tuned default
#         model.fit(X.iloc[train_idx], y.iloc[train_idx])

#         preds  = model.predict(X.iloc[test_idx])
#         rmse   = np.sqrt(mean_squared_error(y.iloc[test_idx], preds))
#         nrmse  = rmse / y.iloc[test_idx].mean()
#         nrmse_folds.append(nrmse)

#     return np.mean(nrmse_folds), np.std(nrmse_folds)
    
def cv_nrmse_catboost(X, y, cv, device="cpu"):
    # nrmse_vals = []
    fold_vals = []                                      # keep fold metrics
    for train_idx, test_idx in cv.split(X):
        # model = CatBoost_TD_Regressor(device=device)
        model = XGB_TD_Regressor(device=device)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds  = model.predict(X.iloc[test_idx])
        rmse   = np.sqrt(mean_squared_error(y.iloc[test_idx], preds))
        # nrmse_vals.append(rmse / y.iloc[test_idx].mean())
        fold_vals.append(rmse / y.iloc[test_idx].mean())
    # return np.mean(nrmse_vals), np.std(nrmse_vals)
    return np.mean(fold_vals), np.std(fold_vals), fold_vals   #

# ------------------------------------------------------------------
# 2. define a wilcoxon test function
# ------------------------------------------------------------------
def paired_tests(x, y, name_x, name_y, alpha=0.05):
    stat_w, p_w = wilcoxon(x, y, alternative="greater")   # “greater” = x > y → y better
    stat_t, p_t = ttest_rel(x, y, alternative="greater")

    print(f"\n--- {name_x} vs {name_y} ---")
    print(f"Wilcoxon p = {p_w:.4g}")
    print(f"Paired-t p = {p_t:.4g}")

    return p_w   # return wilcoxon p for correction

# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
if __name__ == "__main__":

    # ----- paths ---------------------------------------------------
    output_dir = "/work/users/w/a/wayne128/Biomass_ML/Dataset/OutBiomassRaster"
    file_name  = "unc_chao_fia_data.xlsx"
    na_vals    = ['1.#QNB', '1.#INF', '-1.#INF', 'nan', 'NaN', 'inf', '-inf']

    # ----- load tabular data --------------------------------------
    cleaned_df, short_tag = load_and_preprocess_data(file_name, na_vals)

    # ----- # Prepare features and targets variable ----------------------------------------
    exclude_cols = ['ID', 'PLT_CN', 'MEASYEAR', 'hrdwdDRYBIO_AGac_live', 'sftwdDRYBIO_AGac_live',
                    'hrdwd_biomass_tons_ha', 'sftwd_biomass_tons_ha', 'total_biomass_tons_ha',
                    'hrdwd_proportion', 'sftwd_proportion']
    X = cleaned_df.drop(columns=exclude_cols)
    # target
    y_total = cleaned_df["total_biomass_tons_ha"]

    # ----- three Sentinel-2 stacks --------------------------------
    stageA_ids = ["G1a", "G1sw", "G1c"]
    desc_map   = {"G1a": "S2_SUM", "G1sw": "SUM+WIN", "G1c": "S2_ALL"}

    # results = {}
    results_mean_sd = {}
    results_folds   = {}                                 # store all folds
    
    # 10 × 10-fold repeated CV provides the best balance of bias, variance, and runtime 
    # while still giving each plot ten independent out-of-fold evaluations.
    rkf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)

    for gid in stageA_ids:
        print(f"\n=== Stage A model {gid} ===")
        feats = model_feature_sets[gid]

        # drop NaNs in predictors + target
        X = cleaned_df[feats].dropna()
        y = y_total.loc[X.index].dropna()
        X = X.loc[y.index]  
        
        print(f"Size after dropping NaNs in predictors & targets: {len(X)}")
        
        # scale predictors (CatBoost is fine w/o scaling, but scaling makes metrics comparable)
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            index=X.index,
            columns=X.columns,
        )

        # mean_nrmse, sd_nrmse = cv_nrmse_catboost(
        #     X_scaled, y, n_splits=10, device="cpu"  # change to "cuda:0" if GPU
        # )
        
        # mean_nrmse, sd_nrmse = cv_nrmse_catboost(X_scaled, y, rkf, device="cpu")
        
        # print(f"nRMSE = {mean_nrmse:.4f} ± {sd_nrmse:.4f}")
        # results[gid] = (mean_nrmse, sd_nrmse)
        
        mean_, sd_, folds = cv_nrmse_catboost(X_scaled, y, rkf, device="cpu")
        results_mean_sd[gid] = (mean_, sd_)
        results_folds[gid]   = folds                    
        print(f"\n=== {gid}  nRMSE = {mean_:.4f} ± {sd_:.4f}")


    # ----------------------------------------------------------------
    # significant test：G1sw vs G1c vs G1a
    # ----------------------------------------------------------------
    # prepare per-folder nRMSE（the collected data in previous steps）
    a  = np.array(results_folds["G1a"])   # length = fold × repeats
    sw = np.array(results_folds["G1sw"])
    c  = np.array(results_folds["G1c"])

    # ------------------------------------------------------------------
    # 3. compare three pairs
    # ------------------------------------------------------------------
    p_vals = []
    p_vals.append( paired_tests(a,  sw, "G1a",  "G1sw") )  # add winter
    p_vals.append( paired_tests(sw, c, "G1sw", "G1c") )    # add spring and winter
    p_vals.append( paired_tests(a,  c, "G1a",  "G1c") )    # end to end


    # ------------------------------------------------------------------
    # 4. Bonferroni correction
    # ------------------------------------------------------------------
    alpha_raw = 0.05
    alpha_bon = alpha_raw / len(p_vals)       # = 0.05 / 3 ≈ 0.017
    
    print(f"\nBonferroni threshold α_B = {alpha_bon:.3f}")
    
    labels = ["G1a→G1sw", "G1sw→G1c", "G1a→G1c"]
    for lab, p in zip(labels, p_vals):
        mark = "✓ sig" if p < alpha_bon else "✗ non-sign"
        print(f"{lab:10s}: p = {p:.4g}  {mark} (α_B)")


    ## ----- Figure 1 bar chart -------------------------------------
    labels = [f"{g}\n({desc_map[g]})" for g in stageA_ids]
    means  = [results_mean_sd[g][0] for g in stageA_ids]
    sds    = [results_mean_sd[g][1] for g in stageA_ids]
    colors  = ['#3E7CB1', '#66A182', '#F5A623']

    x = np.arange(len(labels))
    
    # ─── Draw the bars ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(x, means, yerr=sds, capsize=5, color=colors,
                  edgecolor='black', linewidth=0.6, width=0.6)

    # Compute the highest top of any error bar
    # Pad it by 20% and set as the y‐limit
    # 1) Compute the highest top of any error bar
    y_max = max(m + s for m, s in zip(means, sds))
    ax.set_ylim(0, y_max*1.20)
    
    # ─── Basic styling ──────────────────────────────────────────────
    ax.set_ylabel('nRMSE', fontsize=14)
    # ax.set_title('Stage A – Sentinel-2 seasonality effect on AGB prediction', pad=12)
    ax.set_xticks(x, labels, fontsize=14)
    
    ax.tick_params(axis='y', labelsize=14)

    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

    # ─── Annotate bar values ───────────────────────────────────────
    for rect, m in zip(bars, means):
        ax.text(
            rect.get_x() + rect.get_width()/2,
            m + sds[0]*0.3,        # small offset above the bar
            f'{m:.3f}',
            ha='center', va='bottom',
            fontsize=14, fontweight='bold'
        )
    
    # ─── Add significance brackets ─────────────────────────────────
    #The below info is preset based on calculation
    dh              = y_max * 0.03      # bracket height step
    bracket_height  = y_max * 1.02      # base height for first bracket
    
    # 1) G1a vs G1sw (p < 0.001 → ***)
    x1, x2 = x[0], x[1]
    y = bracket_height
    ax.plot([x1, x1, x2, x2], [y, y+dh, y+dh, y], lw=1.2, color='green', linestyle='--')
    ax.text((x1+x2)/2, y+dh*0.2, '***', ha='center', va='bottom', color='green', fontsize=12)
    
    # 2) G1sw vs G1c (p < 0.01 → **)
    x1, x2 = x[1], x[2]
    y = bracket_height + dh*1.1
    ax.plot([x1, x1, x2, x2], [y, y+dh, y+dh, y], lw=1.2, color='red', linestyle='--')
    ax.text((x1+x2)/2, y-dh*1.2, '**', ha='center', va='bottom', color='red', fontsize=12)
    
    # 3) G1a vs G1c (p < 0.001 → ***)
    x1, x2 = x[0], x[2]
    y = bracket_height + dh*2.4
    ax.plot([x1, x1, x2, x2], [y, y+dh, y+dh, y], lw=1.2, color='blue',linestyle='--')
    ax.text((x1+x2)/2, y+dh*1.0, '***', ha='center', va='bottom', fontsize=12, color='blue')

    plt.tight_layout()
    fig_out = f"{output_dir}/FigStageA_S2_Seasonal_XGB__{short_tag}.jpg"
    plt.savefig(fig_out, dpi=550, format="jpeg")
    plt.show()
    print(f"\nFigure saved to: {fig_out}")
