import warnings
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, RepeatedKFold
from xgboost import XGBRegressor                      
from typing import Union  # Added for type hint compatibility
from matplotlib import gridspec
from scipy.stats import binned_statistic
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator
from statsmodels.nonparametric.smoothers_lowess import lowess

import shap
from shap import dependence_plot

import joblib
from pathlib import Path

from model_utils  import load_and_preprocess_data
from model_config import model_feature_sets

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import seaborn as sns

def create_elegant_shap_plot(shap_exp, max_display=20, figsize=(16, 10)):
    """
    Create an elegant SHAP plot with feature names in the middle,
    bar chart on the left, and violin plot on the right.
    """
    plt.close('all')
    
    # Get the data for plotting
    shap_values = shap_exp.values
    feature_names = shap_exp.feature_names
    
    # Calculate mean absolute SHAP values and sort
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-max_display:][::-1]
    
    # Prepare data
    top_features = [feature_names[i] for i in top_indices]
    top_mean_abs = mean_abs_shap[top_indices]
    top_shap_values = shap_values[:, top_indices]
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=300)
    
    # Set up positions
    y_positions = np.arange(len(top_features))
    bar_width = 0.4
    
    # Left side: Horizontal bar chart (mean absolute SHAP values)
    bars = ax.barh(y_positions, -top_mean_abs, height=bar_width, 
                   color='#d62728', alpha=0.8, align='center')
    
    # Add values on the bars
    for i, (bar, val) in enumerate(zip(bars, top_mean_abs)):
        ax.text(bar.get_x() - 0.5, bar.get_y() + bar.get_height()/2, 
                f'+{val:.2f}', ha='right', va='center', 
                fontweight='bold', color='#d62728', fontsize=10)
    
    # Right side: Violin plots
    violin_positions = y_positions
    violin_data = [top_shap_values[:, i] for i in range(len(top_features))]
    
    # Create violin plots on the right side
    parts = ax.violinplot(violin_data, positions=violin_positions, 
                         vert=False, widths=bar_width*1.5, showmeans=False, 
                         showmedians=False, showextrema=False)
    
    # Color the violin plots based on SHAP values
    for pc, shap_vals in zip(parts['bodies'], violin_data):
        # Color mapping based on SHAP value magnitude
        colors = plt.cm.RdBu_r(np.linspace(0.2, 0.8, 100))
        pc.set_facecolors(colors[50])  # Neutral color, can be customized
        pc.set_alpha(0.7)
    
    # Feature names in the middle
    max_bar_width = max(top_mean_abs)
    text_x_position = 0  # Middle position
    
    for i, feature in enumerate(top_features):
        ax.text(text_x_position, y_positions[i], feature, 
               ha='center', va='center', fontweight='bold', 
               fontsize=11, bbox=dict(boxstyle='round,pad=0.3', 
                                    facecolor='white', alpha=0.8))
    
    # Formatting
    ax.set_yticks([])
    ax.set_xlabel('SHAP Value Impact', fontsize=12, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    # Set x-axis limits to accommodate both sides
    x_margin = max_bar_width * 0.3
    max_violin_width = max([np.max(np.abs(vals)) for vals in violin_data])
    ax.set_xlim(-max_bar_width - x_margin, max_violin_width + x_margin)
    
    # Add labels for left and right sides
    ax.text(-max_bar_width/2, len(top_features), 'Mean Absolute\nSHAP Value', 
           ha='center', va='bottom', fontweight='bold', fontsize=12)
    ax.text(max_violin_width/2, len(top_features), 'SHAP Value\nDistribution', 
           ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Grid
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    return fig


# ------------------------------------------------------------
# Partial-dependence panel with smaller marginals
# ------------------------------------------------------------
def partial_dep_scatter(
        x: np.ndarray,
        shap_vals: np.ndarray,
        feat_name: str,
        save_path: Union[Path, None] = None,  # Changed to Union for compatibility
        n_bins: int = 25,
        ax: Union[plt.Axes, None] = None,  # Changed to Union for compatibility
):
    """
    Scatter + mean±SD  (small marginals only when ax is None).
    If `save_path` is None, the caller takes care of saving the figure.
    """
    # ------------------------------------------------------------------
    # 1 ▪ Decide where to draw
    # ------------------------------------------------------------------
    own_fig = ax is None
    if own_fig:
        fig = plt.figure(figsize=(5, 5))             # slightly smaller overall
        gs  = gridspec.GridSpec(
            2, 2,
            width_ratios=[4.2, 0.6],                   # 0.6 instead of 1.2
            height_ratios=[0.6, 4.2],                  # idem
            wspace=0.05, hspace=0.05
        )
        ax_sc = fig.add_subplot(gs[1, 0])
        ax_hx = fig.add_subplot(gs[0, 0], sharex=ax_sc)
        ax_hy = fig.add_subplot(gs[1, 1], sharey=ax_sc)
    else:
        ax_sc = ax

    # ------------------------------------------------------------------
    # 2 ▪ Scatter + mean ± SD ribbon
    # ------------------------------------------------------------------
    ax_sc.scatter(x, shap_vals, alpha=0.35, s=18, linewidth=0)

    # mean and ±SD in equal-width bins
    bin_mean, bin_edges, _ = binned_statistic(x, shap_vals, 'mean', n_bins)
    bin_sd,   _,           _ = binned_statistic(x, shap_vals, 'std',  n_bins)
    centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    ax_sc.plot(centres, bin_mean, color='darkorange', lw=2, label='Mean')
    ax_sc.fill_between(
        centres, bin_mean-bin_sd, bin_mean+bin_sd,
        color='orange', alpha=0.25, label='±1 SD'
    )
    ax_sc.axhline(y=0, color='grey', linestyle='--', linewidth=0.5)  # Add y=0 line
    ax_sc.legend(frameon=False, fontsize=9, loc='upper left')
    ax_sc.set_xlabel(feat_name)
    ax_sc.set_ylabel("SHAP value")

    # ── 3  marginal histograms  ─────────────────────────────
    # Marginal histograms without boxes
    if own_fig:
        ax_hx.hist(x, bins=30, color='#44669e', alpha=0.6, edgecolor='#44669e')
        ax_hy.hist(shap_vals, bins=30, orientation='horizontal',
                   color='#44669e', alpha=0.6, edgecolor='#44669e')
        for spine in (*ax_hx.spines.values(), *ax_hy.spines.values()):
            spine.set_visible(False)
        for marg in (ax_hx, ax_hy):
            marg.tick_params(left=False, bottom=False,
                             labelleft=False, labelbottom=False)
        ax_hx.set_ylabel('');  ax_hy.set_xlabel('')

    # Save figure if requested
    if save_path and own_fig:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close(fig)

# ------------------------------------------------------------
# scatter/hex + mean ± SD ribbon + marginals
# ------------------------------------------------------------
def pdp_panel(ax: plt.Axes,
              x: np.ndarray,
              y: np.ndarray,
              feat: str,
              n_bins: int = 25,
              ribbon_color: str = 'darkorange',
              q_low: float = 1,         
              q_high: float = 99,
              stats_q_low=0.5, stats_q_high=99.5) -> None:
    """Draw PDP scatter/hexbin on *ax* and attach marginal histograms.
       Automatically crop coordinate axes by quantiles to focus on the main data area"""
    
    # ------------------------- 1) choose two masks -------------------------
    # mask for the axes + point cloud
    x_lo, x_hi = np.percentile(x, [q_low, q_high])
    y_lo, y_hi = np.percentile(y, [q_low, q_high])
    mask_plot  = (x >= x_lo) & (x <= x_hi) & (y >= y_lo) & (y <= y_hi)
    
    # mask for the line / ribbon       (wider, maybe full range)
    xs_lo, xs_hi = np.percentile(x, [stats_q_low, stats_q_high])
    ys_lo, ys_hi = np.percentile(y, [stats_q_low, stats_q_high])
    mask_stats   = (x >= xs_lo) & (x <= xs_hi) & (y >= ys_lo) & (y <= ys_hi)
    
    # ------------------------- 2) scatter or hexbin ------------------------
    x_plot, y_plot = x[mask_plot], y[mask_plot]
    if len(x_plot) > 9000:                              # dense → hexbin
        ax.hexbin(x_plot, y_plot, gridsize=60, cmap='Blues', mincnt=3, linewidths=0)
    else:                                          # sparse → scatter
        ax.scatter(x_plot, y_plot, s=10, alpha=.25, linewidth=0)

    # ------------------------- 3) mean ± sd on the *wide* mask -------------
    x_stats, y_stats = x[mask_stats], y[mask_stats]
    mean, edges, _ = binned_statistic(x_stats, y_stats, 'mean', n_bins)
    sd,   _,      _ = binned_statistic(x_stats, y_stats, 'std',  n_bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    # ax.plot(centres, mean, color=ribbon_color, lw=2, label='Mean')
    # ax.fill_between(centres, mean-sd, mean+sd,
    #                 color=ribbon_color, alpha=.25, label='±1 SD')
    # To get rid of gap
    valid = ~np.isnan(mean)              # bins that have at least one point
    ax.plot(centres[valid], mean[valid], lw=2, color=ribbon_color, zorder=5)
    ax.fill_between(centres[valid],
                    mean[valid]-sd[valid],
                    mean[valid]+sd[valid],
                    alpha=.25, color=ribbon_color, zorder=4)

    # axes cosmetics
    ax.axhline(0, color='grey', lw=.8, ls='--')
    # ax.set_xlabel(feat)
    ax.set_ylabel("SHAP value")
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    
    # automatic zoom-in and leave 5% buffer
    dx = (x_hi - x_lo) * 0.02
    dy = (y_hi - y_lo) * 0.02
    ax.set_xlim(x_lo - dx, x_hi + dx)
    ax.set_ylim(y_lo - dy, y_hi + dy)

    # marginals
    div   = make_axes_locatable(ax)
    atop  = div.append_axes("top",   size=0.35, pad=0.03, sharex=ax)
    aright= div.append_axes("right", size=0.35, pad=0.03, sharey=ax)
    atop.hist(x, 30, color='#44669e', alpha=.6, edgecolor='#44669e')
    aright.hist(y, 30, orientation='horizontal',
                color='#44669e', alpha=.6, edgecolor='#44669e')
    for sp in (*atop.spines.values(), *aright.spines.values()):
        sp.set_visible(False)
    atop.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
    aright.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)

# ------------------------------------------------------------
# scatter/hex + loess line ± SD ribbon + marginals
# ------------------------------------------------------------
def pdp_loess_panel(ax: plt.Axes,
              x: np.ndarray,
              y: np.ndarray,
              feat: str,
              n_bins: int = 25,
              ribbon_color: str = 'darkorange',
              q_low: float = 1,
              q_high: float = 99,
              stats_q_low: float = 0,
              stats_q_high: float = 100,
              loess_frac: float = 0.25,        # span of the smoother
              ribbon_k: float = 1.0,           # 1 → ±1 σ ; 1.96 → 95 % band
              ) -> None:
    """
    Draw PDP scatter/hexbin on *ax* with a LOESS mean line and a ±k·σ(x) ribbon.
    The axes zoom to the [q_low, q_high] percentiles, while the smoother
    can be fitted on a wider (stats_q_*) window to avoid edge gaps.
    """

    # ------------------------- 1) choose two masks -------------------------
    x_lo, x_hi = np.percentile(x, [q_low, q_high])
    y_lo, y_hi = np.percentile(y, [q_low, q_high])
    mask_plot  = (x >= x_lo) & (x <= x_hi) & (y >= y_lo) & (y <= y_hi)

    xs_lo, xs_hi = np.percentile(x, [stats_q_low, stats_q_high])
    ys_lo, ys_hi = np.percentile(y, [stats_q_low, stats_q_high])
    mask_stats   = (x >= xs_lo) & (x <= xs_hi) & (y >= ys_lo) & (y <= ys_hi)

    # ------------------------- 2) scatter or hexbin ------------------------
    x_plot, y_plot = x[mask_plot], y[mask_plot]
    if len(x_plot) > 9000:                       # dense → hexbin
        ax.hexbin(x_plot, y_plot, gridsize=60, cmap='Blues',
                  mincnt=3, linewidths=0)
    else:                                        # sparse → scatter
        ax.scatter(x_plot, y_plot, s=10, alpha=.25, linewidth=0)

    # ------------------------- 3) LOESS mean ± k·σ ribbon ------------------
    x_stats, y_stats = x[mask_stats], y[mask_stats]

    # sort once – LOWESS is fine unordered, but plotting needs order
    order           = np.argsort(x_stats)
    xs, ys          = x_stats[order], y_stats[order]

    # LOWESS for E[Y|X]
    loess_mean = lowess(ys, xs, frac=loess_frac,
                        it=2,  # robustness iters off; add it=2 for robustness
                        return_sorted=False)

    # LOWESS for Var[Y|X]  →  σ(x)
    resid       = ys - loess_mean
    loess_var   = lowess(resid**2, xs, frac=loess_frac,
                         it=2, return_sorted=False)
    loess_sigma = np.sqrt(np.clip(loess_var, 0, None))   # guard negatives

    # plot
    ax.plot(xs, loess_mean, color=ribbon_color, lw=2, zorder=5)
    ax.fill_between(xs,
                    loess_mean - ribbon_k * loess_sigma,
                    loess_mean + ribbon_k * loess_sigma,
                    color=ribbon_color, alpha=.25, zorder=4)

    # ------------------------- 4) cosmetics -------------------------------
    ax.axhline(0, color='grey', lw=.8, ls='--')
    # ax.set_xlabel(feat)
    ax.set_ylabel("SHAP value", fontsize=14)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))

    # automatic zoom‑in (5 % buffer)
    dx = (x_hi - x_lo) * 0.02
    dy = (y_hi - y_lo) * 0.02
    ax.set_xlim(x_lo - dx, x_hi + dx)
    ax.set_ylim(y_lo - dy, y_hi + dy)

    # ------------------------- 5) marginals -------------------------------
    div    = make_axes_locatable(ax)
    atop   = div.append_axes("top",   size=0.35, pad=0.03, sharex=ax)
    aright = div.append_axes("right", size=0.35, pad=0.03, sharey=ax)

    atop.hist(x, 30, color='#44669e', alpha=.6, edgecolor='#44669e')
    aright.hist(y, 30, orientation='horizontal',
                color='#44669e', alpha=.6, edgecolor='#44669e')

    for sp in (*atop.spines.values(), *aright.spines.values()):
        sp.set_visible(False)
    atop.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
    aright.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)


def fix_shap_labels(ax, shift_amount=5):
    """
    Fix SHAP bar plot labels after inverting x-axis
    Shifts all value labels to the left by a specified amount
    
    Parameters:
    ax: matplotlib axis object
    shift_amount: how much to shift labels to the left (positive number moves left)
    """
    # Wait for all rendering to complete
    plt.draw()
    
    # Get the current x-axis limits (after inversion)
    xlim = ax.get_xlim()
    
    # Collect all text objects and their properties
    texts = list(ax.texts)  # Convert to list to make a copy
    
    # Store text properties before removing
    text_data = []
    for text in texts:
        text_data.append({
            'label': text.get_text(),
            'x': text.get_position()[0],
            'y': text.get_position()[1],
            'va': text.get_va(),
            'ha': text.get_ha(),
            'fontsize': text.get_fontsize(),
            'color': text.get_color()
        })
    
    # Clear all existing texts
    for text in texts:
        text.remove()
    
    # Re-add texts with shifted positioning
    for data in text_data:
        label = data['label']
        
        # Check if this is a value label (contains numbers and +/-)
        is_value_label = any(char.isdigit() or char in ['+', '.'] for char in label)
        
        if is_value_label:
            # Shift the label to the left while maintaining its y position
            # Since x-axis is inverted, subtract to move left
            new_x = data['x'] - shift_amount
            ax.text(new_x, data['y'], label,
                   va='center', ha='center',  # Keep center alignment
                   color='red', fontweight='normal', fontsize=10,
                   transform=ax.transData)
        else:
            # Keep feature names in their original position
            ax.text(data['x'], data['y'], label,
                   va=data['va'], ha=data['ha'],
                   fontsize=data['fontsize'],
                   transform=ax.transData)


                
# ─────────────────────────────────────────────────────────
# MAIN: Stage D – SHAP interactions & importance (XGB-TD)
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ----------------------- config -----------------------
    warnings.filterwarnings("ignore")
    output_dir = "/work/users/w/a/wayne128/Biomass_ML/Dataset/OutBiomassRaster"
    out_dir   = Path(output_dir)
    
    model_group_ID = 'G20'
    force_retraining: bool = False             # True → ignore cache

    # 1 ▪ Load / preprocess
    cleaned_df, _ = load_and_preprocess_data(
        "unc_chao_fia_data.xlsx",
        na_values=["1.#QNB", "1.#INF", "-1.#INF", "nan", "NaN", "inf", "-inf"]
    )

    y = cleaned_df["total_biomass_tons_ha"]
    # X = cleaned_df[model_feature_sets["G19"]].dropna()
    X = cleaned_df[model_feature_sets[model_group_ID]].dropna()

    idx = X.index.intersection(y.index)
    X, y = X.loc[idx], y.loc[idx]

    # Optional scaling
    # scaler        = StandardScaler()
    feature_names = X.columns.tolist()
    # X_scaled      = pd.DataFrame(
    #     scaler.fit_transform(X), index=X.index, columns=feature_names
    # )
    
    X_scaled = X.copy()                      # no scaling for now

    # ---------------- CV & model params ------------------
    # kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)

    # --- TD hyper-parameter block (mapped to XGBRegressor names) ---
    td_params = dict(
        n_estimators       = 1000,
        max_depth          = 9,
        tree_method        = "hist",   # "gpu_hist" if CUDA is available
        max_bin            = 256,
        learning_rate      = 5e-2,     # lr → learning_rate
        min_child_weight   = 2.0,
        reg_lambda         = 0.0,
        subsample          = 0.7,
        objective          = "reg:squarederror",
        random_state       = 42,
    )
    
    # ---------------- cache paths ------------------------
    
    out_dir.mkdir(parents=True, exist_ok=True)
    shap_cache   = out_dir / f"{model_group_ID}_shap_exp.pkl"
    int_cache    = out_dir / f"{model_group_ID}_shap_interaction_values.pkl"
    
    
    # ------------------------------------------------------------------
    # 1 ▪ Use cached objects if allowed and present
    # ------------------------------------------------------------------
    use_cache = (not force_retraining) and shap_cache.exists() and int_cache.exists()
    
    if use_cache:
        # --- instant load, skip the whole CV loop ---------------------
        try:
            shap_exp = joblib.load(shap_cache)
            all_int_vals = joblib.load(int_cache)
            shap_values_all = shap_exp.values
            base_val = shap_exp.base_values[0]
        except Exception as e:
            print(f"Error loading cached files: {e}")
            use_cache = False
    
    if not use_cache:
        # ------------------------------------------------------------------
        # 2 ▪ Run CV → train, explain, collect per‑fold artefacts
        # --- compute from scratch ------------------------
        # ------------------------------------------------------------------
        all_exps, all_int_vals, all_X_test = [], [], []

        for tr, te in cv.split(X_scaled):
            # 2.1 Train model
            model = XGBRegressor(**td_params)
            model.fit(
                X_scaled.iloc[tr], y.iloc[tr],
                eval_set=[(X_scaled.iloc[te], y.iloc[te])],   # validation fold
                verbose=False
            )
    
            # Tree SHAP (native to XGBoost)
            explainer  = shap.TreeExplainer(model, feature_names=feature_names,
                                            model_output="raw")
            X_test     = X_scaled.iloc[te]
            exp_fold    = explainer(X_test)  
            all_exps.append(exp_fold)
            
            # Interaction values (optional) ------------------------------------
            int_vals = explainer.shap_interaction_values(X_test)  # (n_i, p, p)
            all_int_vals.append(int_vals)
            all_X_test.append(X_test)

        # ---------------------------------------------------------------------------
        # 3 ▪ Combine per‑fold objects
        # ---------------------------------------------------------------------------
        values = np.concatenate([exp.values for exp in all_exps], axis=0)
        base_values = np.concatenate([exp.base_values for exp in all_exps], axis=0)
        data = pd.concat(all_X_test, axis=0)  # Aligns with your existing all_X_test
        
        shap_exp = shap.Explanation(values=values, base_values=base_values, data=data, feature_names=feature_names)
        shap_values_all = shap_exp.values  # (N,p) array
        base_val = shap_exp.base_values[0]  # scalar (assuming consistent base across folds)
        
        # Save to cache
        try:
            joblib.dump(shap_exp, shap_cache, compress='zlib')
            joblib.dump(all_int_vals, int_cache, compress='zlib')
        except Exception as e:
            print(f"Error saving cache files: {e}")
    
    
    # =====================================================
    #  Visualisations
    # =====================================================
    # # Interaction heatmap
    # per_fold_mean = [np.abs(iv).mean(axis=0) for iv in all_int_vals]
    # int_mat = np.mean(per_fold_mean, axis=0)
    
    # fig, ax = plt.subplots(figsize=(12, 10))
    
    # sns.heatmap(int_mat, 
    #     xticklabels=feature_names, 
    #     yticklabels=feature_names, 
    #     cmap="viridis", 
    #     annot=True, 
    #     fmt=".2f",
    #     ax=ax)
    # # tick‑label size
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=6)
    # ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=6)
    
    # plt.title("SHAP Interaction Heatmap (XGB TD)")
    # plt.tight_layout()
    # plt.savefig(out_dir / f"{model_group_ID}_shap_interaction_heatmap.png", dpi=120)
    # plt.close()

    # # 3.4‑a  Mean‑|SHAP| bar (top‑20) ------------------------------------------
    # plt.figure(figsize=(10, 8))
    # shap.plots.bar(shap_exp, max_display=20, show=False)
    # plt.title("Top-20 Features by Mean Absolute SHAP (XGB TD)")
    # plt.tight_layout(); 
    # plt.savefig(out_dir / f"{model_group_ID}_shap_bar_plot.png"); 
    # plt.close()
    
    # # 3.5 Layered Violin plot (top-20)
    # plt.figure(figsize=(10, 8))
    # shap.plots.violin(shap_exp, plot_type="layered_violin",
    #               max_display=20, show=False)
    # plt.title("SHAP Layered Violin Plot (XGB TD)")
    # plt.tight_layout(); 
    # plt.savefig(out_dir / f"{model_group_ID}_shap_layered_violin_plot.png"); 
    # plt.close()
    
    ##########################################
    # Bar + Violin two-panel plot ----------------------------------------------
    plt.close('all')  # Close any existing figures
    # fig, axs = plt.subplots(1, 2, figsize=(14, 5), dpi=300,constrained_layout=True)
    fig, axs = plt.subplots(1, 2, figsize=(36, 8), dpi=300)
    # Adjust subplot parameters to give more space for labels
    plt.subplots_adjust(left=0.15, right=0.9, wspace=0.8)
    
    shap.plots.bar(shap_exp, max_display=20, show=False, ax=axs[0])

    axs[0].set_title("(A) Mean Absolute SHAP Value", loc='right', fontsize=14, pad=20)
    
    # Flip the bar plot to face left
    axs[0].invert_xaxis()
    
    # Move y-axis to the right
    axs[0].yaxis.tick_right()
    axs[0].yaxis.set_label_position("right")
    
    # Remove the y-axis tick labels to avoid overlap with right panel
    axs[0].set_yticklabels([])
    
    axs[0].spines['right'].set_visible(True)
    axs[0].spines['left'].set_visible(False)
    
    # FIX THE LABELS - ADD THIS LINE
    # fix_shap_labels(axs[0])
    fix_shap_labels(axs[0], shift_amount=-10)  # Shift more to the left


    plt.sca(axs[1])                 # <‑‑ make axs[1] the active axis
    shap.plots.violin(shap_exp,
                      plot_type="layered_violin",
                      max_display=20,
                      show=False)    # no ax kwarg here!
    axs[1].set_title("(B) SHAP Value Distribution", loc='right', fontsize=14, pad=20)
    
    # Center align and adjust padding from the axis
    axs[1].set_yticklabels(axs[1].get_yticklabels(), ha='center')
    axs[1].tick_params(axis='y', which='major', pad=35)  # Increase pad to move labels away from axis
    
    # fig.suptitle("SHAP Feature Importance (XGB TD)", fontsize=16)
    # plt.tight_layout()
    plt.savefig(out_dir / f"{model_group_ID}_shap_two_panel_plot_bar_violin.png",
        bbox_inches="tight",   # include everything that was drawn
        pad_inches=0.1,dpi=500)
    plt.savefig(out_dir / f"{model_group_ID}_shap_two_panel_plot_bar_violin.pdf",
            bbox_inches="tight",
            pad_inches=0.1)
    plt.close()
    #######################################
    

    
    #######################################
    # Bar + Beeswarm two-panel plot
    plt.close('all')  # Close any existing figures
    # fig, axs = plt.subplots(1, 2, figsize=(14, 5), dpi=300, constrained_layout=True)
    fig, axs = plt.subplots(1, 2, figsize=(36, 8), dpi=300)
    # Adjust subplot parameters to give more space for labels
    plt.subplots_adjust(left=0.15, right=0.9, wspace=0.8)
    
    shap.plots.bar(shap_exp, max_display=20, show=False, ax=axs[0])
    axs[0].set_title("(A) Mean Absolute SHAP Value")
    plt.sca(axs[1])                 # <‑‑ make axs[1] the active axis
    shap.plots.beeswarm(shap_exp, max_display=20, show=False)
    
    axs[1].set_title("(B) SHAP Value Distribution")
    # fig.suptitle("SHAP Feature Importance (XGB TD)", fontsize=16)
    plt.tight_layout()
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_dir / f"{model_group_ID}_shap_two_panel_bar_beeswarm.png",
            # bbox_inches="tight",   # include everything that was drawn
            pad_inches=0.2)       # optional tiny padding)
    plt.close()
    
    
    # ─────────────────────────────────────────────────────────────
    # Partial SHAP dependence plots for the 60 key features
    # ─────────────────────────────────────────────────────────────
    # ------------------------------------------------------------
    # build combined SHAP & feature matrices from earlier section
    # shap_all  : (n_total_samples, n_features)
    # X_concat  : DataFrame with matching rows/columns
    # feature_names : list of column names
    # out_dir : already defined earlier
    # ------------------------------------------------------------
    pdp_dir = out_dir / "partial_dependence_plots"
    pdp_dir.mkdir(exist_ok=True)

    imp = np.abs(shap_exp.values).mean(axis=0)
    top60_idx   = np.argsort(imp)[-60:][::-1]
    # top60_feats = [feature_names[i] for i in top60_idx]
    top60_feats = [(i + 1, feature_names[idx]) for i, idx in enumerate(top60_idx)]  # List of (rank, feat)
    
    ########################################
    # for rank, feat in top60_feats:
    #     # Include rank in filename and pass to function for title
    #     save_path = pdp_dir / f"rank{rank:02d}_{feat}_partial_dep.png"
        
    #     x_vals = shap_exp.data[feat].values
    #     s_vals = shap_exp.values[:, feature_names.index(feat)]
    #     # partial_dep_scatter(x_vals, s_vals, feat, save_path=pdp_dir / f"{feat}_partial_dep.png")
    #     fig, ax = plt.subplots(figsize=(5, 5))
    #     pdp_panel(ax,
    #               x_vals,
    #               s_vals,
    #               feat)
    #     fig.tight_layout()
    #     fig.savefig(save_path, dpi=300)
    #     plt.close(fig)
    ########################################

    # ---------------------------------------------------------------------------
    # 5 ▪ Pick the four features of interest (order is the grid order)
    # ---------------------------------------------------------------------------
    feat_grid = ["NAIP_Profile_10",      # upper‑left
                 "D_alpha",      # upper‑right
                 "Spr_redEdge4",        # lower‑left
                 "Win_COH12",
                 "Spr_EVI",
                 "PALSAR_HH"]        # lower‑right
    Nice_labels_list = ["NAIP canopy‑height model\n10th percentile (m)", 
                        "UAVSAR D‑alpha entropy\n(PolSAR decomposition)", 
                        "Sentinel‑2 red‑edge\nband 4 (spring)", 
                        "Sentinel‑1 Coherence\n(12 day, winter)", 
                        "Sentinel‑2 EVI Index (spring)", 
                        "PALSAR‑2 HH backscatter (dB)"]
    # sanity‑check they exist in your dataframe
    missing = [f for f in feat_grid if f not in feature_names]
    if missing:
        print(f"Warning: Feature(s) not found in data: {missing}")
        feat_grid = [f for f in feat_grid if f in feature_names]
    
    # ---------------------------------------------------------------------------
    # 6 ▪ Build a 2×3 panel figure
    # ---------------------------------------------------------------------------
    if feat_grid:
        fig, axes = plt.subplots(3, 2, figsize=(12, 15), dpi=400)
        
        # for i, feat in enumerate(feat_grid):
        #     if i < len(axes.flat):
        #         ax = axes.flat[i]
        #         x_vals = shap_exp.data[feat].values
        #         s_vals = shap_exp.values[:, feature_names.index(feat)]
        #         partial_dep_scatter(x_vals, s_vals, feat, ax=ax)
        
        labels = list("ABCDEF")
        for ax, feat, lab, labName in zip(axes.flat, feat_grid, labels, Nice_labels_list):
            x_vals = shap_exp.data[feat].values
            s_vals = shap_exp.values[:, feature_names.index(feat)]
            # pdp_panel(ax,
            #           x_vals,
            #           s_vals,
            #           feat,q_low=0.5, q_high=99.5)
                      
            pdp_loess_panel(ax,
                      x_vals,
                      s_vals,
                      feat,q_low=0.5, q_high=99.5, loess_frac= 0.15, ribbon_k=1.0) # 1σ    
            ax.text(0.01, 0.97, f"({lab}) {labName}", transform=ax.transAxes,
                    ha='left', va='top', multialignment='center',  # every line right‑justified
                    fontsize=14)
                    
        # global legend
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right', frameon=False)

        # ---------------------------------------------------------------------------
        # fig.suptitle("Partial-Dependence / SHAP Scatter (Selected Features)", 
        #     fontsize=14, y=.97)
        plt.tight_layout()
        plt.savefig(pdp_dir / "six_feature_partial_dependence.png", 
            dpi=400, bbox_inches='tight')
        plt.close()

