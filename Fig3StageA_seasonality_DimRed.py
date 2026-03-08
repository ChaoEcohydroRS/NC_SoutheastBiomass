# stageA_seasonality_benchmark.py  (with FULL, TOP-k, PCA variants)
# -----------------------------------------------------------------
import warnings, numpy as np, pandas as pd, matplotlib.pyplot as plt
warnings.filterwarnings("ignore",
    message="X does not have valid feature names, but StandardScaler was fitted")

from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics         import mean_squared_error, make_scorer
from sklearn.inspection      import permutation_importance
from sklearn.decomposition   import PCA
from sklearn.compose         import ColumnTransformer
from sklearn.pipeline        import Pipeline

from pytabkit                import CatBoost_TD_Regressor
from model_config            import model_feature_sets  # includes G1a, G1sw, G1c
from model_utils             import load_and_preprocess_data

import torch
torch.set_float32_matmul_precision("high")

# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------

# Scorer for permutation importance: negative RMSE (so higher = better)
neg_rRMSE = make_scorer(
    lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred)),
    greater_is_better=True,
)

def catboost_td(device="cpu"):
    """Instantiate a tuned-default CatBoost regressor."""
    return CatBoost_TD_Regressor(device=device)

def nrmse(y_true, y_pred):
    """Normalized RMSE = RMSE / mean(true)."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse / y_true.mean()

# ---------- 1. Baseline (FULL features) --------------------------
def cv_full(X, y, cv, device="cpu"):
    scores = []
    for tr, te in cv.split(X):
        model = catboost_td(device).fit(X.iloc[tr], y.iloc[tr])
        preds = model.predict(X.iloc[te])
        scores.append(nrmse(y.iloc[te], preds))
    return np.mean(scores), np.std(scores)

# ---------- 2. TOP-k via permutation importance -------------------
def cv_topk(X, y, cv, k=34, n_repeats=5, device="cpu"):
    """
    Cross-validated nRMSE after selecting top-k features by permutation importance
    computed on each training fold.
    """
    scores = []
    for tr, te in cv.split(X):
        # 1. fit full model on train
        base = catboost_td(device).fit(X.iloc[tr], y.iloc[tr])
        # 2. permutation importance on train
        perm = permutation_importance(
            base, X.iloc[tr], y.iloc[tr],
            scoring=neg_rRMSE, n_repeats=n_repeats,
            random_state=42, n_jobs=-1
        )
        # 3. pick top-k (or fewer if p < k)
        k_use    = min(k, X.shape[1])
        top_idx  = np.argsort(perm.importances_mean)[::-1][:k_use]
        top_feats= X.columns[top_idx]
        # 4. refit model on top-k features
        model_k  = catboost_td(device).fit(X.iloc[tr][top_feats], y.iloc[tr])
        # 5. evaluate on test
        preds    = model_k.predict(X.iloc[te][top_feats])
        scores.append(nrmse(y.iloc[te], preds))
    return np.mean(scores), np.std(scores)

# ---------- 3. Season-wise PCA (99% variance) --------------------
def build_season_CT(cols_by_season, var_thr=0.99):
    transformers = []
    for season, cols in cols_by_season.items():
        if cols:
            transformers.append((
                season,
                Pipeline([
                    ("scaler", StandardScaler()),
                    ("pca",    PCA(n_components=var_thr, svd_solver="full"))
                ]),
                cols
            ))
    return ColumnTransformer(transformers, remainder="drop")

def cv_pca(X, y, cv, cols_by_season, var_thr=0.99, device="cpu"):
    """
    Cross-validated nRMSE after applying per-season PCA to retain var_thr variance.
    """
    scores = []
    for tr, te in cv.split(X):
        ct   = build_season_CT(cols_by_season, var_thr)
        Xtr  = ct.fit_transform(X.iloc[tr])
        Xte  = ct.transform(X.iloc[te])
        model= catboost_td(device).fit(Xtr, y.iloc[tr])
        preds= model.predict(Xte)
        scores.append(nrmse(y.iloc[te], preds))
    return np.mean(scores), np.std(scores)

# -----------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------
if __name__ == "__main__":
    
    out_dir  = "/work/users/w/a/wayne128/Biomass_ML/Dataset/OutBiomassRaster"
    
    clean_df, tag  = load_and_preprocess_data(
        "unc_chao_fia_data.xlsx",
        ['1.#QNB','1.#INF','-1.#INF','nan','NaN','inf','-inf']
    )
    y_total  = clean_df["total_biomass_tons_ha"]

    stageA_ids = ["G1a", "G1sw", "G1c"]
    desc_map   = dict(zip(stageA_ids, ["S2_SUM", "SUM+WIN", "S2_ALL"]))

    # 10×10 repeated CV
    repeated_k_fold = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)

    # define season-specific columns (adjust prefixes to actual names, excluding any that contain '_COH'
    cols_by_season = {
        "sum": [c for c in clean_df.columns if "Sum_" in c and "_COH" not in c],
        "win": [c for c in clean_df.columns if "Win_" in c and "_COH" not in c],
        "spr": [c for c in clean_df.columns if "Spr_" in c and "_COH" not in c],
        "fal": [c for c in clean_df.columns if "Fall_" in c and "_COH" not in c]
    }

    all_results = {"FULL": {}, "TOPk": {}, "PCA": {}}

    # -----------------------------------------------------------------
    for gid in stageA_ids:
        feats  = model_feature_sets[gid]
        Xfull  = clean_df[feats].dropna()
        y      = y_total.loc[Xfull.index]
        Xfull  = Xfull.loc[y.index]  # align

        # scale for FULL and TOP-k (PCA rescales internally)
        scaler = StandardScaler()
        Xs     = pd.DataFrame(
                    scaler.fit_transform(Xfull),
                    index=Xfull.index,
                    columns=Xfull.columns
                 )

        # 1. FULL
        all_results["FULL"][gid] = cv_full (Xs, y, repeated_k_fold)
        
        # 2. TOP-k
        all_results["TOPk"][gid] = cv_topk(Xs, y, repeated_k_fold, k=34, n_repeats=5)
        
        # 3. PCA
        
        # Filter cols_by_season to only include columns in Xs
        season_cols_in_X = {
            season: [c for c in cols if c in Xs.columns]
            for season, cols in cols_by_season.items()
        }
        
        all_results["PCA"][gid]  = cv_pca (Xs, y, repeated_k_fold, season_cols_in_X, var_thr=0.99)

    # -----------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------
    print("\n=== Stage A Cross-validated nRMSE (mean ± sd) ===")
    
    for method in ["FULL", "TOPk", "PCA"]:
        print(f"\n{method}")
        for gid in stageA_ids:
            m, s = all_results[method][gid]
            print(f"  {gid:4s}: {m:.4f} ± {s:.4f}")

    # -----------------------------------------------------------------
    # Plot bar charts (one panel per reduction method)
    # -----------------------------------------------------------------
    # Methods and plotting palette
    methods = ["FULL", "TOPk", "PCA"]
    palette = ["#3E7CB1", "#66A182", "#F5A623"]
    
    # X-axis labels and positions
    labels = [f"{g}\n({desc_map[g]})" for g in stageA_ids]
    x = np.arange(len(labels))
    
    # Create 1×3 subplots sharing the y-axis
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    
    for ax, method in zip(axes, methods):
        # Extract means and stds for this reduction method
        means = [all_results[method][g][0] for g in stageA_ids]
        stds  = [all_results[method][g][1] for g in stageA_ids]
    
        # Draw bars with error bars
        bars = ax.bar(
            x, means, yerr=stds, capsize=5,
            color=palette, edgecolor="black", linewidth=0.6
        )
    
        # Format axes
        ax.set_xticks(x, labels, fontsize=9)
        ax.set_title(f"Stage A – {method}", fontsize=11, pad=6)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    
        # Annotate bar values
        for rect, val in zip(bars, means):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                val + 0.01,
                f"{val:.3f}",
                ha="left",
                va="bottom",
                fontsize=9,
                fontweight="bold"
            )
    
    # Y-axis label on the first subplot only
    axes[0].set_ylabel("nRMSE", fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f"{out_dir}/FigStageA_DimRed_Comparison_{tag}.jpg", dpi=550)
    plt.show()
    
    
