import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from pytabkit import CatBoost_TD_Regressor
from statsmodels.nonparametric.smoothers_lowess import lowess

from model_utils import load_and_preprocess_data
from model_config import model_feature_sets


from scipy.stats import pearsonr





# ─────────────────────────────────────────────────────────────────────────────
# MAIN: compute and plot for Stage D
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    output_dir = "/work/users/w/a/wayne128/Biomass_ML/Dataset/OutBiomassRaster"

    # 1. Load data and compute hardwood fraction
    cleaned_df, _ = load_and_preprocess_data(
        'unc_chao_fia_data.xlsx',
        na_values=['1.#QNB','1.#INF','-1.#INF','nan','NaN','inf','-inf']
    )
    hr = cleaned_df['hrdwdDRYBIO_AGac_live']
    sr = cleaned_df['sftwdDRYBIO_AGac_live']
    fHW = hr / (hr + sr)
    fHW = fHW.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Observations
    y = cleaned_df['total_biomass_tons_ha']
    y = y.loc[fHW.index]
    
    # Features for multisensor fusion (G19)
    X = cleaned_df[model_feature_sets['G19']].loc[fHW.index].dropna()
    idx = X.index.intersection(y.index).intersection(fHW.index)
    X = X.loc[idx]
    y = y.loc[idx]
    fHW = fHW.loc[idx]
    
    # Scale predictors
    scaler = StandardScaler().fit(X)
    X_scaled = pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)
    
    # 2. Cross-validated residuals
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    residuals = []
    fHW_vals = []
    
    for train_idx, test_idx in kf.split(X_scaled):
        model = CatBoost_TD_Regressor(n_cv=1, device='cpu')
        model.fit(X_scaled.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X_scaled.iloc[test_idx])
        res = preds - y.iloc[test_idx]
        residuals.append(res.values)
        fHW_vals.append(fHW.iloc[test_idx].values)
    
    residuals = np.concatenate(residuals)
    fHW_vals = np.concatenate(fHW_vals)
    
    
    # ─────────────────────────────────────────────────────────────────────────────
    # CALCULATE MEAN ERRORS BY HARDWOOD FRACTION RANGES
    # ─────────────────────────────────────────────────────────────────────────────
    
    print("\n" + "="*60)
    print("MEAN ERROR ANALYSIS BY HARDWOOD FRACTION")
    print("="*60)
    
    # Define hardwood fraction categories
    categories = [
        ("Pure Softwood", 0.0, 0.2),
        ("Mixed Forest", 0.2, 0.8), 
        ("Pure Hardwood", 0.8, 1.0)
    ]
    
    # Calculate statistics for each category
    for category_name, fHW_min, fHW_max in categories:
        # Filter residuals for this hardwood fraction range
        mask = (fHW_vals >= fHW_min) & (fHW_vals < fHW_max)
        category_residuals = residuals[mask]
        category_fHW = fHW_vals[mask]
        
        if len(category_residuals) > 0:
            # Calculate statistics
            mean_error = np.mean(category_residuals)
            std_error = np.std(category_residuals)
            median_error = np.median(category_residuals)
            mae = np.mean(np.abs(category_residuals))  # Mean Absolute Error
            rmse = np.sqrt(np.mean(category_residuals**2))  # Root Mean Square Error
            
            # Percentiles for error range
            p25 = np.percentile(category_residuals, 25)
            p75 = np.percentile(category_residuals, 75)
            
            print(f"\n{category_name} (fHW: {fHW_min:.1f} - {fHW_max:.1f})")
            print(f"  Sample size: {len(category_residuals):,} plots")
            print(f"  Mean fHW: {np.mean(category_fHW):.3f}")
            print(f"  Mean residual: {mean_error:.1f} Mg ha⁻¹")
            print(f"  Std deviation: {std_error:.1f} Mg ha⁻¹")
            print(f"  Median residual: {median_error:.1f} Mg ha⁻¹")
            print(f"  Mean absolute error: {mae:.1f} Mg ha⁻¹")
            print(f"  RMSE: {rmse:.1f} Mg ha⁻¹")
            print(f"  IQR: [{p25:.1f}, {p75:.1f}] Mg ha⁻¹")
            print(f"  Error range (±1 std): ±{std_error:.1f} Mg ha⁻¹")
        else:
            print(f"\n{category_name} (fHW: {fHW_min:.1f} - {fHW_max:.1f})")
            print("  No data points in this range")
    
    # Additional analysis: Fine-grained bins
    print("\n" + "-"*60)
    print("DETAILED ANALYSIS BY 0.1 HARDWOOD FRACTION BINS")
    print("-"*60)
    
    detailed_bins = np.arange(0.0, 1.1, 0.1)
    for i in range(len(detailed_bins)-1):
        fHW_min = detailed_bins[i]
        fHW_max = detailed_bins[i+1]
        mask = (fHW_vals >= fHW_min) & (fHW_vals < fHW_max)
        bin_residuals = residuals[mask]
        
        if len(bin_residuals) > 0:
            mean_error = np.mean(bin_residuals)
            std_error = np.std(bin_residuals)
            mae = np.mean(np.abs(bin_residuals))
            
            print(f"fHW {fHW_min:.1f}-{fHW_max:.1f}: n={len(bin_residuals):3d}, "
                  f"Mean={mean_error:6.1f}, MAE={mae:5.1f}, Std={std_error:5.1f} Mg ha⁻¹")
    
    # Overall statistics
    print("\n" + "-"*60)
    print("OVERALL MODEL PERFORMANCE")
    print("-"*60)
    overall_mean = np.mean(residuals)
    overall_std = np.std(residuals)
    overall_mae = np.mean(np.abs(residuals))
    overall_rmse = np.sqrt(np.mean(residuals**2))
    
    print(f"Total samples: {len(residuals):,}")
    print(f"Overall mean residual: {overall_mean:.1f} Mg ha⁻¹")
    print(f"Overall std deviation: {overall_std:.1f} Mg ha⁻¹") 
    print(f"Overall MAE: {overall_mae:.1f} Mg ha⁻¹")
    print(f"Overall RMSE: {overall_rmse:.1f} Mg ha⁻¹")
    print(f"Hardwood fraction range: {fHW_vals.min():.3f} - {fHW_vals.max():.3f}")
    
    # Test for systematic bias trend
    correlation, p_value = pearsonr(fHW_vals, residuals)
    print(f"\nCorrelation between fHW and residuals: {correlation:.3f} (p={p_value:.3f})")
    if p_value < 0.05:
        if correlation > 0:
            print("→ Significant positive bias: model underestimates with increasing hardwood fraction")
        else:
            print("→ Significant negative bias: model overestimates with increasing hardwood fraction")
    else:
        print("→ No significant systematic bias with hardwood fraction")
    
    print("\n" + "="*60)
    
    
    
    # ─────────────────────────────────────────────────────────────────────────────
    # CONTINUE WITH ORIGINAL PLOTTING CODE
    # ─────────────────────────────────────────────────────────────────────────────
    
    
    # 3. LOESS smoothing with 95% CI by bootstrap
    grid = np.linspace(fHW_vals.min(), fHW_vals.max(), 200)
    loess_mean = lowess(residuals, fHW_vals, frac=0.3, return_sorted=False)
    
    # Bootstrap for CI
    B = 100
    boot = np.zeros((B, grid.size))
    for b in range(B):
        ixs = np.random.choice(len(fHW_vals), len(fHW_vals), replace=True)
        boot[b] = lowess(residuals[ixs], fHW_vals[ixs], frac=0.3, xvals=grid)
    
    ci_low = np.percentile(boot, 2.5, axis=0)
    ci_high = np.percentile(boot, 97.5, axis=0)
    
    # 4. Violin envelopes by fHW bins
    bins = np.linspace(0, 1, 11)
    centers = 0.5*(bins[:-1]+bins[1:])
    violin_data = [residuals[(fHW_vals>=bins[i])&(fHW_vals<bins[i+1])]
                   for i in range(len(bins)-1)]
    
    fig, ax = plt.subplots(figsize=(6,4))
    
    # Scatter
    ax.scatter(fHW_vals, residuals, s=10, alpha=0.3, color='gray')
    
    # Violins
    parts = ax.violinplot(violin_data, positions=centers, widths=0.08,
                          showmeans=False, showmedians=False, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor('#66A182')
        pc.set_edgecolor('black')
        pc.set_alpha(0.5)
    
    # LOESS line + CI band
    loess_grid = lowess(residuals, fHW_vals, frac=0.3, xvals=grid)
    ax.plot(grid, loess_grid, color='black', lw=2)
    ax.fill_between(grid, ci_low, ci_high, color='black', alpha=0.2)
    
    # Styling
    ax.set_xlabel('Hardwood fraction $f_{HW}$')
    ax.set_ylabel('Residual (Predicted $-$ Observed AGB) Mg ha$^{-1}$')
    # ax.set_title('Stage D – Residual sensitivity of the ALL model to $f_{HW}$', pad=12)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    
    plt.tight_layout()
    fig_out = f"{output_dir}/Fig6StageD_WoodRatioResidual_benchmark.png"
    fig.savefig(fig_out, dpi=300)
    plt.show()
    print(f"\nSaved Figure to: {fig_out}")
