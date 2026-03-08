import numpy as np
import rasterio
import os
from rasterio.plot import show
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from scipy import stats
from matplotlib.patches import Rectangle

# Define ablation analysis with VALIDATION PERFORMANCE data
ANALYSES = {
    'ablation_analysis': {
        'baseline': 'G20',
        'comparisons': ['G30', 'G31', 'G32', 'G33', 'G34'],
        'semantic_names': {
            'G20': 'FULL_MODEL (All Features)',
            'G30': 'No_NAIP_Optical',
            'G31': 'No_UAVSAR_Polarimetry', 
            'G32': 'No_3D_Structure',
            'G33': 'No_SAR_Coherence',
            'G34': 'Satellite_Only',
        },
        'feature_contributions': {
            'G30': 'Without NAIP Optical',
            'G31': 'Without UAVSAR Polarimetry',
            'G32': 'Without 3D Structure',
            'G33': 'Without SAR Coherence',
            'G34': 'Without Airborne Data',
        },
        # VALIDATION PERFORMANCE - Total Biomass
        'validation_performance': {
            'G20': {'Test_R2': 0.752, 'Test_RMSE': 34.17},
            'G30': {'Test_R2': 0.753, 'Test_RMSE': 34.09},
            'G31': {'Test_R2': 0.696, 'Test_RMSE': 37.83},
            'G32': {'Test_R2': 0.149, 'Test_RMSE': 67.13},
            'G33': {'Test_R2': 0.736, 'Test_RMSE': 35.27},
            'G34': {'Test_R2': 0.291, 'Test_RMSE': 61.30},
        },
        'description': 'Ablation Analysis: Feature Contributions with Validation-Guided Interpretation'
    }
}


def load_raster(filepath):
    """Load a raster file and return the array and metadata."""
    with rasterio.open(filepath) as src:
        data = src.read(1)
        profile = src.profile
        transform = src.transform
        crs = src.crs
    return data, profile, transform, crs


def create_mask_from_biomass(masked_biomass_path):
    """Extract binary mask from pre-masked biomass file where 0 = non-forest."""
    if not Path(masked_biomass_path).exists():
        print(f"⚠️  WARNING: Masked biomass file not found: {masked_biomass_path}")
        return None

    try:
        with rasterio.open(masked_biomass_path) as src:
            masked_biomass = src.read(1)

        # Create binary mask: 1 where biomass exists (forest), 0 where it's 0 (non-forest)
        mask = (masked_biomass != 0).astype(np.float32)

        forest_pixels = np.sum(mask > 0)
        total_pixels = mask.size
        print(f"✓ Created mask from masked biomass: {forest_pixels:,} forest pixels ({forest_pixels/total_pixels*100:.1f}%)")

        return mask

    except Exception as e:
        print(f"⚠️  WARNING: Error creating mask from biomass: {e}")
        import traceback
        traceback.print_exc()
        return None


def apply_mask_to_data(data, mask):
    """Apply binary mask to data via multiplication."""
    if mask is None:
        return data

    # Multiply: biomass * mask (forest=1, non-forest=0)
    masked_data = data * mask

    # Convert 0s to NaN for cleaner visualization
    masked_data[masked_data == 0] = np.nan

    return masked_data


def calculate_statistics(full_model, ablated_model, mask=None):
    """Calculate statistics for spatial sensitivity analysis."""
    if mask is not None:
        full_masked = full_model[mask]
        ablated_masked = ablated_model[mask]
    else:
        full_masked = full_model.flatten()
        ablated_masked = ablated_model.flatten()
    
    valid_mask = ~(np.isnan(full_masked) | np.isnan(ablated_masked))
    full_valid = full_masked[valid_mask]
    ablated_valid = ablated_masked[valid_mask]
    
    if len(full_valid) == 0:
        return None
    
    difference = full_model - ablated_model
    difference_flat = difference[~np.isnan(difference)].flatten()
    abs_difference_flat = np.abs(difference_flat)
    
    stats_dict = {
        'mean_abs_difference': np.mean(abs_difference_flat),
        'median_abs_difference': np.median(abs_difference_flat),
        'std_difference': np.std(difference_flat),
    }
    
    return stats_dict, difference


def plot_figure1_feature_importance(results_df, output_dir):
    """
    Figure 1: Feature Importance Summary (Ablation Analysis)
    Two panel figure showing R2 drop and RMSE increase
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Sort by validation performance impact
    results_df = results_df.sort_values('r2_drop', ascending=True)
    
    # Prepare data
    feature_labels = results_df['feature_name'].tolist()
    x_pos = np.arange(len(feature_labels))
    
    # Define color scheme based on impact
    colors = []
    for r2_drop in results_df['r2_drop']:
        if r2_drop > 0.5:
            colors.append('#D32F2F')  # Critical - Dark red
        elif r2_drop > 0.05:
            colors.append('#FF6F00')  # High - Orange
        elif r2_drop > 0.01:
            colors.append('#FBC02D')  # Moderate - Yellow
        else:
            colors.append('#7CB342')  # Minimal - Green
    
    # Panel a: R² Drop (Validation Performance)
    ax1 = axes[0]
    bars1 = ax1.barh(x_pos, results_df['r2_drop'], color=colors,
                     alpha=0.65, edgecolor='black', linewidth=1.5)

    # Add threshold line for "high impact"
    threshold = 0.05
    ax1.axvline(x=threshold, color='orange', linestyle='--', linewidth=2,
                alpha=0.7, label=f'High impact threshold')

    ax1.set_yticks(x_pos)
    ax1.set_yticklabels(feature_labels, fontsize=11)
    ax1.set_xlabel('R² Drop (Validation)', fontsize=12)
    ax1.set_title('(a) Feature Importance by Validation Performance\n(Higher = More Critical)',
                  loc='left', horizontalalignment='left', fontsize=13, pad=15)
    ax1.grid(True, alpha=0.35, axis='x')
    ax1.invert_yaxis()
    ax1.legend(loc='upper right', fontsize=10)

    # Add value labels on bars
    for i, (idx, row) in enumerate(results_df.iterrows()):
        ax1.text(row['r2_drop'] + 0.01, i, f"{row['r2_drop']:.3f}",
                va='center', fontsize=9)

    # Panel b: RMSE Increase (Prediction Error Impact)
    ax2 = axes[1]
    bars2 = ax2.barh(x_pos, results_df['rmse_increase_pct'], color=colors,
                     alpha=0.65, edgecolor='black', linewidth=1.5)

    ax2.set_yticks(x_pos)
    ax2.set_yticklabels([])  # Remove y-axis labels for right panel
    ax2.set_xlabel('RMSE Increase (%)', fontsize=12)
    ax2.set_title('(b) Prediction Error Impact\n(Validation-Based)',
                  loc='left', horizontalalignment='left', fontsize=13, pad=15)
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.invert_yaxis()
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(results_df.iterrows()):
        ax2.text(row['rmse_increase_pct'] + 2, i, f"{row['rmse_increase_pct']:.1f}%",
                va='center', fontsize=9)

    plt.tight_layout()

    output_path_png = output_dir / "Figure1_Feature_Importance_Validation.png"
    output_path_pdf = output_dir / "Figure1_Feature_Importance_Validation.pdf"
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight', transparent=True)
    plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()

    print(f"\n✓ Saved Figure 1 to {output_path_png}")
    print(f"✓ Saved Figure 1 to {output_path_pdf}")
    return output_path_png


def plot_figure2_spatial_patterns(full_model_data, difference_maps, feature_info,
                                   profile, output_dir):
    """
    Figure 2: Spatial Pattern of Feature Contributions
    Multi-panel map showing WHERE different features matter
    Focuses on top 4 most important features based on validation performance
    """

    # Select top 4 features by importance (already sorted)
    top_features = feature_info[:4]

    # Create figure with 2x2 layout: 4 difference maps
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.05, wspace=0.1)

    # Calculate extent and convert from UTM to Lat/Lon
    from rasterio.transform import xy
    from rasterio.warp import transform as warp_transform

    transform = profile['transform']
    src_crs = profile['crs']
    nrows, ncols = full_model_data.shape

    # Get corner coordinates in source CRS (UTM)
    left_utm, top_utm = xy(transform, 0, 0, offset='ul')
    right_utm, bottom_utm = xy(transform, nrows, ncols, offset='ul')

    # Transform from UTM (EPSG:32617) to Lat/Lon (EPSG:4326)
    dst_crs = 'EPSG:4326'

    # Transform corners
    lon_left, lat_top = warp_transform(src_crs, dst_crs, [left_utm], [top_utm])
    lon_right, lat_bottom = warp_transform(src_crs, dst_crs, [right_utm], [bottom_utm])

    # Use actual calculated extent (no override)
    extent = [lon_left[0], lon_right[0], lat_bottom[0], lat_top[0]]  # [lon_min, lon_max, lat_min, lat_max]

    print(f"   Geographic extent: Lon [{extent[0]:.4f}° to {extent[1]:.4f}°], Lat [{extent[2]:.4f}° to {extent[3]:.4f}°]")

    # Define display extent (what portion to show) - adjust this as needed
    display_lat_max = 35.3  # Maximum latitude to display

    # Panels a, b, c, d: Top 4 feature difference maps
    difference_cmap = 'YlOrRd'  # Yellow-Orange-Red for differences

    panel_labels = ['(a)', '(b)', '(c)', '(d)']
    positions = [gs[0, 0], gs[0, 1], gs[1, 0], gs[1, 1]]

    # Find global vmax for difference maps for consistent scaling
    vmax_diff = max([info['mean_abs_diff'] * 3 for info in top_features])  # 3x mean for range

    # Import formatters for tick labels
    from matplotlib.ticker import MultipleLocator, FuncFormatter

    for idx, (pos, label) in enumerate(zip(positions, panel_labels)):
        if idx < len(top_features):
            feature_data = top_features[idx]
            diff_map = difference_maps[feature_data['feature_name']]

            ax = fig.add_subplot(pos)

            # Plot absolute difference
            abs_diff = np.abs(diff_map)
            im = ax.imshow(abs_diff, cmap=difference_cmap, vmin=0, vmax=vmax_diff, extent=extent, aspect='auto', interpolation='nearest')

            # Add panel label and title inside the panel
            title_text = (f"{label} {feature_data['feature_name']}\n"
                         f"Mean Difference: {feature_data['mean_abs_diff']:.1f} Mg/ha | "
                         f"R² Drop: {feature_data['r2_drop']:.3f}")
            ax.text(0.02, 0.98, title_text, transform=ax.transAxes, fontsize=12,
                   verticalalignment='top', horizontalalignment='left',
                   color='black', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3))
            ax.tick_params(labelsize=11, pad=-17, direction='in', length=4)
            ax.tick_params(axis='y', rotation=90)  # Vertical y-axis labels
            # Adjust vertical alignment of rotated y-labels
            for tick_label in ax.yaxis.get_ticklabels():
                tick_label.set_verticalalignment('center')
            ax.grid(True, color='lightgray', linestyle='--', linewidth=0.5, alpha=0.7)

            # Set grid interval: 0.4 degrees for longitude, 0.3 degrees for latitude
            ax.xaxis.set_major_locator(MultipleLocator(0.4))
            ax.yaxis.set_major_locator(MultipleLocator(0.3))

            # Format tick labels with degree symbol
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.1f}°'))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, p: f'{y:.1f}°'))

            # Set display limits (crop view without distorting data)
            ax.set_ylim(extent[2], display_lat_max)

            # Add colorbar only for the last panel (d)
            if idx == len(top_features) - 1:
                # Reduce width to 80% with smaller fraction, move closer to panel
                cbar = plt.colorbar(im, ax=ax, fraction=0.037, pad=0.02, shrink=0.8)
                cbar.set_label('|Difference| (Mg/ha)', fontsize=10)

    output_path_png = output_dir / "Figure2_Spatial_Feature_Patterns.png"
    output_path_pdf = output_dir / "Figure2_Spatial_Feature_Patterns.pdf"
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved Figure 2 to {output_path_png}")
    print(f"✓ Saved Figure 2 to {output_path_pdf}")


def run_analysis(raster_dir, output_dir, model_type, analysis_name, analysis_config, masked_biomass_path=None):
    """Run ablation analysis and generate both figures."""

    baseline_name = analysis_config['baseline']
    comparisons = analysis_config['comparisons']
    semantic_names = analysis_config['semantic_names']
    feature_contributions = analysis_config['feature_contributions']
    validation_perf = analysis_config['validation_performance']

    # Create mask from pre-masked biomass file if provided
    mask = None
    if masked_biomass_path:
        print(f"\n🌲 Creating forest mask from masked biomass file...")
        mask = create_mask_from_biomass(masked_biomass_path)

    # Load baseline (full model)
    baseline_path = Path(raster_dir) / f"Total_biomass_tons_ha_{baseline_name}_{model_type}_completeSubplot.tif"
    print(f"\n📊 Loading baseline model: {baseline_path}")

    full_model, profile, transform, crs = load_raster(baseline_path)

    # Apply forest mask if created
    if mask is not None:
        full_model = apply_mask_to_data(full_model, mask)
    
    # Store results for Figure 1
    results_list = []
    
    # Store difference maps for Figure 2
    difference_maps = {}
    
    baseline_r2 = validation_perf[baseline_name]['Test_R2']
    baseline_rmse = validation_perf[baseline_name]['Test_RMSE']
    
    print(f"\n{'='*70}")
    print(f"Processing {len(comparisons)} feature ablations...")
    print(f"{'='*70}")
    
    for comp_name in comparisons:
        comp_path = Path(raster_dir) / f"Total_biomass_tons_ha_{comp_name}_{model_type}_completeSubplot.tif"

        if not comp_path.exists():
            print(f"⚠️  WARNING: {comp_path} not found, skipping...")
            continue
        
        print(f"\n  Processing: {semantic_names[comp_name]}")
        
        # Load ablated model
        ablated_model, _, _, _ = load_raster(comp_path)

        # Apply forest mask if available
        if mask is not None:
            ablated_model = apply_mask_to_data(ablated_model, mask)

        # Calculate statistics
        stats_dict, difference_map = calculate_statistics(full_model, ablated_model)
        
        # Get validation performance
        comp_r2 = validation_perf[comp_name]['Test_R2']
        comp_rmse = validation_perf[comp_name]['Test_RMSE']
        
        r2_drop = baseline_r2 - comp_r2
        rmse_increase = comp_rmse - baseline_rmse
        rmse_increase_pct = (rmse_increase / baseline_rmse) * 100
        
        feature_name = feature_contributions[comp_name]
        
        # Store for Figure 1
        results_list.append({
            'feature_name': feature_name,
            'r2_drop': r2_drop,
            'rmse_increase_pct': rmse_increase_pct,
            'mean_abs_difference': stats_dict['mean_abs_difference'],
            'model_name': comp_name
        })
        
        # Store difference map for Figure 2
        difference_maps[feature_name] = difference_map
        
        print(f"    R² drop: {r2_drop:.3f} | RMSE increase: {rmse_increase_pct:.1f}% | "
              f"Mean spatial diff: {stats_dict['mean_abs_difference']:.1f} Mg/ha")
    
    # Create DataFrame and sort by importance
    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values('r2_drop', ascending=False)
    
    print(f"\n{'='*70}")
    print("Generating figures...")
    print(f"{'='*70}")
    
    # Generate Figure 1: Feature Importance
    plot_figure1_feature_importance(results_df, Path(output_dir))
    
    # Prepare data for Figure 2: top 3 features
    top_3_features = []
    for _, row in results_df.head(4).iterrows():
        top_3_features.append({
            'feature_name': row['feature_name'],
            'mean_abs_diff': row['mean_abs_difference'],
            'r2_drop': row['r2_drop'],
        })
    
    # Generate Figure 2: Spatial Patterns
    # plot_figure2_spatial_patterns(full_model, difference_maps, top_3_features, 
    #                               profile, Path(output_dir))
    plot_figure2_spatial_patterns(full_model, difference_maps, top_3_features, 
                                   profile, Path(output_dir))
    
    return results_df


def main():
    """Main function to generate the two key manuscript figures."""

    # Set paths - UPDATE THESE TO YOUR PATHS
    raster_dir = '../RS_Disturbance_Pairing/Input/'
    output_dir = '.'
    masked_biomass_path = '../RS_Disturbance_Pairing/Input/Total_biomass_tons_ha_G20_ensemble_completeSubplot_maskedOut.tif'
    os.makedirs(output_dir, exist_ok=True)

    model_type = 'xgb'
    
    print("\n" + "="*80)
    print("  MANUSCRIPT FIGURE GENERATION")
    print("  Generating 2 key figures for landscape ecology journal")
    print("="*80)
    print("\n📋 Output:")
    print("   Figure 1: Feature Importance Summary (validation-based)")
    print("   Figure 2: Spatial Pattern of Feature Contributions (top 4 features)")
    print("\n🎯 This addresses reviewer comment:")
    print("   'WHERE do different features matter most spatially?'")
    print("="*80)
    
    for analysis_name, analysis_config in ANALYSES.items():
        results = run_analysis(raster_dir, output_dir, model_type,
                              analysis_name, analysis_config, masked_biomass_path=masked_biomass_path)
    
    print(f"\n{'='*80}")
    print("✅ FIGURE GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"\n📁 Figures saved to: {output_dir}")
    print("\n📊 Generated figures:")
    print("   1. Figure1_Feature_Importance_Validation.png")
    print("      → Shows which features matter for model performance")
    print("   2. Figure2_Spatial_Feature_Patterns.png")
    print("      → Shows WHERE the top 4 features contribute spatially")
    print("\n💡 Manuscript guidance:")
    print("   • Figure 1 goes in 'Ablation Analysis' subsection")
    print("   • Figure 2 addresses coauthor's comment about spatial patterns")
    print("   • Caption for Figure 2 should describe any spatial patterns you observe")
    print("     (e.g., 'most important in complex terrain', 'uniform across landscape', etc.)")
    print("="*80)


if __name__ == "__main__":
    main()
