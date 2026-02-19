# Forest Disturbance Analysis — FIA × Remote Sensing Study

## Purpose

This directory contains the disturbance analysis scripts produced in response to a peer-reviewer concern about the temporal mismatch between FIA ground measurements (2015–2022) and Sentinel-2 imagery (2018–2019) used in the biomass modelling study.

**Reviewer concern:** Forest disturbances occurring between 2015 and 2022 could introduce noise when FIA plot biomass measurements are matched to RS pixel values from a fixed 2018–2019 acquisition window.

**Response strategy:**
1. **Spatial evidence** — Two Google Earth Engine scripts map forest disturbance across the study region using independent RS products (GFW Hansen and LandTrendr), and confirm that disturbance is minimal at the FIA plot locations during 2015–2022.
2. **Tabular evidence** — One Python script uses the FIA tabular data directly (no confidential location data needed) to show plot-level biomass stability and low disturbance rates across all measurement years.

---

## Files

### Google Earth Engine Scripts (paste into [code.earthengine.google.com](https://code.earthengine.google.com))

| File | Dataset | What it does |
|------|---------|--------------|
| `DisturbanceAnalysisGFW.js` | Hansen Global Forest Change v1.11 (UMD) | Maps annual forest loss year (2015–2022) filtered to areas ≥ 30% tree cover; exports `loss_year` raster |
| `DisturbanceAnalysisLandtrendr.js` | Landsat 5/7/8/9 Collection 2 SR | Runs LandTrendr spectral segmentation on NBR time series (2010–2023); detects and exports the greatest-loss disturbance year and magnitude within 2015–2022 |

#### Key GEE configuration (edit at top of each script)

```javascript
var study_area  = ee.Geometry.Rectangle([-125, 24, -66, 50]);  // replace with your AOI
var filter_start = 2015;    // FIA measurement window start
var filter_end   = 2022;    // FIA measurement window end
var min_tree_cover = 30;    // GFW: minimum % tree cover (baseline 2000)
var disturbance_threshold = -100;  // LandTrendr: NBR magnitude threshold
```

#### GEE export outputs

| Export file | Bands | Source |
|-------------|-------|--------|
| `gfw_forest_loss.tif` | `loss_year` (year of loss, float) | GFW |
| `landtrendr_disturbance.tif` | `disturbance_year`, `disturbance_magnitude` | LandTrendr |

#### LandTrendr technical notes

- NBR is computed with sensor-correct band indices: L5/L7 use `SR_B4`/`SR_B7`; L8/L9 use `SR_B5`/`SR_B7`.
- Disturbance extraction uses the **segment-based eMapR approach** (consecutive vertex pairs → sort by NBR magnitude) to avoid the `Array arguments must have same length` GEE error that occurs with direct vertex sorting.
- `minObservationsNeeded = 6` is appropriate for the 14-year time series (2010–2023).

---

### Python Script

| File | Inputs | Outputs |
|------|--------|---------|
| `DisturbanceSummaryFIA.py` | `unc_chao_fia_data.xlsx` (same file used in modelling) | Summary table CSV/XLSX + figure JPG |

#### What `DisturbanceSummaryFIA.py` does

1. **Replicates the identical filtering pipeline** from `model_utils.load_and_preprocess_data()`:
   - Keeps only MEASYEAR 2015–2022
   - Requires complete plots (all 4 subplots present)
   - Excludes zero-biomass plots
   - So the analysed plot population is exactly the modelling dataset

2. **Dynamically detects FIA disturbance/treatment columns** in both subplot sheets and the `fia_plot` sheet:
   - Disturbance codes: `DSTRBCD1`, `DSTRBCD2`, `DSTRBCD3`
   - Disturbance years: `DSTRBYR1`, `DSTRBYR2`, `DSTRBYR3`
   - Treatment codes: `TRTCD1`, `TRTCD2`, `TRTCD3`
   - Works correctly whether these columns are present or absent

3. **Classifies plots by temporal relationship to RS imagery** (2018–2019):
   - Pre-RS (2015–2017)
   - Concurrent (2018–2019)
   - Post-RS (2020–2022)

4. **Outputs**:

| Output file | Contents |
|-------------|----------|
| `fia_disturbance_summary_table.csv` | Per-year: N plots, mean/median/SD/min/max AGB, disturbance counts & % |
| `fia_disturbance_summary_table.xlsx` | Same, multi-sheet: By_Year, Temporal_Groups, Disturbance_Codes, Treatment_Codes |
| `fia_temporal_group_summary.csv` | Pre/Concurrent/Post-RS group statistics |
| `fia_disturbance_summary.jpg` | 2–3 panel figure (see below) |

#### Figure panels

- **(A)** Box plot of total AGB (Mg/ha) by FIA measurement year — bars colour-coded by pre/concurrent/post-RS group. A stable AGB distribution across years is direct evidence that no large-scale disturbance affected the plot population.
- **(B)** Bar chart of plot count per measurement year — shows the temporal distribution of FIA sampling effort.
- **(C)** (Only rendered if disturbance code columns exist) Grouped bar chart of % plots with any disturbance, natural disturbance only, and harvest/treatment, per year.

#### Running the script

```bash
# On the HPC cluster (same environment as the modelling scripts)
cd /path/to/unc_chao_fia_data.xlsx/directory
python DisturbanceSummaryFIA.py
```

Adjust `FILE_NAME`, `OUTPUT_DIR`, `RS_START`, `RS_END` at the top of the script if paths or imagery dates differ.

Dependencies: `numpy`, `pandas`, `matplotlib`, `openpyxl`

---

### `BiomassChangeFIA.py`

| File | Inputs | Outputs |
|------|--------|---------|
| `BiomassChangeFIA.py` | `unc_chao_fia_data.xlsx` | `biomass_change_by_treatment.csv/xlsx`, `biomass_change_summary.jpg` |

Addresses the specific concern that **cutting (TRTCD=10, ~24% of plots)** may bias AGB estimates. Two complementary analyses:

**Approach A — Removed biomass** (all 305 plots)
- FIA subplot data records harvested trees in `*_removed` columns
- Computes `removed_AGB` and `removed fraction = removed / (live + removed)` per plot
- Stratified by disturbance class: No disturbance | Cutting only | Natural dist. only | Both
- A low removed fraction (e.g. <5%) means cutting was minor thinning, not stand-clearing

**Approach B — Re-measurement ΔAGB** (paired plots with prior cycle in file)
- Links each current plot to its prior measurement via `PREV_PLT_CN` in `fia_plot`
- Computes `ΔAGB_live = current − prior` and annual change rate `ΔAGB / REMPER`
- Stratified by the same disturbance classes

**Treatment-year timing check**
- Uses `TRTYR1–3` to determine how long before the RS imagery (2018–2019) each cutting event occurred
- Categorises cuts as: After RS | ≤2 yrs before RS | 3–5 yrs | 6–10 yrs | >10 yrs
- Old cuts allow forest recovery; only recent cuts (≤2 yr) are a genuine matching concern

#### Figure panels
- **(A)** Live AGB box plot by disturbance class
- **(B)** Removed AGB fraction (%) by class — key metric for reviewer
- **(C)** ΔAGB box plot (only if paired prior-cycle data found)
- **(D)** Bar chart of cutting timing relative to RS imagery

---

## Workflow Summary

```
Reviewer concern: temporal mismatch FIA (2015-2022) vs RS (2018-2019)
         │
         ├── Spatial response (GEE)
         │     ├── DisturbanceAnalysisGFW.js          → gfw_forest_loss.tif
         │     └── DisturbanceAnalysisLandtrendr.js   → landtrendr_disturbance.tif
         │           (overlay with FIA plot locations in GIS to show minimal disturbance)
         │
         └── Tabular response (Python)
               └── DisturbanceSummaryFIA.py
                     ├── fia_disturbance_summary_table.xlsx
                     ├── fia_temporal_group_summary.csv
                     └── fia_disturbance_summary.jpg
```

**Interpretation for reviewer response:**
- GFW and LandTrendr layers show forest disturbance is spatially limited and does not cluster at FIA plot locations during 2015–2022.
- The FIA summary table shows AGB distributions are consistent across all measurement years with no systematic decline, and (if disturbance codes are present) that very few plots carry any FIA-recorded disturbance flag.
- Together these two lines of evidence support the claim that temporal mismatch between FIA measurements and RS imagery does not introduce significant systematic noise.

---

## Data Confidentiality Note

FIA plot location coordinates (`LAT`, `LON` from the `fia_plot` sheet) are protected by USFS confidentiality agreements and are **not** included in this repository. The Python script uses only tabular biomass and disturbance attributes. The GEE spatial analysis operates over the full study region without requiring plot-level coordinates.

---

## References

- Hansen, M. C., et al. (2013). High-resolution global maps of 21st-century forest cover change. *Science*, 342(6160), 850–853.
- Kennedy, R. E., et al. (2010). Detecting trends in forest disturbance and recovery using yearly Landsat time series: 1. LandTrendr — Temporal segmentation algorithms. *Remote Sensing of Environment*, 114(12), 2897–2910.
- eMapR LT-GEE library: [https://github.com/eMapR/LT-GEE](https://github.com/eMapR/LT-GEE)
- USFS FIA Database Manual v9.1 — disturbance code definitions (§2.5.8).
