# Forest Disturbance Analysis — FIA × Remote Sensing

Supporting analysis for:
**AI-Powered Multisensor Fusion for Forest Biomass Mapping: Photogrammetric Canopy Profiles Improve Estimates in Southeastern North Carolina**

Zenodo: 10.5281/zenodo.18688899

---

## Overview

This folder contains scripts and results for quantifying forest disturbance (2015–2022) across the study region (~557,000 ha, southeastern NC) using GFW Hansen and LandTrendr, relative to the Sentinel-2 imagery window (2018–2019).

---

## Scripts

### GEE Scripts (run in [code.earthengine.google.com](https://code.earthengine.google.com))

| File | Description |
|------|-------------|
| `DisturbanceAnalysisGFW.js` | Annual forest loss 2015–2022 from Hansen GFC v1.11; exports `gfw_forest_loss.tif` |
| `DisturbanceAnalysisLandtrendr.js` | LandTrendr NBR segmentation on Landsat time series; exports disturbance year and magnitude |

Edit these variables at the top of each script to match your study area:

```javascript
var study_area  = ee.Geometry.Rectangle([-125, 24, -66, 50]);
var filter_start = 2015;
var filter_end   = 2022;
var min_tree_cover = 30;        // GFW only
var disturbance_threshold = -100;  // LandTrendr only
```

**LandTrendr note:** `lt_end_year` is set to 2025 (not 2022) to avoid right-edge bias — LandTrendr needs at least 2 post-disturbance composites to confirm a breakpoint. The detection filter is still 2015–2022.

Outputs saved on Longleaf HPC: `/work/users/w/a/wayne128/Biomass_ML/Dataset/OutBiomassRaster/Disturbance`

### Python Script

**`A_Review_TimeMismatch_DisturbanceAnalysis.py`**

Reprojects the two disturbance rasters to the biomass grid, applies the forest mask, and computes biomass-weighted disturbance metrics by year and by temporal proximity to the RS window. Outputs the 4-panel supplementary figure and summary CSV/XLSX.

Set `GFW_YEAR_OFFSET = 2000` if the GEE export uses raw Hansen encoding (15–22); set to `0` if 2000 was already added. Update `MODEL_RMSE` if the biomass model changes.

---

## Key Results

Study domain: 557,121 ha | pixel size: 25 m | mean AGB: 103.7 Mg/ha | model RMSE: 34.17 Mg/ha

### Cumulative disturbance 2015–2022

| | GFW | LandTrendr |
|--|-----|------------|
| Disturbed area | 64,011 ha (11.5%) | 34,749 ha (6.2%) |
| Biomass-weighted fraction | 10.7% | 4.8% |
| Worst-case domain AGB shift | 11.1 Mg/ha (0.33× RMSE) | 5.0 Mg/ha (0.15× RMSE) |

### Per-year breakdown

#### GFW Hansen

| Year | Area (%) | Bio-wtd (%) | Density ratio | Mean AGB (Mg/ha) | WC shift (Mg/ha) |
|------|----------|-------------|---------------|-----------------|-----------------|
| 2015 | 1.31 | 0.75 | 0.57 | 59.3 | 0.78 |
| 2016 | 1.42 | 0.97 | 0.69 | 71.0 | 1.01 |
| 2017 | 1.67 | 1.26 | 0.75 | 78.2 | 1.31 |
| **2018** | **1.70** | **1.26** | **0.74** | **76.5** | **1.30** |
| **2019** | **1.50** | **1.68** | **1.13** | **116.7** | **1.74** |
| 2020 | 1.26 | 1.49 | 1.19 | 123.1 | 1.55 |
| 2021 | 1.35 | 1.72 | 1.27 | 132.0 | 1.78 |
| 2022 | 1.28 | 1.57 | 1.23 | 128.0 | 1.63 |

Bold = RS imagery window (2018–2019). Density ratio = mean AGB in disturbed pixels / mean AGB across all forest pixels.

#### LandTrendr (`lt_end_year = 2025`)

| Year | Area (%) | Bio-wtd (%) | Density ratio | Mean AGB (Mg/ha) | WC shift (Mg/ha) | LT/GFW |
|------|----------|-------------|---------------|-----------------|-----------------|--------|
| 2015 | 0.98 | 0.50 | 0.51 | 53.1 | 0.52 | 0.75 |
| 2016 | 0.76 | 0.44 | 0.59 | 61.0 | 0.46 | 0.53 |
| 2017 | 0.84 | 0.57 | 0.68 | 70.9 | 0.60 | 0.50 |
| **2018** | **0.85** | **0.63** | **0.74** | **76.4** | **0.65** | **0.50** |
| **2019** | **1.27** | **1.13** | **0.89** | **92.4** | **1.18** | **0.85** |
| 2020 | 0.80 | 0.80 | 1.00 | 103.3 | 0.83 | 0.64 |
| 2021 | 0.41 | 0.43 | 1.05 | 109.4 | 0.45 | 0.30 |
| 2022 | 0.32 | 0.31 | 0.96 | 99.7 | 0.32 | 0.25 |

The LT/GFW gap reflects the algorithmic difference: GFW detects any canopy loss; LandTrendr requires a sustained NBR decline. Both products bracket the true disturbance signal.

### Temporal proximity to RS window

Plots measured outside 2018–2019 face a temporal gap between FIA measurement and imagery. Disturbances in those gap years create a (low AGB, high spectral) training inconsistency; pre- and post-RS gaps produce the same directional effect.

| Period | GFW area (%) | GFW WC shift | LT area (%) | LT WC shift |
|--------|-------------|-------------|------------|------------|
| Pre-RS gap (2015–2017) | 4.40% | 3.10 Mg/ha | 2.58% | 1.58 Mg/ha |
| RS window (2018–2019) | 3.20% | 3.05 Mg/ha | 2.12% | 1.83 Mg/ha |
| Post-RS gap (2020–2022) | 3.85% | 4.96 Mg/ha | 1.53% | 1.60 Mg/ha |
| **Gap years combined** | **8.25%** | **8.06 Mg/ha** | **4.11%** | **3.18 Mg/ha** |

Even under the worst-case assumption (all disturbed pixels drop to zero AGB), the combined gap-year effect is 8.06 Mg/ha (GFW) or 3.18 Mg/ha (LandTrendr) — well below model RMSE.

---

## Data Confidentiality

FIA plot coordinates are not included. GEE analysis covers the full study domain without plot-level location data, per USFS confidentiality requirements.

---

## References

- Hansen et al. (2013). *Science* 342, 850–853.
- Kennedy et al. (2010). *Remote Sensing of Environment* 114, 2897–2910.
- eMapR LT-GEE: https://github.com/eMapR/LT-GEE
