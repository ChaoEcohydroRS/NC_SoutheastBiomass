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

## Data Confidentiality

FIA plot coordinates are not included. GEE analysis covers the full study domain without plot-level location data, per USFS confidentiality requirements.

---

## References

- Hansen et al. (2013). *Science* 342, 850–853.
- Kennedy et al. (2010). *Remote Sensing of Environment* 114, 2897–2910.
- eMapR LT-GEE: https://github.com/eMapR/LT-GEE
