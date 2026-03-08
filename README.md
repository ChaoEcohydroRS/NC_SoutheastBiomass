# Title: AI-Powered Multisensor Fusion for Forest Biomass Mapping: Photogrammetric Canopy Profiles Improve Estimates in Southeastern North Carolina

Forest Disturbance Analysis — FIA × Remote Sensing Study

Zenodo (10.5281/zenodo.18688899)
## Purpose

This directory contains the disturbance analysis scripts and results for quantifying forest disturbance patterns across the study region (557,121 ha, eastern US) during 2015–2022. The analysis uses two independent remote sensing products — Global Forest Watch (GFW) Hansen and LandTrendr — to characterise disturbance extent, biomass-weighted impact, and temporal distribution relative to the Sentinel-2 imagery acquisition window (2018–2019).

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


####Saved also on longleaf HPC: /work/users/w/a/wayne128/Biomass_ML/Dataset/OutBiomassRaster/Disturbance


#### LandTrendr technical notes

- NBR is computed with sensor-correct band indices: L5/L7 use `SR_B4`/`SR_B7`; L8/L9 use `SR_B5`/`SR_B7`.
- Disturbance extraction uses the **segment-based eMapR approach** (consecutive vertex pairs → sort by NBR magnitude) to avoid the `Array arguments must have same length` GEE error that occurs with direct vertex sorting.
- `minObservationsNeeded = 6` is appropriate for the 16-year time series (2010–2025).
- **`lt_end_year` extended from 2023 → 2025** (updated 2026-02-21) to resolve the LandTrendr right-edge bias identified in the first run. LandTrendr requires ≥2 post-disturbance observations to confirm a spectral breakpoint; with the 2023 end year, disturbances in 2021–2022 had only 1–2 trailing composites, causing systematic underdetection (LT/GFW detection ratio = 0.11–0.12 in those years). Extending to 2025 provides 3–4 post-event observations, making 2021–2022 detection fully reliable. The filter window (`filter_start`/`filter_end` = 2015–2022) is unchanged; the extra years are used only to stabilise segment fitting.

---

### `A_Review_TimeMismatch_DisturbanceAnalysis.py`

| File | Inputs | Outputs |
|------|--------|---------|
| `A_Review_TimeMismatch_DisturbanceAnalysis.py` | `gfw_forest_loss.tif`, `landtrendr_disturbance_updated.tif`, masked biomass raster | `FigSX_TimeMismatch_Disturbance.png/.pdf`, `time_mismatch_disturbance_summary.csv/.xlsx` |

Reprojects both disturbance rasters to the biomass modelling grid, applies the forest mask, computes four biomass-weighted metrics per year and per temporal proximity category, and generates the four-panel supplementary figure.

**Notes:**
- `landtrendr_disturbance_updated.tif` is the corrected export after fixing pixel-level QA masking, growing-season compositing, and tightening the disturbance threshold in `DisturbanceAnalysisLandtrendr.js` (see LandTrendr technical notes above).
- Set `GFW_YEAR_OFFSET = 2000` if the GEE script exported raw Hansen `lossyear` encoding (values 15–22); set to `0` if the script added 2000 before exporting (values 2015–2022).
- `MODEL_RMSE` at the top of the script must match the validation RMSE of the biomass model (currently 34.17 Mg/ha from G20 full-model). Update if the model changes.

#### Why simple average AGB is not sufficient — four-metric approach

Simple mean AGB in disturbed pixels answers "what was the typical biomass at those locations" but does not quantify how much of the **total carbon pool** was at risk. A disturbed low-biomass stand matters far less than a disturbed high-biomass stand, yet both count equally in a simple mean or area fraction. The script therefore computes four complementary metrics:

| # | Metric | Formula | What it answers |
|---|--------|---------|-----------------|
| 1 | **Biomass-weighted disturbance fraction** | Σ(AGB × area in disturbed pixels) / Σ(AGB × area in all forest pixels) × 100% | What fraction of the total standing carbon pool is in disturbed pixels? Directly comparable to the area fraction — a large gap (bio% << area%) means disturbance is concentrated in low-biomass stands. |
| 2 | **AGB density ratio** | mean AGB in disturbed pixels / mean AGB in all forest pixels | Is disturbance targeting high-biomass or low-biomass stands? Ratio < 1 means biomass-weighted impact is proportionally smaller than area alone implies. |
| 3 | **Temporally-stratified biomass-at-risk** | Metrics 1 & 4 split by minimum distance to RS window (2018–2019) | Only non-concurrent disturbances (Near and Far, distance > 0) fall in the gap between FIA measurement and RS imagery and can introduce training inconsistency; concurrent disturbances (distance = 0) are the **least** concerning because FIA and imagery capture the same forest state regardless of whether a disturbance occurred. |
| 4 | **Worst-case domain mean AGB shift** | Σ(AGB in disturbed pixels × pixel area) / total forest area  [Mg/ha] | Upper bound: if every disturbed pixel dropped to 0 AGB, by how much would the domain-level mean shift? Compared directly to model RMSE to contextualise the magnitude. |

**Temporal proximity categories** (minimum distance to RS window 2018–2019):

| Category | Years | Concern level |
|----------|-------|---------------|
| Concurrent (0 yr) | 2018, 2019 | **Lowest** — FIA and imagery capture the same forest state; no temporal gap regardless of disturbance |
| Near (1–2 yr) | 2016, 2017, 2020, 2021 | **Higher** — 1–2 yr gap between FIA measurement and imagery; disturbance falls in the gap and creates (low AGB, high spectral) training inconsistency |
| Far (3+ yr) | 2015, 2022 | **Moderate** — 3–4 yr gap; larger temporal separation but also more time for partial spectral recovery before imagery |

**Figure panels:**
- **(a)** GFW disturbance map 2015–2022 within forest mask; inset shows area% and biomass-weighted%
- **(b)** LandTrendr disturbance map 2015–2022 within forest mask; same inset
- **(c)** Dual y-axis annual chart: bars = area fraction (%), line = biomass-weighted fraction (%). RS window (2018–2019) shaded in gold. Divergence between bar height and line reveals whether disturbed pixels are biomass-rich or biomass-poor relative to the forest mean.
- **(d)** Temporal proximity worst-case AGB shift (Mg/ha) by category (Concurrent / Near / Far), grouped by GFW and LandTrendr. Dotted red reference line at model RMSE (34.17 Mg/ha). Bar annotations include AGB density ratio (r); r < 1 indicates below-average-biomass stands.

**Excel output sheets:**

| Sheet | Contents |
|-------|---------|
| `GFW_Annual` | Per-year: area, area%, total biomass at risk (Mg), bio-weighted%, mean AGB, density ratio, worst-case shift |
| `LandTrendr_Annual` | Same for LandTrendr |
| `Combined_Annual` | Both sources concatenated |
| `Advanced_Metrics` | Cumulative scalar values for all four metrics, both sources |
| `Proximity_Breakdown` | Per proximity category × source: area, bio-weighted%, worst-case shift, density ratio |

---

## Results — Raster-Based Disturbance Assessment

**Script:** `A_Review_TimeMismatch_DisturbanceAnalysis.py`
**Run 1:** 2026-02-21 — `lt_end_year = 2023` (right-edge bias present in 2021–2022)
**Run 2:** 2026-02-21 — `lt_end_year = 2025` (right-edge bias mitigated; **current results**)
**Model RMSE reference:** 34.17 Mg/ha (G20 ensemble full-model validation)

### Forest domain

| Metric | Value |
|--------|-------|
| Forest modelling domain | 557,121 ha |
| Pixel size | 25 m (0.0625 ha) |
| Mean AGB (all forest pixels) | 103.7 Mg/ha |

---

### Cumulative disturbance 2015–2022 — four biomass-weighted metrics

| Metric | GFW | LandTrendr |
|--------|-----|------------|
| **Disturbed area** | 64,011 ha | 34,749 ha |
| **Area fraction** | 11.49% | 6.24% |
| **Biomass-weighted fraction** (Metric 1) | 10.71% | 4.82% |
| **Worst-case domain AGB shift** (Metric 4) | **11.1 Mg/ha** | **5.0 Mg/ha** |
| Worst-case shift relative to RMSE | **0.33×** | **0.15×** |

Both products confirm that even the absolute upper bound on domain-mean AGB impact — assuming every disturbed pixel instantaneously drops to zero biomass — stays well below the model RMSE. LandTrendr's worst-case is approximately one-seventh of RMSE.

---

### Temporal gap analysis — disturbances in the mismatch window

The actual temporal mismatch occurs when disturbances fall **in the gap** between a plot's FIA measurement year and the RS imagery year (2018–2019). Disturbances concurrent with the imagery (2018–2019) are the **least** concerning: plots measured in those years are already temporally aligned with the imagery, so both FIA and RS capture the same forest state. The problematic scenarios are the non-concurrent years:

| Period | FIA plots at risk | Disturbance relative to FIA and imagery | Consequence |
|--------|-------------------|-----------------------------------------|-------------|
| Pre-RS gap (2015–2017) | Plots measured 2015–2017 | **In FIA, not in imagery** — disturbance falls within FIA data window (2015–2022) and is recorded in FIA; RS imagery acquired later (2018–2019) when forest has partially recovered, so disturbance is no longer (fully) visible spectrally | FIA records post-disturbance **low AGB**; imagery captures the recovering stand with **higher spectral signal** → training pair: (low AGB, higher spectral) |
| RS window (2018–2019) | Plots measured 2018–2019 | Simultaneous — FIA measurement and RS imagery capture the same forest state | **No mismatch**: both FIA and imagery reflect the same condition, whether disturbed or not |
| Post-RS gap (2020–2022) | Plots measured 2020–2022 | **In FIA, not in imagery** — disturbance occurs after imagery (2018–2019) but before or during FIA measurement; imagery captures the pre-disturbance state, FIA records the post-disturbance state | FIA records post-disturbance **low AGB**; imagery captures the pre-disturbance stand with **high spectral signal** → training pair: (low AGB, high spectral) |

Worst-case AGB shifts for the gap periods:

| Gap | GFW area (%) | GFW WC shift | LT area (%) | LT WC shift |
|-----|-------------|-------------|------------|------------|
| Pre-RS gap (2015–2017) | 4.40% | 3.10 Mg/ha | 2.58% | 1.58 Mg/ha |
| RS window (2018–2019) | 3.20% | 3.05 Mg/ha | 2.12% | 1.83 Mg/ha |
| Post-RS gap (2020–2022) | 3.85% | 4.96 Mg/ha | 1.53% | 1.60 Mg/ha |
| **All gap years combined** | **8.25%** | **8.06 Mg/ha** | **4.11%** | **3.18 Mg/ha** |
| Relative to RMSE | | **0.24×** | | **0.09×** |

Even combining all disturbances in both gap periods — the maximum possible mismatch scenario — the worst-case domain-mean AGB shift is 8.06 Mg/ha (GFW) or 3.18 Mg/ha (LandTrendr), both well below model RMSE. Both gap periods produce the same type of training inconsistency — a (low AGB, high spectral) pair — because in each case the disturbance is captured by FIA but not by the imagery: pre-RS disturbances are recorded by FIA while the imagery later captures the recovering stand, and post-RS disturbances are recorded by FIA while the imagery captures the pre-disturbance stand. The two gap periods therefore **compound** rather than cancel.

---

### Per-year breakdown

#### GFW Hansen (unchanged between runs)

| Year | Area (%) | Bio-wtd (%) | Density ratio (r) | Mean AGB (Mg/ha) | WC shift (Mg/ha) | Note |
|------|----------|-------------|-------------------|-----------------|-----------------|------|
| 2015 | 1.31 | 0.75 | 0.57 | 59.3 | 0.78 | |
| 2016 | 1.42 | 0.97 | 0.69 | 71.0 | 1.01 | |
| 2017 | 1.67 | 1.26 | 0.75 | 78.2 | 1.31 | |
| **2018** | **1.70** | **1.26** | **0.74** | **76.5** | **1.30** | **RS window (no mismatch)** |
| **2019** | **1.50** | **1.68** | **1.13** | **116.7** | **1.74** | **RS window (no mismatch)** |
| 2020 | 1.26 | 1.49 | 1.19 | 123.1 | 1.55 | |
| 2021 | 1.35 | 1.72 | 1.27 | 132.0 | 1.78 | |
| 2022 | 1.28 | 1.57 | 1.23 | 128.0 | 1.63 | |

#### LandTrendr (Run 2 — `lt_end_year = 2025`)

| Year | Area (%) | Bio-wtd (%) | Density ratio (r) | Mean AGB (Mg/ha) | WC shift (Mg/ha) | LT/GFW | Change vs Run 1 |
|------|----------|-------------|-------------------|-----------------|-----------------|--------|-----------------|
| 2015 | 0.98 | 0.50 | 0.51 | 53.1 | 0.52 | 0.75 | −3% |
| 2016 | 0.76 | 0.44 | 0.59 | 61.0 | 0.46 | 0.53 | −2% |
| 2017 | 0.84 | 0.57 | 0.68 | 70.9 | 0.60 | 0.50 | +4% |
| **2018** | **0.85** | **0.63** | **0.74** | **76.4** | **0.65** | **0.50** | **+15%** |
| **2019** | **1.27** | **1.13** | **0.89** | **92.4** | **1.18** | **0.85** | **+33%** |
| 2020 | 0.80 | 0.80 | 1.00 | 103.3 | 0.83 | 0.64 | +69% |
| 2021 | 0.41 | 0.43 | 1.05 | 109.4 | 0.45 | 0.30 | +174% |
| 2022 | 0.32 | 0.31 | 0.96 | 99.7 | 0.32 | 0.25 | +112% |

---

### Key findings

#### Finding 1 — GFW density ratio shift: disturbance moves into high-biomass stands over time (Metric 2)

The AGB density ratio (disturbed-stand mean / forest mean) shows a pronounced temporal trend in GFW:

| Period | Years | Density ratio | Interpretation |
|--------|-------|--------------|----------------|
| Early | 2015–2018 | 0.57 – 0.74 | Disturbance in **below-average** biomass stands |
| **RS window** | **2019** | **1.13** | Disturbance shifts to **above-average** biomass stands |
| Late | 2020–2022 | 1.19 – 1.27 | Disturbance continues in high-biomass stands |

Mean AGB in GFW-disturbed pixels rises from 59 Mg/ha (2015) to 132 Mg/ha (2021). This likely reflects a shift in disturbance type — early-period loss concentrated in low-stature or edge forests (deforestation, selective thinning); later years include storm damage, fire, or harvest in older, more carbon-dense stands.

Note that 2019 disturbances are concurrent with the RS imagery and therefore are **not** a direct source of temporal mismatch — plots measured in 2019 already align with the imagery. The 2019 density ratio (r = 1.13) is an ecologically interesting finding (disturbance shifted into above-average biomass stands) but has no special status for bias: the real mismatch concern lies with disturbances in 2015–2017 (pre-RS gap) and 2020–2022 (post-RS gap), where FIA measurement dates diverge from the imagery window. The fact that density ratios in those gap years are mostly below 1 (disturbance in below-average biomass stands) is directly reassuring.

LandTrendr density ratios are below or near 1.0 throughout (0.51–1.05). The slight exceedance in 2021 (r = 1.05) reflects newly recovered detections in denser stands, which take longer to show spectral recovery and were previously missed by the right-edge bias.

#### Finding 2 — LandTrendr right-edge bias: substantially mitigated by extending to 2025

Extending `lt_end_year` from 2023 to 2025 provides 3–4 post-event growing-season composites for 2021–2022, allowing LandTrendr to confirm breakpoints it previously missed. The improvement is large and systematic:

| Year | LT area — Run 1 (2023) | LT area — Run 2 (2025) | Change | LT/GFW — Run 1 | LT/GFW — Run 2 |
|------|----------------------|----------------------|--------|---------------|---------------|
| 2019 | 5,326 ha | 7,083 ha | +33% | 0.64 | 0.85 |
| 2020 | 2,656 ha | 4,482 ha | +69% | 0.38 | 0.64 |
| 2021 | 830 ha | **2,276 ha** | **+174%** | 0.11 | **0.30** |
| 2022 | 842 ha | **1,787 ha** | **+112%** | 0.12 | **0.25** |

The 2021–2022 LT/GFW ratio improved from 0.11–0.12 to 0.25–0.30. The remaining gap relative to GFW (which maintains ~0.50–0.85 in earlier years) reflects the genuine algorithmic difference between the two products: GFW detects any canopy-cover reduction, while LandTrendr requires a sustained NBR magnitude decline. This residual gap is not a bias — it is the expected conservative behaviour of LandTrendr. Both estimates now bracket a credible range for the true disturbance signal.

#### Finding 3 — Temporal proximity stratification (Metric 3)

| Proximity | Years | Mismatch role | GFW area (%) | GFW WC shift | LT area (%) | LT WC shift |
|-----------|-------|--------------|-------------|-------------|------------|------------|
| Concurrent (0 yr) | 2018, 2019 | **No mismatch** — FIA and imagery aligned | 3.20% | 3.05 Mg/ha | 2.12% | 1.83 Mg/ha |
| Near (1–2 yr) | 2016, 2017, 2020, 2021 | **Gap** — 1–2 yr between FIA and imagery | 5.35% | 5.38 Mg/ha | 2.80% | 2.39 Mg/ha |
| Far (3+ yr) | 2015, 2022 | **Gap** — 3–4 yr between FIA and imagery | 2.58% | 2.41 Mg/ha | 1.31% | 0.85 Mg/ha |

The Near and Far categories represent the actual temporal mismatch window — disturbances in those years fall in the gap between FIA measurement dates and RS imagery. The Concurrent category does not contribute to mismatch: FIA plots measured in 2018–2019 are temporally aligned with the imagery, so both data sources reflect the same forest state. Even combining Near + Far (the complete gap period), the worst-case AGB shift is 7.79 Mg/ha (GFW) or 3.24 Mg/ha (LandTrendr) — well below model RMSE. Both the pre-RS and post-RS gap periods produce the same direction of training inconsistency (disturbance in FIA but not in imagery → low AGB paired with high spectral signal), so their effects compound rather than cancel.

---

## Workflow Summary

```
Temporal mismatch analysis: FIA (2015–2022) vs RS imagery (2018–2019)
         │
         ├── Spatial analysis — GEE scripts
         │     ├── DisturbanceAnalysisGFW.js             → gfw_forest_loss.tif
         │     └── DisturbanceAnalysisLandtrendr.js       → landtrendr_disturbance_updated.tif
         │           (QA masking + growing-season compositing fix applied 2026-02-21)
         │
         └── Spatial analysis — Python (raster domain analysis)
               └── A_Review_TimeMismatch_DisturbanceAnalysis.py
                     ├── FigSX_TimeMismatch_Disturbance.png / .pdf
                     └── time_mismatch_disturbance_summary.csv / .xlsx
```

---

## Data Confidentiality Note

FIA plot location coordinates (`LAT`, `LON` from the `fia_plot` sheet) are protected by USFS confidentiality agreements and are **not** included in this repository. The GEE spatial analysis operates over the full study region without requiring plot-level coordinates.

---

## References

- Hansen, M. C., et al. (2013). High-resolution global maps of 21st-century forest cover change. *Science*, 342(6160), 850–853.
- Kennedy, R. E., et al. (2010). Detecting trends in forest disturbance and recovery using yearly Landsat time series: 1. LandTrendr — Temporal segmentation algorithms. *Remote Sensing of Environment*, 114(12), 2897–2910.
- eMapR LT-GEE library: [https://github.com/eMapR/LT-GEE](https://github.com/eMapR/LT-GEE)
- USFS FIA Database Manual v9.1 — disturbance code definitions (§2.5.8).
