
/*
Forest Disturbance Analysis using LandTrendr
Copy and paste this into the Google Earth Engine Code Editor
*/

// =============================================================================
// CONFIGURATION - MODIFY THESE PARAMETERS
// =============================================================================

// Study area (example: bounding box)
// Replace with your area of interest or draw a geometry in GEE Code Editor
var study_area = ee.Geometry.Rectangle([-125, 24, -66, 50]);

// LandTrendr time series window (extend back for proper segmentation)
var lt_start_year = 2010;  // Extended start for LandTrendr
var lt_end_year = 2023;    // Covers full FIA period (2015-2022)

// Filter window for disturbance display (FIA measurement period)
var filter_start = 2015;
var filter_end = 2022;

// LandTrendr parameters
// NOTE: minObservationsNeeded must be <= number of years in time series
var lt_params = {
  maxSegments: 6,
  spikeThreshold: 0.9,
  vertexCountOvershoot: 3,
  preventOneYearRecovery: true,
  recoveryThreshold: 0.25,
  pvalThreshold: 0.05,
  bestModelProportion: 0.75,
  minObservationsNeeded: 6
};

// NBR change magnitude threshold (negative = disturbance)
var disturbance_threshold = -100;

// =============================================================================
// FUNCTIONS
// =============================================================================

function getLandsatCollection(year_start, year_end, aoi) {
  /**
   * Build merged Landsat collection with NBR for LandTrendr input
   */

  // L5/L7: NIR = SR_B4, SWIR2 = SR_B7
  function addNBR_L57(image) {
    var nbr = image.normalizedDifference(['SR_B4', 'SR_B7']).multiply(1000).rename('NBR');
    return image.addBands(nbr);
  }

  // L8/L9: NIR = SR_B5, SWIR2 = SR_B7  (SR_B4 on L8/L9 is Red, not NIR)
  function addNBR_L89(image) {
    var nbr = image.normalizedDifference(['SR_B5', 'SR_B7']).multiply(1000).rename('NBR');
    return image.addBands(nbr);
  }

  // Landsat 5
  var l5 = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
    .filterBounds(aoi)
    .filterDate(year_start + '-01-01', year_end + '-12-31')
    .filter(ee.Filter.lt('CLOUD_COVER', 50))
    .map(addNBR_L57);

  // Landsat 7
  var l7 = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
    .filterBounds(aoi)
    .filterDate(year_start + '-01-01', year_end + '-12-31')
    .filter(ee.Filter.lt('CLOUD_COVER', 50))
    .map(addNBR_L57);

  // Landsat 8
  var l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterBounds(aoi)
    .filterDate(year_start + '-01-01', year_end + '-12-31')
    .filter(ee.Filter.lt('CLOUD_COVER', 50))
    .map(addNBR_L89);

  // Landsat 9
  var l9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
    .filterBounds(aoi)
    .filterDate(year_start + '-01-01', year_end + '-12-31')
    .filter(ee.Filter.lt('CLOUD_COVER', 50))
    .map(addNBR_L89);

  return l5.merge(l7).merge(l8).merge(l9);
}

function extractDisturbanceFromLT(lt_result, threshold) {
  /**
   * Extract greatest disturbance segment from LandTrendr output.
   * Uses the segment-based approach (consecutive vertex pairs) following the
   * eMapR LT-GEE library, which avoids the "Array arguments must have same
   * length along one axis" error caused by sorting vertex rows directly.
   *
   * LandTrendr array layout: rows = [year, src, fitted, isVertex, ...],
   *                          cols = one per year in the time series.
   */

  var lt = lt_result.select('LandTrendr');

  // Isolate vertex observations (isVertex = row index 3)
  var vertexMask = lt.arraySlice(0, 3, 4);
  var vertices   = lt.arrayMask(vertexMask);   // [4 rows × n_vertices cols]

  // Build segment arrays from consecutive vertex pairs
  var left  = vertices.arraySlice(1, 0, -1);   // all vertices except last
  var right = vertices.arraySlice(1, 1, null); // all vertices except first

  var startYear = left.arraySlice(0, 0, 1);    // year at segment start
  var startVal  = left.arraySlice(0, 2, 3);    // fitted NBR at segment start (row 2)
  var endYear   = right.arraySlice(0, 0, 1);   // year at segment end
  var endVal    = right.arraySlice(0, 2, 3);   // fitted NBR at segment end

  var mag = endVal.subtract(startVal);         // negative = NBR loss = disturbance
  var dur = endYear.subtract(startYear);

  // Stack segment attributes into a [5 × n_segments] array.
  // Each band is [1 × n_segments]; toArray(0) concatenates along axis 0.
  var segInfo = ee.Image.cat([startYear, endYear, startVal, endVal, mag]).toArray(0);

  // Sort segments by magnitude ascending — most negative (greatest loss) goes first.
  var sortKey = segInfo.arraySlice(0, 4, 5);   // magnitude row [1 × n_segments]
  var sorted  = segInfo.arraySort(sortKey);

  // First column after sort = segment with greatest NBR loss
  var best = sorted.arraySlice(1, 0, 1);       // [5 × 1]

  // Year of disturbance = end year of the loss segment
  var dist_year = best.arraySlice(0, 1, 2)
    .arrayProject([0]).arrayFlatten([['yod']]).toInt16();

  var magnitude = best.arraySlice(0, 4, 5)
    .arrayProject([0]).arrayFlatten([['magnitude']]);

  // Keep only pixels where loss magnitude exceeds threshold
  var is_dist = magnitude.lt(threshold);

  return {
    year: dist_year.updateMask(is_dist),
    magnitude: magnitude.updateMask(is_dist)
  };
}

// =============================================================================
// MAIN ANALYSIS
// =============================================================================

print('Building Landsat collection...');
var landsat_collection = getLandsatCollection(lt_start_year, lt_end_year, study_area);

print('Preparing LandTrendr input...');
var lt_collection = landsat_collection.select(['NBR']);

// Create annual median composites for LandTrendr input
var years = ee.List.sequence(lt_start_year, lt_end_year);

var annual_collection = ee.ImageCollection(years.map(function(year) {
  year = ee.Number(year);
  var composite = lt_collection
    .filter(ee.Filter.calendarRange(year, year, 'year'))
    .median()
    .set('system:time_start', ee.Date.fromYMD(year, 6, 1).millis());
  return composite;
}));

// Run LandTrendr
print('Running LandTrendr...');
var lt_params_with_data = {
  timeSeries: annual_collection,
  maxSegments: lt_params.maxSegments,
  spikeThreshold: lt_params.spikeThreshold,
  vertexCountOvershoot: lt_params.vertexCountOvershoot,
  preventOneYearRecovery: lt_params.preventOneYearRecovery,
  recoveryThreshold: lt_params.recoveryThreshold,
  pvalThreshold: lt_params.pvalThreshold,
  bestModelProportion: lt_params.bestModelProportion,
  minObservationsNeeded: lt_params.minObservationsNeeded
};

var lt_result = ee.Algorithms.TemporalSegmentation.LandTrendr(lt_params_with_data);

// Extract disturbances across full time series
print('Extracting LandTrendr disturbances...');
var lt_disturbance = extractDisturbanceFromLT(lt_result, disturbance_threshold);
var lt_year_all = lt_disturbance.year;
var lt_magnitude_all = lt_disturbance.magnitude;

// Filter results to study display period
print('Filtering to study period: ' + filter_start + '-' + filter_end);
var lt_period_mask = lt_year_all.gte(filter_start).and(lt_year_all.lte(filter_end));
var lt_year = lt_year_all.updateMask(lt_period_mask);
var lt_magnitude = lt_magnitude_all.updateMask(lt_period_mask);

// =============================================================================
// VISUALIZATION
// =============================================================================

print('Creating map visualization...');

Map.centerObject(study_area, 6);
Map.addLayer(study_area, {color: 'white'}, 'Study Area', false);

var year_vis = {
  min: filter_start,
  max: filter_end,
  palette: ['yellow', 'orange', 'red']
};

var magnitude_vis = {
  min: -500,
  max: 0,
  palette: ['white', 'red']
};

// Disturbance year (color-coded)
Map.addLayer(
  lt_year.clip(study_area),
  year_vis,
  'LandTrendr Disturbance Year',
  true
);

// Binary disturbance mask
Map.addLayer(
  lt_year.mask().selfMask().clip(study_area),
  {palette: ['red']},
  'LandTrendr Disturbance (Binary)',
  false
);

// NBR change magnitude
Map.addLayer(
  lt_magnitude.clip(study_area),
  magnitude_vis,
  'LandTrendr Disturbance Magnitude',
  false
);

// =============================================================================
// STATISTICS
// =============================================================================

print('Calculating disturbance statistics...');
print('Study period: ' + filter_start + '-' + filter_end);

var pixel_area = ee.Image.pixelArea();

// Total disturbance area
var total_area = pixel_area
  .updateMask(lt_year.mask())
  .reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: study_area,
    scale: 30,
    maxPixels: 1e13
  });

print('\n=== LandTrendr Disturbance Area Statistics (hectares) ===');
print('Total disturbance area:', ee.Number(total_area.get('area')).divide(10000));

// Annual disturbance breakdown
print('\n--- Annual Disturbance Breakdown ---');
var annual_disturbance = ee.List.sequence(filter_start, filter_end).map(function(year) {
  year = ee.Number(year);
  var annual_area = pixel_area
    .updateMask(lt_year.eq(year))
    .reduceRegion({
      reducer: ee.Reducer.sum(),
      geometry: study_area,
      scale: 30,
      maxPixels: 1e13
    });
  return ee.Feature(null, {
    year: year,
    area_ha: ee.Number(annual_area.get('area')).divide(10000)
  });
});

print(ee.FeatureCollection(annual_disturbance));

// =============================================================================
// EXPORT TASK (OPTIONAL)
// =============================================================================

// Stack year and magnitude into a single 2-band image for export
var lt_export = ee.Image.cat([
  lt_year.rename('disturbance_year'),
  lt_magnitude.rename('disturbance_magnitude')
]).toFloat().clip(study_area);

Export.image.toDrive({
  image: lt_export,
  description: 'landtrendr_disturbance',
  folder: 'GEE_Exports',
  fileNamePrefix: 'landtrendr_disturbance',
  region: study_area,
  scale: 30,
  maxPixels: 1e13,
  crs: 'EPSG:4326'
});

/*
Export.image.toAsset({
  image: lt_export,
  description: 'landtrendr_disturbance',
  assetId: 'users/YOUR_USERNAME/landtrendr_disturbance',
  region: study_area,
  scale: 30,
  maxPixels: 1e13,
  crs: 'EPSG:4326'
});
*/

print('\nAnalysis complete! Check the Layers panel and Console for results.');
print('Export task submitted: landtrendr_disturbance (bands: disturbance_year, disturbance_magnitude)');
