
/*
Forest Disturbance Analysis using Global Forest Watch (GFW)
Copy and paste this into the Google Earth Engine Code Editor
*/

// =============================================================================
// CONFIGURATION - MODIFY THESE PARAMETERS
// =============================================================================

// Study area (example: bounding box)
// Replace with your area of interest or draw a geometry in GEE Code Editor
var study_area = ee.Geometry.Rectangle([-125, 24, -66, 50]);

// Filter window for disturbance detection (FIA measurement period)
var filter_start = 2015;
var filter_end = 2022;

// Minimum tree cover % for forest definition (baseline year 2000)
var min_tree_cover = 30;

// =============================================================================
// FUNCTIONS
// =============================================================================

function getGFWLoss(year_start, year_end, tree_cover_threshold) {
  /**
   * Get Global Forest Watch forest loss data filtered by time window and tree cover
   */

  var gfw = ee.Image('UMD/hansen/global_forest_change_2023_v1_11');

  // Get loss year (0 = no loss, 1-23 = year of loss since 2000)
  var loss_year = gfw.select('lossyear');

  // Convert to actual year and filter by time window
  var loss_year_actual = loss_year.add(2000);
  var loss_mask = loss_year_actual.gte(year_start).and(loss_year_actual.lte(year_end));

  var loss_year_filtered = loss_year_actual.updateMask(loss_mask);

  // Get tree cover (baseline year 2000)
  var tree_cover = gfw.select('treecover2000');

  // Restrict to areas with sufficient tree cover
  var forest_mask = tree_cover.gte(tree_cover_threshold);

  return loss_year_filtered.updateMask(forest_mask);
}

// =============================================================================
// MAIN ANALYSIS
// =============================================================================

print('Loading Global Forest Watch data...');
var gfw_loss = getGFWLoss(filter_start, filter_end, min_tree_cover);

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

// GFW loss year (continuous, color-coded by year)
Map.addLayer(
  gfw_loss.clip(study_area),
  year_vis,
  'GFW Forest Loss Year',
  true
);

// Binary disturbance mask
Map.addLayer(
  gfw_loss.mask().selfMask().clip(study_area),
  {palette: ['red']},
  'GFW Forest Loss (Binary)',
  false
);

// =============================================================================
// STATISTICS
// =============================================================================

print('Calculating disturbance statistics...');
print('Study period: ' + filter_start + '-' + filter_end);

var pixel_area = ee.Image.pixelArea();

// Total loss area
var total_area = pixel_area
  .updateMask(gfw_loss.mask())
  .reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: study_area,
    scale: 30,
    maxPixels: 1e13
  });

print('\n=== GFW Forest Loss Area Statistics (hectares) ===');
print('Total forest loss area:', ee.Number(total_area.get('area')).divide(10000));

// Annual loss breakdown
print('\n--- Annual Loss Breakdown ---');
var annual_loss = ee.List.sequence(filter_start, filter_end).map(function(year) {
  year = ee.Number(year);
  var annual_area = pixel_area
    .updateMask(gfw_loss.eq(year))
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

print(ee.FeatureCollection(annual_loss));

// =============================================================================
// EXPORT TASK (OPTIONAL)
// =============================================================================

// GFW provides loss year only; there is no change-magnitude band in the Hansen product.
var gfw_export = gfw_loss.rename('loss_year').toFloat().clip(study_area);

Export.image.toDrive({
  image: gfw_export,
  description: 'gfw_forest_loss',
  folder: 'GEE_Exports',
  fileNamePrefix: 'gfw_forest_loss',
  region: study_area,
  scale: 30,
  maxPixels: 1e13,
  crs: 'EPSG:4326'
});

/*
Export.image.toAsset({
  image: gfw_export,
  description: 'gfw_forest_loss',
  assetId: 'users/YOUR_USERNAME/gfw_forest_loss',
  region: study_area,
  scale: 30,
  maxPixels: 1e13,
  crs: 'EPSG:4326'
});
*/

print('\nAnalysis complete! Check the Layers panel and Console for results.');
print('Export task submitted: gfw_forest_loss (band: loss_year)');
