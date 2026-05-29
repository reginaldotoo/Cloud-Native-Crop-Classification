# Cloud-Native Crop Classification

Multi-temporal crop type classification for agricultural land in Manitoba, Canada — built entirely on cloud-native geospatial infrastructure. No data downloads, no intermediate files. The pipeline queries satellite imagery directly from Microsoft Planetary Computer, processes it in memory with Dask, and classifies crops using Random Forest.

The goal was to build a scalable, reproducible crop classification pipeline using the kind of open, Python-based tools used in production remote sensing systems.

## What it does

The pipeline takes a study area boundary and training polygons as input, and produces a classified crop map as output. Everything in between — image search, loading, cloud masking, compositing, feature engineering, classification — happens programmatically in a single script.

**Study area:** Macdonald municipality, Manitoba, Canada  
**Imagery:** Sentinel-2 L2A (optical) + Sentinel-1 RTC (SAR), May–September 2024  
**Classes:** Cereals (Spring Wheat, Oats, Barley), Canola, Soybeans, Corn  
**Resolution:** 10 meters

## How it works

The pipeline pulls Sentinel-2 and Sentinel-1 scenes from Planetary Computer's STAC catalog using `pystac-client`. `stackstac` converts the search results into lazy Xarray arrays backed by Dask, meaning nothing actually downloads until computation is triggered. This lets you chain operations on a ~80 GB image stack without needing 80 GB of RAM.

For each month (May through September), the pipeline builds a composite from cloud-masked S2 scenes and speckle-filtered S1 scenes. Four vegetation indices (NDVI, EVI, NDRE, PSRI) are derived from the optical bands, and three SAR features (VV dB, VH dB, cross-ratio) are computed from the radar data. That gives 17 features per month, or 85 features total across the five-month growing season.

A crop mask from AAFC's Annual Crop Inventory (downloaded at runtime from Canada's public REST endpoint) restricts classification to agricultural pixels only. Training polygons are rasterized onto the 10m grid, and a Random Forest classifier is trained with a polygon-level train/test split to prevent spatial autocorrelation from inflating accuracy.

## Results

| Metric | Value |
|--------|-------|
| Overall Accuracy | 96.47% |
| Cohen's Kappa | 0.9508 |
| Test samples | 14,457 pixels |
| Train/test split | By polygon (70/30), stratified |

Per-class performance:

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Cereals | 1.00 | 1.00 | 1.00 |
| Canola | 1.00 | 1.00 | 1.00 |
| Soybeans | 0.88 | 0.99 | 0.93 |
| Corn | 0.98 | 0.73 | 0.84 |

Corn recall is the main weakness — 465 Corn pixels were misclassified as Soybeans, which is a well-known spectral confusion pair in Prairie agriculture. A more sophisticated speckle filter (e.g. Gamma MAP or multi-temporal Quegan) would likely improve SAR feature discrimination and help separate these two classes.

## Design decisions

- **Sentinel-1 RTC over raw GRD:** The RTC collection from Planetary Computer comes pre-calibrated with terrain correction and gamma flattening already applied, eliminating three preprocessing steps. A spatial mean filter handles residual speckle — simpler than a Gamma MAP or multi-temporal Quegan filter, but sufficient for classification features when paired with monthly compositing.
- **Polygon-level train/test split:** Splitting by pixel inflates accuracy to ~99.9% due to spatial autocorrelation between adjacent pixels in the same field. Splitting by polygon gives honest numbers.
- **AAFC crop mask at runtime:** Rather than bundling a static reference raster, the pipeline downloads the AAFC Annual Crop Inventory from Canada's public REST endpoint and caches it locally. This keeps the repo lightweight and ensures the mask matches the target year.
- **Single-script design:** Intentionally self-contained for readability and reproducibility. A production system would use config files, logging, and modular packaging.

## Tech stack

- **pystac-client** — STAC catalog search
- **planetary-computer** — Azure Blob Storage URL signing
- **stackstac** — STAC items to lazy Xarray DataArrays
- **xarray / dask** — chunked, lazy multi-dimensional array computation
- **rioxarray / rasterio** — CRS handling, GeoTIFF I/O, polygon rasterization
- **geopandas** — shapefile I/O and reprojection
- **scipy** — spatial speckle filtering
- **scikit-learn** — Random Forest classification and evaluation
- **matplotlib** — visualization

## Data sources

- **Sentinel-2 L2A** — surface reflectance, 10 spectral bands, via Planetary Computer
- **Sentinel-1 RTC** — radiometrically terrain-corrected gamma naught backscatter, via Planetary Computer
- **AAFC Annual Crop Inventory** — 30m crop type map, via Government of Canada Open Data

## Usage

```
pip install pystac-client planetary-computer stackstac xarray dask rioxarray geopandas scikit-learn scipy matplotlib requests
```

Edit the paths at the top of `cloud_native_pipeline.py` to point to your ROI shapefile and training polygons, then:

```
python cloud_native_pipeline.py
```

The script takes about 15–20 minutes end to end, mostly spent on the data transfer from Azure during the final `.compute()` step. The AAFC raster is cached locally after the first download.

## Notes

- The polygon-level train/test split is critical. A pixel-level split inflates accuracy to ~99.9% due to spatial autocorrelation between adjacent pixels within the same field. The polygon-level split gives honest numbers.
- Planetary Computer's signed URLs expire after a few hours. If the script fails with a CURL/DNS error partway through, run it again from the top to get fresh tokens.
- This pipeline is intentionally single-script and self-contained. For a production system you'd want config files, logging, and modular packaging — but for a portfolio project, readability and reproducibility come first.
