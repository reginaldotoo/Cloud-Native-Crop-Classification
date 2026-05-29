
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import xarray as xr
import geopandas as gpd
import stackstac
import planetary_computer as pc
from pystac_client import Client
from scipy.ndimage import uniform_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import rioxarray  # noqa: F401
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import rasterio
import requests
import time
import os


# CONFIGURATION

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Paths ---
ROI_PATH = os.path.join(SCRIPT_DIR, "data", "roi", "Macdonald.shp")
TRAINING_PATH = os.path.join(SCRIPT_DIR, "data", "training", "Macdonald_Training.shp")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")

# --- Sensor / temporal ---
YEAR = 2024
MONTHS = [5, 6, 7, 8, 9]
MONTH_NAMES = ["May", "Jun", "Jul", "Aug", "Sep"]
MAX_CLOUD_COVER = 50

S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
S1_BANDS = ["vv", "vh"]

RESOLUTION = 10
CHUNKSIZE = 2048
EPSG = 32614

# SCL values to mask: 1=Saturated, 3=Cloud Shadow, 8=Cloud Med,
# 9=Cloud High, 10=Cirrus, 11=Snow/Ice
SCL_MASK_VALUES = [1, 3, 8, 9, 10, 11]

# Speckle filter kernel size (matches GEE Gamma MAP kernel=7)
SPECKLE_KERNEL = 7

# Classification
CLASS_NAMES = ["Cereals", "Canola", "Soybeans", "Corn"]
CLASS_FIELD = "Class"
RF_N_ESTIMATORS = 200
RF_RANDOM_STATE = 42
TEST_SIZE = 0.3
MAX_SAMPLES = 50000

# STAC
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


# 1. LOAD ROI

def load_roi(path):
    """Read ROI shapefile and return GeoDataFrame + bbox in EPSG:4326."""
    roi = gpd.read_file(path)
    roi_4326 = roi.to_crs("EPSG:4326")
    bbox = list(roi_4326.total_bounds)
    print(f"ROI loaded: {path}")
    print(f"  CRS: {roi.crs}")
    print(f"  Bbox (4326): {[round(b, 4) for b in bbox]}")
    return roi, bbox


# 2. DOWNLOAD AAFC CROP MASK

# AAFC crop codes for target classes
AAFC_CROP_CODES = [146, 136, 133, 153, 158, 147]  # Spring Wheat, Oats, Barley, Canola, Soybeans, Corn
AAFC_REMAP =      [  1,   1,   1,   2,   3,   4]  # Cereals, Canola, Soybeans, Corn


def download_aafc(bbox, year, output_dir):
    """
    Download AAFC Annual Crop Inventory for the bbox from the public
    ESRI REST ImageServer. No API key, no authentication.
    """
    aafc_path = os.path.join(output_dir, f"aafc_{year}.tif")
    if os.path.exists(aafc_path):
        print(f"  AAFC already downloaded: {aafc_path}")
        return aafc_path

    base_url = (
        f"https://agriculture.canada.ca/imagery-images/rest/services/"
        f"annual_crop_inventory/{year}/ImageServer/exportImage"
    )

    width = int(abs(bbox[2] - bbox[0]) * 111000 / 30)
    height = int(abs(bbox[3] - bbox[1]) * 111000 / 30)

    params = {
        "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "bboxSR": "4326",
        "imageSR": "32614",
        "size": f"{width},{height}",
        "format": "tiff",
        "pixelType": "U8",
        "interpolation": "RSP_NearestNeighbor",
        "f": "image",
    }

    print(f"  Downloading AAFC {year} crop inventory...")
    response = requests.get(base_url, params=params, timeout=120)
    response.raise_for_status()

    os.makedirs(output_dir, exist_ok=True)
    with open(aafc_path, "wb") as f:
        f.write(response.content)

    print(f"  Saved: {aafc_path} ({len(response.content) / 1024:.0f} KB)")
    return aafc_path


def build_crop_mask(aafc_path, feature_stack):
    """
    Read AAFC raster, remap to 4 classes, resample to match the feature
    stack grid, and return a boolean mask (True = crop pixel).
    """
    with rasterio.open(aafc_path) as src:
        aafc_data = src.read(1)
        aafc_transform = src.transform
        aafc_crs = src.crs

    # Remap to 4 classes (0 = non-crop)
    remapped = np.zeros_like(aafc_data, dtype=np.uint8)
    for code, group in zip(AAFC_CROP_CODES, AAFC_REMAP):
        remapped[aafc_data == code] = group

    # Create xarray DataArray with spatial coords
    ny_aafc, nx_aafc = remapped.shape
    x_aafc = np.linspace(
        aafc_transform.c + aafc_transform.a / 2,
        aafc_transform.c + aafc_transform.a * (nx_aafc - 0.5),
        nx_aafc,
    )
    y_aafc = np.linspace(
        aafc_transform.f + aafc_transform.e / 2,
        aafc_transform.f + aafc_transform.e * (ny_aafc - 0.5),
        ny_aafc,
    )

    aafc_da = xr.DataArray(
        remapped, dims=["y", "x"],
        coords={"y": y_aafc, "x": x_aafc},
    )
    aafc_da = aafc_da.rio.write_crs(aafc_crs)

    # Reproject/resample to match feature stack grid
    target_y = feature_stack.coords["y"].values
    target_x = feature_stack.coords["x"].values
    aafc_resampled = aafc_da.interp(
        y=target_y, x=target_x, method="nearest",
    ).fillna(0).astype(np.uint8)

    crop_mask = aafc_resampled > 0
    n_crop = int(crop_mask.sum().values)
    n_total = crop_mask.size
    print(f"  Crop mask: {n_crop}/{n_total} pixels ({100 * n_crop / n_total:.1f}% cropland)")

    return crop_mask


# 2. STAC SEARCH

def search_stac(collection, bbox, datetime_range):
    """Query Planetary Computer STAC catalog."""
    catalog = Client.open(STAC_URL, modifier=pc.sign_inplace)
    query_params = {"collections": [collection], "bbox": bbox, "datetime": datetime_range}
    if collection == "sentinel-2-l2a":
        query_params["query"] = {"eo:cloud_cover": {"lt": MAX_CLOUD_COVER}}
    items = catalog.search(**query_params).item_collection()
    print(f"  {collection}: {len(items)} scenes")
    return items


# 3. LOAD SENTINEL-2


def load_s2(items, bbox):
    """Load S2 spectral bands + SCL as lazy Xarray arrays."""
    stack = stackstac.stack(
        items,
        assets=S2_BANDS,
        epsg=EPSG,
        resolution=RESOLUTION,
        bounds_latlon=bbox,
        chunksize=CHUNKSIZE,
        dtype=np.float64,
        rescale=False,
    )
    stack = stack / 10000.0

    scl = stackstac.stack(
        items,
        assets=["SCL"],
        epsg=EPSG,
        resolution=RESOLUTION,
        bounds_latlon=bbox,
        chunksize=CHUNKSIZE,
        rescale=False,
    ).squeeze("band", drop=True)

    print(f"  S2 stack: {stack.shape} ({stack.nbytes / 1e9:.1f} GB uncompressed)")
    return stack, scl


# 4. LOAD SENTINEL-1 RTC

def load_s1(items, bbox):
    """
    Load S1 RTC (gamma naught) as lazy Xarray array.
    PC's sentinel-1-rtc already includes radiometric calibration,
    geometric terrain correction, and gamma flattening.
    """
    stack = stackstac.stack(
        items,
        assets=S1_BANDS,
        epsg=EPSG,
        resolution=RESOLUTION,
        bounds_latlon=bbox,
        chunksize=CHUNKSIZE,
        dtype=np.float64,
        rescale=False,
    )
    print(f"  S1 RTC stack: {stack.shape} ({stack.nbytes / 1e9:.1f} GB uncompressed)")
    return stack


# 5. CLOUD MASKING (S2)

def apply_cloud_mask(stack, scl):
    """Mask clouds/shadows/snow using SCL band."""
    bad_mask = scl.isin(SCL_MASK_VALUES)
    return stack.where(~bad_mask)


# 6. SPECKLE FILTERING (S1)

def apply_speckle_filter(s1_stack, kernel_size=SPECKLE_KERNEL):
    """
    Apply spatial mean filter to each S1 scene.
    Replaces the Gamma MAP + Quegan filter from GEE.
    """
    def filter_scene(scene):
        return xr.apply_ufunc(
            lambda x: uniform_filter(x, size=kernel_size),
            scene,
            dask="parallelized",
            output_dtypes=[np.float64],
        )

    filtered_scenes = []
    for t in range(s1_stack.sizes["time"]):
        filtered_scenes.append(filter_scene(s1_stack.isel(time=t)))
    filtered = xr.concat(filtered_scenes, dim="time")
    filtered["time"] = s1_stack["time"]
    return filtered


# 7. MONTHLY COMPOSITES

def compute_monthly_composites(stack, year, months):
    """Group by month, take pixel-wise median."""
    composites = {}
    for month in months:
        month_mask = stack["time"].dt.month == month
        month_data = stack.sel(time=month_mask)
        if month_data.sizes["time"] > 0:
            composites[month] = month_data.median(dim="time")
        else:
            composites[month] = None
            print(f"  WARNING: no scenes for month {month}")
    return composites


# 8. TEMPORAL INTERPOLATION (S2)

def temporal_interpolation(composites, months):
    """Fill NaN gaps using nearest valid neighbors (before/after averaging)."""
    sorted_months = sorted(months)
    filled = {}

    for i, month in enumerate(sorted_months):
        current = composites[month]
        if current is None:
            current = xr.full_like(composites[sorted_months[0]], np.nan)

        before = None
        for j in range(i - 1, -1, -1):
            if composites[sorted_months[j]] is not None:
                before = composites[sorted_months[j]]
                break

        after = None
        for j in range(i + 1, len(sorted_months)):
            if composites[sorted_months[j]] is not None:
                after = composites[sorted_months[j]]
                break

        filled_current = current.copy()
        nan_mask = current.isnull()

        if before is not None and after is not None:
            filled_current = filled_current.where(~nan_mask, (before + after) / 2.0)
        elif before is not None:
            filled_current = filled_current.where(~nan_mask, before)
        elif after is not None:
            filled_current = filled_current.where(~nan_mask, after)

        filled[month] = filled_current
    return filled


# 9. VEGETATION INDICES

def _strip_coords(da):
    """Keep only y and x coords, drop all stackstac metadata."""
    drop = [c for c in da.coords if c not in {"y", "x"}]
    return da.drop_vars(drop, errors="ignore")


def compute_indices(s2_composite):
    """Compute NDVI, EVI, NDRE, PSRI from an S2 monthly composite."""
    nir = _strip_coords(s2_composite.sel(band="B08"))
    red = _strip_coords(s2_composite.sel(band="B04"))
    blue = _strip_coords(s2_composite.sel(band="B02"))
    green = _strip_coords(s2_composite.sel(band="B03"))
    re1 = _strip_coords(s2_composite.sel(band="B05"))
    re2 = _strip_coords(s2_composite.sel(band="B06"))
    nir_narrow = _strip_coords(s2_composite.sel(band="B8A"))

    ndvi = ((nir - red) / (nir + red)).expand_dims(band=["NDVI"])
    evi = (2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))).expand_dims(band=["EVI"])
    ndre = ((nir_narrow - re1) / (nir_narrow + re1)).expand_dims(band=["NDRE"])
    psri = ((red - green) / re2).expand_dims(band=["PSRI"])

    return xr.concat([ndvi, evi, ndre, psri], dim="band")


# 10. S1 DERIVED FEATURES

def compute_s1_features(s1_composite):
    """Convert to dB and compute cross-ratio."""
    vv = _strip_coords(s1_composite.sel(band="vv"))
    vh = _strip_coords(s1_composite.sel(band="vh"))

    vv_db = (10 * np.log10(vv.where(vv > 0))).expand_dims(band=["VV_dB"])
    vh_db = (10 * np.log10(vh.where(vh > 0))).expand_dims(band=["VH_dB"])
    cr = (vh / vv.where(vv > 0)).expand_dims(band=["CR"])

    return xr.concat([vv_db, vh_db, cr], dim="band")


# 11. BUILD 85-FEATURE STACK

def build_feature_stack(s2_composites, s1_composites, months):
    """
    Combine S2 bands + indices + S1 features per month.
    Output: (85, y, x) = 5 months x 17 bands
    """
    all_layers = []
    band_names = []

    for month in months:
        month_str = f"{month:02d}"

        # S2: strip metadata, compute indices, merge
        s2 = s2_composites[month]
        drop = [c for c in s2.coords if c not in {"band", "y", "x"}]
        s2 = s2.drop_vars(drop, errors="ignore")

        indices = compute_indices(s2)
        s2_full = xr.concat([s2, indices], dim="band")
        for b in s2_full["band"].values:
            band_names.append(f"{b}_{month_str}")
        all_layers.append(s2_full)

        # S1: dB + cross-ratio
        s1 = compute_s1_features(s1_composites[month])
        for b in s1["band"].values:
            band_names.append(f"{b}_{month_str}")
        all_layers.append(s1)

    feature_stack = xr.concat(all_layers, dim="band")
    feature_stack["band"] = band_names

    print(f"\nFeature stack: {feature_stack.shape}")
    print(f"  {len(band_names)} features = {len(months)} months x 17 bands")
    return feature_stack


# 12. SAMPLE TRAINING DATA

def sample_training_data(feature_stack, training_path, roi_crs):
    """
    Rasterize training polygons and extract ALL pixels within them.
    Returns X, y, and polygon_ids for polygon-level train/test splitting.
    """
    training = gpd.read_file(training_path)
    print(f"\nTraining data: {len(training)} polygons")
    print(f"  Classes: {training[CLASS_FIELD].value_counts().to_dict()}")

    # Reproject to raster CRS
    raster_crs = feature_stack.rio.crs
    if raster_crs is None:
        raster_crs = f"EPSG:{EPSG}"
    training = training.to_crs(raster_crs)

    # Build raster grid info from feature stack
    y_coords = feature_stack.coords["y"].values
    x_coords = feature_stack.coords["x"].values
    ny, nx = len(y_coords), len(x_coords)

    res_x = abs(x_coords[1] - x_coords[0])
    res_y = abs(y_coords[1] - y_coords[0])

    x_min = x_coords.min() - res_x / 2
    x_max = x_coords.max() + res_x / 2
    y_min = y_coords.min() - res_y / 2
    y_max = y_coords.max() + res_y / 2
    transform = from_bounds(x_min, y_min, x_max, y_max, nx, ny)

    # Burn class labels into raster
    class_shapes = [(geom, label) for geom, label in
                    zip(training.geometry, training[CLASS_FIELD])]
    label_raster = rasterize(
        class_shapes, out_shape=(ny, nx), transform=transform,
        fill=0, dtype=np.int32,
    )

    # Burn polygon IDs (1-indexed) into a separate raster
    poly_shapes = [(geom, i + 1) for i, geom in enumerate(training.geometry)]
    poly_id_raster = rasterize(
        poly_shapes, out_shape=(ny, nx), transform=transform,
        fill=0, dtype=np.int32,
    )

    # Report
    unique, counts = np.unique(label_raster[label_raster > 0], return_counts=True)
    print("  Labeled pixels per class:")
    for cls, cnt in zip(unique, counts):
        name = CLASS_NAMES[cls - 1] if cls <= len(CLASS_NAMES) else f"Class {cls}"
        print(f"    {name}: {cnt}")
    total_labeled = label_raster[label_raster > 0].size
    print(f"  Total labeled pixels: {total_labeled}")

    # Extract features at labeled pixels
    labeled_mask = label_raster > 0
    labeled_rows, labeled_cols = np.where(labeled_mask)

    print("Extracting pixel values at training polygons...")
    y_sel = xr.DataArray(y_coords[labeled_rows], dims="points")
    x_sel = xr.DataArray(x_coords[labeled_cols], dims="points")

    sampled = feature_stack.sel(x=x_sel, y=y_sel, method="nearest")
    sampled = sampled.compute()

    X = sampled.values.T
    y = label_raster[labeled_mask]
    polygon_ids = poly_id_raster[labeled_mask]

    # Drop NaN
    valid = ~np.any(np.isnan(X), axis=1)
    n_dropped = np.sum(~valid)
    if n_dropped > 0:
        print(f"  Dropped {n_dropped} samples with NaN values")
    X = X[valid]
    y = y[valid]
    polygon_ids = polygon_ids[valid]

    # Cap at MAX_SAMPLES
    if len(X) > MAX_SAMPLES:
        print(f"  Subsampling from {len(X)} to {MAX_SAMPLES} (stratified)...")
        idx = np.arange(len(X))
        _, idx_keep = train_test_split(
            idx, test_size=MAX_SAMPLES, random_state=RF_RANDOM_STATE, stratify=y
        )
        X = X[idx_keep]
        y = y[idx_keep]
        polygon_ids = polygon_ids[idx_keep]

    print(f"  Final: {len(X)} samples x {X.shape[1]} features")
    print(f"  Unique polygons: {len(np.unique(polygon_ids))}")
    return X, y, polygon_ids


# 13. CLASSIFY

def train_and_evaluate(X, y, polygon_ids):
    """
    Train RF with polygon-level train/test split.
    All pixels from a polygon go to either train or test, never both.
    This prevents spatial autocorrelation from inflating accuracy.
    """
    # Get unique polygons and their class (majority class per polygon)
    unique_polys = np.unique(polygon_ids)
    poly_classes = np.array([
        np.bincount(y[polygon_ids == pid]).argmax()
        for pid in unique_polys
    ])

    # Split polygons, not pixels
    train_polys, test_polys = train_test_split(
        unique_polys, test_size=TEST_SIZE, random_state=RF_RANDOM_STATE,
        stratify=poly_classes,
    )

    train_mask = np.isin(polygon_ids, train_polys)
    test_mask = np.isin(polygon_ids, test_polys)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"\nPolygon-level split:")
    print(f"  Train: {len(train_polys)} polygons ({len(X_train)} pixels)")
    print(f"  Test:  {len(test_polys)} polygons ({len(X_test)} pixels)")

    print(f"\nTraining RF ({RF_N_ESTIMATORS} trees)...")
    clf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        random_state=RF_RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\n" + "=" * 50)
    print("CLASSIFICATION RESULTS")
    print("=" * 50)
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    oa = np.trace(cm) / cm.sum()
    p_e = sum(
        cm.sum(axis=0)[i] * cm.sum(axis=1)[i] for i in range(len(CLASS_NAMES))
    ) / (cm.sum() ** 2)
    kappa = (oa - p_e) / (1 - p_e)

    print(f"\nOverall Accuracy: {oa:.4f}")
    print(f"Kappa: {kappa:.4f}")
    return clf


def predict_map(clf, feature_stack, output_path):
    """Run wall-to-wall prediction and save classified GeoTIFF."""
    print("\nPredicting full AOI...")
    print("(This triggers the full data download — may take 10-20 min)")

    t0 = time.time()
    stack_computed = feature_stack.compute()
    t1 = time.time()
    print(f"  Data loaded in {(t1 - t0) / 60:.1f} min")

    n_bands, ny, nx = stack_computed.shape
    flat = stack_computed.values.reshape(n_bands, -1).T

    valid = ~np.any(np.isnan(flat), axis=1)
    predictions = np.zeros(flat.shape[0], dtype=np.uint8)
    if valid.sum() > 0:
        predictions[valid] = clf.predict(flat[valid])

    classified = predictions.reshape(ny, nx)

    classified_da = xr.DataArray(
        classified[np.newaxis, :, :],
        dims=["band", "y", "x"],
        coords={"y": stack_computed.coords["y"], "x": stack_computed.coords["x"]},
    )
    classified_da = classified_da.rio.write_crs(
        stack_computed.rio.crs or f"EPSG:{EPSG}"
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    classified_da.rio.to_raster(output_path, driver="GTiff", dtype="uint8")
    print(f"  Saved: {output_path}")
    return classified


# 14. VISUALIZATION
def plot_rgb_composites(s2_composites, months):
    """Plot monthly RGB composites."""
    fig, axes = plt.subplots(1, len(months), figsize=(20, 4))
    for i, month in enumerate(months):
        rgb = s2_composites[month].sel(band=["B04", "B03", "B02"])
        rgb_np = rgb.compute().values
        rgb_np = np.clip(rgb_np, 0, 0.4) / 0.4
        rgb_np = np.moveaxis(rgb_np, 0, -1)
        axes[i].imshow(rgb_np)
        axes[i].set_title(MONTH_NAMES[i])
        axes[i].axis("off")
    plt.suptitle(f"Monthly RGB — Macdonald {YEAR}", fontsize=14)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"rgb_composites_{YEAR}.png")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_classification(classified, output_path):
    """Plot classified map."""
    cmap = ListedColormap(["#92a55b", "#d6ff70", "#cc9933", "#ffff99"])
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    im = ax.imshow(classified, cmap=cmap, vmin=1, vmax=4)
    cbar = plt.colorbar(im, ax=ax, ticks=[1, 2, 3, 4])
    cbar.ax.set_yticklabels(CLASS_NAMES)
    ax.set_title(f"Crop Classification — Macdonald {YEAR}")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


# MAIN
def main():
    print("=" * 60)
    print(f"Cloud-Native Crop Classification Pipeline")
    print(f"Study Area: Macdonald, Manitoba | Year: {YEAR}")
    print(f"Features: 17 bands x 5 months = 85")
    print("=" * 60)

    # 1. Load ROI
    roi, bbox = load_roi(ROI_PATH)

    # 2. Search STAC
    datetime_range = f"{YEAR}-05-01/{YEAR}-09-30"
    print(f"\nSearching STAC ({datetime_range})...")
    s2_items = search_stac("sentinel-2-l2a", bbox, datetime_range)
    s1_items = search_stac("sentinel-1-rtc", bbox, datetime_range)

    # 3. Load as lazy arrays
    print("\nLoading imagery (lazy)...")
    s2_stack, scl = load_s2(s2_items, bbox)
    s1_stack = load_s1(s1_items, bbox)

    # 4. Cloud mask S2
    print("\nApplying cloud mask...")
    s2_masked = apply_cloud_mask(s2_stack, scl)

    # 5. Speckle filter S1
    print("Applying speckle filter...")
    s1_filtered = apply_speckle_filter(s1_stack)

    # 6. Monthly composites
    print("\nComputing monthly composites...")
    s2_monthly = compute_monthly_composites(s2_masked, YEAR, MONTHS)
    s1_monthly = compute_monthly_composites(s1_filtered, YEAR, MONTHS)

    # 7. Temporal interpolation (S2 only)
    print("Applying temporal interpolation...")
    s2_filled = temporal_interpolation(s2_monthly, MONTHS)

    # 8. Build 85-feature stack
    features = build_feature_stack(s2_filled, s1_monthly, MONTHS)

    # 9. Download AAFC and apply crop mask
    print("\nApplying crop mask...")
    aafc_path = download_aafc(bbox, YEAR, OUTPUT_DIR)
    crop_mask = build_crop_mask(aafc_path, features)
    features = features.where(crop_mask)

    # 10. Sample training data and classify
    X, y, polygon_ids = sample_training_data(features, TRAINING_PATH, roi.crs)
    clf = train_and_evaluate(X, y, polygon_ids)

    # 11. Predict full map
    classified_path = os.path.join(OUTPUT_DIR, f"classified_macdonald_{YEAR}.tif")
    classified = predict_map(clf, features, classified_path)

    # 12. Visualize
    plot_rgb_composites(s2_filled, MONTHS)
    plot_classification(
        classified,
        os.path.join(OUTPUT_DIR, f"classified_macdonald_{YEAR}.png"),
    )

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
