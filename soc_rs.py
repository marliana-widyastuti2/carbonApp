import os, glob, warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.ops import unary_union
import rasterio as rio
from rasterio.warp import transform_bounds
from rasterio.mask import mask
from rasterio.merge import merge as rio_merge
from rasterio.io import MemoryFile
from pyproj import Transformer
from scipy.signal import savgol_filter
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
from cubist import Cubist
import spectral as spy

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

EXCEL_PATH  = r"/home/marliana/shared_folder/CarbonApp/data/Data collection -Carbon Project updated fomular02022016.xlsx" 
SHEET_NAME  = "Data"

COL_LAT     = "GPSCoreLocation Latitude"
COL_LON     = "GPSCoreLocation Longitude"
COL_SITE    = "Sampling site"
COL_FARM    = "Farm"
TARGET_COL  = "TOC_est"   # summed by site -> SOC

RASTER_DIR  = r"/home/marliana/shared_folder/CarbonApp/ssd/Nicolas_dataset/ACT NSW QLD LANDSAT 9 PANSHARPENED"
SHP_DIR     = r"/home/marliana/shared_folder/CarbonApp/shapefiles"

OUT_DIR     = r"/home/marliana/shared_folder/CarbonApp/OUT_SOC"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_CSV     = "training_sites_landsat7bands.csv"
OUT_TRAIN_ENRICHED_CSV = "training_sites_with_features_and_predictions.csv"
OUT_CV_TABLE_CSV       = "cubist_cv_results_21vars.csv"

PRODUCT_BANDS = 7
BANDS       = [f"B{i}" for i in range(1, 8)]
DERIV_BANDS = [f"{b}_d1" for b in BANDS]
CR_BANDS    = [f"CR{b}" for b in BANDS]

# Landsat band centers (must match B1..B7 order)
WAVELENGTHS = np.array([443, 482, 561, 655, 865, 1609, 2201], dtype=float)

# SG settings 
SG_WINDOW    = 3
SG_POLYORDER = 2
SG_DERIV     = 1

# CV grid
N_FOLDS           = 10
RANDOM_SEED       = 0
GRID_N_COMMITTEES = [1, 5, 10, 25]
GRID_NEIGHBORS    = [1, 3, 5, 7, 9]
GRID_N_RULES      = [10, 20, 30, 40, 50]

# Prediction nodata in outputs
PREDICTION_NODATA = -9999.0

# Training sampling strictness
REQUIRE_ALL_7_BANDS_VALID = True

EXPORT_ENRICHED_TRAINING_TABLE = True


# ============================================================
# BLOCK 2 — GENERIC HELPERS
# ============================================================
def list_rasters(raster_dir):
    exts = ("*.tif", "*.tiff")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(raster_dir, "**", ext), recursive=True))
    return [f for f in files if os.path.isfile(f)]

def is_7band_stack(path):
    try:
        with rio.open(path) as src:
            return src.count >= 7
    except Exception:
        return False

def raster_footprint_wgs84(raster_path):
    try:
        with rio.open(raster_path) as src:
            if src.crs is None:
                return None
            b = src.bounds
            return transform_bounds(src.crs, "EPSG:4326", b.left, b.bottom, b.right, b.top)
    except Exception:
        return None

def rasters_intersecting_farm(raster_paths, farm_geom_wgs84):
    farm_bbox_poly = box(*farm_geom_wgs84.bounds)
    keep = []
    for rp in raster_paths:
        tb = raster_footprint_wgs84(rp)
        if tb is None:
            continue
        if box(*tb).intersects(farm_bbox_poly):
            keep.append(rp)
    return keep

def _clean_vals7(vals, nodata_vals, nodata_scalar):
    v = np.array(vals[:PRODUCT_BANDS], dtype=float) if len(vals) >= PRODUCT_BANDS else np.full((PRODUCT_BANDS,), np.nan, dtype=float)

    # per-band nodata preferred
    if nodata_vals is not None and len(nodata_vals) >= PRODUCT_BANDS:
        for b in range(PRODUCT_BANDS):
            nd = nodata_vals[b]
            if nd is not None and np.isfinite(nd) and v[b] == float(nd):
                v[b] = np.nan
    elif nodata_scalar is not None and np.isfinite(nodata_scalar):
        v = np.where(v == float(nodata_scalar), np.nan, v)

    n_valid = int(np.isfinite(v).sum())
    return v, n_valid


# ============================================================
# BLOCK 3 — GEOMETRY
# ============================================================
def load_farm_geom_wgs84(shp_path):
    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        raise ValueError(f"Empty shapefile: {shp_path}")
    if gdf.crs is None:
        warnings.warn(f"{shp_path} has no CRS. Assuming EPSG:4326.")
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf[gdf.geometry.notnull() & (~gdf.geometry.is_empty)].copy()
    geom = unary_union(gdf.geometry.values)
    if not isinstance(geom, (Polygon, MultiPolygon)):
        raise ValueError(f"Farm geometry must be Polygon/MultiPolygon. Got: {type(geom)}")
    return geom

def _geom_wgs84_to_src(geom_wgs84, dst_crs):
    tfm = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)

    def transform_polygon(p: Polygon):
        xs, ys = p.exterior.coords.xy
        new_ext = [tfm.transform(x, y) for x, y in zip(xs, ys)]
        new_holes = []
        for ring in p.interiors:
            xs2, ys2 = ring.coords.xy
            new_holes.append([tfm.transform(x, y) for x, y in zip(xs2, ys2)])
        return Polygon(new_ext, new_holes)

    if isinstance(geom_wgs84, Polygon):
        return transform_polygon(geom_wgs84)
    return MultiPolygon([transform_polygon(p) for p in geom_wgs84.geoms])


# ============================================================
# BLOCK 4 — TRAINING CSV FROM EXCEL + LANDSAT SAMPLING
# ============================================================
def load_excel_points(excel_path, sheet_name):
    df = pd.read_excel(excel_path, sheet_name=sheet_name, engine="openpyxl")

    needed = [COL_LAT, COL_LON, COL_SITE, COL_FARM, TARGET_COL]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required Excel columns: {missing}")

    df = df.dropna(subset=[COL_LAT, COL_LON]).copy()
    df[COL_LAT] = pd.to_numeric(df[COL_LAT], errors="coerce")
    df[COL_LON] = pd.to_numeric(df[COL_LON], errors="coerce")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=[COL_LAT, COL_LON]).copy()

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[COL_LON], df[COL_LAT]),
        crs="EPSG:4326"
    )
    return gdf

def build_training_csv():
    print("Loading Excel points...")
    gdf_all = load_excel_points(EXCEL_PATH, SHEET_NAME)

    print("Summing target per sampling site (grouping depths)...")
    site_sum = (
        gdf_all.groupby(COL_SITE, as_index=False)[TARGET_COL]
              .sum(min_count=1)
              .rename(columns={TARGET_COL: "SOC"})
    )

    coords = (
        gdf_all.groupby(COL_SITE)
              .agg({COL_LAT: "first", COL_LON: "first", COL_FARM: "first"})
              .reset_index()
    )
    df_sites = coords.merge(site_sum, on=COL_SITE, how="left")

    print("Scanning raster folder for 7-band stacks...")
    rasters = [r for r in list_rasters(RASTER_DIR) if is_7band_stack(r)]
    if not rasters:
        raise RuntimeError("No >=7-band GeoTIFF found in RASTER_DIR.")

    raster_index = []
    for rp in rasters:
        tb = raster_footprint_wgs84(rp)
        if tb is None:
            continue
        minx, miny, maxx, maxy = tb
        raster_index.append({"raster": rp, "minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy})

    for b in range(1, PRODUCT_BANDS + 1):
        df_sites[f"B{b}"] = np.nan

    n_no_bbox = 0
    n_bbox_but_invalid = 0
    n_ok = 0

    print("Sampling rasters at site coordinates (TRY ALL bbox matches until valid)...")
    for i in tqdm(range(len(df_sites))):
        lon = float(df_sites.loc[i, COL_LON])
        lat = float(df_sites.loc[i, COL_LAT])

        candidates = [
            rb["raster"] for rb in raster_index
            if (rb["minx"] <= lon <= rb["maxx"]) and (rb["miny"] <= lat <= rb["maxy"])
        ]
        if not candidates:
            n_no_bbox += 1
            continue

        got = False
        for candidate in candidates:
            try:
                with rio.open(candidate) as src:
                    if src.crs is None or src.count < 7:
                        continue

                    tfm = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                    x, y = tfm.transform(lon, lat)

                    vals = next(src.sample([(x, y)]))
                    if getattr(vals, "mask", None) is not None:
                        vals = np.where(vals.mask, np.nan, vals.data)

                    vals7, n_valid = _clean_vals7(vals, list(src.nodatavals) if src.nodatavals is not None else None, src.nodata)

                    if REQUIRE_ALL_7_BANDS_VALID:
                        if n_valid != 7:
                            continue
                    else:
                        if n_valid == 0:
                            continue

                    for b in range(1, 8):
                        df_sites.loc[i, f"B{b}"] = float(vals7[b - 1])

                    got = True
                    n_ok += 1
                    break
            except Exception:
                continue

        if not got:
            n_bbox_but_invalid += 1

    out_csv = os.path.join(OUT_DIR, OUT_CSV)
    df_sites.to_csv(out_csv, index=False)

    print(f"✅ Training CSV saved: {out_csv}")
    print(f"   sites total            : {len(df_sites)}")
    print(f"   no bbox match          : {n_no_bbox}")
    print(f"   bbox but invalid       : {n_bbox_but_invalid}")
    print(f"   successfully sampled   : {n_ok}")

    return out_csv


# ============================================================
# BLOCK 5 — FEATURES + CUBIST TRAINING
# ============================================================
def concordance_cc(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 2:
        return np.nan
    x = y_true[m]; y = y_pred[m]
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    if vx <= 0 or vy <= 0:
        return np.nan
    sx, sy = np.sqrt(vx), np.sqrt(vy)
    rho = np.corrcoef(x, y)[0, 1]
    return (2 * rho * sx * sy) / (vx + vy + (mx - my) ** 2)

def sg_first_derivative_per_row(X_2d):
    return np.apply_along_axis(
        lambda row: savgol_filter(
            row, window_length=SG_WINDOW, polyorder=SG_POLYORDER, deriv=SG_DERIV, mode="nearest"
        ),
        axis=1,
        arr=np.asarray(X_2d, dtype=float)
    )

def continuum_removed_spy(X_2d, wavelengths):
    X = np.asarray(X_2d, dtype=float)
    w = np.asarray(wavelengths, dtype=float).reshape(-1)
    if X.ndim != 2:
        raise ValueError("X_2d must be 2D (n_rows, n_bands)")
    if X.shape[1] != w.size:
        raise ValueError("wavelengths length must match n_bands")
    cr = spy.remove_continuum(X, w)
    return np.asarray(cr, dtype=float)

def build_features_21_df(df):
    raw = df[BANDS].astype(float).copy()
    raw_np = raw.to_numpy()
    deriv_np = sg_first_derivative_per_row(raw_np)
    cr_np = continuum_removed_spy(raw_np, WAVELENGTHS)

    X = pd.concat(
        [
            raw.reset_index(drop=True),
            pd.DataFrame(deriv_np, columns=DERIV_BANDS),
            pd.DataFrame(cr_np, columns=CR_BANDS),
        ],
        axis=1
    )
    return X

def cv_score_global_oof(X, y, n_committees, neighbors, n_rules, n_folds=5, seed=42):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    y = np.asarray(y, dtype=float)
    y_oof = np.full_like(y, np.nan, dtype=float)

    for tr_idx, va_idx in kf.split(X):
        model = Cubist(n_committees=n_committees, neighbors=neighbors, n_rules=n_rules)
        model.fit(X.iloc[tr_idx, :], y[tr_idx])
        y_oof[va_idx] = model.predict(X.iloc[va_idx, :])

    m = np.isfinite(y) & np.isfinite(y_oof)
    if m.sum() < 2:
        return np.nan, np.nan, np.nan, np.nan

    yt = y[m]; yp = y_oof[m]
    rmse = root_mean_squared_error(yt, yp)
    iqr = np.percentile(yt, 75) - np.percentile(yt, 25)
    rpiq = (iqr / rmse) if rmse > 0 else np.nan
    corr = np.corrcoef(yt, yp)[0, 1]
    r2 = corr**2 if np.isfinite(corr) else np.nan
    ccc = concordance_cc(yt, yp)
    return float(rmse), float(rpiq), float(r2), float(ccc)

def tune_hyperparams(X, y, n_folds, seed):
    records = []
    best_key = None
    best_params = None

    for nc in GRID_N_COMMITTEES:
        for nb in GRID_NEIGHBORS:
            for nr in GRID_N_RULES:
                rmse, rpiq, r2, ccc = cv_score_global_oof(X, y, nc, nb, nr, n_folds=n_folds, seed=seed)

                records.append({
                    "n_committees": nc,
                    "neighbors": nb,
                    "n_rules": nr,
                    "cv_rmse": rmse,
                    "cv_rpiq": rpiq,
                    "cv_r2": r2,
                    "cv_ccc": ccc
                })

                key = (
                    rmse if np.isfinite(rmse) else np.inf,
                    -(rpiq if np.isfinite(rpiq) else -np.inf),
                    -(r2 if np.isfinite(r2) else -np.inf)
                )
                if (best_key is None) or (key < best_key):
                    best_key = key
                    best_params = {"n_committees": nc, "neighbors": nb, "n_rules": nr}

    results_df = pd.DataFrame(records).sort_values(["cv_rmse", "cv_rpiq"], ascending=[True, False])
    return best_params, results_df

def train_final_model(training_csv):
    print("\nLoading training CSV for modeling...")
    df = pd.read_csv(training_csv)

    cols_needed = ["SOC"] + BANDS
    df_train = df.dropna(subset=cols_needed).copy()
    if df_train.empty:
        raise RuntimeError("No valid training rows after dropping NaNs in SOC and bands. Check sampling.")

    y = df_train["SOC"].astype(float).values
    X = build_features_21_df(df_train)

    print("\nTuning Cubist hyperparameters with K-fold CV (GLOBAL OOF scoring)...")
    best_params, cv_table = tune_hyperparams(X, y, n_folds=N_FOLDS, seed=RANDOM_SEED)
    print("Best params:", best_params)

    # GLOBAL OOF for best config
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    y_oof = np.full_like(y, np.nan, dtype=float)

    for tr_idx, va_idx in kf.split(X):
        m = Cubist(**best_params)
        m.fit(X.iloc[tr_idx, :], y[tr_idx])
        y_oof[va_idx] = m.predict(X.iloc[va_idx, :])

    msk = np.isfinite(y) & np.isfinite(y_oof)
    yt = y[msk]; yp = y_oof[msk]

    rmse = root_mean_squared_error(yt, yp)
    mae  = mean_absolute_error(yt, yp)
    corr = np.corrcoef(yt, yp)[0, 1]
    r2   = corr**2 if np.isfinite(corr) else 0.0
    iqr  = np.percentile(yt, 75) - np.percentile(yt, 25)
    rpiq = (iqr / rmse) if rmse > 0 else np.nan
    ccc  = concordance_cc(yt, yp)

    print("\n=== OOF CV Metrics (GLOBAL) ===")
    print(f"Train rows   : {len(df_train)}")
    print(f"R² (Pearson²): {r2:.4f}")
    print(f"RMSE         : {rmse:.4f}")
    print(f"MAE          : {mae:.4f}")
    print(f"RPIQ         : {rpiq:.4f}")
    print(f"CCC          : {ccc:.4f}")

    cv_table.to_csv(os.path.join(OUT_DIR, OUT_CV_TABLE_CSV), index=False)

    final_model = Cubist(**best_params)
    final_model.fit(X, y)

    if EXPORT_ENRICHED_TRAINING_TABLE:
        df_out = df_train[[COL_SITE, COL_LAT, COL_LON, COL_FARM, "SOC"] + BANDS].copy().reset_index(drop=True)

        raw_np = df_out[BANDS].to_numpy(dtype=float)
        df_out[DERIV_BANDS] = pd.DataFrame(sg_first_derivative_per_row(raw_np), columns=DERIV_BANDS)
        df_out[CR_BANDS]    = pd.DataFrame(continuum_removed_spy(raw_np, WAVELENGTHS), columns=CR_BANDS)

        df_out["SOC_pred_oof"] = y_oof
        df_out["SOC_pred_fit"] = final_model.predict(build_features_21_df(df_out))

        df_out.to_csv(os.path.join(OUT_DIR, OUT_TRAIN_ENRICHED_CSV), index=False)

    return final_model, best_params, rmse


# ============================================================
# BLOCK 6 — MOSAIC-THEN-PREDICT (NO IMPUTATION)
# ============================================================
def _features_21_from_pixels_no_impute(pix_7):
    """
    Build 21 features WITHOUT imputing missing bands.
    Valid pixel rule:
      - valid if ALL 7 bands are finite
      - if ANY band is NaN => invalid => output stays NODATA
    """
    X = np.asarray(pix_7, dtype=float)
    valid = np.all(np.isfinite(X), axis=1)
    if valid.sum() == 0:
        return None, valid

    Xv = X[valid]
    deriv = sg_first_derivative_per_row(Xv)
    cr    = continuum_removed_spy(Xv, WAVELENGTHS)
    feats = np.concatenate([Xv, deriv, cr], axis=1)
    return feats, valid

def clip_raster_to_farm_as_memdataset(raster_path, farm_geom_wgs84):
    """
    Clip a 7-band raster to farm polygon and return a rasterio dataset in memory (MemoryFile).
    Using filled=False => masked array; we keep nodata via mask later.
    """
    src = rio.open(raster_path)
    if src.crs is None:
        src.close()
        return None

    geom_src = _geom_wgs84_to_src(farm_geom_wgs84, src.crs)

    # masked array output
    out_img, out_transform = mask(
        src,
        [geom_src.__geo_interface__],
        crop=True,
        filled=False
    )

    # Ensure we have 7 bands
    if out_img.shape[0] < 7:
        src.close()
        return None

    out_img = out_img[:7, :, :]  # keep first 7

    # Build a small in-memory GTiff dataset for merging
    meta = src.meta.copy()
    meta.update({
        "driver": "GTiff",
        "count": 7,
        "height": out_img.shape[1],
        "width": out_img.shape[2],
        "transform": out_transform,
        # keep dtype as float32 for later conversion to NaN reliably
        "dtype": "float32",
        "nodata": src.nodata if (src.nodata is not None and np.isfinite(src.nodata)) else None
    })

    src.close()

    memfile = MemoryFile()
    ds = memfile.open(**meta)
    # Write with mask applied: masked -> nodata if nodata exists, else write raw and later use mask
    arr = out_img.filled(meta["nodata"]) if meta["nodata"] is not None else out_img.filled(np.nan)
    ds.write(arr.astype("float32"))
    return (memfile, ds)  # caller must close both

def mosaic_clipped_datasets(clipped_datasets):
    """
    Merge multiple clipped datasets into one mosaic array (7, H, W) and transform.
    Assumes all share same CRS (they should, if your tiles are consistent).
    """
    datasets = [ds for (_, ds) in clipped_datasets]
    mosaic, transform = rio_merge(datasets)  # mosaic shape: (7, H, W)
    return mosaic, transform, datasets[0].crs

def predict_soc_from_mosaic(model, mosaic_7, mosaic_transform, mosaic_crs, out_tif):
    """
    Predict SOC on a 7-band mosaic. NO IMPUTATION:
      - any pixel with NaN in any band => output = PREDICTION_NODATA
    """
    block = mosaic_7[:7, :, :].astype("float32")

    # Convert nodata to NaN:
    # Many of your intermediate clips may carry nodata. We treat ANY non-finite as invalid.
    # Also convert PREDICTION_NODATA if it appears (rare here).
    block = np.where(block == PREDICTION_NODATA, np.nan, block)

    bands, h, w = block.shape
    flat = block.reshape(bands, -1).T  # (Npix, 7)

    feats, valid = _features_21_from_pixels_no_impute(flat)

    pred_flat = np.full((flat.shape[0],), PREDICTION_NODATA, dtype="float32")
    if feats is not None and valid.sum() > 0:
        pred = model.predict(feats)
        pred_flat[valid] = np.asarray(pred, dtype="float32")

    pred_block = pred_flat.reshape(h, w).astype("float32")

    meta = {
        "driver": "GTiff",
        "count": 1,
        "dtype": "float32",
        "nodata": PREDICTION_NODATA,
        "height": h,
        "width": w,
        "transform": mosaic_transform,
        "crs": mosaic_crs
    }

    os.makedirs(os.path.dirname(out_tif), exist_ok=True)
    with rio.open(out_tif, "w", **meta) as dst:
        dst.write(pred_block, 1)


# ============================================================
# BLOCK 7 — FARM WORKFLOW (MOSAIC THEN CLIP/PREDICT)
# ============================================================
def run_farms_predict_mosaic(farms):
    """
    End-to-end workflow:
    1) Build training CSV from Excel + raster sampling
    2) Tune + train final Cubist model
    3) For each farm:
         - find intersecting rasters
         - clip each to farm
         - mosaic clipped pieces
         - predict SOC ONCE
         - write ONE output raster per farm
    """
    training_csv = build_training_csv()
    model, best_params, rmse = train_final_model(training_csv)
    print("\nFinal model params:", best_params)
    print("OOF RMSE:", rmse)

    all_rasters = [r for r in list_rasters(RASTER_DIR) if is_7band_stack(r)]
    if not all_rasters:
        raise RuntimeError("No 7-band stacks found in RASTER_DIR.")

    results = {}

    for farm in farms:
        print("\n====================================================")
        print(f"FARM: {farm}")
        print("====================================================")

        shp_path = os.path.join(SHP_DIR, f"{farm}.shp")
        if not os.path.exists(shp_path):
            raise RuntimeError(f"Missing farm shapefile: {shp_path}")

        farm_geom_wgs84 = load_farm_geom_wgs84(shp_path)
        farm_dir = os.path.join(OUT_DIR, str(farm))
        os.makedirs(farm_dir, exist_ok=True)

        rasters = rasters_intersecting_farm(all_rasters, farm_geom_wgs84)
        if not rasters:
            print(f"⚠️ No Landsat rasters intersect farm {farm}. Skipping.")
            results[str(farm)] = {"pred_raster": None, "shp": shp_path, "n_tiles": 0}
            continue

        print(f"Tiles intersecting farm bbox: {len(rasters)}")
        clipped = []
        try:
            # 1) clip each raster to farm (small pieces)
            for rp in rasters:
                out = clip_raster_to_farm_as_memdataset(rp, farm_geom_wgs84)
                if out is not None:
                    clipped.append(out)

            if not clipped:
                print(f"⚠️ No clipped pieces produced for {farm} (CRS missing or <7 bands).")
                results[str(farm)] = {"pred_raster": None, "shp": shp_path, "n_tiles": len(rasters)}
                continue

            # 2) mosaic clipped pieces
            mosaic7, mosaic_transform, mosaic_crs = mosaic_clipped_datasets(clipped)

            # 3) predict once on mosaic and write one output
            out_pred = os.path.join(farm_dir, f"{farm}_SOC_PRED_MOSAIC_CLIPPED.tif")
            print(f"[{farm}] Predicting mosaic -> {os.path.basename(out_pred)}")
            predict_soc_from_mosaic(model, mosaic7, mosaic_transform, mosaic_crs, out_pred)

            print(f"✅ [{farm}] Wrote mosaic prediction raster: {out_pred}")
            results[str(farm)] = {"pred_raster": out_pred, "shp": shp_path, "n_tiles": len(rasters), "n_clipped": len(clipped)}

        finally:
            # close memory datasets
            for memfile, ds in clipped:
                try:
                    ds.close()
                except Exception:
                    pass
                try:
                    memfile.close()
                except Exception:
                    pass

    return results
