import rasterio as rio
import numpy as np
import pandas as pd
from rasterio.mask import mask

import stratify
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"


def extract_to_csv():
    mean_file = OUTPUT_DIR / "clipped_SOC_mean.tif"
    var_file  = OUTPUT_DIR / "clipped_SOC_var.tif"

    with rio.open(mean_file) as src_mean, rio.open(var_file) as src_var:
        mean = src_mean.read(1)
        var  = src_var.read(1)

        nodata = src_mean.nodata
        transform = src_mean.transform

        rows, cols = np.meshgrid(
            np.arange(mean.shape[0]),
            np.arange(mean.shape[1]),
            indexing="ij"
        )

        xs, ys = rio.transform.xy(transform, rows, cols)
        xs = np.array(xs).ravel()
        ys = np.array(ys).ravel()

        mean_flat = mean.ravel()
        var_flat  = var.ravel()

    # ---- mask nodata ----
    mask = np.ones_like(mean_flat, dtype=bool)
    if nodata is not None:
        mask &= mean_flat != nodata
        mask &= var_flat  != nodata

    mask &= ~np.isnan(mean_flat)
    mask &= ~np.isnan(var_flat)

    # ---- build dataframe ----
    df = pd.DataFrame({
        "Xg": xs[mask],
        "Yg": ys[mask],
        "SOC": mean_flat[mask],
        "s2_SOC": var_flat[mask],
    })

    # ---- save to csv ----
    out_csv = OUTPUT_DIR / "SOC_points.csv"
    df.to_csv(out_csv, index=False)
