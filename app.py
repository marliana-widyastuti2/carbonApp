import os
import tempfile
import zipfile
from pathlib import Path

import streamlit as st
import geopandas as gpd
import rasterio
from rasterio.mask import mask


st.set_page_config(page_title="CarbonSampling App", layout="centered")
st.title("Clip raster by shapefile (local raster)")

st.write("Pick a local raster from a folder, then upload a **zipped shapefile** to clip it.")

# --- Local raster selection ---
raster_dir = st.text_input("Local raster folder (absolute path)", value=str(Path.home()))
raster_dir_path = Path(raster_dir).expanduser()

tifs = []
if raster_dir_path.exists() and raster_dir_path.is_dir():
    tifs = sorted([p for p in raster_dir_path.glob("*.tif")] + [p for p in raster_dir_path.glob("*.tiff")])

if not tifs:
    st.warning("No .tif/.tiff found in that folder yet.")
    raster_path = None
else:
    raster_path = st.selectbox("Choose a raster", options=[str(p) for p in tifs])

# --- Vector upload ---
shp_zip = st.file_uploader("Shapefile (ZIP)", type=["zip"])

# --- Output options ---
out_dir = st.text_input("Output folder (absolute path)", value=str(raster_dir_path))
out_name = st.text_input("Output filename", value="clipped.tif")

col1, col2 = st.columns(2)
crop_to_geom = col1.checkbox("Crop to geometry bounds", value=True)
all_touched = col2.checkbox("All touched (more inclusive edges)", value=False)

run_btn = st.button("Clip")

def _extract_zip_to_dir(zip_bytes: bytes, out_dir: str) -> Path:
    zpath = Path(out_dir) / "shape.zip"
    zpath.write_bytes(zip_bytes)

    with zipfile.ZipFile(zpath, "r") as zf:
        zf.extractall(out_dir)

    shp_files = list(Path(out_dir).glob("**/*.shp"))
    if not shp_files:
        raise ValueError("No .shp found in the ZIP. Ensure the zip contains .shp/.shx/.dbf/.prj.")
    return shp_files[0]

def _clip_raster(raster_path: str, vector_path: str, out_path: str, crop: bool, all_touched: bool) -> None:
    gdf = gpd.read_file(vector_path)
    if gdf.empty:
        raise ValueError("Vector file has no features.")

    # dissolve to one geometry
    geom = gdf.geometry.unary_union

    with rasterio.open(raster_path) as src:
        if gdf.crs is None:
            raise ValueError("Shapefile CRS missing (.prj missing).")
        if src.crs is None:
            raise ValueError("Raster CRS missing.")
        if gdf.crs != src.crs:
            gdf2 = gdf.to_crs(src.crs)
            geom = gdf2.geometry.unary_union

        out_image, out_transform = mask(
            src,
            [geom],
            crop=crop,
            all_touched=all_touched,
            nodata=src.nodata,
            filled=True
        )

        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        if out_meta.get("nodata", None) is None:
            if "float" in str(out_meta["dtype"]).lower():
                out_meta["nodata"] = -9999.0
            else:
                out_meta["nodata"] = 0

        # ensure output folder exists
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(out_path, "w", **out_meta) as dst:
            dst.write(out_image)

if run_btn:
    if raster_path is None:
        st.error("No raster selected.")
    elif shp_zip is None:
        st.error("Please upload a zipped shapefile.")
    else:
        try:
            out_dir_path = Path(out_dir).expanduser()
            out_path = out_dir_path / out_name

            with tempfile.TemporaryDirectory() as tmpdir:
                shp_path = _extract_zip_to_dir(shp_zip.read(), tmpdir)
                _clip_raster(str(raster_path), str(shp_path), str(out_path),
                            crop=crop_to_geom, all_touched=all_touched)

            st.success(f"Done! Saved to:\n{out_path}")
        except Exception as e:
            st.exception(e)
