import streamlit as st
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.io import MemoryFile
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
from io import BytesIO
import re

st.set_page_config(page_title="SCS Curve Number & Retention App", layout="wide")

st.title("SCS Curve Number (TR-55) & Retention Map Generator")

st.write(
    """
    This app:
    1. Reads **OpenLandMap soil texture** and **ESA CCI LULC** rasters,
    2. Converts soil texture → Hydrologic Soil Group (HSG A–D),
    3. Converts CCI LULC → land-use categories,
    4. Computes **Curve Number (CN)** using TR-55 (AMC-II),
    5. Computes **Potential Maximum Retention** S (mm, SI units),
    6. Names the maps using the **LULC filename** (e.g., LULC 1999 → CN 1999, Retention 1999).
    """
)

# -------------------------------------------------------
# 1. Upload inputs
# -------------------------------------------------------
soil_file = st.file_uploader(
    "Upload SOIL raster (GeoTIFF, OpenLandMap USDA texture classes 1–12)",
    type=["tif", "tiff"],
)
lulc_file = st.file_uploader(
    "Upload LULC raster (GeoTIFF, ESA CCI Land Cover, e.g., LULC_1999.tif)",
    type=["tif", "tiff"],
)

# -------------------------------------------------------
# 2. Mappings
# -------------------------------------------------------

# OpenLandMap soil texture → Hydrologic Soil Group (HSG)
soil_to_hsg = {
    12: "A",  # Sand
    11: "A",  # Loamy sand
    9:  "A",  # Sandy loam

    7:  "B",  # Loam
    8:  "B",  # Silt loam
    10: "B",  # Silt

    6:  "C",  # Sandy clay loam
    4:  "C",  # Clay loam
    5:  "C",  # Silty clay loam

    1:  "D",  # Clay
    2:  "D",  # Silty clay
    3:  "D",  # Sandy clay
}
hsg_letter_to_id = {"A": 1, "B": 2, "C": 3, "D": 4}

# ESA CCI LULC → meaningful categories
cci_to_category = {
    0:  None,             # No data
    10: "cropland",
    11: "cropland",
    12: "cropland",
    20: "cropland",
    30: "mosaic_cropland",
    40: "mosaic_cropland",

    50: "forest",
    60: "forest",
    61: "forest",
    62: "forest",
    70: "forest",
    71: "forest",
    72: "forest",
    80: "forest",
    81: "forest",
    82: "forest",
    90: "forest",
    100:"forest",

    110:"grassland",
    120:"shrubland",
    121:"shrubland",
    122:"shrubland",
    130:"grassland",

    140:"barren",
    150:"barren",
    151:"barren",
    152:"barren",
    153:"barren",
    200:"barren",
    201:"barren",
    202:"barren",
    220:"barren",

    160:"water",
    170:"water",
    180:"water",
    210:"water",

    190:"urban",
}

# TR-55 CN table (AMC-II) – retention in SI units (mm) via formula
cn_table = {
    "cropland":        {"A": 64, "B": 75, "C": 82, "D": 85},  # Row crops (good)
    "mosaic_cropland": {"A": 60, "B": 72, "C": 80, "D": 84},  # Small grain (good)
    "grassland":       {"A": 39, "B": 61, "C": 74, "D": 80},  # Pasture/grassland good
    "shrubland":       {"A": 35, "B": 56, "C": 70, "D": 77},  # Brush fair
    "forest":          {"A": 30, "B": 55, "C": 70, "D": 77},  # Woods good
    "barren":          {"A": 72, "B": 82, "C": 87, "D": 89},  # Dirt / bare
    "urban":           {"A": 98, "B": 98, "C": 98, "D": 98},  # Fully paved
    "water":           {"A":100, "B":100, "C":100, "D":100},  # Open water
}

# -------------------------------------------------------
# 3. Helper functions
# -------------------------------------------------------
def save_to_temp(uploaded_file, suffix=".tif"):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()
    return tmp.name

def reproject_soil_to_lulc(soil_path, lulc_path):
    """Reproject soil raster to the grid (extent/resolution/CRS) of the LULC raster."""
    with rasterio.open(soil_path) as soil_src, rasterio.open(lulc_path) as lulc_src:
        soil = soil_src.read(1)
        soil_nodata = soil_src.nodata if soil_src.nodata is not None else 0

        soil_reproj = np.full((lulc_src.height, lulc_src.width),
                              soil_nodata, dtype=soil.dtype)

        reproject(
            source=soil,
            destination=soil_reproj,
            src_transform=soil_src.transform,
            src_crs=soil_src.crs,
            dst_transform=lulc_src.transform,
            dst_crs=lulc_src.crs,
            resampling=Resampling.nearest
        )

        return soil_reproj, soil_nodata, lulc_src.profile

def soil_to_hsg_id(soil_arr, soil_nodata):
    hsg_id = np.full_like(soil_arr, -1, dtype=np.int16)
    valid = soil_arr != soil_nodata
    for code, letter in soil_to_hsg.items():
        mask = (soil_arr == code) & valid
        hsg_id[mask] = hsg_letter_to_id[letter]
    return hsg_id

def classify_lulc_categories(lulc_arr):
    """Map LULC codes to category IDs and return array + mapping."""
    cat_arr = np.full(lulc_arr.shape, -1, dtype=np.int16)
    codes = np.unique(lulc_arr)
    cats = []
    for code in codes:
        cat = cci_to_category.get(int(code))
        if cat is not None:
            cats.append(cat)
    cats = sorted(set(cats))
    cat_to_id = {c: i for i, c in enumerate(cats)}
    for code in codes:
        cat = cci_to_category.get(int(code))
        if cat is None:
            continue
        mask = (lulc_arr == code)
        cat_arr[mask] = cat_to_id[cat]
    return cat_arr, cat_to_id

def compute_cn_and_retention(soil_arr, soil_nodata, lulc_arr, profile):
    """Compute HSG, CN and S (mm) from arrays."""
    hsg_id = soil_to_hsg_id(soil_arr, soil_nodata)
    cn_nodata = -9999.0
    cn = np.full_like(lulc_arr, cn_nodata, dtype=np.float32)

    id_to_letter = {v: k for k, v in hsg_letter_to_id.items()}
    unl = np.unique(lulc_arr)
    unknown_codes = []

    for lc_val in unl:
        cat = cci_to_category.get(int(lc_val))
        if cat is None:
            unknown_codes.append(int(lc_val))
            continue
        lc_mask = (lulc_arr == lc_val)
        for hid, hletter in id_to_letter.items():
            sg_mask = (hsg_id == hid)
            mask = lc_mask & sg_mask
            if not np.any(mask):
                continue
            cn_val = cn_table[cat][hletter]
            cn[mask] = cn_val

    cn_profile = profile.copy()
    cn_profile.update(dtype="float32", nodata=cn_nodata)

    # Retention S (mm), SI: S = (25400 / CN) - 254
    S_nodata = -9999.0
    S = np.full_like(cn, S_nodata, dtype=np.float32)
    valid = (cn != cn_nodata) & (cn > 0)
    S[valid] = (25400.0 / cn[valid]) - 254.0  # mm

    S_profile = profile.copy()
    S_profile.update(dtype="float32", nodata=S_nodata)

    return hsg_id, cn, cn_profile, S, S_profile, sorted(set(unknown_codes))

def array_to_geotiff_bytes(arr, profile):
    with MemoryFile() as memfile:
        with memfile.open(**profile) as ds:
            ds.write(arr, 1)
        data = memfile.read()
    return BytesIO(data)

def plot_raster(arr, title, cbar_label=None, vmin=None, vmax=None,
                ticks=None, ticklabels=None):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(arr, interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax)
    if cbar_label:
        cbar.set_label(cbar_label)
    if ticks is not None and ticklabels is not None:
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticklabels)
    fig.tight_layout()
    return fig

def infer_year_from_filename(filename: str) -> str:
    """Extract a 4-digit year from filename, else return generic label."""
    match = re.search(r"(19|20)\d{2}", filename)
    if match:
        return match.group(0)
    else:
        return "LULC"

# -------------------------------------------------------
# 4. Main workflow
# -------------------------------------------------------
if soil_file and lulc_file:
    if st.button("Generate CN & Retention Maps"):
        with st.spinner("Processing..."):

            # Infer "year" from LULC filename
            lulc_name = lulc_file.name
            year_label = infer_year_from_filename(lulc_name)

            # Save inputs to temp
            soil_path_tmp = save_to_temp(soil_file)
            lulc_path_tmp = save_to_temp(lulc_file)

            # Reproject soil to LULC grid
            soil_reproj, soil_nodata, lulc_profile = reproject_soil_to_lulc(
                soil_path_tmp, lulc_path_tmp
            )

            # Read full LULC
            with rasterio.open(lulc_path_tmp) as lulc_src:
                lulc_full = lulc_src.read(1)
                profile = lulc_src.profile

            # Compute HSG, CN, S
            hsg_id, cn, cn_profile, S, S_profile, unknown_codes = compute_cn_and_retention(
                soil_reproj, soil_nodata, lulc_full, profile
            )

        if unknown_codes:
            st.warning(
                f"These CCI LULC codes were not mapped to categories and remain NoData in CN: {unknown_codes}"
            )

        st.subheader(f"Maps for {year_label}")

        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)

        # LULC categories map (names as legend)
        lulc_cat_arr, cat_to_id = classify_lulc_categories(lulc_full)
        cat_ids = list(cat_to_id.values())
        cat_names = list(cat_to_id.keys())

        with col1:
            st.markdown(f"**LULC Map ({year_label}) – Category Names**")
            fig_lulc = plot_raster(
                lulc_cat_arr,
                f"LULC {year_label}",
                cbar_label="Category",
                ticks=cat_ids,
                ticklabels=cat_names,
            )
            st.pyplot(fig_lulc)

        # HSG map
        with col2:
            st.markdown("**Hydrologic Soil Group (HSG)**")
            hsg_ticks = [1, 2, 3, 4]
            hsg_labels = ["A", "B", "C", "D"]
            fig_hsg = plot_raster(
                hsg_id,
                "HSG",
                cbar_label="HSG",
                ticks=hsg_ticks,
                ticklabels=hsg_labels,
            )
            st.pyplot(fig_hsg)

        # CN map: legend fixed 60–100
        with col3:
            st.markdown(f"**Curve Number (CN) – {year_label}**")
            fig_cn = plot_raster(
                cn,
                f"Curve Number {year_label}",
                cbar_label="CN",
                vmin=60,
                vmax=100,
            )
            st.pyplot(fig_cn)

        # Retention map (mm)
        with col4:
            st.markdown(f"**Potential Maximum Retention S (mm) – {year_label}**")
            fig_S = plot_raster(
                S,
                f"Retention S (mm) {year_label}",
                cbar_label="S (mm)",
            )
            st.pyplot(fig_S)

        # Downloads
        st.subheader("Download Outputs (GeoTIFF)")

        cn_bytes = array_to_geotiff_bytes(cn, cn_profile)
        st.download_button(
            label=f"Download CN GeoTIFF ({year_label})",
            data=cn_bytes,
            file_name=f"CurveNumber_{year_label}.tif",
            mime="image/tiff",
        )

        S_bytes = array_to_geotiff_bytes(S, S_profile)
        st.download_button(
            label=f"Download Retention S (mm) GeoTIFF ({year_label})",
            data=S_bytes,
            file_name=f"Retention_S_mm_{year_label}.tif",
            mime="image/tiff",
        )

        # Clean temp files
        try:
            os.remove(soil_path_tmp)
            os.remove(lulc_path_tmp)
        except Exception:
            pass

else:
    st.info("Please upload both a SOIL raster and a LULC raster to enable processing.")
