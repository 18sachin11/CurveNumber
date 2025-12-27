import streamlit as st
import rasterio
from rasterio.warp import reproject, Resampling
import rasterio.io
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
from io import BytesIO

st.set_page_config(page_title="Curve Number (SCS-CN) Map Generator", layout="wide")

st.title("Curve Number (SCS-CN) Map Generator")

st.write(
    """
    **Inputs required**
    - SOIL raster: OpenLandMap soil texture (USDA textural classes, codes 1–12).
    - LULC raster: ESA CCI Land Cover (values 0–220).

    **Outputs**
    - Hydrologic Soil Group (HSG) map (A/B/C/D).
    - LULC map (CCI codes).
    - Curve Number (CN) map and downloadable CN GeoTIFF.
    """
)

# -------------------------------------------------------------------
# 1. Upload widgets
# -------------------------------------------------------------------
soil_file = st.file_uploader(
    "Upload SOIL raster (GeoTIFF, OpenLandMap soil texture)", type=["tif", "tiff"]
)
lulc_file = st.file_uploader(
    "Upload LULC raster (GeoTIFF, ESA CCI Land Cover)", type=["tif", "tiff"]
)

# -------------------------------------------------------------------
# 2. Mappings
# -------------------------------------------------------------------
# OpenLandMap soil texture -> Hydrologic Soil Group (HSG)
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

# ESA CCI LULC → broader SCS land-use categories
# Based on your legend PDF.
cci_to_category = {
    0:  None,              # No data
    10: "cropland",        # Cropland, rainfed
    11: "cropland",        # Herbaceous cover (cropland)
    12: "cropland",        # Tree or shrub cover (cropland)
    20: "cropland",        # Cropland, irrigated or post-flooding

    30: "mosaic_cropland", # Mosaic cropland / natural veg
    40: "mosaic_cropland", # Mosaic nat. veg / cropland

    # Tree cover classes (broadleaf & needleleaf, all decid/evergreen)
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

    # Mosaics dominated by woody cover or herbaceous cover
    100: "forest",         # Mosaic tree & shrub / herbaceous
    110: "grassland",      # Mosaic herbaceous / tree & shrub

    # Shrub & grass
    120: "shrubland",
    121: "shrubland",
    122: "shrubland",
    130: "grassland",

    # Lichens/mosses, sparse veg, bare, snow/ice → treat as "barren"
    140: "barren",
    150: "barren",
    151: "barren",
    152: "barren",
    153: "barren",
    200: "barren",
    201: "barren",
    202: "barren",
    220: "barren",         # Permanent snow & ice

    # Flooded / wetlands and water → treat as "water" (high CN)
    160: "water",
    170: "water",
    180: "water",
    210: "water",

    # Urban
    190: "urban",
}

# CN look-up table (AMC-II). Adjust to your preferred values if needed.
cn_table = {
    "cropland":        {"A": 67, "B": 78, "C": 85, "D": 89},
    "mosaic_cropland": {"A": 69, "B": 80, "C": 86, "D": 90},
    "grassland":       {"A": 39, "B": 61, "C": 74, "D": 80},
    "shrubland":       {"A": 35, "B": 56, "C": 70, "D": 77},
    "forest":          {"A": 30, "B": 55, "C": 70, "D": 77},
    "barren":          {"A": 77, "B": 86, "C": 91, "D": 94},
    "urban":           {"A": 77, "B": 85, "C": 90, "D": 92},
    "water":           {"A": 100, "B": 100, "C": 100, "D": 100},
}

# -------------------------------------------------------------------
# 3. Helper functions
# -------------------------------------------------------------------
def save_to_temp(uploaded_file, suffix=".tif"):
    """Save uploaded file to a temporary GeoTIFF and return its path."""
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
            resampling=Resampling.nearest,
        )

        return soil_reproj, soil_nodata, lulc_src.profile


def soil_to_hsg_id(soil_reproj, soil_nodata):
    """Convert soil texture code raster → HSG ID raster (1=A,2=B,3=C,4=D)."""
    hsg_id = np.full_like(soil_reproj, -1, dtype=np.int16)
    valid = soil_reproj != soil_nodata
    for code, letter in soil_to_hsg.items():
        mask = (soil_reproj == code) & valid
        hsg_id[mask] = hsg_letter_to_id[letter]
    return hsg_id


def compute_cn(soil_reproj, soil_nodata, lulc_path):
    """Compute CN, and return CN array, LULC array, HSG array, profile and unknown LULC codes."""
    with rasterio.open(lulc_path) as lulc_src:
        lulc = lulc_src.read(1)
        lulc_nodata = lulc_src.nodata if lulc_src.nodata is not None else 0
        profile = lulc_src.profile

    hsg_id = soil_to_hsg_id(soil_reproj, soil_nodata)
    cn_nodata = -9999.0
    cn = np.full_like(lulc, cn_nodata, dtype=np.float32)

    id_to_letter = {v: k for k, v in hsg_letter_to_id.items()}

    unique_lulc = np.unique(lulc[lulc != lulc_nodata])
    unknown_lulc_codes = []

    for lc in unique_lulc:
        category = cci_to_category.get(int(lc))
        if category is None:
            unknown_lulc_codes.append(int(lc))
            continue

        lc_mask = lulc == lc

        for hsg_code, hsg_letter in id_to_letter.items():
            sg_mask = hsg_id == hsg_code
            mask = lc_mask & sg_mask
            if not np.any(mask):
                continue
            cn_val = cn_table[category][hsg_letter]
            cn[mask] = cn_val

    profile.update(dtype="float32", nodata=cn_nodata)
    return cn, lulc, hsg_id, profile, sorted(set(unknown_lulc_codes))


def cn_to_geotiff_bytes(cn, profile):
    """Write CN raster to an in-memory GeoTIFF and return as BytesIO."""
    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(**profile) as ds:
            ds.write(cn, 1)
        data = memfile.read()
    return BytesIO(data)


def plot_raster(arr, title, cbar_label=None):
    """Return a matplotlib Figure for a 2D raster."""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(arr, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax)
    if cbar_label:
        cbar.set_label(cbar_label)
    fig.tight_layout()
    return fig

# -------------------------------------------------------------------
# 4. Main workflow
# -------------------------------------------------------------------
if soil_file and lulc_file:
    if st.button("Generate Curve Number Map"):
        with st.spinner("Processing rasters and generating CN map..."):

            soil_path_tmp = save_to_temp(soil_file)
            lulc_path_tmp = save_to_temp(lulc_file)

            soil_reproj, soil_nodata, ref_profile = reproject_soil_to_lulc(
                soil_path_tmp, lulc_path_tmp
            )

            cn, lulc, hsg_id, profile, unknown_codes = compute_cn(
                soil_reproj, soil_nodata, lulc_path_tmp
            )

        if unknown_codes:
            st.warning(
                f"These CCI LULC codes were not mapped and remain NoData in CN: {unknown_codes}"
            )

        # ----------------- Show maps -----------------
        st.subheader("Maps")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Hydrologic Soil Group (HSG)**  \n1=A, 2=B, 3=C, 4=D")
            fig_hsg = plot_raster(hsg_id, "HSG", "HSG ID")
            st.pyplot(fig_hsg)

        with col2:
            st.markdown("**LULC (CCI Codes)**")
            fig_lulc = plot_raster(lulc, "LULC", "CCI code")
            st.pyplot(fig_lulc)

        with col3:
            st.markdown("**Curve Number (CN)**")
            fig_cn = plot_raster(cn, "Curve Number", "CN")
            st.pyplot(fig_cn)

        # ----------------- Download CN GeoTIFF -----------------
        st.subheader("Download Curve Number GeoTIFF")

        cn_bytes = cn_to_geotiff_bytes(cn, profile)
        st.download_button(
            label="Download CN GeoTIFF",
            data=cn_bytes,
            file_name="CurveNumber.tif",
            mime="image/tiff",
        )

        # Clean temp files
        try:
            os.remove(soil_path_tmp)
            os.remove(lulc_path_tmp)
        except Exception:
            pass
else:
    st.info("Please upload both a SOIL raster and a LULC raster to enable CN generation.")
