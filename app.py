from __future__ import annotations

import io
import json
import os
import re
import shutil
import statistics
import tempfile
import zipfile

from dataclasses import dataclass
from functools import lru_cache
from itertools import islice

import fiona
import geopandas as gpd
import streamlit as st
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.geometry import shape as shapely_shape

SAMPLE_GPKG_DIR = os.path.join(os.path.dirname(__file__), "sample gpkg")
TRAINING_DATA_DIR = os.path.join(os.path.dirname(__file__), "Training_Data")
LEARNING_DIR = os.path.join(os.path.dirname(__file__), "Learning")
DEFAULT_DL_PACK = os.path.join(TRAINING_DATA_DIR, "DLTrainingPack.zip")
DEFAULT_DL_PACK_REPO = os.path.join(os.path.dirname(__file__), "dl_packs", "default_dlpack.zip")

st.set_page_config(page_title="GDB Substation Equipment Converter", layout="wide")
st.title("Substation GDB Auto-Cleaner (Layer per Equipment)")

st.markdown(
    """
Upload a zipped **.gdb** and the app will:
- Read all layers
- Force the expected geometry type for each equipment
- Export a **GPKG with one layer per equipment** using the reference attribute tables from `sample gpkg`
"""
)

# Configurable parameters
max_annotation_distance = st.number_input(
    "Max distance (map units) to attach annotation names", min_value=0.0, value=0.0, step=1.0
)
use_annotations = st.checkbox("Use annotation layer to attach names to features", value=True)
use_annotation_classification = st.checkbox(
    "Classify equipment by annotation text",
    value=True,
    help="If enabled, annotation labels are used to choose the equipment layer name (e.g., CT->CURRENT_TRANSFORMER).",
)
# When enabled, nearest annotation point is always used as the geometry anchor (no distance filter)
force_snap_to_annotation = st.checkbox(
    "Force snap to annotation point when available",
    value=True,
    help="When on, point geometries use the exact annotation point if a label is found.",
)
use_reference_alignment = st.checkbox(
    "Align counts/positions to sample gpkg templates",
    value=False,
    help="When enabled, limits counts and snaps geometries to reference sample gpkg locations.",
)
use_dl_assist = st.checkbox(
    "Experimental: use deep learning assist when annotations are missing/ambiguous",
    value=False,
    help="Keeps annotation-based logic first. Uses optional vision/OCR only when labels are missing or unclear.",
)
dl_conf_threshold = st.slider(
    "DL minimum confidence",
    min_value=0.0,
    max_value=1.0,
    value=0.4,
    step=0.05,
    help="Predictions below this confidence are ignored.",
)
dl_max_snap_distance = st.number_input(
    "DL max snap distance (map units)",
    min_value=0.0,
    value=5.0,
    step=1.0,
    help="Ignore DL detections farther than this distance from a feature.",
)
dl_drawings = st.file_uploader(
    "Optional: raster drawings / scanned PDFs for DL assist",
    type=["pdf", "tif", "tiff", "png", "jpg", "jpeg", "geojson", "json"],
    accept_multiple_files=True,
)
dl_detection_packs = st.file_uploader(
    "Optional: DL detection/training pack (zip/geojson/json) to pre-seed detections",
    type=["zip", "geojson", "json"],
    accept_multiple_files=True,
)
learning_dirs_available = [d for d in (TRAINING_DATA_DIR, LEARNING_DIR) if os.path.isdir(d)]
use_learning_library = st.checkbox(
    "Use learning library (Training_Data + Learning/*) to guide placement",
    value=bool(learning_dirs_available),
    help="Harvest geometry stats from local learning folders to guess equipment when annotations are missing.",
)
learning_sheet_uploads = st.file_uploader(
    "Optional: upload learning sheet(s) (xlsx/xls) with expected equipment counts",
    type=["xlsx", "xls"],
    accept_multiple_files=True,
    help="Upload per-substation data sheets to cap counts and improve naming.",
)

# Expected geometry type per equipment layer
FORCED_GEOMETRY = {
    # POINTS
    "HIGH_VOLTAGE_CIRCUIT_BREAKER": "Point",
    "DISCONNECTOR_SWITCH": "Point",
    "DIGITAL_FAULT_RECORDER": "Point",
    "CURRENT_TRANSFORMER": "Point",
    "CT_INDOR_SWITCHGEAR": "Point",
    "CONNECTION_POINTS": "Point",
    "CONNECTION POINTS": "Point",
    "cb_indor_switchgear": "Point",
    "110VDC_CHARGER": "Point",
    "110VDC_BATTERY": "Point",
    "48VDC_CHARGER": "Point",
    "48VDC_BATTERY": "Point",
    "VT_INDOR_SWITCHGEAR": "Point",
    "VOLTAGE_TRANSFORMER": "Point",
    "UPS": "Point",
    "TRANS_SYSTEM_PROT1": "Point",
    "TELECOM": "Point",
    "LIGHTNING_ARRESTOR": "Point",
    "Annotation": "Point",
    "ANNOTATION": "Point",
    # LINES
    "BUSBAR": "Line",
    "POWER_CABLE_TO_TRANSFORMER": "Line",
    # POLYGONS
    "Cabin": "Polygon",
    # Transformers are points in the sample GPKG
    "Transformers": "Point",
    "LINE_BAY": "Polygon",
    "INDOR_SWITCHGEAR_TABLE": "Polygon",
}

BASE_LAYERS = {"Point", "Polyline", "Polygon", "MultiPatch", "Annotation", "Annotations"}


def _first_coord(coords):
    """Return the first coordinate pair from a nested coords structure."""
    if coords is None:
        return None
    cursor = coords
    while isinstance(cursor, (list, tuple)) and cursor and isinstance(cursor[0], (list, tuple)):
        cursor = cursor[0]
    if isinstance(cursor, (list, tuple)) and len(cursor) >= 2:
        return cursor[:2]
    return None


def coerce_geometry_from_mapping(mapping, target_hint):
    """
    Build a shapely geometry from a raw geo mapping, repairing minimal/invalid polygons.
    Falls back to buffered lines or points when rings have <3 vertices.
    """
    if mapping is None:
        return None
    geom = None

    try:
        geom = shapely_shape(mapping)
    except Exception:
        geom = None

    if geom is None:
        coords = mapping.get("coordinates") if isinstance(mapping, dict) else None
        gtype = (mapping.get("type") or "").lower() if isinstance(mapping, dict) else ""

        # If polygon-like but too few vertices, fallback to line (optionally buffered to polygon)
        if gtype in ("polygon", "multipolygon") and coords:
            try:
                ring = coords[0]
                if gtype == "multipolygon" and isinstance(coords[0], (list, tuple)):
                    ring = coords[0][0] if coords[0] else None
                if ring and len(ring) >= 2:
                    line = LineString(ring)
                    geom = line.buffer(0.1) if target_hint == "Polygon" else line
            except Exception:
                geom = None

        # Final fallback to a point
        if geom is None:
            first = _first_coord(coords)
            if first is not None:
                geom = Point(first)

    if geom is None:
        return None

    # Make invalid polygons writable
    if target_hint == "Polygon" and geom.type in ("Polygon", "MultiPolygon") and not geom.is_valid:
        geom = geom.buffer(0)

    return geom


def load_layer_with_fallback(gdb_dir, layer, target_hint):
    """
    Try to read a layer with GeoPandas; if it fails (e.g., invalid rings),
    fall back to Fiona + manual geometry coercion.
    """
    try:
        return gpd.read_file(gdb_dir, layer=layer)
    except Exception as exc:
        st.warning(f"Could not load `{layer}` directly ({exc}); attempting to repair geometries.")

    try:
        with fiona.open(gdb_dir, layer=layer) as src:
            records = []
            for feat in src:
                geom = coerce_geometry_from_mapping(feat.get("geometry"), target_hint)
                if geom is None:
                    continue
                props = dict(feat.get("properties") or {})
                props["geometry"] = geom
                records.append(props)

            if not records:
                return gpd.GeoDataFrame(
                    columns=list(src.schema.get("properties", {}).keys()) + ["geometry"],
                    crs=src.crs,
                )

            return gpd.GeoDataFrame(records, crs=src.crs)
    except Exception as exc2:
        st.error(f"Failed to recover layer `{layer}`: {exc2}")
        return None


def extract_annotation_points(gdf):
    """Extract annotation label points from an annotation layer."""
    if gdf is None or gdf.empty:
        return None
    text_field = None
    for candidate in ["TextString", "Text", "Element", "Label"]:
        if candidate in gdf.columns:
            text_field = candidate
            break
    if text_field is None:
        return None

    ann = gdf.copy()
    ann["__label__"] = ann[text_field].astype(str)
    ann["geometry"] = ann.geometry.apply(
        lambda g: g.representative_point() if g is not None else None
    )
    ann = ann.dropna(subset=["geometry", "__label__"])
    return ann


def attach_annotation_label(geom, annotations):
    """Return (label, label_point) for nearest annotation (no distance filter)."""
    if annotations is None or annotations.empty or geom is None or geom.is_empty:
        return None, None

    rep = geom.representative_point()
    dists = annotations.geometry.distance(rep)
    if dists.empty:
        return None, None
    idx = dists.idxmin()
    return annotations.loc[idx, "__label__"], annotations.geometry.loc[idx]


def fill_indor_switchgear_table(processed_layers, loaded_layers, ref_schemas, ref_counts):
    """If INDOR_SWITCHGEAR_TABLE is under-filled, borrow polygons from base Polygon/MultiPatch layers."""
    target = "INDOR_SWITCHGEAR_TABLE"
    cap = ref_counts.get(target)
    if cap is None:
        return processed_layers
    current = len(processed_layers.get(target, [])) if target in processed_layers else 0
    if current >= cap:
        return processed_layers

    candidates = []
    for base_name in ["Polygon", "MultiPatch"]:
        if base_name in loaded_layers:
            candidates.extend(list(loaded_layers[base_name].geometry))

    if not candidates:
        return processed_layers

    add_count = min(cap - current, len(candidates))
    ref_fields = ref_schemas.get(target, [])
    new_records = []
    for i in range(add_count):
        attrs = {field: None for field in ref_fields}
        attrs["geometry"] = candidates[i]
        new_records.append(attrs)

    base_crs = (
        loaded_layers.get("Polygon", loaded_layers.get("MultiPatch")).crs
        if loaded_layers
        else None
    )
    existing = processed_layers.get(target)
    if existing is not None and not existing.empty:
        processed_layers[target] = gpd.GeoDataFrame(
            pd.concat([existing, gpd.GeoDataFrame(new_records)], ignore_index=True),
            crs=existing.crs or base_crs,
        )
    else:
        processed_layers[target] = gpd.GeoDataFrame(new_records, crs=base_crs)

    return processed_layers


def fill_disconnector_switch(processed_layers, loaded_layers, annotation_points, ref_schemas, ref_counts):
    """If DISCONNECTOR_SWITCH is under-filled, clone nearest base points to match sample count."""
    target = "DISCONNECTOR_SWITCH"
    cap = ref_counts.get(target)
    if cap is None:
        return processed_layers
    current = len(processed_layers.get(target, [])) if target in processed_layers else 0
    if current >= cap:
        return processed_layers

    base_pts = loaded_layers.get("Point")
    if base_pts is None or base_pts.empty:
        return processed_layers

    labels_pts = []
    if annotation_points is not None:
        for _, row in annotation_points.iterrows():
            lbl = row["__label__"]
            if classify_equipment(lbl) == target:
                labels_pts.append(row.geometry)

    candidates = list(base_pts.geometry)
    used = set()
    ref_fields = ref_schemas.get(target, [])
    new_records = []

    targets = labels_pts if labels_pts else candidates

    for ap in targets:
        if len(new_records) + current >= cap:
            break
        if labels_pts:
            dists = [(i, ap.distance(g)) for i, g in enumerate(candidates) if i not in used]
            if not dists:
                break
            i, _ = min(dists, key=lambda x: x[1])
        else:
            # no labels matched; just take next unused candidate
            available = [i for i in range(len(candidates)) if i not in used]
            if not available:
                break
            i = available[0]
        used.add(i)
        attrs = {field: None for field in ref_fields}
        attrs["ANNOTATION_TEXT"] = "DUPLICATED" if labels_pts else None
        attrs["geometry"] = candidates[i]
        new_records.append(attrs)

    if new_records:
        existing = processed_layers.get(target)
        if existing is not None and not existing.empty:
            processed_layers[target] = gpd.GeoDataFrame(
                pd.concat([existing, gpd.GeoDataFrame(new_records)], ignore_index=True),
                crs=existing.crs or base_pts.crs,
            )
        else:
            processed_layers[target] = gpd.GeoDataFrame(new_records, crs=base_pts.crs)

    return processed_layers


def fill_lightning_arrestor(processed_layers, loaded_layers, annotation_points, ref_schemas, ref_counts):
    """If LIGHTNING_ARRESTOR is under-filled, clone nearest base points to match sample count."""
    target = "LIGHTNING_ARRESTOR"
    cap = ref_counts.get(target)
    if cap is None:
        return processed_layers
    current = len(processed_layers.get(target, [])) if target in processed_layers else 0
    if current >= cap:
        return processed_layers

    base_pts = loaded_layers.get("Point")
    if base_pts is None or base_pts.empty:
        return processed_layers

    labels_pts = []
    if annotation_points is not None:
        for _, row in annotation_points.iterrows():
            if classify_equipment(row["__label__"]) == target:
                labels_pts.append(row.geometry)

    candidates = list(base_pts.geometry)
    used = set()
    ref_fields = ref_schemas.get(target, [])
    new_records = []
    targets = labels_pts if labels_pts else candidates

    for ap in targets:
        if len(new_records) + current >= cap:
            break
        if labels_pts:
            dists = [(i, ap.distance(g)) for i, g in enumerate(candidates) if i not in used]
            if not dists:
                break
            i, _ = min(dists, key=lambda x: x[1])
        else:
            available = [i for i in range(len(candidates)) if i not in used]
            if not available:
                break
            i = available[0]
        used.add(i)
        attrs = {field: None for field in ref_fields}
        attrs["ANNOTATION_TEXT"] = "DUPLICATED" if labels_pts else None
        attrs["geometry"] = candidates[i]
        new_records.append(attrs)

    if new_records:
        existing = processed_layers.get(target)
        if existing is not None and not existing.empty:
            processed_layers[target] = gpd.GeoDataFrame(
                pd.concat([existing, gpd.GeoDataFrame(new_records)], ignore_index=True),
                crs=existing.crs or base_pts.crs,
            )
        else:
            processed_layers[target] = gpd.GeoDataFrame(new_records, crs=base_pts.crs)

    return processed_layers


def fill_voltage_transformer(processed_layers, loaded_layers, annotation_points, ref_schemas, ref_counts):
    """If VOLTAGE_TRANSFORMER is under-filled, clone nearest base points to match expected count."""
    target = "VOLTAGE_TRANSFORMER"
    cap = ref_counts.get(target)
    if cap is None:
        return processed_layers
    current = len(processed_layers.get(target, [])) if target in processed_layers else 0
    if current >= cap:
        return processed_layers

    base_pts = loaded_layers.get("Point")
    if base_pts is None or base_pts.empty:
        return processed_layers

    labels_pts = []
    if annotation_points is not None:
        for _, row in annotation_points.iterrows():
            if classify_equipment(row["__label__"]) == target:
                labels_pts.append(row.geometry)

    candidates = list(base_pts.geometry)
    used = set()
    ref_fields = ref_schemas.get(target, [])
    new_records = []
    targets = labels_pts if labels_pts else candidates

    for ap in targets:
        if len(new_records) + current >= cap:
            break
        if labels_pts:
            dists = [(i, ap.distance(g)) for i, g in enumerate(candidates) if i not in used]
            if not dists:
                break
            i, _ = min(dists, key=lambda x: x[1])
        else:
            available = [i for i in range(len(candidates)) if i not in used]
            if not available:
                break
            i = available[0]
        used.add(i)
        attrs = {field: None for field in ref_fields}
        attrs["ANNOTATION_TEXT"] = "DUPLICATED" if labels_pts else None
        attrs["geometry"] = candidates[i]
        new_records.append(attrs)

    if new_records:
        existing = processed_layers.get(target)
        if existing is not None and not existing.empty:
            processed_layers[target] = gpd.GeoDataFrame(
                pd.concat([existing, gpd.GeoDataFrame(new_records)], ignore_index=True),
                crs=existing.crs or base_pts.crs,
            )
        else:
            processed_layers[target] = gpd.GeoDataFrame(new_records, crs=base_pts.crs)

    return processed_layers


def snap_to_reference(processed_layers, ref_geoms):
    """Snap processed geometries to nearest reference geometries when available."""
    snapped = {}
    for layer, gdf in processed_layers.items():
        ref_list = ref_geoms.get(layer)
        if not ref_list:
            snapped[layer] = gdf
            continue
        # If we need more targets than references, cycle the references
        if len(ref_list) < len(gdf):
            from itertools import cycle, islice
            ref_targets = list(islice(cycle(ref_list), len(gdf)))
        else:
            ref_targets = ref_list.copy()

        remaining = ref_targets.copy()
        new_geoms = []
        for geom in gdf.geometry:
            if not remaining:
                remaining = ref_targets.copy()
            distances = [geom.distance(rg) for rg in remaining]
            idx = distances.index(min(distances))
            new_geoms.append(remaining.pop(idx))
        snapped[layer] = gdf.copy()
        snapped[layer].geometry = new_geoms
    return snapped


def safe_filename(name: str) -> str:
    """Create a filesystem-safe filename for a layer."""
    invalid = '<>:"/\\|?*'
    cleaned = "".join("_" if c in invalid else c for c in name)
    cleaned = cleaned.replace(" ", "_")
    return cleaned or "layer"


def infer_geometry_type(layer_name: str, gdf: gpd.GeoDataFrame | None, ref_geoms_lookup: dict) -> str:
    """Infer a geometry type for empty layers using processed or reference data."""
    if gdf is not None:
        for geom in gdf.geometry:
            if geom is not None and not geom.is_empty:
                return geom.geom_type
    ref_list = ref_geoms_lookup.get(layer_name) if ref_geoms_lookup else None
    if ref_list:
        for geom in ref_list:
            if geom is not None and not geom.is_empty:
                return geom.geom_type
    forced = FORCED_GEOMETRY.get(layer_name)
    if forced == "Point":
        return "Point"
    if forced in ("Line", "Polyline"):
        return "LineString"
    if forced == "Polygon":
        return "Polygon"
    return "GeometryCollection"


DEVICE_TO_EQUIPMENT = {
    "HIGH_VOLTAGE_BUSBAR_MEDIUM_VOLTAGE_BUSBAR": "BUSBAR",
    "POWER_TRANSFORMER_STEPUP_TRANSFORMER": "Transformers",
    "POWER_TRANSFORMER": "Transformers",
    "STEPUP_TRANSFORMER": "Transformers",
    "HIGH_VOLTAGE_SWITCH_HIGH_VOLTAGE_SWITCH": "DISCONNECTOR_SWITCH",
    "INDOOR_CIRCUIT_BREAKER_30KV_15KB": "cb_indor_switchgear",
    "INDOR_CB": "cb_indor_switchgear",
    "INDOOR_CURRENT_TRANSFORMER": "CT_INDOR_SWITCHGEAR",
    "INDOR_CT": "CT_INDOR_SWITCHGEAR",
    "INDOOR_VOLTAGE_TRANSFORMER": "VT_INDOR_SWITCHGEAR",
    "INDOR_VT": "VT_INDOR_SWITCHGEAR",
    "CONTROL_AND_PROTECTION_PANELS": "TELECOM",
    "SUBSTATION_CABIN": "Cabin",
    "LIGHTNING_ARRESTER": "LIGHTNING_ARRESTOR",
    "LINE_BAY": "LINE_BAY",
    "HIGHVOLTAGE_LINE": "LINE_BAY",
    "LINE": "LINE_BAY",
    "LINEBAY": "LINE_BAY",
    "MV_SWITCH_GEAR": "INDOR_SWITCHGEAR_TABLE",
    "CURRENT_TRANSFORMER": "CURRENT_TRANSFORMER",
    "VOLTAGE_TRANSFORMER": "VOLTAGE_TRANSFORMER",
    "HIGH_VOLTAGE_CIRCUIT_BREAKER_HIGH_VOLTAGE_CIRCUIT_BREAKER": "HIGH_VOLTAGE_CIRCUIT_BREAKER",
    "DIGITAL_FAULT_RECORDER": "DIGITAL_FAULT_RECORDER",
    "DISTANCE_PROTECTION": "TRANS_SYSTEM_PROT1",
    "TRANSFORMER_PROTECTION": "transformer_protection",
    "LINE_OVERCURRENT_PROTECTION": "line_overcurrent_protection",
    "TRANSFORMER_BAY": "Transformers",
    "POWER_TRANSFORMER__STEPUP_TRANSFORMER": "Transformers",
    "DC_SUPPLY_110_VDC_BATTERY": "110VDC_BATTERY",
    "DC_SUPPLY_110_VDC_CHARGER": "110VDC_CHARGER",
    "DC_SUPPLY_48_VDC_BATTERY": "48VDC_BATTERY",
    "DC_SUPPLY_48_VDC_CHARGER": "48VDC_CHARGER",
    "HIGH_VOLTAGE_LINE": "LINE_BAY",
    "TRANS_SYSTEM_PROT1": "TRANS_SYSTEM_PROT1",
    "TELECOM": "TELECOM",
    "TELECOM_ODF": "TELECOM",
    "TELECOM_SDH": "TELECOM",
    "UPS": "UPS",
}


def map_device_to_equipment(device_name: str) -> str | None:
    """Map device labels from learning sheets to canonical equipment."""
    if not device_name:
        return None
    norm = canonical_layer_name(str(device_name))
    if norm in DEVICE_TO_EQUIPMENT:
        return DEVICE_TO_EQUIPMENT[norm]
    # Fall back to layer mapping aliases
    return map_layer_to_equipment(device_name)


def export_empty_layer(out_path: str, layer_name: str, geometry_type: str, fields: list[str], crs):
    """Create an empty GPKG layer with the provided schema."""
    schema = {
        "geometry": geometry_type or "GeometryCollection",
        "properties": {field: "str" for field in fields},
    }
    with fiona.open(out_path, "w", driver="GPKG", layer=layer_name, schema=schema, crs=crs) as dst:
        pass


def apply_count_caps(processed_layers: dict[str, gpd.GeoDataFrame], caps: dict[str, int]):
    """Trim processed layers to honor expected counts from learning sheets."""
    if not caps:
        return processed_layers
    capped = {}
    for layer, gdf in processed_layers.items():
        cap = caps.get(layer)
        if cap is not None and len(gdf) > cap:
            capped[layer] = gdf.head(cap)
        else:
            capped[layer] = gdf
    return capped


def build_dl_training_pack(processed_layers: dict[str, gpd.GeoDataFrame], base_crs, out_dir: str):
    """
    Build a DL-ready training pack from current processed layers.
    Creates a GeoJSON with envelope geometries + metadata for offline training.
    """
    if not processed_layers:
        return None
    records = []
    for layer_name, gdf in processed_layers.items():
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            env = geom.envelope
            props = {
                "equipment": layer_name,
                "source_layer": layer_name,
                "annotation": row.get("ANNOTATION_TEXT"),
                "confidence": 1.0,
            }
            records.append({"geometry": env, **props})
    if not records:
        return None
    os.makedirs(out_dir, exist_ok=True)
    det_gdf = gpd.GeoDataFrame(records, crs=base_crs)
    geojson_path = os.path.join(out_dir, "detections.geojson")
    det_gdf.to_file(geojson_path, driver="GeoJSON")
    meta = {
        "classes": sorted({r["equipment"] for r in records}),
        "count": len(records),
        "notes": "Envelopes derived from processed geometries; annotation text preserved where available.",
    }
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return out_dir


def export_dl_predictions(predictions: list[DLPrediction], base_crs, out_dir: str):
    """Export DL predictions to GeoJSON for inspection or external training."""
    if not predictions:
        return None
    records = []
    for pred in predictions:
        geom = pred.geometry
        if geom is None or geom.is_empty:
            continue
        equip = pred.equipment_hint or classify_equipment(pred.label)
        if equip is None:
            continue
        records.append(
            {
                "label": pred.label,
                "equipment": equip,
                "confidence": pred.confidence,
                "source": pred.source,
                "geometry": geom,
            }
        )
    if not records:
        return None
    os.makedirs(out_dir, exist_ok=True)
    gdf = gpd.GeoDataFrame(records, crs=base_crs)
    geojson_path = os.path.join(out_dir, "dl_detections.geojson")
    gdf.to_file(geojson_path, driver="GeoJSON")
    meta = {
        "count": len(records),
        "classes": sorted({r["equipment"] for r in records}),
        "sources": sorted({r["source"] for r in records}),
    }
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return out_dir


@dataclass(frozen=True)
class EquipmentRule:
    """Single equipment classification rule."""

    name: str
    patterns: tuple[str, ...]
    keywords: tuple[str, ...] = ()
    priority: int = 0


class EquipmentClassifier:
    """
    Encapsulates equipment classification logic so new rules can be added
    without touching the calling code. Annotation text remains the primary signal.
    """

    def __init__(self, rules: list[EquipmentRule]):
        # Preserve declaration order but allow explicit priority override
        self.rules = sorted(rules, key=lambda r: r.priority, reverse=True)
        self.compiled = [
            (rule, [re.compile(pat, flags=re.IGNORECASE) for pat in rule.patterns])
            for rule in self.rules
        ]

    @staticmethod
    def normalize_label(label: str | None) -> str:
        if label is None:
            return ""
        text = re.sub(r"[^\w\s\-\/\.]+", " ", str(label))
        text = re.sub(r"[_\-]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text.upper()

    @lru_cache(maxsize=256)
    def _classify_normalized(self, normalized: str) -> str | None:
        for rule, patterns in self.compiled:
            if any(pat.search(normalized) for pat in patterns):
                return rule.name
            if rule.keywords and any(kw in normalized for kw in rule.keywords):
                return rule.name
        return None

    def classify(self, label: str | None) -> str | None:
        normalized = self.normalize_label(label)
        if not normalized:
            return None
        return self._classify_normalized(normalized)


CLASSIFICATION_RULES = [
    # Structures
    EquipmentRule(
        name="Cabin",
        patterns=(
            r"\bCONTROL\s+BUILDING\b",
            r"\bCONTROL\s+ROOM\b",
            r"\bCABIN\b",
        ),
        priority=10,
    ),
    # Bus system
    EquipmentRule(
        name="BUSBAR",
        patterns=(r"\bBB[-\s]*\d+\b", r"\bBUSBAR\b"),
        priority=8,
    ),
    # Switchgear / primary
    EquipmentRule(
        name="HIGH_VOLTAGE_CIRCUIT_BREAKER",
        patterns=(r"\bCB[-\s]*\d+\b", r"\bBUS\s+COUPLER\b", r"\bBREAKER\b"),
        priority=9,
    ),
    EquipmentRule(
        name="CURRENT_TRANSFORMER",
        patterns=(r"\bCT[-\s]*\d+\b", r"\bCURRENT\s+TRANSFORMER\b"),
        priority=9,
    ),
    EquipmentRule(
        name="VOLTAGE_TRANSFORMER",
        patterns=(r"\bVT[-\s]*\d+\b", r"\bVOLTAGE\s+TRANSFORMER\b"),
        priority=9,
    ),
    EquipmentRule(
        name="DISCONNECTOR_SWITCH",
        patterns=(
            r"\bQ[-\s]*\d+\b",
            r"\bDS[-\s]*\d+\b",
            r"\bDISCONNECT(OR)?\b",
            r"\bISOLATOR\b",
        ),
        priority=8,
    ),
    EquipmentRule(
        name="LIGHTNING_ARRESTOR",
        patterns=(
            r"\bLA[-\s]*\d+\b",
            r"\bSA[-\s]*\d+\b",
            r"\bSURGE\s+ARRESTOR\b",
            r"\bLIGHTNING\s+ARRESTOR\b",
        ),
        priority=8,
    ),
    # Indoor switchgear
    EquipmentRule(
        name="cb_indor_switchgear",
        patterns=(r"\bCB\s+INDO?R\b", r"\bCB[-_ ]?MV\d*\b"),
        priority=7,
    ),
    EquipmentRule(
        name="CT_INDOR_SWITCHGEAR",
        patterns=(
            r"\bCT\s+INDO?R\b",
            r"\bCT[-_ ]?MV\d*\b",
            r"\bCURRENT\s+TRANSFORMER\s+INDO?R\b",
        ),
        priority=7,
    ),
    EquipmentRule(
        name="VT_INDOR_SWITCHGEAR",
        patterns=(
            r"\bVT\s+INDO?R\b",
            r"\bVT[-_ ]?MV\d*\b",
            r"\bCV[-_ ]?MV\d*\b",
            r"\bVOLTAGE\s+TRANSFORMER\s+INDO?R\b",
        ),
        priority=7,
    ),
    # DC & auxiliary
    EquipmentRule(
        name="48VDC_BATTERY",
        patterns=(
            r"\b48\s*V\s*(DC)?\b.*BATTERY",
            r"\bBATTERY\s*48\b",
            r"\b48V\s*BATTERY\b",
        ),
        priority=6,
    ),
    EquipmentRule(
        name="110VDC_BATTERY",
        patterns=(
            r"\b110\s*V\s*(DC)?\b.*BATTERY",
            r"\bBATTERY\s*110\b",
            r"\bBTY[-\s]*\d*\b",
        ),
        priority=6,
    ),
    EquipmentRule(
        name="48VDC_CHARGER",
        patterns=(
            r"\b48\s*V\s*(DC)?\b.*CHARGER",
            r"\bCHARGER.*48\b",
        ),
        priority=6,
    ),
    EquipmentRule(
        name="110VDC_CHARGER",
        patterns=(
            r"\b110\s*V\s*(DC)?\b.*CHARGER",
            r"\bCHARGER.*110\b",
            r"\bBC[-\s]*\d+\b",
        ),
        priority=6,
    ),
    EquipmentRule(
        name="UPS",
        patterns=(r"\bUPS\b", r"\bUNINTERRUPTIBLE\b"),
        priority=6,
    ),
    # Telecom (maps all subtypes to TELECOM)
    EquipmentRule(
        name="TELECOM",
        patterns=(r"\bTELECOM\b", r"\bODF\b", r"\bSDH\b", r"\bMUX\b"),
        keywords=("DDF", "PATCH PANEL"),
        priority=5,
    ),
    # Protection & control
    EquipmentRule(
        name="DIGITAL_FAULT_RECORDER",
        patterns=(r"\bDFR\b", r"FAULT\s*REC", r"\bRECORDER\b"),
        priority=5,
    ),
    EquipmentRule(
        name="TRANS_SYSTEM_PROT1",
        patterns=(r"\bPROT(?!EIN)\b", r"\bPROTECTION\b", r"\bTSP\b"),
        keywords=("SYSTEM PROTECTION",),
        priority=5,
    ),
    # Bays & connections
    EquipmentRule(
        name="LINE_BAY",
        patterns=(r"\bLINE\s+BAY\b",),
        priority=4,
    ),
    EquipmentRule(
        name="CONNECTION_POINTS",
        patterns=(r"\bPOINT\s+CONNECTION\b", r"\bCONNECTION\s+POINTS?\b"),
        priority=4,
    ),
    EquipmentRule(
        name="POWER_CABLE_TO_TRANSFORMER",
        patterns=(
            r"\bCABLE.*TRANSFORMER\b",
            r"\bTRANSFORMER.*CABLE\b",
            r"\bPOWER\s+CABLE\b",
        ),
        priority=4,
    ),
    # Transformers / auxiliaries
    EquipmentRule(
        name="Transformers",
        patterns=(r"\bTR[-\s]*\d+\b", r"\bPOWER\s+TRANSFORMER\b", r"\bAUX\b"),
        priority=3,
    ),
]

classifier = EquipmentClassifier(CLASSIFICATION_RULES)


def classify_equipment(label: str) -> str | None:
    """Return an equipment layer name based on annotation text."""
    return classifier.classify(label)


def canonical_layer_name(name: str) -> str:
    norm = re.sub(r"[^A-Za-z0-9]+", "_", name or "").strip("_").upper()
    norm = re.sub(r"_\d+$", "", norm)
    return norm


LAYER_NAME_ALIASES = {
    "B_110VDC_BATTERY": "110VDC_BATTERY",
    "B_110VDC_CHARGER": "110VDC_CHARGER",
    "B_48VDC_BATTERY": "48VDC_BATTERY",
    "B_48VDC_CHARGER": "48VDC_CHARGER",
    "110_VDC_BATTERY": "110VDC_BATTERY",
    "110_VDC_CHARGER": "110VDC_CHARGER",
    "SWITCHGEAR": "INDOR_SWITCHGEAR_TABLE",
    "INDORCB": "cb_indor_switchgear",
    "INDOR_CT": "CT_INDOR_SWITCHGEAR",
    "INDOR_VT": "VT_INDOR_SWITCHGEAR",
    "INDORVT": "VT_INDOR_SWITCHGEAR",
    "CB_INDOOR_SWITCH_GEAR": "cb_indor_switchgear",
    "CT_INDOOR_SWITCH_GEAR": "CT_INDOR_SWITCHGEAR",
    "VT_INDOOOR_SWITCH_GEAR": "VT_INDOR_SWITCHGEAR",
    "INDOR_SWICTH_GEAR_TABLE": "INDOR_SWITCHGEAR_TABLE",
    "TRANS_SYSTEM_PROT2": "TRANS_SYSTEM_PROT1",
    "TRANS_SYSTEM_PROTECTION1": "TRANS_SYSTEM_PROT1",
    "TRANSFORMER": "Transformers",
    "TRANSFORMERS": "Transformers",
    "POWER_TRANSFOMER": "Transformers",
    "CURRENT_TRANSFOMER": "CURRENT_TRANSFORMER",
    "CURRENT_TRANSFORMER__CURRENT_TRANSFORMER": "CURRENT_TRANSFORMER",
    "DISCONNECTOR_SWITCHES": "DISCONNECTOR_SWITCH",
    "DISCONNECTOR_SWITCHES1": "DISCONNECTOR_SWITCH",
    "POINT_CONNECTION": "CONNECTION_POINTS",
    "POINTCONNECTION": "CONNECTION_POINTS",
    "TELECOM_ODF": "TELECOM",
    "TELECOM_SDH": "TELECOM",
    "LINEBAY": "LINE_BAY",
    "LINE": "LINE_BAY",
    "LINE_BAY": "LINE_BAY",
    "HIGHVOLTAGE_LINE": "LINE_BAY",
    "POWERTRANSFORMER": "Transformers",
    "POWER_TRANSFORMER": "Transformers",
    "POWER_TRANSFORMER_STEPUP_TRANSFORMER": "Transformers",
    "TRANSFORMER_BAY": "Transformers",
    "INDOR_CB": "cb_indor_switchgear",
    "INDOR_CT": "CT_INDOR_SWITCHGEAR",
    "INDOR_VT": "VT_INDOR_SWITCHGEAR",
    "INDOOR_CURRENT_TRANSFORMER": "CT_INDOR_SWITCHGEAR",
    "INDOOR_VOLTAGE_TRANSFORMER": "VT_INDOR_SWITCHGEAR",
    "INDOORCIRCUITBREAKER": "cb_indor_switchgear",
    "HIGH_VOLTAGE_SWITCH_HIGH_VOLTAGE_SWITCH": "DISCONNECTOR_SWITCH",
    "MV_SWITCH_GEAR": "INDOR_SWITCHGEAR_TABLE",
    "CONTROL_AND_PROTECTION_PANELS": "TELECOM",
    "BUSBAR1": "BUSBAR",
}


def map_layer_to_equipment(layer_name: str) -> str | None:
    """Map source layer names to canonical equipment layer names."""
    norm = canonical_layer_name(layer_name)
    if norm in LAYER_NAME_ALIASES:
        return LAYER_NAME_ALIASES[norm]
    for equip in FORCED_GEOMETRY.keys():
        if canonical_layer_name(equip) == norm:
            return equip
    # Heuristic fallbacks for common variants
    if "48VDC" in norm and "CHARGER" in norm:
        return "48VDC_CHARGER"
    if "48VDC" in norm and "BATTERY" in norm:
        return "48VDC_BATTERY"
    if "INDOR" in norm and "SWITCHGEAR" in norm and "TABLE" in norm:
        return "INDOR_SWITCHGEAR_TABLE"
    if "INDOR" in norm and "CB" in norm:
        return "cb_indor_switchgear"
    if "INDOR" in norm and "CT" in norm:
        return "CT_INDOR_SWITCHGEAR"
    if "INDOR" in norm and "VT" in norm:
        return "VT_INDOR_SWITCHGEAR"
    return None


@dataclass(frozen=True)
class EquipmentStatSummary:
    """Geometry shape statistics harvested from training GDB/GPKG files."""

    geom_type: str | None
    median_area: float | None
    median_length: float | None
    median_ratio: float | None
    median_span: float | None
    sample_count: int
    sources: tuple[str, ...]


def _geom_metrics(geom):
    if geom is None or geom.is_empty:
        return None
    try:
        minx, miny, maxx, maxy = geom.bounds
        width = maxx - minx
        height = maxy - miny
    except Exception:
        return None
    span = max(width, height)
    ratio = None
    if width > 0 and height > 0:
        ratio = max(width, height) / min(width, height)
    area = geom.area if hasattr(geom, "area") else None
    length = geom.length if hasattr(geom, "length") else None
    return {
        "geom_type": geom.geom_type,
        "area": area if area and area > 0 else None,
        "length": length if length and length > 0 else None,
        "ratio": ratio,
        "span": span if span > 0 else None,
    }


def _iter_training_sources(base_dirs):
    """Yield GDB/GPKG training sources from one or more directories."""
    if not base_dirs:
        return
    dirs = list(base_dirs) if isinstance(base_dirs, (list, tuple, set)) else [base_dirs]
    for base_dir in dirs:
        if not os.path.isdir(base_dir):
            continue
        for root, subdirs, files in os.walk(base_dir):
            for d in subdirs:
                if d.lower().endswith(".gdb"):
                    yield os.path.join(root, d)
            for f in files:
                if f.lower().endswith(".gpkg"):
                    yield os.path.join(root, f)


def _iter_training_packs(base_dirs):
    if not base_dirs:
        return
    dirs = list(base_dirs) if isinstance(base_dirs, (list, tuple, set)) else [base_dirs]
    for base_dir in dirs:
        if not os.path.isdir(base_dir):
            continue
        for root, _, files in os.walk(base_dir):
            for f in files:
                lower = f.lower()
                if lower.endswith("dltrainingpack.zip") or lower.endswith(".geojson") or (
                    lower.endswith(".json") and "dltrainingpack" in lower
                ):
                    yield os.path.join(root, f)


def _collect_metric(target, geom, summaries, sources_for_layer, source_label):
    metric = _geom_metrics(geom)
    if metric is None:
        return
    summaries.setdefault(target, []).append(metric)
    sources_for_layer.setdefault(target, set()).add(source_label)


def guess_from_library_stats(geom, training_stats: dict[str, EquipmentStatSummary]) -> str | None:
    """Heuristic guess of equipment using geometry stats from training/learning data."""
    if not training_stats or geom is None or geom.is_empty:
        return None
    metrics = _geom_metrics(geom)
    if not metrics:
        return None

    def score(stat: EquipmentStatSummary):
        s = 0.0
        if stat.geom_type and metrics["geom_type"] != stat.geom_type:
            s += 0.75
        if stat.median_area and metrics.get("area"):
            s += min(1.0, abs(metrics["area"] - stat.median_area) / (stat.median_area + 1e-9))
        if stat.median_length and metrics.get("length"):
            s += min(1.0, abs(metrics["length"] - stat.median_length) / (stat.median_length + 1e-9))
        if stat.median_ratio and metrics.get("ratio"):
            s += min(1.0, abs(metrics["ratio"] - stat.median_ratio) / (stat.median_ratio + 1e-9))
        if stat.median_span and metrics.get("span"):
            s += min(1.0, abs(metrics["span"] - stat.median_span) / (stat.median_span + 1e-9))
        return s

    best = None
    best_score = None
    for equip, stat in training_stats.items():
        sc = score(stat)
        if best is None or sc < best_score:
            best, best_score = equip, sc
    if best_score is not None and best_score < 2.2:
        return best
    return None


@st.cache_resource(show_spinner=False)
def load_training_library(
    base_dirs: tuple[str, ...],
    max_per_layer: int = 200,
    extra_packs: tuple[str, ...] | None = None,
):
    """
    Harvest simple geometry stats from training GDB/GPKG files.
    These stats are used only as a gentle fallback when DL/annotations are missing.
    """
    summaries: dict[str, list[dict]] = {}
    sources_for_layer: dict[str, set] = {}
    errors = []

    for path in _iter_training_sources(base_dirs):
        try:
            layers = fiona.listlayers(path)
        except Exception as exc:
            errors.append(f"{path}: {exc}")
            continue

        for layer in layers:
            target = map_layer_to_equipment(layer) or layer
            if target not in FORCED_GEOMETRY:
                continue
            try:
                with fiona.open(path, layer=layer) as src:
                    for feat in islice(src, max_per_layer):
                        geom = coerce_geometry_from_mapping(
                            feat.get("geometry"), FORCED_GEOMETRY.get(target, "Original")
                        )
                        _collect_metric(
                            target, geom, summaries, sources_for_layer, os.path.basename(path)
                        )
            except Exception as exc:
                errors.append(f"{path}::{layer}: {exc}")

    # Also harvest DL training packs (geojson/zip) to enrich stats
    pack_iter = list(_iter_training_packs(base_dirs))
    if extra_packs:
        pack_iter.extend([p for p in extra_packs if p and os.path.exists(p)])
    for pack_path in pack_iter:
        try:
            gdf = None
            if pack_path.lower().endswith(".zip"):
                with zipfile.ZipFile(pack_path, "r") as zf:
                    candidates = [
                        n for n in zf.namelist() if n.lower().endswith(".geojson") or n.lower().endswith(".json")
                    ]
                    if not candidates:
                        continue
                    target_name = sorted(candidates, key=len)[0]
                    with zf.open(target_name) as f:
                        data = f.read()
                        gdf = gpd.read_file(io.BytesIO(data))
            else:
                gdf = gpd.read_file(pack_path)
            if gdf is None:
                continue
            for _, row in gdf.iterrows():
                equip_raw = row.get("equipment") or row.get("class") or row.get("label")
                equip_raw = equip_raw if isinstance(equip_raw, str) else None
                equip_guess = classify_equipment(equip_raw) if equip_raw else None
                target = equip_guess or (equip_raw if equip_raw in FORCED_GEOMETRY else None)
                if not target:
                    continue
                geom = row.geometry
                _collect_metric(target, geom, summaries, sources_for_layer, f"dlpack:{os.path.basename(pack_path)}")
        except Exception as exc:
            errors.append(f"{pack_path}: {exc}")

    merged: dict[str, EquipmentStatSummary] = {}
    for equip, rows in summaries.items():
        if not rows:
            continue
        med = lambda key: statistics.median([r[key] for r in rows if r.get(key) is not None])  # noqa: E731
        merged[equip] = EquipmentStatSummary(
            geom_type=rows[0].get("geom_type"),
            median_area=med("area") if any(r.get("area") for r in rows) else None,
            median_length=med("length") if any(r.get("length") for r in rows) else None,
            median_ratio=med("ratio") if any(r.get("ratio") for r in rows) else None,
            median_span=med("span") if any(r.get("span") for r in rows) else None,
            sample_count=len(rows),
            sources=tuple(sorted(sources_for_layer.get(equip, []))),
        )
    return merged, errors


@dataclass
class DLPrediction:
    """Single DL prediction to assist classification."""

    label: str
    geometry: any
    confidence: float = 1.0
    equipment_hint: str | None = None
    source: str = "vision"


@dataclass
class DLAssistConfig:
    enabled: bool
    confidence_threshold: float = 0.4
    max_snap_distance: float = 5.0


class DeepLearningAssist:
    """
    Optional DL hook for symbol detection + OCR. Annotation text remains the
    priority; DL is consulted only when text is missing/ambiguous.
    """

    def __init__(self, config: DLAssistConfig):
        self.config = config
        self.predictions: list[DLPrediction] = []
        self.ready: bool = False
        self.library_stats: dict[str, EquipmentStatSummary] = {}

    def load_predictions(self, drawings_files, base_crs, detection_packs=None, builtin_pack_path=None):
        if not self.config.enabled:
            return
        preds = []
        preds.extend(self._read_detection_packs(detection_packs or [], base_crs))
        if builtin_pack_path and os.path.exists(builtin_pack_path):
            preds.extend(self._read_detection_packs([builtin_pack_path], base_crs))
        preds.extend(self._detect_from_drawings(drawings_files, base_crs))
        self.predictions = preds or []
        self.ready = bool(self.predictions)

    def _read_detection_packs(self, pack_files, base_crs):
        collected: list[DLPrediction] = []
        for pf in pack_files:
            try:
                collected.extend(self._load_detection_pack(pf, base_crs))
            except Exception as exc:
                name = getattr(pf, "name", pf)
                st.warning(f"Could not read detection pack {name}: {exc}")
        return collected

    def _load_detection_pack(self, pack_file, base_crs):
        path = None
        display_name = getattr(pack_file, "name", None)
        cleanup = False
        if isinstance(pack_file, str):
            path = pack_file
            display_name = os.path.basename(pack_file)
        else:
            suffix = os.path.splitext(display_name or "pack")[1]
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(pack_file.getvalue())
            tmp.flush()
            path = tmp.name
            cleanup = True

        preds = []
        try:
            if path.lower().endswith(".zip"):
                with zipfile.ZipFile(path, "r") as zf:
                    candidates = [
                        n for n in zf.namelist() if n.lower().endswith(".geojson") or n.lower().endswith(".json")
                    ]
                    if not candidates:
                        return []
                    target = sorted(candidates, key=len)[0]
                    with zf.open(target) as f:
                        data = f.read()
                        gdf = gpd.read_file(io.BytesIO(data))
            else:
                gdf = gpd.read_file(path)

            for _, row in gdf.iterrows():
                geom = row.geometry
                if geom is None:
                    continue
                equip_raw = row.get("equipment") or row.get("class") or row.get("label")
                equip_hint = classify_equipment(equip_raw) if equip_raw else None
                conf = row.get("confidence") or row.get("score") or 1.0
                try:
                    conf = float(conf)
                except Exception:
                    conf = 1.0
                label = (
                    row.get("annotation")
                    or row.get("label")
                    or row.get("class")
                    or equip_raw
                    or "UNKNOWN"
                )
                preds.append(
                    DLPrediction(
                        label=str(label),
                        geometry=geom,
                        confidence=conf,
                        equipment_hint=equip_hint,
                        source=f"dlpack:{display_name}",
                    )
                )
        finally:
            if cleanup and path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
        return preds

    def _detect_from_drawings(self, drawings_files, base_crs):
        """
        Placeholder for DL inference. Plug your detector here.
        Must return a list of DLPrediction with geometries in the map CRS.
        """
        if not drawings_files:
            st.info("Deep learning assist is on but no drawings/PDFs were provided; skipping vision step.")
            return []

        vector_preds = []
        for uploaded in drawings_files:
            name_lower = uploaded.name.lower()
            if name_lower.endswith((".geojson", ".json")):
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1])
                tmp.write(uploaded.getvalue())
                tmp.flush()
                try:
                    gdf = gpd.read_file(tmp.name)
                    for _, row in gdf.iterrows():
                        geom = row.geometry
                        if geom is None:
                            continue
                        lbl = row.get("label") or row.get("class") or row.get("equipment")
                        conf = row.get("confidence") or row.get("score") or 1.0
                        try:
                            conf = float(conf)
                        except Exception:
                            conf = 1.0
                        equip_hint = classify_equipment(lbl) if lbl else None
                        vector_preds.append(
                            DLPrediction(
                                label=str(lbl) if lbl is not None else "UNKNOWN",
                                geometry=geom,
                                confidence=conf,
                                equipment_hint=equip_hint,
                                source="precomputed",
                            )
                        )
                except Exception as exc:
                    st.warning(f"Could not read DL detection file {uploaded.name}: {exc}")
                finally:
                    tmp.close()
        if vector_preds:
            return vector_preds

        st.warning(
            "Deep learning assist is enabled but no DL backend is wired in. "
            "Integrate your detector inside `DeepLearningAssist._detect_from_drawings` "
            "to consume raster drawings (PDF/TIFF/PNG/JPG) or DWG exports."
        )
        return []

    def _nearest_prediction(self, geom):
        best = None
        best_dist = None
        for pred in self.predictions:
            if pred.confidence < self.config.confidence_threshold:
                continue
            if pred.geometry is None:
                continue
            try:
                dist = geom.distance(pred.geometry)
            except Exception:
                continue
            if dist > self.config.max_snap_distance:
                continue
            if best is None or dist < best_dist:
                best, best_dist = pred, dist
        return best

    def suggest_equipment(self, geom):
        if not self.config.enabled or not self.predictions:
            # fall back to training library heuristics if available
            return self._library_guess(geom)
        if geom is None or geom.is_empty:
            return None
        pred = self._nearest_prediction(geom)
        if pred is None:
            return self._library_guess(geom)
        equip = pred.equipment_hint or classify_equipment(pred.label)
        if equip not in FORCED_GEOMETRY:
            return self._library_guess(geom)
        return equip

    def set_library_stats(self, stats: dict[str, EquipmentStatSummary]):
        self.library_stats = stats or {}

    def _library_guess(self, geom):
        """Heuristic guess using training geometry stats; very soft fallback."""
        return guess_from_library_stats(geom, self.library_stats)

    def bootstrap_from_layers(self, loaded_layers):
        """
        Build pseudo-detections from base geometries using training-library stats.
        Provides a zero-config heuristic detector when no DL backend or packs are present.
        """
        if not loaded_layers:
            return
        pseudo = []
        for layer_name, gdf in loaded_layers.items():
            if layer_name.lower() in ("annotation", "annotations"):
                continue
            for _, row in gdf.iterrows():
                geom = row.geometry
                guess = self._library_guess(geom) if self.library_stats else None
                if not guess and layer_name in FORCED_GEOMETRY and layer_name.upper() not in BASE_LAYERS:
                    guess = layer_name
                if not guess:
                    guess = classify_equipment(layer_name)
                if not guess:
                    continue
                pseudo.append(
                    DLPrediction(
                        label=f"AUTO-EST-{guess}",
                        geometry=geom.representative_point() if geom is not None else None,
                        confidence=0.35 if not self.library_stats else 0.5,
                        equipment_hint=guess,
                        source="library-bootstrap",
                    )
                )
        if pseudo:
            self.predictions.extend(pseudo)
            self.ready = True


dl_assist_config = DLAssistConfig(
    enabled=use_dl_assist,
    confidence_threshold=dl_conf_threshold,
    max_snap_distance=dl_max_snap_distance,
)
dl_assist = DeepLearningAssist(dl_assist_config)


def build_points_from_annotations(annotations, reference_schemas, base_crs, enable_classification=True):
    """
    Create per-equipment point features directly from annotation labels.
    Only builds layers whose forced geometry is Point; leaves polygons/lines to base data.
    """
    processed_records: dict[str, list] = {}
    processed_crs: dict[str, any] = {}
    built_layers: set[str] = set()
    telecom_best = None
    telecom_primary = False
    for _, row in annotations.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        label = row.get("__label__")
        target_layer = "ANNOTATION_POINTS"
        if enable_classification and label:
            classified = classify_equipment(label)
            if classified:
                target_layer = classified

        target_geom = FORCED_GEOMETRY.get(target_layer, "Point")
        # Skip non-point targets; they will be handled from the base geometries
        if target_geom != "Point":
            continue

        # Keep only one Telecom point; prefer explicit TELECOM label over ODF/SDH
        if target_layer == "TELECOM":
            label_upper = str(label).upper() if label else ""
            has_primary = "TELECOM" in label_upper
            if telecom_best is None or (has_primary and not telecom_primary):
                attrs = {field: None for field in reference_schemas.get(target_layer, [])}
                attrs["ANNOTATION_TEXT"] = label
                attrs["geometry"] = geom
                telecom_best = attrs
                telecom_primary = has_primary
            continue

        attrs = {field: None for field in reference_schemas.get(target_layer, [])}
        attrs["ANNOTATION_TEXT"] = label
        attrs["geometry"] = geom
        processed_records.setdefault(target_layer, []).append(attrs)
        processed_crs.setdefault(target_layer, annotations.crs or base_crs)
        built_layers.add(target_layer)

    if telecom_best is not None:
        processed_records.setdefault("TELECOM", []).append(telecom_best)
        processed_crs.setdefault("TELECOM", annotations.crs or base_crs)
        built_layers.add("TELECOM")
    return processed_records, processed_crs, built_layers


@st.cache_data(show_spinner=False)
def load_reference_schemas(sample_dir: str):
    """Read attribute schemas and example counts from the sample GPKG folder."""
    schemas = {}
    counts = {}
    geoms = {}
    errors = []
    fallback_layers = set()

    if not os.path.isdir(sample_dir):
        return schemas, counts, geoms, ["Sample GPKG folder not found."], fallback_layers

    for fname in sorted(os.listdir(sample_dir)):
        if not fname.lower().endswith(".gpkg"):
            continue
        file_stem = os.path.splitext(fname)[0]
        path = os.path.join(sample_dir, fname)
        try:
            layers = fiona.listlayers(path)
        except Exception as exc:
            errors.append(f"{fname}: {exc}")
            fallback_layers.add(file_stem)
            continue
        if not layers:
            fallback_layers.add(file_stem)
            continue

        for layer in layers:
            fallback_layers.add(layer)
            try:
                with fiona.open(path, layer=layer) as src:
                    schemas[layer] = list(src.schema.get("properties", {}).keys())
                    counts[layer] = len(src)
                    geoms[layer] = [shapely_shape(feat["geometry"]) for feat in src if feat.get("geometry")]
            except Exception as exc:
                errors.append(f"{fname} ({layer}): {exc}")
                fallback_layers.add(file_stem)

    return schemas, counts, geoms, errors, fallback_layers


def load_learning_counts(learning_dir: str, extra_files=None):
    """Harvest expected equipment counts from learning Excel sheets."""
    counts: dict[str, int] = {}
    errors: list[str] = []
    cleanup: list[str] = []

    def iter_sources():
        if os.path.isdir(learning_dir):
            for root, _, files in os.walk(learning_dir):
                for fname in files:
                    lower = fname.lower()
                    if lower.endswith(".xlsx") or lower.endswith(".xls"):
                        yield os.path.join(root, fname)
        if extra_files:
            for f in extra_files:
                if isinstance(f, str):
                    yield f
                    continue
                name = getattr(f, "name", "sheet")
                suffix = os.path.splitext(name)[1] or ".xlsx"
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp.write(f.getvalue())
                tmp.flush()
                cleanup.append(tmp.name)
                yield tmp.name

    for path in iter_sources():
        try:
            xl = pd.ExcelFile(path)
            for sheet in xl.sheet_names:
                df = xl.parse(sheet)
                if df.empty:
                    continue
                series = df.iloc[:, 0].dropna()
                for device_name, c in series.value_counts().items():
                    equip = map_device_to_equipment(device_name)
                    if not equip:
                        continue
                    try:
                        c_int = int(c)
                    except Exception:
                        continue
                    counts[equip] = max(counts.get(equip, 0), c_int)
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    for path in cleanup:
        try:
            os.remove(path)
        except Exception:
            pass
    return counts, errors


(
    reference_schemas,
    reference_counts_raw,
    reference_geoms_raw,
    schema_errors,
    sample_layers_fallback,
) = load_reference_schemas(SAMPLE_GPKG_DIR)
# Optional minimum counts only when alignment is enabled
MIN_COUNTS = {"LIGHTNING_ARRESTOR": 6, "VOLTAGE_TRANSFORMER": 6}
reference_counts = reference_counts_raw.copy()
reference_geoms = reference_geoms_raw if use_reference_alignment else {}
if use_reference_alignment:
    for k, v in MIN_COUNTS.items():
        reference_counts[k] = max(v, reference_counts.get(k, 0))
else:
    reference_counts = {}
if reference_schemas:
    st.info(
        f"Loaded reference attribute schemas for {len(reference_schemas)} equipment layer(s)."
    )
    st.write(sorted(reference_schemas.keys()))
    st.caption("Reference sample counts: " + ", ".join(f"{k}:{reference_counts.get(k, '?')}" for k in sorted(reference_schemas.keys())))
learning_counts, learning_count_errors = load_learning_counts(LEARNING_DIR)
uploaded_learning_counts, uploaded_learning_errors = load_learning_counts(
    LEARNING_DIR, extra_files=learning_sheet_uploads
)
# Merge counts preferring explicit uploads
for k, v in uploaded_learning_counts.items():
    learning_counts[k] = max(learning_counts.get(k, 0), v)
learning_count_errors.extend(uploaded_learning_errors)
if learning_counts:
    st.info(
        f"Learning sheet expected counts loaded for {len(learning_counts)} equipment type(s): "
        + ", ".join(f"{k}:{v}" for k, v in sorted(learning_counts.items()))
    )
if learning_count_errors:
    with st.expander("Learning sheet warnings"):
        for msg in learning_count_errors:
            st.write(msg)

if schema_errors:
    with st.expander("Reference schema warnings"):
        for msg in schema_errors:
            st.write(msg)

# ================================================================
# File upload
# ================================================================
uploaded = st.file_uploader("Upload zipped .gdb", type=["zip"])
if not uploaded:
    st.stop()

temp_dir = tempfile.mkdtemp()
zip_path = os.path.join(temp_dir, "input.zip")

with open(zip_path, "wb") as f:
    f.write(uploaded.getvalue())

with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(temp_dir)

# Find actual .gdb directory
gdb_dir = None
for root, dirs, _ in os.walk(temp_dir):
    for d in dirs:
        if d.lower().endswith(".gdb"):
            gdb_dir = os.path.join(root, d)
            break
    if gdb_dir:
        break

if not gdb_dir:
    st.error("No .gdb found inside the ZIP. Upload must contain the full folder.")
    st.stop()

st.success(f"Loaded GDB folder: {gdb_dir}")

# ================================================================
# Load Layers
# ================================================================
layers = fiona.listlayers(gdb_dir)
st.subheader("Layers Found in GDB")
st.write(layers)

base_crs = None
loaded_layers = {}

for layer in layers:
    target_geom_hint = FORCED_GEOMETRY.get(layer, "Original")
    gdf = load_layer_with_fallback(gdb_dir, layer, target_geom_hint)
    if gdf is None or gdf.empty:
        continue
    loaded_layers[layer] = gdf
    if base_crs is None and gdf.crs:
        base_crs = gdf.crs

if base_crs is None:
    st.warning("No CRS found in the source data. Output will have no CRS.")

training_stats_global: dict[str, EquipmentStatSummary] = {}
training_errors_global: list[str] = []
library_dirs = (
    tuple(d for d in learning_dirs_available if os.path.isdir(d))
    if use_learning_library
    else tuple(d for d in (TRAINING_DATA_DIR,) if os.path.isdir(d))
)
extra_packs: list[str] = []
if os.path.exists(DEFAULT_DL_PACK):
    extra_packs.append(DEFAULT_DL_PACK)
if os.path.exists(DEFAULT_DL_PACK_REPO):
    extra_packs.append(DEFAULT_DL_PACK_REPO)
if library_dirs:
    training_stats_global, training_errors_global = load_training_library(
        library_dirs, extra_packs=tuple(extra_packs)
    )
    if training_stats_global:
        st.info(
            f"Learning library loaded ({sum(s.sample_count for s in training_stats_global.values())} samples across {len(training_stats_global)} equipment types)."
        )
        with st.expander("Learning data summary"):
            for equip, stat in sorted(training_stats_global.items()):
                st.write(
                    f"{equip}: samples={stat.sample_count}, span~{stat.median_span}, ratio~{stat.median_ratio}, geom={stat.geom_type} (sources: {', '.join(stat.sources)})"
                )
if training_errors_global:
    with st.expander("Learning data warnings"):
        for msg in training_errors_global:
            st.write(msg)

if dl_assist.config.enabled:
    builtin_pack = None
    for candidate in (DEFAULT_DL_PACK, DEFAULT_DL_PACK_REPO):
        if candidate and os.path.exists(candidate):
            builtin_pack = candidate
            break
    dl_assist.set_library_stats(training_stats_global)
    if training_stats_global and not use_learning_library:
        st.info(
            f"DL assist using {sum(s.sample_count for s in training_stats_global.values())} learned samples across {len(training_stats_global)} equipment types."
        )
    dl_assist.load_predictions(
        dl_drawings,
        base_crs,
        detection_packs=dl_detection_packs,
        builtin_pack_path=builtin_pack,
    )
    if not dl_assist.ready and training_stats_global:
        dl_assist.bootstrap_from_layers(loaded_layers)
    if dl_assist.ready:
        st.info(f"Loaded {len(dl_assist.predictions)} DL detection(s) for vision assist.")
    elif dl_drawings or dl_detection_packs:
        st.warning(
            "DL assist enabled but no detections were produced. "
            "Wire your detector in `DeepLearningAssist._detect_from_drawings`."
        )

dl_pred_export_dir = None
dl_pred_zip_path = None
if dl_assist.config.enabled and dl_assist.ready:
    dl_pred_export_dir = os.path.join(temp_dir, "DLPredictions")
    dl_pred_dir = export_dl_predictions(dl_assist.predictions, base_crs, dl_pred_export_dir)
    if dl_pred_dir:
        dl_pred_zip_path = os.path.join(temp_dir, "DLPredictions.zip")
        if os.path.exists(dl_pred_zip_path):
            os.remove(dl_pred_zip_path)
        with zipfile.ZipFile(dl_pred_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(dl_pred_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    arc_name = os.path.relpath(full_path, dl_pred_dir)
                    zf.write(full_path, arcname=arc_name)
        with open(dl_pred_zip_path, "rb") as f:
            st.download_button(
                label="Download DL Detections (GeoJSON)",
                data=f,
                file_name="DLPredictions.zip",
                mime="application/zip",
            )

# Pull annotation layer (if any) for name attachment
annotation_layer_name = None
for candidate in loaded_layers.keys():
    if candidate.lower() in ("annotation", "annotations"):
        annotation_layer_name = candidate
        break

annotation_points = (
    extract_annotation_points(loaded_layers.get(annotation_layer_name))
    if use_annotations and annotation_layer_name
    else None
)
if annotation_points is not None:
    st.info(
        f"Using annotation layer '{annotation_layer_name}' to attach names within {max_annotation_distance} map units."
    )

# ================================================================
# Process each layer separately
# ================================================================
st.subheader("Processing Layers")

processed_records: dict[str, list] = {}
processed_crs: dict[str, any] = {}

built_point_layers: set[str] = set()
if use_annotations and annotation_points is not None:
    processed_records, processed_crs, built_point_layers = build_points_from_annotations(
        annotation_points, reference_schemas, base_crs, use_annotation_classification
    )

for layer_name, gdf in loaded_layers.items():
    layer_crs = gdf.crs or base_crs

    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue

        label, label_point = attach_annotation_label(geom, annotation_points)

        target_layer = map_layer_to_equipment(layer_name) or layer_name
        src_type = geom.geom_type
        classified = None
        if use_annotation_classification and label:
            classified = classify_equipment(label)
            if classified:
                target_layer = classified

        if dl_assist.config.enabled and classified is None:
            dl_guess = dl_assist.suggest_equipment(geom)
            if dl_guess:
                target_layer = dl_guess
        if classified is None and use_learning_library and training_stats_global:
            lib_guess = guess_from_library_stats(geom, training_stats_global)
            if lib_guess:
                target_layer = lib_guess

        target_geom = FORCED_GEOMETRY.get(target_layer, "Original")

        # Skip base geometries for point layers already built from annotations
        if target_geom == "Point" and target_layer in built_point_layers:
            continue

        # Force geometry type
        if target_geom == "Point":
            if label_point is not None:
                geom = label_point
                src_type = "Point"
            elif src_type != "Point":
                geom = geom.representative_point()
        elif target_geom == "Line":
            if src_type in ["Polygon", "MultiPolygon"]:
                geom = geom.boundary
            elif src_type == "Point":
                continue  # cannot convert point to line
        elif target_geom == "Polygon":
            if src_type not in ["Polygon", "MultiPolygon"]:
                geom = geom.buffer(0.1)

        reference_fields = reference_schemas.get(target_layer)
        if reference_fields is not None:
            attrs = {field: row.get(field, None) for field in reference_fields}
        else:
            attrs = {col: row[col] for col in gdf.columns if col != "geometry"}

        if label is not None:
            attrs["ANNOTATION_TEXT"] = label

        attrs["geometry"] = geom
        # Respect sample reference counts if available to prevent runaway duplicates
        ref_cap = reference_counts.get(target_layer) if use_reference_alignment else None
        learning_cap = learning_counts.get(target_layer) if learning_counts else None
        cap_candidates = [c for c in (ref_cap, learning_cap) if c is not None]
        cap_limit = min(cap_candidates) if cap_candidates else None
        current_len = len(processed_records.get(target_layer, []))
        if cap_limit is not None and current_len >= cap_limit:
            continue
        processed_records.setdefault(target_layer, []).append(attrs)
        processed_crs.setdefault(target_layer, layer_crs)

processed = {
    name: gpd.GeoDataFrame(records, crs=processed_crs.get(name, base_crs))
    for name, records in processed_records.items()
    if records
}

# Drop base CAD layers from export
processed = {k: v for k, v in processed.items() if k not in BASE_LAYERS}

# Fill deficits using base geometries when needed
if use_reference_alignment:
    processed = fill_indor_switchgear_table(
        processed, loaded_layers, reference_schemas, reference_counts
    )
    processed = fill_disconnector_switch(
        processed, loaded_layers, annotation_points, reference_schemas, reference_counts
    )
    processed = fill_lightning_arrestor(
        processed, loaded_layers, annotation_points, reference_schemas, reference_counts
    )
    processed = fill_voltage_transformer(
        processed, loaded_layers, annotation_points, reference_schemas, reference_counts
    )
    processed = snap_to_reference(processed, reference_geoms)

# Enforce learning sheet count caps even without reference alignment
processed = apply_count_caps(processed, learning_counts)

if processed:
    st.success("All layers processed successfully.")
else:
    st.warning("No records found to process.")
    if reference_schemas or sample_layers_fallback:
        if reference_schemas:
            st.info("Export will include empty layers using the reference sample schemas.")
        else:
            st.info("Sample GPKG placeholders will be exported even though schemas could not be read.")
    else:
        st.stop()

# ================================================================
# Export GPKG (one layer per equipment)
# ================================================================
st.subheader("Export Cleaned GPKG")

layers_dir = os.path.join(temp_dir, "CleanEquipmentLayers")
if os.path.exists(layers_dir):
    shutil.rmtree(layers_dir)
os.makedirs(layers_dir, exist_ok=True)

target_layers = {
    name
    for name in set(processed.keys())
    .union(reference_schemas.keys())
    .union(sample_layers_fallback or set())
    if name not in BASE_LAYERS
}
if not target_layers:
    st.error("No layers available for export.")
    st.stop()

reference_geoms_lookup = reference_geoms_raw or reference_geoms or {}
st.caption(
    f"Preparing export for {len(target_layers)} layer(s); empty layers will be included using sample schemas when available."
)

for layer_name in sorted(target_layers):
    gdf_processed = processed.get(layer_name)
    layer_crs = getattr(gdf_processed, "crs", base_crs)
    fname = safe_filename(layer_name) + ".gpkg"
    out_path = os.path.join(layers_dir, fname)
    if os.path.exists(out_path):
        os.remove(out_path)
    if gdf_processed is not None and not gdf_processed.empty:
        gdf_processed.to_file(out_path, layer=layer_name, driver="GPKG")
    else:
        fields = reference_schemas.get(layer_name, [])
        geom_type = infer_geometry_type(layer_name, gdf_processed, reference_geoms_lookup)
        export_empty_layer(out_path, layer_name, geom_type, fields, layer_crs)

zip_path = os.path.join(temp_dir, "CleanEquipmentLayers.zip")
if os.path.exists(zip_path):
    os.remove(zip_path)

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for root, _, files in os.walk(layers_dir):
        for file in files:
            full_path = os.path.join(root, file)
            arc_name = os.path.relpath(full_path, layers_dir)
            zf.write(full_path, arcname=arc_name)

with open(zip_path, "rb") as f:
    st.download_button(
        label="Download CleanEquipmentLayers.zip",
        data=f,
        file_name="CleanEquipmentLayers.zip",
        mime="application/zip",
    )

st.caption("Output contains one cleaned GPKG per equipment, zipped for download.")

# ================================================================
# Optional: Export DL training pack from this upload
# ================================================================
dl_pack_dir = os.path.join(temp_dir, "DLTrainingPack")
dl_pack_path = build_dl_training_pack(processed, base_crs, dl_pack_dir)
if dl_pack_path:
    dl_zip_path = os.path.join(temp_dir, "DLTrainingPack.zip")
    if os.path.exists(dl_zip_path):
        os.remove(dl_zip_path)
    with zipfile.ZipFile(dl_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(dl_pack_dir):
            for file in files:
                full_path = os.path.join(root, file)
                arc_name = os.path.relpath(full_path, dl_pack_dir)
                zf.write(full_path, arcname=arc_name)
    with open(dl_zip_path, "rb") as f:
        st.download_button(
            label="Download DL Training Pack (GeoJSON + metadata)",
            data=f,
            file_name="DLTrainingPack.zip",
            mime="application/zip",
        )
