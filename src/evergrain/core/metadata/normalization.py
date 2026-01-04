import re
from collections import defaultdict
from typing import List
from evergrain.core.models.metadata import MetadataRow


def normalize_metadata(rows: List[MetadataRow]) -> List[MetadataRow]:
    """Apply all normalization steps to metadata rows."""
    rows = _normalize_years(rows)
    rows = _resolve_cluster_names(rows)
    rows = _assign_undeclared_clusters(rows)
    rows = _assign_ids(rows)
    return rows


# ----------------------------
# Private Helper Functions
# ----------------------------


def _normalize_years(rows: List[MetadataRow]) -> List[MetadataRow]:
    """Normalize 2-digit years to 4-digit years (e.g., 99 → 1999, 23 → 2023)."""
    for row in rows:
        if row.year is not None and 0 <= row.year < 100:
            row.year += 2000 if row.year < 50 else 1900
    return rows


def _resolve_cluster_names(rows: List[MetadataRow]) -> List[MetadataRow]:
    """Normalize cluster names for typo tolerance (e.g., 'tC12' → 'tightCluster12')."""
    for row in rows:
        cluster_raw = (row.Cluster or "").strip()
        if not cluster_raw:
            continue
        cluster_lower = cluster_raw.lower()
        match = re.search(r"\d+", cluster_lower)
        if not match:
            raise ValueError(f"Row {row.row_num}: Invalid Cluster='{cluster_raw}', no numeric suffix.")
        number = match.group()
        if re.match(r"t.*c.*\d+", cluster_lower):
            row.Cluster = f"tightCluster{number}"
        elif re.match(r"l.*c.*\d+", cluster_lower):
            row.Cluster = f"looseCluster{number}"
        else:
            raise ValueError(f"Row {row.row_num}: Invalid Cluster='{cluster_raw}'.")
    return rows


def _assign_undeclared_clusters(rows: List[MetadataRow]) -> List[MetadataRow]:
    """Assign 'tightClusterN' to rows with missing Cluster within each (Event, Scene) group."""
    scenes = defaultdict(list)
    for row in rows:
        if row.Event and row.Scene:
            scenes[(row.Event, row.Scene)].append(row)

    for scene_rows in scenes.values():
        used_indices = set()
        unclustered = []
        for row in scene_rows:
            if row.Cluster and isinstance(row.Cluster, str) and row.Cluster.startswith("tightCluster"):
                try:
                    idx = int(row.Cluster[12:])
                    used_indices.add(idx)
                except (ValueError, IndexError):
                    pass
            elif row.Cluster is None:
                unclustered.append(row)
        next_idx = 1
        for row in unclustered:
            while next_idx in used_indices:
                next_idx += 1
            row.Cluster = f"tightCluster{next_idx}"
            used_indices.add(next_idx)
            next_idx += 1
    return rows


# ID assignment functions

TEMP_EVENT_PREFIX = "__TEMP_EVENT_"
TEMP_SCENE_PREFIX = "__TEMP_SCENE_"
TEMP_CLUSTER_PREFIX = "__TEMP_CLUSTER_"


def _fill_none_with_temporaries(rows: List[MetadataRow]) -> List[MetadataRow]:
    """
    Replace None values in Event, Scene, Cluster with temporary unique strings.
    Uses global TEMP_*_PREFIX constants.
    """
    temp_event_counter = 1
    temp_scene_counter = 1
    temp_cluster_counter = 1

    for row in rows:
        # Replace None Event with temporary
        if row.Event is None:
            row.Event = f"{TEMP_EVENT_PREFIX}{temp_event_counter}"
            temp_event_counter += 1

        # Replace None Scene with temporary
        if row.Scene is None:
            row.Scene = f"{TEMP_SCENE_PREFIX}{temp_scene_counter}"
            temp_scene_counter += 1

        # Replace None Cluster with temporary
        if row.Cluster is None:
            row.Cluster = f"{TEMP_CLUSTER_PREFIX}{temp_cluster_counter}"
            temp_cluster_counter += 1

    return rows


def _cleanup_temporaries(rows: List[MetadataRow]) -> List[MetadataRow]:
    """
    Restore temporary strings back to None.
    Uses global TEMP_*_PREFIX constants.
    """
    for row in rows:
        # Restore Event
        if row.Event and row.Event.startswith(TEMP_EVENT_PREFIX):
            row.Event = None

        # Restore Scene
        if row.Scene and row.Scene.startswith(TEMP_SCENE_PREFIX):
            row.Scene = None

        # Restore Cluster
        if row.Cluster and row.Cluster.startswith(TEMP_CLUSTER_PREFIX):
            row.Cluster = None

    return rows


def _assign_ids(rows: List[MetadataRow]) -> List[MetadataRow]:
    """Assign IDs and unique_group_id based on Event, Scene, and Cluster names."""

    rows = _fill_none_with_temporaries(rows)

    event_counter = defaultdict(lambda: len(event_counter) + 1)
    scene_counter = defaultdict(lambda: len(scene_counter) + 1)
    cluster_counter = defaultdict(lambda: len(cluster_counter) + 1)

    for row in rows:
        # Assign Event ID
        if row.Event is not None:
            key = row.Event.lower().strip()
            if key:
                row.event_id = event_counter[key]

        # Assign Scene ID
        if row.Scene is not None:
            key = row.Scene.lower().strip()
            if key:
                row.scene_id = scene_counter[key]

        # Assign Cluster ID
        if row.Cluster is not None:
            key = row.Cluster.lower().strip()
            if key:
                row.cluster_id = cluster_counter[key]

        # Create unique_group_id
        if row.event_id is not None and row.scene_id is not None and row.cluster_id is not None:
            row.unique_group_id = f"{row.event_id:03d}-{row.scene_id:03d}-{row.cluster_id:03d}"

    rows = _cleanup_temporaries(rows)

    return rows
