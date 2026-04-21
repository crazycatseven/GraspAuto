"""Topology-aware MANO vertex grouping for the v6.0 LoST tokenizer."""

from __future__ import annotations

import pickle
import warnings
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

GROUP_NAMES = ("thumb", "index", "middle", "ring", "little", "palm")
GROUP_ID_BY_NAME = {name: idx for idx, name in enumerate(GROUP_NAMES)}

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MANO_MODEL_PATH = PROJECT_ROOT / "assets" / "mano_v1_2" / "models" / "MANO_RIGHT.pkl"

# MANO right-hand kinematic order in the shipped model:
#   0 wrist
#   1-3 index, 4-6 middle, 7-9 ring, 10-12 little, 13-15 thumb
_JOINT_TO_GROUP_ID = {
    0: GROUP_ID_BY_NAME["palm"],
    1: GROUP_ID_BY_NAME["index"],
    2: GROUP_ID_BY_NAME["index"],
    3: GROUP_ID_BY_NAME["index"],
    4: GROUP_ID_BY_NAME["middle"],
    5: GROUP_ID_BY_NAME["middle"],
    6: GROUP_ID_BY_NAME["middle"],
    7: GROUP_ID_BY_NAME["ring"],
    8: GROUP_ID_BY_NAME["ring"],
    9: GROUP_ID_BY_NAME["ring"],
    10: GROUP_ID_BY_NAME["little"],
    11: GROUP_ID_BY_NAME["little"],
    12: GROUP_ID_BY_NAME["little"],
    13: GROUP_ID_BY_NAME["thumb"],
    14: GROUP_ID_BY_NAME["thumb"],
    15: GROUP_ID_BY_NAME["thumb"],
}


def _load_mano_arrays(mano_pkl_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with mano_pkl_path.open("rb") as handle:
            mano_data = pickle.load(handle, encoding="latin1")
    faces = np.asarray(mano_data["f"], dtype=np.int64)
    weights = np.asarray(mano_data["weights"], dtype=np.float32)
    return faces, weights


def _build_face_adjacency(num_vertices: int, faces: np.ndarray) -> list[list[int]]:
    neighbors = [set() for _ in range(num_vertices)]
    for tri in faces:
        a, b, c = (int(v) for v in tri)
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    return [sorted(v) for v in neighbors]


def _connected_components(vertices: np.ndarray, adjacency: Sequence[Sequence[int]]) -> list[np.ndarray]:
    if vertices.size == 0:
        return []

    allowed = set(int(v) for v in vertices.tolist())
    unseen = set(allowed)
    components: list[np.ndarray] = []
    while unseen:
        start = unseen.pop()
        queue = [start]
        component = [start]
        for current in queue:
            for neighbor in adjacency[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
                    component.append(neighbor)
        components.append(np.asarray(sorted(component), dtype=np.int64))
    return components


def _pick_reassignment_label(
    component: np.ndarray,
    labels: np.ndarray,
    adjacency: Sequence[Sequence[int]],
    weights: np.ndarray,
    current_group_id: int,
) -> int:
    boundary_labels: list[int] = []
    for vertex in component.tolist():
        for neighbor in adjacency[int(vertex)]:
            neighbor_label = int(labels[neighbor])
            if neighbor_label != current_group_id:
                boundary_labels.append(neighbor_label)

    if boundary_labels:
        return Counter(boundary_labels).most_common(1)[0][0]

    mean_joint_weights = weights[component].mean(axis=0)
    for joint_id in np.argsort(mean_joint_weights)[::-1].tolist():
        group_id = _JOINT_TO_GROUP_ID[int(joint_id)]
        if group_id != current_group_id:
            return group_id
    return GROUP_ID_BY_NAME["palm"]


def _repair_disconnected_groups(
    labels: np.ndarray,
    adjacency: Sequence[Sequence[int]],
    weights: np.ndarray,
) -> np.ndarray:
    repaired = labels.copy()
    for _ in range(8):
        changed = False
        for group_id in range(len(GROUP_NAMES)):
            vertices = np.flatnonzero(repaired == group_id)
            if vertices.size == 0:
                raise RuntimeError(f"MANO group {GROUP_NAMES[group_id]!r} became empty.")
            components = _connected_components(vertices, adjacency)
            if len(components) <= 1:
                continue

            largest_idx = int(np.argmax([component.size for component in components]))
            for comp_idx, component in enumerate(components):
                if comp_idx == largest_idx:
                    continue
                new_group_id = _pick_reassignment_label(
                    component=component,
                    labels=repaired,
                    adjacency=adjacency,
                    weights=weights,
                    current_group_id=group_id,
                )
                repaired[component] = new_group_id
                changed = True
        if not changed:
            break

    for group_id in range(len(GROUP_NAMES)):
        vertices = np.flatnonzero(repaired == group_id)
        components = _connected_components(vertices, adjacency)
        if len(components) != 1:
            raise RuntimeError(
                f"Failed to derive a connected MANO topology group for {GROUP_NAMES[group_id]!r}."
            )
    return repaired


@lru_cache(maxsize=4)
def _cached_group_arrays(mano_pkl_path: str) -> tuple[tuple[np.ndarray, ...], tuple[int, ...]]:
    resolved = Path(mano_pkl_path).expanduser().resolve()
    faces, weights = _load_mano_arrays(resolved)
    dominant_joint = weights.argmax(axis=1)
    labels = np.asarray([_JOINT_TO_GROUP_ID[int(joint)] for joint in dominant_joint], dtype=np.int64)
    adjacency = _build_face_adjacency(num_vertices=weights.shape[0], faces=faces)
    labels = _repair_disconnected_groups(labels=labels, adjacency=adjacency, weights=weights)

    group_arrays = tuple(np.flatnonzero(labels == group_id).astype(np.int64) for group_id in range(len(GROUP_NAMES)))
    group_sizes = tuple(int(group.size) for group in group_arrays)
    return group_arrays, group_sizes


def build_vertex_groups(mano_pkl_path: str | Path = MANO_MODEL_PATH) -> Dict[str, np.ndarray]:
    """Return semantic MANO vertex groups keyed by finger/palm name."""
    groups, _ = _cached_group_arrays(str(Path(mano_pkl_path)))
    return {name: groups[idx].copy() for idx, name in enumerate(GROUP_NAMES)}


def build_vertex_group_tensors(
    mano_pkl_path: str | Path = MANO_MODEL_PATH,
    device: torch.device | str | None = None,
) -> Dict[str, torch.Tensor]:
    """Return semantic MANO vertex groups as ``torch.long`` tensors."""
    groups = build_vertex_groups(mano_pkl_path=mano_pkl_path)
    return {
        name: torch.as_tensor(indices, dtype=torch.long, device=device)
        for name, indices in groups.items()
    }


def build_vertex_group_masks(
    mano_pkl_path: str | Path = MANO_MODEL_PATH,
    device: torch.device | str | None = None,
) -> Dict[str, torch.Tensor]:
    """Return boolean ``(778,)`` masks for each semantic group."""
    group_tensors = build_vertex_group_tensors(mano_pkl_path=mano_pkl_path, device=device)
    masks: Dict[str, torch.Tensor] = {}
    for name, indices in group_tensors.items():
        mask = torch.zeros(778, dtype=torch.bool, device=indices.device)
        mask[indices] = True
        masks[name] = mask
    return masks


def stacked_vertex_group_mask(
    mano_pkl_path: str | Path = MANO_MODEL_PATH,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the stacked semantic mask tensor with shape ``(6, 778)``."""
    masks = build_vertex_group_masks(mano_pkl_path=mano_pkl_path, device=device)
    return torch.stack([masks[name] for name in GROUP_NAMES], dim=0)


def vertex_group_sizes(mano_pkl_path: str | Path = MANO_MODEL_PATH) -> Dict[str, int]:
    """Return the number of MANO vertices assigned to each semantic group."""
    _, sizes = _cached_group_arrays(str(Path(mano_pkl_path)))
    return {name: int(sizes[idx]) for idx, name in enumerate(GROUP_NAMES)}
