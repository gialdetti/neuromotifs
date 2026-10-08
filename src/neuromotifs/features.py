"""Pairwise features: the geometry of every ordered neuron pair, as the paper's models see it.

Two stages, as in the original pipeline: the *positions* table lists every ordered pair
``(from, to)``, ``from != to``, with the Cartesian coordinates of both endpoints; the *geometry*
enrichment adds the offset ``to - from``, its vertical sign, and the spherical coordinates of the
offset and of the source.

Positions are in the model frame (``z`` vertical). Spherical coordinates are radius ``rho``,
azimuth ``az = atan2(y, x)`` and inclination ``el = atan2(hypot(x, y), z)`` from the ``z`` axis.
``from_rho, from_az, from_el`` are taken about the circuit's origin, as in the paper, so the
position-dependent model is not translation-invariant.
"""

import numpy as np
import pandas as pd

XYZ = ["x", "y", "z"]


def pairwise_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    """``(from, to)`` indices of all ``n * (n - 1)`` ordered pairs, row-major, diagonal excluded.

    ``M[from, to] = column`` scatters a per-pair column back into an ``n x n`` matrix.
    """
    return np.nonzero(~np.eye(n, dtype=bool))


def pairwise_features(positions) -> pd.DataFrame:
    """The full feature table of all ordered pairs: the positions table, enriched with geometry."""
    f = pairwise_positions(positions)
    f = enrich_pairwise_geometry(f)
    return f


def pairwise_positions(positions) -> pd.DataFrame:
    """All ordered pairs with their endpoint coordinates.

    Columns: ``from, to, from_x, from_y, from_z, to_x, to_y, to_z``.
    """
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must have shape (n, 3), got {positions.shape}")
    if not np.isfinite(positions).all():
        raise ValueError("positions must be finite")

    i, j = pairwise_indices(len(positions))
    xyz = pd.DataFrame(positions, columns=XYZ)
    return (
        pd.DataFrame({"from": i, "to": j})
        .join(xyz.add_prefix("from_"), on="from")
        .join(xyz.add_prefix("to_"), on="to")
    )


def enrich_pairwise_geometry(pairs: pd.DataFrame) -> pd.DataFrame:
    """Add the offset and the spherical coordinates to a positions table.

    Adds ``dx, dy, dz, sign_dz``, ``offset_rho, offset_az, offset_el`` and
    ``from_rho, from_az, from_el``.
    """
    source, target = _xyz(pairs, "from"), _xyz(pairs, "to")
    offset = target - source
    return pairs.assign(
        dx=offset[:, 0],
        dy=offset[:, 1],
        dz=offset[:, 2],
        sign_dz=np.sign(offset[:, 2]),
        **spherical_coordinates(offset, prefix="offset"),
        **spherical_coordinates(source, prefix="from"),
    )


def spherical_coordinates(xyz: np.ndarray, prefix: str) -> dict[str, np.ndarray]:
    """Columns ``{prefix}_rho, {prefix}_az, {prefix}_el`` of the rows ``(x, y, z)``."""
    x, y, z = np.asarray(xyz, dtype=float).T
    r_xy = np.hypot(x, y)
    return {
        f"{prefix}_rho": np.hypot(r_xy, z),
        f"{prefix}_az": np.arctan2(y, x),
        f"{prefix}_el": np.arctan2(r_xy, z),
    }


def _xyz(pairs: pd.DataFrame, prefix: str) -> np.ndarray:
    return pairs[[f"{prefix}_{c}" for c in XYZ]].to_numpy()
