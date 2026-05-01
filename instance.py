"""Dataclasses for the polling place location problem."""
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
import pickle


@dataclass
class Instance:
    """A procedurally generated polling place location problem instance.

    Geometry:
        bounds: (xmin, ymin, xmax, ymax) of the county, in km.
        precinct_label_grid: (G, G) integer raster; each cell labeled with its precinct id (0..N-1).
        grid_xs, grid_ys: 1D arrays of length G with x/y coordinates of grid cell centers.
        precinct_centroids: (N, 2) centroid of each precinct in km.
        precinct_areas: (N,) area in km^2.
        precinct_voters: (N,) registered voter count per precinct (integer).

    Sites and distances:
        site_locations: (M, 2) candidate site locations.
        site_capacity: (M,) capacity in voters.
        site_types: list of M strings.
        distance_matrix: (N, M) precinct-to-site Euclidean distance.

    Optimization parameters:
        K: budget on opened sites.

    Metadata:
        seed, params: for reproducibility.
    """
    bounds: Tuple[float, float, float, float]
    precinct_label_grid: np.ndarray
    grid_xs: np.ndarray
    grid_ys: np.ndarray
    precinct_centroids: np.ndarray
    precinct_areas: np.ndarray
    precinct_voters: np.ndarray
    site_locations: np.ndarray
    site_capacity: np.ndarray
    site_types: List[str]
    distance_matrix: np.ndarray
    K: int
    seed: int
    params: Dict[str, Any]

    def __setstate__(self, state):
        """Restore from pickle. Older pickles may carry deprecated fields
        (river_polyline, bridges, distance_matrix_raw, precinct_demographics,
        landmarks, lakes, highway_polyline, zones) — silently discard them
        rather than blow up; the dataclass has no slots for them anymore."""
        deprecated = {"landmarks", "lakes", "highway_polyline", "zones",
                       "river_polyline", "bridges", "distance_matrix_raw",
                       "precinct_demographics"}
        clean = {k: v for k, v in state.items() if k not in deprecated}
        self.__dict__.update(clean)

    @property
    def n_precincts(self) -> int:
        return len(self.precinct_centroids)

    @property
    def n_sites(self) -> int:
        return len(self.site_locations)

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> 'Instance':
        with open(path, 'rb') as f:
            return pickle.load(f)


@dataclass
class Solution:
    """A solution to a polling place instance.

    x: (M,) binary, 1 if site opened.
    y: (N, M) binary, y[i,j]=1 if precinct i assigned to site j.
    objective: total voter-weighted distance.
    solver_status: e.g. 'heuristic-greedy+local-search' or 'optimal'.
    metadata: extra solver-specific info.
    """
    x: np.ndarray
    y: np.ndarray
    objective: float
    solver_status: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> 'Solution':
        with open(path, 'rb') as f:
            return pickle.load(f)
