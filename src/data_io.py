# data_io.py - helpers for reading the raw PathSnap files
import os
import numpy as np
import pandas as pd
from pyproj import Transformer

EXTENSIONS = (".track", ".nodes", ".arcs", ".route")


def trip_id_of(root, folder, filename):
    """The trip id is the name of the folder holding the file.
    If the file sits directly in the partition folder, use the file name instead."""
    if os.path.abspath(root) == os.path.abspath(folder):
        return os.path.splitext(filename)[0]
    return os.path.basename(root)


def find_trips(folder):
    """Return a sorted list of trip ids found in a partition folder."""
    trip_ids = set()
    for root, _, files in os.walk(folder):
        for name in files:
            if os.path.splitext(name)[1] in EXTENSIONS:
                trip_ids.add(trip_id_of(root, folder, name))
    return sorted(trip_ids)


def find_file(folder, trip_id, ext):
    """Path of the file with this extension for this trip, or None if missing."""
    for root, _, files in os.walk(folder):
        for name in files:
            if os.path.splitext(name)[1] == ext and trip_id_of(root, folder, name) == trip_id:
                return os.path.join(root, name)
    return None


def load_track(path):
    # no header: longitude, latitude, timestamp in seconds
    return pd.read_csv(path, sep=r"\s+", header=None,
                       names=["lon", "lat", "t"])


def load_nodes(path):
    # no header: longitude, latitude. The row number is the node id
    return pd.read_csv(path, sep=r"\s+", header=None, names=["lon", "lat"])


def load_arcs(path):
    # no header: from_node, to_node. The row number is the arc id
    return pd.read_csv(path, sep=r"\s+", header=None,
                       names=["from_node", "to_node"])


def load_route(path):
    # one column: the arc ids the vehicle went through, in order
    return pd.read_csv(path, sep=r"\s+", header=None, names=["arc_id"])


def load_trip(folder, trip_id):
    """Load everything we have for one trip into a dict.
    A missing file just gives None, so we can detect it later."""
    trip = {"id": trip_id}
    loaders = [(".track", load_track), (".nodes", load_nodes),
               (".arcs", load_arcs), (".route", load_route)]
    for ext, loader in loaders:
        path = find_file(folder, trip_id, ext)
        trip[ext[1:]] = loader(path) if path else None
    return trip


def make_projector(lon_mean, lat_mean):
    """Return a function that converts lon/lat to metres (UTM zone for the area)."""
    zone = int((lon_mean + 180) / 6) + 1
    epsg = (32600 if lat_mean >= 0 else 32700) + zone
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)

    def project(lon, lat):
        x, y = transformer.transform(np.asarray(lon), np.asarray(lat))
        return np.column_stack([x, y])

    return project, epsg
