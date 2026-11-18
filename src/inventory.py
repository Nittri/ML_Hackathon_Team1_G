# inventory.py - build a table describing every trip in a partition
# usage (from the repo root):  python src/inventory.py data/train
import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
from data_io import (find_trips, find_file, load_track, load_nodes,
                     load_arcs, load_route, make_projector)


def describe_trip(folder, trip_id):
    row = {"trip": trip_id}

    paths = {ext: find_file(folder, trip_id, ext)
             for ext in [".track", ".nodes", ".arcs", ".route"]}
    for ext, path in paths.items():
        row["has" + ext] = path is not None
        row["bytes" + ext] = os.path.getsize(path) if path else 0

    if paths[".track"]:
        track = load_track(paths[".track"])
        gaps = track["t"].diff().dropna()
        row["n_points"] = len(track)
        row["n_missing"] = int(track.isna().sum().sum())
        row["pct_1s_gaps"] = round((gaps == 1).mean() * 100, 1)
        row["max_gap_s"] = gaps.max()
        row["time_sorted"] = bool(track["t"].is_monotonic_increasing)
        row["lon_min"], row["lon_max"] = track["lon"].min(), track["lon"].max()
        row["lat_min"], row["lat_max"] = track["lat"].min(), track["lat"].max()

        # trip length in metres, using a projected coordinate system
        project, _ = make_projector(track["lon"].mean(), track["lat"].mean())
        xy = project(track["lon"], track["lat"])
        steps = np.sqrt((np.diff(xy, axis=0) ** 2).sum(axis=1))
        row["length_m"] = round(steps.sum(), 1)
        row["max_step_m"] = round(steps.max(), 1)

    if paths[".nodes"] and paths[".arcs"]:
        nodes = load_nodes(paths[".nodes"])
        arcs = load_arcs(paths[".arcs"])
        graph = nx.Graph()
        graph.add_nodes_from(range(len(nodes)))
        graph.add_edges_from(zip(arcs["from_node"], arcs["to_node"]))
        row["n_nodes"] = len(nodes)
        row["n_arcs"] = len(arcs)
        row["n_components"] = nx.number_connected_components(graph)
        row["n_isolated"] = len(list(nx.isolates(graph)))
        row["max_node_id_in_arcs"] = int(arcs.max().max())

    if paths[".route"]:
        route = load_route(paths[".route"])
        row["route_len"] = len(route)
        row["max_route_arc_id"] = int(route["arc_id"].max())

    return row


def main():
    folder = sys.argv[1]
    name = os.path.basename(folder.rstrip("/"))
    trips = find_trips(folder)
    print(f"{name}: found {len(trips)} trips")

    rows = [describe_trip(folder, trip) for trip in trips]
    table = pd.DataFrame(rows)

    os.makedirs("results", exist_ok=True)
    out = f"results/inventory_{name}.csv"
    table.to_csv(out, index=False)
    print("saved", out)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(table.describe())
    files = ["has.track", "has.nodes", "has.arcs", "has.route"]
    print("trips missing a file:", int((~table[files].all(axis=1)).sum()))
    print("trips with a gap > 5 s:", int((table["max_gap_s"] > 5).sum()))
    print("trips with more than 1 component:", int((table["n_components"] > 1).sum()))


if __name__ == "__main__":
    main()
