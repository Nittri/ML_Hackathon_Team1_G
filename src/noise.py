# noise.py - how far is each GPS point from the reference route?
# usage (from the repo root):  python src/noise.py data/train
import os
import sys
import numpy as np
import pandas as pd
import shapely
from shapely import STRtree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data_io import find_trips, load_trip, make_projector

RADII = [10, 20, 30, 50, 75, 100]   # metres


def point_to_route_distances(trip):
    """For every GPS point: distance (m) to the nearest route arc, and which
    position in the route list that arc has."""
    track, nodes, arcs, route = trip["track"], trip["nodes"], trip["arcs"], trip["route"]
    project, _ = make_projector(track["lon"].mean(), track["lat"].mean())

    route_ids = route["arc_id"].values
    start_nodes = arcs["from_node"].values[route_ids]
    end_nodes = arcs["to_node"].values[route_ids]
    nodes_xy = project(nodes["lon"], nodes["lat"])
    segments = np.stack([nodes_xy[start_nodes], nodes_xy[end_nodes]], axis=1)

    tree = STRtree(shapely.linestrings(segments))
    points = shapely.points(project(track["lon"], track["lat"]))
    pairs, dist = tree.query_nearest(points, return_distance=True, all_matches=False)
    return dist, pairs[1]


def main():
    folder = sys.argv[1]
    name = os.path.basename(folder.rstrip("/"))
    trips = find_trips(folder)

    rows = []
    all_dist = []
    for i, trip_id in enumerate(trips, start=1):
        print(f"measuring {trip_id} ({i}/{len(trips)})")
        dist, position = point_to_route_distances(load_trip(folder, trip_id))
        all_dist.append(dist)
        rows.append({
            "trip": trip_id,
            "n_points": len(dist),
            "median_m": round(float(np.median(dist)), 1),
            "p95_m": round(float(np.percentile(dist, 95)), 1),
            "p99_m": round(float(np.percentile(dist, 99)), 1),
            "max_m": round(float(dist.max()), 1),
            # how often does the nearest route arc move backwards along the route?
            "pct_nearest_goes_back": round(float((np.diff(position) < 0).mean()) * 100, 1),
        })
    table = pd.DataFrame(rows)
    all_dist = np.concatenate(all_dist)

    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    table.to_csv(f"results/noise_{name}.csv", index=False)

    print()
    print(f"partition: {name}, {len(all_dist)} GPS points")
    print("mean distance to route (m):", round(float(all_dist.mean()), 1))
    print("median:", round(float(np.median(all_dist)), 1))
    for p in [90, 95, 99, 99.9]:
        print(f"{p}th percentile (m):", round(float(np.percentile(all_dist, p)), 1))
    print("share of points within a radius of the route:")
    for r in RADII:
        print(f"  within {r:>3} m: {round(float((all_dist <= r).mean()) * 100, 2)} %")
    print("five trips with the largest 99th percentile:")
    print(table.nlargest(5, "p99_m")[["trip", "median_m", "p99_m", "max_m"]])
    print("trips where the nearest route arc goes backwards most often:")
    print(table.nlargest(5, "pct_nearest_goes_back")[["trip", "pct_nearest_goes_back"]])

    plt.figure(figsize=(7, 4))
    plt.hist(all_dist[all_dist <= 100], bins=100)
    plt.xlabel("distance from GPS point to nearest reference route arc (m)")
    plt.ylabel("number of GPS points")
    plt.title(f"GPS noise relative to the reference route ({name} partition)")
    plt.tight_layout()
    plt.savefig(f"figures/noise_{name}.png", dpi=150)
    print("saved figures/noise_" + name + ".png")


if __name__ == "__main__":
    main()
