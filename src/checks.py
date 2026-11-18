# checks.py - row-level checks on the reference route, the network and the time gaps
# usage (from the repo root):  python src/checks.py data/train
import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
from data_io import find_trips, load_trip, make_projector


def check_trip(folder, trip_id):
    trip = load_trip(folder, trip_id)
    track, nodes, arcs, route = trip["track"], trip["nodes"], trip["arcs"], trip["route"]
    row = {"trip": trip_id}

    # 1. do the ids really point to things that exist?
    ids = route["arc_id"].values
    row["route_ids_valid"] = bool(ids.max() < len(arcs))
    row["arc_nodes_valid"] = bool(arcs.max().max() < len(nodes))

    # 2. does each route arc start where the previous one ended?
    ends = arcs["to_node"].values[ids[:-1]]
    starts = arcs["from_node"].values[ids[1:]]
    row["pct_route_chained"] = round(float((ends == starts).mean()) * 100, 1)
    row["n_repeated_arcs"] = int(len(ids) - len(set(ids)))

    # 3. do arcs come in both directions? (is the network two-way?)
    pairs = set(zip(arcs["from_node"].values, arcs["to_node"].values))
    has_reverse = [(b, a) in pairs for a, b in pairs]
    row["pct_arcs_with_reverse"] = round(float(np.mean(has_reverse)) * 100, 1)

    # 4. does the whole route sit inside one connected piece of the network?
    graph = nx.Graph()
    graph.add_edges_from(zip(arcs["from_node"].values, arcs["to_node"].values))
    component_of = {}
    for number, component in enumerate(nx.connected_components(graph)):
        for node in component:
            component_of[node] = number
    route_nodes = set(arcs["from_node"].values[ids]) | set(arcs["to_node"].values[ids])
    row["route_components"] = len({component_of[n] for n in route_nodes})

    # 5. time gaps and what the vehicle did across them
    gaps = np.diff(track["t"].values)
    project, _ = make_projector(track["lon"].mean(), track["lat"].mean())
    xy = project(track["lon"], track["lat"])
    steps = np.sqrt((np.diff(xy, axis=0) ** 2).sum(axis=1))
    biggest = int(np.argmax(gaps))
    row["max_gap_s"] = float(gaps[biggest])
    row["dist_across_max_gap_m"] = round(float(steps[biggest]), 1)
    row["n_gaps_over_5s"] = int((gaps > 5).sum())
    row["n_gaps_over_60s"] = int((gaps > 60).sum())
    row["n_duplicate_times"] = int((gaps == 0).sum())
    row["n_time_going_back"] = int((gaps < 0).sum())
    one_second = gaps == 1
    row["n_jumps_over_100m_in_1s"] = int((steps[one_second] > 100).sum())
    return row


def main():
    folder = sys.argv[1]
    name = os.path.basename(folder.rstrip("/"))
    trips = find_trips(folder)
    rows = []
    for i, trip_id in enumerate(trips, start=1):
        print(f"checking {trip_id} ({i}/{len(trips)})")
        rows.append(check_trip(folder, trip_id))
    table = pd.DataFrame(rows)

    os.makedirs("results", exist_ok=True)
    out = f"results/checks_{name}.csv"
    table.to_csv(out, index=False)
    print("saved", out)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("route ids valid in every trip:", table["route_ids_valid"].all())
    print("arc node ids valid in every trip:", table["arc_nodes_valid"].all())
    print("route components per trip:")
    print(table["route_components"].value_counts())
    columns = ["pct_route_chained", "pct_arcs_with_reverse", "n_repeated_arcs",
               "n_gaps_over_5s", "n_gaps_over_60s", "dist_across_max_gap_m",
               "n_duplicate_times", "n_time_going_back", "n_jumps_over_100m_in_1s"]
    print(table[columns].describe())
    print("five trips with the biggest gaps:")
    print(table.nlargest(5, "max_gap_s")[["trip", "max_gap_s", "dist_across_max_gap_m"]])


if __name__ == "__main__":
    main()
