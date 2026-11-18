# build_candidates.py - one row per (GPS point, candidate road arc)
# usage (from the repo root):  python src/build_candidates.py data/train train 3
#   the last number is optional: how many trips to do (leave it out for all)
import os
import sys
import numpy as np
import pandas as pd
import shapely
from shapely import STRtree
from data_io import load_track, load_nodes, load_arcs, load_route, make_projector

RADIUS = 50      # metres: arcs closer than this to a GPS point are candidates
MAX_CAND = 30    # keep at most the 30 nearest candidates per point


def angle_diff(a, b):
    """Smallest difference between two directions in degrees (0 to 180)."""
    return np.abs((a - b + 180) % 360 - 180)


def build_trip(folder, trip_id):
    base = f"{folder}/{trip_id}/{trip_id}"
    track = load_track(base + ".track")
    nodes = load_nodes(base + ".nodes")
    arcs = load_arcs(base + ".arcs")
    route = load_route(base + ".route")

    # 1. Everything into metres, using this trip's own projection
    project, _ = make_projector(track["lon"].mean(), track["lat"].mean())
    pts_xy = project(track["lon"], track["lat"])
    nodes_xy = project(nodes["lon"], nodes["lat"])

    # 2. One straight line per arc (arc id = row number in the .arcs file)
    start = arcs["from_node"].values
    end = arcs["to_node"].values
    seg = np.stack([nodes_xy[start], nodes_xy[end]], axis=1)
    lines = shapely.linestrings(seg)
    points = shapely.points(pts_xy)

    # 3. Candidates: every arc within RADIUS metres of each point
    tree = STRtree(lines)
    pt_idx, arc_idx = tree.query(points, predicate="dwithin", distance=RADIUS)

    # 4. Label: the closest arc of the reference route counts as correct
    route_ids = route["arc_id"].values
    route_tree = STRtree(lines[route_ids])
    pairs, _ = route_tree.query_nearest(points, return_distance=True, all_matches=False)
    correct_arc = np.zeros(len(points), dtype=int)
    correct_arc[pairs[0]] = route_ids[pairs[1]]

    # 5. Features that describe the (point, arc) pair
    dist_m = shapely.distance(points[pt_idx], lines[arc_idx])
    along_frac = shapely.line_locate_point(lines[arc_idx], points[pt_idx], normalized=True)
    arc_len = shapely.length(lines[arc_idx])

    # direction of each arc, and direction the vehicle is moving (to the next point)
    d = seg[arc_idx, 1, :] - seg[arc_idx, 0, :]
    arc_bearing = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    step = np.diff(pts_xy, axis=0)
    step = np.vstack([step, step[-1:]])          # last point reuses the previous step
    step_len = np.hypot(step[:, 0], step[:, 1])
    heading = np.degrees(np.arctan2(step[:, 1], step[:, 0]))
    heading_diff = angle_diff(heading[pt_idx], arc_bearing)
    heading_diff[step_len[pt_idx] < 1] = 90      # parked: heading means nothing, use neutral 90

    # speed (m/s) from the previous point to this one; 0 for the first point
    dt = np.diff(track["t"].values)
    prev_speed = np.where(dt > 0, step_len[:-1] / np.where(dt > 0, dt, 1), 0)
    speed = np.concatenate([[0], prev_speed])

    df = pd.DataFrame({
        "trip_id": trip_id,
        "point_idx": pt_idx,
        "arc_id": arc_idx,
        "label": (arc_idx == correct_arc[pt_idx]).astype(int),
        "dist_m": dist_m,
        "along_frac": along_frac,
        "heading_diff": heading_diff,
        "speed": speed[pt_idx],
        "arc_len": arc_len,
    })

    # 6. Rank candidates by distance for each point and keep the nearest MAX_CAND
    df = df.sort_values(["point_idx", "dist_m"])
    df["dist_rank"] = df.groupby("point_idx").cumcount()
    df = df[df["dist_rank"] < MAX_CAND]

    # 7. Candidate miss rate: points where the correct arc is not among the candidates
    hit_points = df.loc[df["label"] == 1, "point_idx"].nunique()
    return df, len(points), hit_points


def main():
    folder = sys.argv[1]
    partition = sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
    trip_ids = sorted(d for d in os.listdir(folder) if os.path.isdir(f"{folder}/{d}"))[:limit]

    os.makedirs("data/candidates", exist_ok=True)       # data/ is git-ignored
    out = f"data/candidates/{partition}_candidates.csv"
    if os.path.exists(out):
        os.remove(out)

    total_pts, total_hits, total_rows = 0, 0, 0
    for i, trip_id in enumerate(trip_ids):
        df, n_pts, hits = build_trip(folder, trip_id)
        df.to_csv(out, mode="a", header=(i == 0), index=False)   # write trip by trip
        total_pts += n_pts
        total_hits += hits
        total_rows += len(df)
        print(f"{trip_id}: {n_pts} points, {len(df) / n_pts:.1f} candidates/point, "
              f"miss rate {1 - hits / n_pts:.1%}")

    print(f"\n{partition}: {total_rows} rows, overall miss rate {1 - total_hits / total_pts:.2%}")
    print("saved to", out)


main()