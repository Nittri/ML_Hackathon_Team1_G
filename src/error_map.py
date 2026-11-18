# error_map.py - draw the Logistic Regression mistakes of ONE train trip on a Folium map
# usage (from the repo root):
#   python src/error_map.py 00000011            (whole trip)
#   python src/error_map.py 00000011 0 800      (only GPS points 0 to 799)
# Partition used: train (out-of-fold predictions, so the model never saw this trip)
import os
import sys
import numpy as np
import pandas as pd
import folium
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from data_io import load_trip, make_projector

FEATURES = ["dist_m", "along_frac", "heading_diff", "speed", "arc_len", "dist_rank"]

trip_id = sys.argv[1]
start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
end = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9

# 1. Same data and same folds as error_analysis.py
t = pd.read_csv("data/candidates/train_candidates.csv", dtype={"trip_id": str})
has_correct = t.groupby(["trip_id", "point_idx"])["label"].transform("max") == 1
t = t[has_correct].reset_index(drop=True)

t["prob"] = 0.0
for tr, va in GroupKFold(n_splits=5).split(t, groups=t["trip_id"]):
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(C=10, class_weight="balanced", max_iter=1000))
    model.fit(t.loc[tr, FEATURES], t.loc[tr, "label"])
    t.loc[va, "prob"] = model.predict_proba(t.loc[va, FEATURES])[:, 1]

# 2. The arc the model picked and the correct arc, for every GPS point
picked = t.loc[t.groupby(["trip_id", "point_idx"])["prob"].idxmax(),
               ["trip_id", "point_idx", "arc_id", "label", "dist_m"]]
truth = t.loc[t["label"] == 1, ["trip_id", "point_idx", "arc_id", "dist_m"]]
m = picked.merge(truth, on=["trip_id", "point_idx"], suffixes=("_pred", "_true"))
m["wrong"] = m["label"] == 0

# 3. Keep only the trip we want to draw
g = m[m["trip_id"] == trip_id].sort_values("point_idx").reset_index(drop=True)
if len(g) == 0:
    print("no points found for trip", trip_id)
    sys.exit(1)
print(f"trip {trip_id}: points {len(g)}   wrong {g['wrong'].sum()}   "
      f"error rate {g['wrong'].mean():.3f}")

# 4. Load the raw files of this trip
trip = load_trip("data/train", trip_id)
track, nodes, arcs, route = trip["track"], trip["nodes"], trip["arcs"], trip["route"]
f, to = arcs["from_node"].values, arcs["to_node"].values
p, q = g["arc_id_pred"].values, g["arc_id_true"].values

# 5. Kind of mistake: same road in the opposite direction, or a different road?
reverse = (f[p] == to[q]) & (to[p] == f[q])
g["kind"] = np.where(~g["wrong"], "correct",
                     np.where(reverse, "opposite direction", "different road"))
print(g["kind"].value_counts().to_string())

# 6. Sanity check: is point_idx really the row number in the .track file?
# We measure the distance (metres) from that track row to the correct arc ourselves
# and compare it with dist_m stored in the candidate table. They should match closely.
project, epsg = make_projector(track["lon"].mean(), track["lat"].mean())
pts = project(track["lon"].values, track["lat"].values)
nxy = project(nodes["lon"].values, nodes["lat"].values)
pt = pts[g["point_idx"].values]
a, b = nxy[f[q]], nxy[to[q]]
ab, ap = b - a, pt - a
share = np.clip((ap * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9), 0, 1)
d = np.linalg.norm(pt - (a + share[:, None] * ab), axis=1)
diff = np.abs(d - g["dist_m_true"].values)
print(f"check: median |my distance - dist_m| = {np.median(diff):.3f} m, "
      f"95th percentile = {np.percentile(diff, 95):.3f} m  (small = point_idx is right)")

# 7. Build the map (only the chosen window of points)
w = g[(g["point_idx"] >= start) & (g["point_idx"] < end)]
ll = track.loc[w["point_idx"].values, ["lat", "lon"]].values
fmap = folium.Map(location=[ll[:, 0].mean(), ll[:, 1].mean()], zoom_start=16,
                  tiles="OpenStreetMap")

def arc_line(arc_id):
    # a road segment as two corners: [lat, lon] of the from node and the to node
    return [[nodes.lat[f[arc_id]], nodes.lon[f[arc_id]]],
            [nodes.lat[to[arc_id]], nodes.lon[to[arc_id]]]]

# green = reference route
layer_route = folium.FeatureGroup(name="Reference route (green)")
for arc_id in route["arc_id"].values:
    folium.PolyLine(arc_line(arc_id), color="green", weight=5, opacity=0.6).add_to(layer_route)
layer_route.add_to(fmap)

# orange = the wrong road the model picked
layer_pick = folium.FeatureGroup(name="Wrong arc picked (orange)")
for arc_id in w.loc[w["wrong"], "arc_id_pred"].unique():
    folium.PolyLine(arc_line(arc_id), color="orange", weight=4, opacity=0.8).add_to(layer_pick)
layer_pick.add_to(fmap)

# blue = GPS point, model correct;  red = GPS point, model wrong
layer_ok = folium.FeatureGroup(name="GPS point, correct (blue)")
layer_bad = folium.FeatureGroup(name="GPS point, wrong (red)")
for (_, row), (lat, lon) in zip(w.iterrows(), ll):
    wrong = row["wrong"]
    text = (f"point {int(row['point_idx'])}<br>picked arc {int(row['arc_id_pred'])}"
            f"<br>true arc {int(row['arc_id_true'])}<br>{row['kind']}")
    folium.CircleMarker([lat, lon], radius=4 if wrong else 3,
                        color="red" if wrong else "blue", fill=True,
                        fill_opacity=0.9, popup=text).add_to(layer_bad if wrong else layer_ok)
layer_ok.add_to(fmap)
layer_bad.add_to(fmap)

fmap.fit_bounds([[ll[:, 0].min(), ll[:, 1].min()], [ll[:, 0].max(), ll[:, 1].max()]])
folium.LayerControl().add_to(fmap)
legend = ('<div style="position:fixed;bottom:20px;left:20px;z-index:9999;background:white;'
          'padding:8px;border:1px solid grey;font-size:13px">'
          f'<b>Train trip {trip_id} (out-of-fold)</b><br>'
          '<span style="color:green">&#9632;</span> reference route<br>'
          '<span style="color:blue">&#9679;</span> GPS point, correct<br>'
          '<span style="color:red">&#9679;</span> GPS point, wrong<br>'
          '<span style="color:orange">&#9632;</span> wrong arc picked</div>')
fmap.get_root().html.add_child(folium.Element(legend))

os.makedirs("maps", exist_ok=True)
out = f"maps/errors_train_{trip_id}.html"
fmap.save(out)
print("saved", out)
