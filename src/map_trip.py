# map_trip.py - draw a window of one trip on a Folium map
# usage: python src/map_trip.py data/train 00000034 0 400
#        (partition folder, trip id, first GPS point, how many points)
import os
import sys
import numpy as np
import folium
from data_io import load_trip

MARGIN = 0.003   # degrees around the GPS points (roughly 300 m)


def arc_segments(arcs, nodes, ids):
    """Turn arc ids into [[lat, lon], [lat, lon]] pairs that folium can draw."""
    lon = nodes["lon"].values
    lat = nodes["lat"].values
    starts = arcs["from_node"].values[ids]
    ends = arcs["to_node"].values[ids]
    return [[[lat[a], lon[a]], [lat[b], lon[b]]] for a, b in zip(starts, ends)]


def main():
    folder, trip_id = sys.argv[1], sys.argv[2]
    first = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    count = int(sys.argv[4]) if len(sys.argv) > 4 else 400

    trip = load_trip(folder, trip_id)
    nodes, arcs, route = trip["nodes"], trip["arcs"], trip["route"]
    window = trip["track"].iloc[first:first + count]

    # which nodes fall inside the window (plus a small margin)?
    lon = nodes["lon"].values
    lat = nodes["lat"].values
    inside = ((lon > window["lon"].min() - MARGIN) & (lon < window["lon"].max() + MARGIN) &
              (lat > window["lat"].min() - MARGIN) & (lat < window["lat"].max() + MARGIN))
    arc_inside = inside[arcs["from_node"].values] | inside[arcs["to_node"].values]
    network_ids = np.where(arc_inside)[0]

    route_ids = route["arc_id"].values
    route_ids = route_ids[arc_inside[route_ids]]

    print("points drawn:", len(window))
    print("network arcs drawn:", len(network_ids))
    print("route arcs drawn:", len(route_ids))

    m = folium.Map(location=[window["lat"].mean(), window["lon"].mean()],
                   zoom_start=17, control_scale=True)

    network = folium.FeatureGroup(name="Road network (grey)")
    folium.PolyLine(arc_segments(arcs, nodes, network_ids),
                    color="gray", weight=2, opacity=0.6).add_to(network)
    network.add_to(m)

    reference = folium.FeatureGroup(name="Reference route (green)")
    folium.PolyLine(arc_segments(arcs, nodes, route_ids),
                    color="green", weight=5, opacity=0.8).add_to(reference)
    reference.add_to(m)

    gps = folium.FeatureGroup(name="GPS points (blue)")
    for index, row in window.iterrows():
        folium.CircleMarker([row["lat"], row["lon"]], radius=3, color="blue",
                            fill=True, fill_opacity=0.8,
                            tooltip=f"point {index}, t={row['t']:.0f}").add_to(gps)
    gps.add_to(m)

    folium.LayerControl().add_to(m)

    os.makedirs("maps", exist_ok=True)
    partition = os.path.basename(folder.rstrip("/"))
    out = f"maps/{partition}_{trip_id}_{first}_{count}.html"
    m.save(out)
    print("saved", out)


if __name__ == "__main__":
    main()
