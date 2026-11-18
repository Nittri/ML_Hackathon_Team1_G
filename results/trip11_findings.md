# Trip 00000011: label ambiguity on an out-and-back route
Partition: train. All numbers are from my own runs (Logistic Regression, C = 10,
grouped 5-fold out-of-fold predictions).

- 2828 points, 1448 wrong (error rate 0.512). 1414 of the wrong points pick the
  same road in the opposite direction; 34 pick a different road.
- Timestamps are in order (0 of 2827 steps go backwards). The route chains
  forward (564 of 564 consecutive arcs).
- 99.8% of the route's 565 arcs have their reverse arc in the route too.
  Median over the 80 train trips: 0.0. Next highest trip: 0.429.
- The label arc points against the direction of motion (heading_diff > 90) at
  50.8% of points, and 95.9% of those are between 170 and 180 degrees. At those
  points the reverse arc is a candidate (100%) and matches the motion
  (median heading_diff 2.5 degrees).
- Reading (an inference, not proven): the vehicle drove the road out and back.
  "Label = nearest route arc" cannot tell the outbound arc from the return arc,
  so the label is ambiguous here.

## Other trips
- Excluding trip 00000011, label arcs against the motion are mostly 90-170
  degrees (about 78%); only 21.7% are 170-180 degrees. Median distance to the
  label arc: 5.8 m. So this ambiguity does not explain the other trips.
  The cause is not tested yet.
