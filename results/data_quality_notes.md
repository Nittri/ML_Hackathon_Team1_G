# Data quality notes (before cleaning)
Partitions used: train (80 trips), student_test (10 trips). Evaluator data not used.

## 1. Files and missing values
- What I checked: that every trip folder has the .track, .nodes, .arcs and .route files, and that the tracks have no empty values.
- What I found: all four files exist for every trip in both partitions, and there are no missing values in the tracks.
- Why it matters: there is no need to impute or drop rows for missing data.

## 2. Time gaps
- What I checked: the time difference between consecutive GPS points in each trip.
- What I found: 45 of 80 train trips have a gap above 5 s, against 2 of 10 in student_test. The biggest gaps (00000090: 153,064 s, 00000078: 52,810 s) moved only 3.3 m and 2.5 m, so the vehicle was parked. A few gaps moved far: 00000055 (184 s, 888 m), 00000070 (152 s, 541 m), 00000099 (25 s, 463 m), so the vehicle was moving and the signal was lost.
- Why it matters: the HMM needs two rules, one for parked gaps (stay put) and one for moving gaps (allow a longer jump between roads).

## 3. GPS jumps
- What I checked: points that move more than 100 m in 1 s.
- What I found: this happens only in train, up to 13 times in one trip. Student_test has none.
- Why it matters: a car cannot move that fast, so these are GPS glitches. They make train noisier than student_test.

## 4. Road network (components, one-way arcs)
- What I checked: network size, number of disconnected components, and how many arcs have a reverse arc.
- What I found: trips are spread across the world, so each trip needs its own projection into metres. Train trips average 24,169 nodes and 48,461 arcs. Almost every trip's network has several disconnected components (train mean 39, max 421). On average 92% of arcs have a reverse arc (minimum 35% in train).
- Why it matters: distances must be computed in metres per trip. Arcs are directed, so the HMM must use directed transitions, and a path between two arcs may not exist if they are on different components.

## 5. Reference route (valid ids, chaining, repeated arcs)
- What I checked: that arc and node ids are valid, that consecutive route arcs connect end to start, that the route stays in one component, and how often arcs repeat.
- What I found: all ids are valid in all 90 trips. On average 99.9% of consecutive arcs chain correctly (minimum 98.1% in train). The route stays in one component in all trips. Repeated arcs average 0.8 per train trip (max 29) and 25.5 per student_test trip (max 255 in one trip).
- Why it matters: the route is reliable enough to build labels from. Loops and repeated arcs mean label alignment must not assume each arc appears only once.

## 6. GPS noise relative to the reference route
- What I checked: the distance from each GPS point to the nearest arc of the reference route.
- What I found: median 5.6 m (train) and 5.2 m (student_test). The 95th percentile is 140.3 m in train and 16.8 m in student_test. Within 50 m: 89.2% of train points and 99.86% of student_test points.
- Why it matters: most points are close to the true road, but train has a heavy tail of far-off points. The candidate radius must be chosen from train only, and the candidate miss rate must be measured.

## 7. Problem trips (00000054, 00000033, 00000002)
- What I found: 00000054 (median 138.6 m) and 00000033 (median 122.4 m) have points consistently far from the reference route. 00000002 has a normal median but a 99th percentile of 482 m.
- What I do NOT know yet: whether the cause is an incomplete or wrong reference route, or a GPS offset. A teammate is cleaning the dataset, so I will ask for the rules used.

## 8. Decisions and open questions
- Per-trip projection into metres is used for all distance calculations.
- The HMM will use directed arcs and two different rules for time gaps.
- Starting candidate radius: 50 m plus at least the nearest ~20 arcs, chosen from train only.
- Student_test is used for comparing models, not for tuning radius or similar choices.
- Open: the exact cleaning rules from my teammate, and whether to exclude or flag the problem trips.