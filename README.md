# Find-Optimal-Racing-Line
This repository provides a optimization-based trajectory planning pipeline for autonomous racing vehicles. It generates a theoretically optimal racing line and a physically feasible velocity profile from raw, noisy track boundary data.

## Pipeline Overview
1. **Track Parsing & Parameterization:** Extracts raw cone data (left/right boundaries) and fits cubic B-splines to generate a smooth mathematical track corridor.
2. **Global Trajectory Optimization:** Formulates the minimum-curvature racing line problem as a strictly convex QP, ensuring a global optimum.
3. **Velocity Profiling:** Calculates analytical curvature to determine lateral velocity limits, followed by a forward-backward kinematic sweep to enforce longitudinal acceleration/deceleration constraints.

## Dataset & Acknowledgements
The track dataset utilized in this project for testing and visualization are sourced from the open-source Formula Student Driverless (FSD) Racetrack Dataset.
We highly appreciate the contribution of the original authors to the autonomous racing community:
Dataset Repository: iv461/fsd_racetrack_dataset
(Please ensure you have cloned or downloaded the dataset into the ⁠dataset/⁠ directory to run the examples seamlessly).

