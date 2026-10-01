# Racing Line Planner

Generate a racing line and a periodic speed profile for a closed track from ordered cone boundaries. The planner uses NumPy, SciPy, PyYAML, and Matplotlib; it does not require Gurobi or a commercial solver license.

The default objective is estimated lap time. An alternative objective minimizes the arc-length integral of squared curvature. Both use local nonlinear optimization under a simplified vehicle model: global optimality and real-vehicle feasibility are not guaranteed.

## Requirements

- Python 3.11 or newer. Development and regression tests were run with Python 3.13.
- Dependencies listed in `requirements.txt`.

## Quick Start

Clone the repository and run the following commands from its root directory:

```sh
git clone https://github.com/Captain0261/Find-Optimal-Racing-Line.git
cd Find-Optimal-Racing-Line
python -m venv .venv
```

Activate the environment in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or activate it on Linux/macOS:

```sh
source .venv/bin/activate
```

Install dependencies, extract the included dataset, and run the planner:

```sh
python -m pip install -r requirements.txt
python -m zipfile -e track_dataset.zip .
python racing_line.py --data-dir ./dataset --track 1
```

The archive contains nine track layouts under `dataset/` and a dataset overview image. The program displays a speed-colored racing line and a speed-versus-distance plot.

Always pass `--data-dir` when using the extracted dataset. Without that argument, the existing local-workspace default is `../fsd_racetrack_dataset-main/dataset`, relative to the script directory.

## Command-Line Examples

Export CSV, JSON, and PNG files without opening a plot window:

```sh
python racing_line.py --data-dir ./dataset --track 1 --no-show --output ./results/track1
```

Optimize curvature instead of lap time and load vehicle parameters:

```sh
python racing_line.py --data-dir ./dataset --objective curvature --config vehicle.example.json
```

Change the discretization and iteration budget:

```sh
python racing_line.py --data-dir ./dataset --track 2 --controls 64 --samples 1000 --max-iterations 200
```

Run regression tests and validate the dataset:

```sh
python -m unittest -v test_racing_line
python validate_dataset.py --data-dir ./dataset
python validate_dataset.py --data-dir ./dataset --tracks 5 6 --output ./results/recheck.json
```

Use `python racing_line.py --help` for all options. The default discretization uses 48 optimization control points and 600 evaluation samples, with a limit of 200 iterations per optimization stage. The sample count must be at least four times the control-point count.

## Planning Pipeline

1. **Validate the original track corridor.** Ordered cones define two nested, closed polygons. The planner rejects invalid coordinates, consecutive duplicate points, self-intersecting boundaries, and intersections between the two boundaries. Boundary smoothing does not change the admissible region.
2. **Construct corresponding cross-sections.** Arc-length alignment initializes a reference line. Its normal lines are intersected with the original boundaries to obtain left/right pairs. Ambiguous or intersecting cross-sections are rejected.
3. **Fit a periodic path.** The first control point is appended explicitly before fitting a periodic cubic spline, so the final unique point is retained. Positions, signed curvature, and segment lengths come from the same spline. Gaussian quadrature includes the closing segment.
4. **Enforce geometric clearance.** Constraints use actual distances to the original boundary polygons. A bound on the spline derivative accounts for gaps between evaluation samples and produces a lower bound on continuous-path clearance.
5. **Optimize the path.** A curvature objective supplies an initial feasible line. By default, a second SLSQP stage minimizes estimated lap time and recomputes the speed profile for every candidate path. Path selection therefore accounts for acceleration and braking capability.
6. **Compute periodic speed.** Squared speeds are relaxed over every edge, including the final-to-first edge, until convergence. Constraints include a coupled longitudinal/lateral friction ellipse, maximum speed, steering angle, and steering rate. Each discrete edge uses the greater endpoint speed and absolute curvature for conservative checks.
7. **Validate the result.** The planner checks clearance, sampled curvature, the discrete friction ellipse, and sampled path self-intersections. A non-converged solver may return its best validated feasible candidate with a warning and an explicit status. If no feasible candidate exists, planning fails.

The lap-time stage uses an objective tolerance of `1e-5`; the curvature stage uses `1e-7`. Final geometric and dynamic acceptance checks are performed independently of the solver's success flag.

## Vehicle Configuration

Edit `vehicle.example.json` and pass it with `--config`. Distances are in meters, speeds in meters per second, accelerations in meters per second squared, and angles in radians.

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `width` | 1.2 | Vehicle width |
| `length` | 0.0 | Zero selects a width-based disk; a positive value enables a conservative circumscribed disk for the vehicle rectangle |
| `safety_margin` | 0.2 | Additional clearance outside the modeled footprint |
| `wheelbase` | 1.6 | Wheelbase used to relate curvature to steering angle |
| `max_steer` | 0.6 | Maximum steering angle |
| `max_steer_rate` | 1.2 | Maximum steering rate in radians per second |
| `a_lat_max` | 10.0 | Lateral acceleration capability |
| `a_acc` | 5.0 | Longitudinal acceleration capability |
| `a_dec` | 8.0 | Braking deceleration capability |
| `v_max` | 25.0 | Maximum speed |

With the default disk model, the required center-to-boundary clearance is `width / 2 + safety_margin = 0.8 m`. The default does **not** account for actual vehicle length.

When `length` is positive, the required clearance becomes `hypot(width, length) / 2 + safety_margin`. This encloses the entire rectangular footprint at any heading, but can reject narrow corridors that an orientation-aware rectangle model might traverse.

These are example parameters, not a calibrated racing vehicle. Set them to match the intended vehicle before interpreting the results.

## Outputs

`--output` specifies a filename prefix. For example, `--output ./results/track1` writes:

- **CSV:** `s_m,x_m,y_m,signed_curvature_per_m,speed_mps,segment_length_m`. The first point is not repeated. The final row's segment length closes the loop back to the first point.
- **JSON:** Estimated lap time, path length, clearance lower bound, acceleration and steering metrics, vehicle and optimizer settings, and solver status. It also stores the path's control points; append the first point and fit a periodic cubic spline on uniform knots in `[0, 1]` to reconstruct the position curve.
- **PNG:** The racing line colored by speed and a velocity profile that includes the closing segment.

Generated results are local artifacts and are not required to run the planner.

## Python API

```python
from racing_line import Vehicle, load_track, plan_racing_line

track = load_track("dataset", track_id=1)
vehicle = Vehicle(width=1.2, safety_margin=0.2)
plan = plan_racing_line(track, vehicle, objective="lap_time")

print(plan.metrics)
print(plan.solver)
```

`Plan` contains sampled positions, signed curvature, segment lengths, velocity, lateral speed limits, control-station offsets, metrics, solver status, and spline control points. Pass the original cone polygons to `Track` to retain validation against the original boundaries.

Additional interfaces:

- `generate_velocity_profile(...)` returns `(lateral_speed_limit, velocity, accumulated_distance, signed_curvature)`.
- `get_optimal_line(..., N=None)` infers the input length and rejects a conflicting explicit `N`. Its third return value belongs to the reconstructed optimization control stations, not the original boundary samples. Do not combine those offsets element by element with the input boundaries; prefer `plan_racing_line` for new code.
- A repeated closing point is removed from closed inputs. Returned sample arrays contain only unique points.

## Tests and Validation

The regression suite covers periodic interpolation and derivatives, the closing-edge acceleration limit, start-index invariance of the speed profile, the friction ellipse, steering rate, analytical circular-track speed, curvature sign, invalid inputs, reversed or shifted boundaries, narrow corridors, vehicle footprint configuration, and the different outcomes of curvature and lap-time objectives on a circular track.

The optional nine-track corridor test uses the adjacent `../fsd_racetrack_dataset-main/dataset` directory and is skipped when that directory is absent. To validate the extracted repository dataset explicitly, run `validate_dataset.py --data-dir ./dataset`.

The dataset validator records each track's metrics and solver status. It exits with a nonzero status if any run fails validation, does not converge, or no tracks are found.

## Model Limitations

- The planner finds local solutions; it does not prove a globally fastest racing line.
- The speed model is a simplified point-mass model for steady flying laps, not standing starts.
- Tire slip, load transfer, aerodynamic effects, speed-dependent power/braking capability, and actuator delay are not modeled.
- Dynamics and steering checks apply to the discretized model. Increasing resolution can change the estimated lap time; continuous real-vehicle feasibility is not established by these checks.
- The continuous geometric clearance bound uses the configured disk footprint. Actual vehicle dimensions and track uncertainty must be reflected in the configuration.

Use resolution checks and closed-loop trajectory-tracking simulation when assessing a planned trajectory.

## Project Files

| File | Purpose |
| --- | --- |
| `racing_line.py` | Planner, command-line interface, plotting, and export |
| `test_racing_line.py` | Regression tests |
| `validate_dataset.py` | Batch track validation |
| `vehicle.example.json` | Example vehicle configuration |
| `requirements.txt` | Python dependencies |
| `track_dataset.zip` | Existing bundled dataset archive |

## Dataset Credit

The track data comes from the FSD Racetrack Dataset by StarkStrom Augsburg, released with *Lane Detection using Graph Search and Geometric Constraints for Formula Student Driverless* by Ivo Ivanov and Carsten Markgraf. The dataset documentation identifies its license as LGPLv3 and reports cone-position accuracy of approximately 0.2-0.3 m. Dataset licensing is separate from the planner code.

Dataset repository: [iv461/fsd_racetrack_dataset](https://github.com/iv461/fsd_racetrack_dataset).
