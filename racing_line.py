"""Closed-track racing-line planning with explicit geometric validation.

Units: metres, seconds, radians. The optimizer is local and the tire model is
a conservative point-mass friction ellipse, not a full vehicle simulator.
"""
from dataclasses import dataclass, asdict
from collections import OrderedDict
from pathlib import Path
import argparse
import json
import warnings

import numpy as np
import yaml
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize


def _points(value, name="points"):
    p = np.asarray(value, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 4 or not np.isfinite(p).all():
        raise ValueError(f"{name} must contain at least four finite 2D points")
    if np.linalg.norm(p[-1] - p[0]) < 1e-9:
        p = p[:-1]
    if len(p) < 4 or np.any(np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1) < 1e-8):
        raise ValueError(f"{name} contains duplicate consecutive points")
    return p.copy()


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _area(p):
    return np.sum(_cross(p, np.roll(p, -1, axis=0))) / 2


def _segments_cross(a, b, c, d):
    """Intersection including touches, with broadcasting."""
    ab, cd = b - a, d - c
    cross1, cross2 = _cross(ab, c - a), _cross(ab, d - a)
    cross3, cross4 = _cross(cd, a - c), _cross(cd, b - c)
    overlap = np.all(np.maximum(np.minimum(a, b), np.minimum(c, d)) <=
                     np.minimum(np.maximum(a, b), np.maximum(c, d)) + 1e-10, axis=-1)
    return (cross1 * cross2 <= 1e-16) & (cross3 * cross4 <= 1e-16) & overlap


def _check_simple(p, name):
    n = len(p)
    for i in range(n):
        js = np.arange(i + 2, n)
        if i == 0:
            js = js[js != n - 1]
        if np.any(_segments_cross(p[i], p[(i + 1) % n], p[js], p[(js + 1) % n])):
            raise ValueError(f"{name} is self-intersecting or touches itself")


def _inside(p, poly):
    a, b = poly, np.roll(poly, -1, axis=0)
    x, y = p[:, 0, None], p[:, 1, None]
    dy = b[:, 1] - a[:, 1]
    crossing = (a[:, 1] > y) != (b[:, 1] > y)
    safe_dy = np.where(np.abs(dy) > 1e-15, dy, 1.0)
    x_hit = a[:, 0] + (y - a[:, 1]) * (b[:, 0] - a[:, 0]) / safe_dy
    return np.count_nonzero(crossing & (x < x_hit), axis=1) % 2 == 1


def _distance(p, poly):
    a, d = poly, np.roll(poly, -1, axis=0) - poly
    x, y = p[:, 0, None] - a[:, 0], p[:, 1, None] - a[:, 1]
    t = np.clip((x * d[:, 0] + y * d[:, 1]) / np.sum(d * d, axis=1), 0, 1)
    return np.sqrt(np.min((x - t * d[:, 0]) ** 2 + (y - t * d[:, 1]) ** 2, axis=1))


def _resample_polygon(p, n):
    closed = np.vstack((p, p[0]))
    s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))]
    target = np.linspace(0, s[-1], n, endpoint=False)
    return np.column_stack([np.interp(target, s, closed[:, j]) for j in range(2)])


def periodic_spline(points):
    """All unique points participate; append the first point only for fitting."""
    p = _points(points)
    ds = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
    s = np.r_[0., np.cumsum(ds)]
    return CubicSpline(s, np.vstack((p, p[0])), bc_type="periodic"), s[:-1]


def draw_track(cone_data, boundary_data):
    try:
        return tuple(_points([cone_data[i] for i in boundary_data[side]], side)
                     for side in ("left", "right"))
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Invalid cone IDs or boundary structure: {exc}") from exc


def load_track(data_dir, track_id=1):
    directory = Path(data_dir)
    with (directory / f"cone_map_{track_id}.yaml").open(encoding="utf-8") as f:
        cones = yaml.safe_load(f)
    with (directory / f"boundaries_{track_id}.yaml").open(encoding="utf-8") as f:
        boundaries = yaml.safe_load(f)
    return Track(*draw_track(cones, boundaries))


@dataclass
class Vehicle:
    width: float = 1.2
    safety_margin: float = 0.2
    # Zero selects a disk of radius width/2. A positive length uses the
    # circumscribed disk of the rectangle, conservative for every heading.
    length: float = 0.0
    wheelbase: float = 1.6
    max_steer: float = 0.6
    max_steer_rate: float = 1.2
    a_lat_max: float = 10.0
    a_acc: float = 5.0
    a_dec: float = 8.0
    v_max: float = 25.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value < 0 or (name not in ("length", "safety_margin") and value == 0):
                raise ValueError(f"Invalid vehicle parameter: {name}")
        if self.max_steer >= np.pi / 2:
            raise ValueError("max_steer must be less than pi/2")

    @property
    def radius(self):
        return np.hypot(self.width, self.length) / 2 + self.safety_margin

    @property
    def max_curvature(self):
        return np.tan(self.max_steer) / self.wheelbase


class Track:
    def __init__(self, left, right):
        self.left, self.right = _points(left, "left"), _points(right, "right")
        for name, p in (("left", self.left), ("right", self.right)):
            _check_simple(p, name)
        if abs(_area(self.left)) > abs(_area(self.right)):
            self.outer, self.inner = self.left, self.right
        else:
            self.outer, self.inner = self.right, self.left
        if not _inside(self.inner, self.outer).all():
            raise ValueError("Expected two nested, closed track boundaries")
        for a, b in zip(self.left, np.roll(self.left, -1, axis=0)):
            if np.any(_segments_cross(a, b, self.right, np.roll(self.right, -1, axis=0))):
                raise ValueError("Track boundaries intersect")

    def clearance(self, p):
        distance = np.minimum(_distance(p, self.left), _distance(p, self.right))
        valid = _inside(p, self.outer) & ~_inside(p, self.inner)
        return np.where(valid, distance, -distance)

    @staticmethod
    def _normal_hits(center, normal, boundary):
        a, edge = boundary, np.roll(boundary, -1, axis=0) - boundary
        denom = _cross(normal[:, None, :], edge)
        safe = np.where(np.abs(denom) > 1e-10, denom, 1.)
        offset = a - center[:, None, :]
        t = _cross(offset, edge) / safe
        q = _cross(offset, normal[:, None, :]) / safe
        valid = (np.abs(denom) > 1e-10) & (q >= -1e-9) & (q <= 1 + 1e-9)
        score = np.where(valid, np.abs(t), np.inf)
        index = np.argmin(score, axis=1)
        if not np.isfinite(score[np.arange(len(center)), index]).all():
            raise ValueError("Cannot construct a normal cross-section at every station")
        return t[np.arange(len(center)), index]

    def cross_sections(self, n):
        if not isinstance(n, (int, np.integer)) or n < 8:
            raise ValueError("At least eight track stations are required")
        # Arc-length alignment only seeds a reference. Actual corridor stations
        # are intersections of its normals with the original cone polygons.
        m = max(256, n)
        l, r = _resample_polygon(self.left, m), _resample_polygon(self.right, m)
        if _area(l) * _area(r) < 0:
            r = r[::-1]
        shift = min(range(m), key=lambda j: np.mean(np.sum((l - np.roll(r, j, axis=0)) ** 2, axis=1)))
        r = np.roll(r, shift, axis=0)
        center = _resample_polygon((l + r) / 2, n)
        reference, s = periodic_spline(center)
        tangent = reference(s, 1)
        normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        tl = self._normal_hits(center, normal, self.left)
        tr = self._normal_hits(center, normal, self.right)
        if np.any(tl * tr >= 0) or np.any(self.clearance(center) <= 0):
            raise ValueError("Ambiguous corridor correspondence; inspect boundary ordering")
        left = center + tl[:, None] * normal
        right = center + tr[:, None] * normal
        for i in range(n):
            js = np.arange(i + 1, n)
            if np.any(_segments_cross(left[i], right[i], left[js], right[js])):
                raise ValueError("Normal cross-sections intersect; reference is ambiguous")
        return left, right


def fit_track(left_bound, right_bound, N=1000):
    """Legacy interface; returns corresponding cross-sections, without smoothing away cones."""
    left, right = Track(left_bound, right_bound).cross_sections(N)
    return left[:, 0], left[:, 1], right[:, 0], right[:, 1]


def _curvature(spline, u):
    d1, d2 = spline(u, 1), spline(u, 2)
    speed = np.linalg.norm(d1, axis=1)
    if np.any(speed < 1e-8):
        raise ValueError("Degenerate path tangent")
    return _cross(d1, d2) / speed ** 3


def _sample_spline(spline, n):
    period = spline.x[-1] - spline.x[0]
    u = np.linspace(spline.x[0], spline.x[-1], n, endpoint=False)
    step = period / n
    # Integrate the same spline used for positions and curvature, including seam.
    nodes, weights = np.polynomial.legendre.leggauss(4)
    uq = u[:, None] + (nodes + 1) * step / 2
    ds = (np.linalg.norm(spline(uq, 1), axis=2) @ weights) * step / 2
    # Cubic Bezier derivative control vectors bound speed everywhere. Thus the
    # nearest sample is at most guard metres away along the continuous spline.
    h = np.diff(spline.x)
    p0, p1 = spline(spline.x[:-1]), spline(spline.x[1:])
    d0, d2 = spline(spline.x[:-1], 1), spline(spline.x[1:], 1)
    dmid = 3 * (p1 - p0) / h[:, None] - d0 - d2
    speed_bound = np.max(np.linalg.norm(np.vstack((d0, dmid, d2)), axis=1))
    guard = speed_bound * step / 2
    return spline(u), _curvature(spline, u), ds, guard


def speed_profile(curvature, ds, vehicle=None, tolerance=1e-10, max_iterations=20000):
    """Periodic relaxation in v^2 with a coupled friction ellipse on every edge.

    The greater endpoint curvature and speed are used on each discrete edge.
    This is conservative compared with using only its starting sample.
    """
    vehicle = vehicle or Vehicle()
    k, ds = np.asarray(curvature, float), np.asarray(ds, float)
    if k.ndim != 1 or k.shape != ds.shape or len(k) < 4 or not np.isfinite(k).all() or not np.isfinite(ds).all() or np.any(ds <= 0):
        raise ValueError("Curvature and positive segment lengths must be equal-length finite vectors")
    if np.max(np.abs(k)) > vehicle.max_curvature + 1e-6:
        raise ValueError("Path exceeds the configured steering angle limit")
    edge_k = np.maximum(np.abs(k), np.abs(np.roll(k, -1)))
    edge_cap = np.minimum(vehicle.v_max ** 2, vehicle.a_lat_max / np.maximum(edge_k, 1e-15))
    steering = np.arctan(vehicle.wheelbase * k)
    steering_change = np.abs(np.roll(steering, -1) - steering)
    rate_cap = (vehicle.max_steer_rate * ds / np.maximum(steering_change, 1e-15)) ** 2
    edge_cap = np.minimum(edge_cap, rate_cap)
    w = np.minimum(edge_cap, np.roll(edge_cap, 1))
    b = edge_k / vehicle.a_lat_max
    b_squared = b * b
    accel_step, brake_step = 2 * ds * vehicle.a_acc, 2 * ds * vehicle.a_dec
    accel_denom, brake_denom = 1 + (accel_step * b) ** 2, 1 + (brake_step * b) ** 2

    def reachable(low, a, denominator):
        return (low + a * np.sqrt(np.maximum(0., denominator - b_squared * low * low))) / denominator

    for _ in range(max_iterations):
        before = w.copy()
        # Vectorized simultaneous relaxation is independent of the start index.
        forward = np.roll(reachable(before, accel_step, accel_denom), 1)
        backward = reachable(np.roll(before, -1), brake_step, brake_denom)
        w = np.minimum(before, np.minimum(forward, backward))
        if np.max(before - w) < tolerance:
            break
    else:
        raise RuntimeError("Periodic speed relaxation did not converge")
    velocity = np.sqrt(np.maximum(w, 0.))
    if np.any(velocity < 1e-6):
        raise ValueError("Path requires effectively zero speed")
    lateral = np.sqrt(np.minimum(vehicle.v_max ** 2, vehicle.a_lat_max / np.maximum(np.abs(k), 1e-15)))
    return lateral, velocity


def generate_velocity_profile(opt_x, opt_y, a_lat_max=10, a_acc=5, a_dec=8, v_max=25, *, vehicle=None):
    """Legacy four-array return; curvature is now signed."""
    p = _points(np.column_stack((opt_x, opt_y)))
    spline, stations = periodic_spline(p)
    k = _curvature(spline, stations)
    ends = np.r_[stations[1:], spline.x[-1]]
    nodes, weights = np.polynomial.legendre.leggauss(4)
    uq = stations[:, None] + (nodes + 1) * (ends - stations)[:, None] / 2
    ds = (np.linalg.norm(spline(uq, 1), axis=2) @ weights) * (ends - stations) / 2
    vehicle = vehicle or Vehicle(a_lat_max=a_lat_max, a_acc=a_acc, a_dec=a_dec, v_max=v_max)
    lateral, velocity = speed_profile(k, ds, vehicle)
    return lateral, velocity, np.r_[0., np.cumsum(ds[:-1])], k


def metrics(points, curvature, ds, velocity, track, vehicle, guard=0.):
    vnext = np.roll(velocity, -1)
    ax = (vnext ** 2 - velocity ** 2) / (2 * ds)
    edge_lat = np.maximum(velocity ** 2, vnext ** 2) * np.maximum(np.abs(curvature), np.abs(np.roll(curvature, -1)))
    ellipse = (ax / np.where(ax >= 0, vehicle.a_acc, vehicle.a_dec)) ** 2 + (edge_lat / vehicle.a_lat_max) ** 2
    steer = np.arctan(vehicle.wheelbase * curvature)
    rate = np.abs(np.roll(steer, -1) - steer) / ds * np.maximum(velocity, vnext)
    return {
        "lap_time_s": float(np.sum(2 * ds / (velocity + vnext))),
        "length_m": float(ds.sum()),
        "min_clearance_lower_bound_m": float(np.min(track.clearance(points)) - guard),
        "required_clearance_m": float(vehicle.radius),
        "max_acceleration_mps2": float(max(0., ax.max())),
        "max_deceleration_mps2": float(max(0., -ax.min())),
        "seam_acceleration_mps2": float(ax[-1]),
        "max_lateral_acceleration_mps2": float(edge_lat.max()),
        "max_friction_ellipse": float(ellipse.max()),
        "max_abs_curvature_per_m": float(np.max(np.abs(curvature))),
        "max_steering_rate_radps": float(rate.max()),
    }


@dataclass
class Plan:
    points: np.ndarray
    curvature: np.ndarray
    ds: np.ndarray
    velocity: np.ndarray
    lateral_velocity: np.ndarray
    alpha: np.ndarray
    metrics: dict
    solver: list
    control_points: np.ndarray


def plan_racing_line(track, vehicle=None, *, objective="lap_time", control_points=48,
                     samples=600, max_iterations=200):
    """Local path optimization; lap_time re-solves speed for every candidate.

    No claim of global optimality. Constraints are checked against original
    boundary polygons, and the returned continuous spline has a clearance guard.
    """
    vehicle = vehicle or Vehicle()
    if objective not in ("lap_time", "curvature"):
        raise ValueError("objective must be lap_time or curvature")
    if control_points < 8 or samples < 4 * control_points or max_iterations < 1:
        raise ValueError("Require >=8 controls, >=4 samples/control and positive iterations")
    left, right = track.cross_sections(control_points)
    width = np.linalg.norm(right - left, axis=1)
    if np.any(width <= 2 * vehicle.radius):
        raise ValueError("Track is too narrow for the configured vehicle and margin")
    low, high = vehicle.radius / width, 1 - vehicle.radius / width
    knots = np.linspace(0, 1, control_points + 1)
    # SLSQP asks for the same finite-difference candidates separately for the
    # objective and constraints. Reuse those geometries without changing math.
    cache = OrderedDict()
    solver_info = []

    def evaluate(alpha):
        key = alpha.tobytes()
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        control = left + alpha[:, None] * (right - left)
        spline = CubicSpline(knots, np.vstack((control, control[0])), bc_type="periodic")
        p, k, ds, guard = _sample_spline(spline, samples)
        clearance = track.clearance(p) - vehicle.radius - guard
        steering_constraint = vehicle.max_curvature - np.abs(k)
        constraints = np.r_[clearance, steering_constraint]
        curvature_cost = float(np.sum((k * k + np.roll(k * k, -1)) * ds / 2))
        value = (p, k, ds, guard, constraints, curvature_cost)
        cache[key] = value
        if len(cache) > 4 * (control_points + 1):
            cache.popitem(last=False)
        return value

    alpha = np.full(control_points, .5)
    curvature_seed_time = None
    stages = ["curvature"] + (["lap_time"] if objective == "lap_time" else [])
    for stage in stages:
        best = [np.inf, None]

        def cost(a):
            p, k, ds, guard, constraints, curvature_cost = evaluate(a)
            if stage == "curvature":
                value = curvature_cost
            else:
                # Steering violations are handled by constraints. Temporarily
                # relax that check for infeasible line-search candidates only.
                relaxed = Vehicle(**{**asdict(vehicle), "max_steer": np.pi / 2 - 1e-6})
                _, v = speed_profile(k, ds, relaxed)
                value = float(np.sum(2 * ds / (v + np.roll(v, -1))))
            if constraints.min() >= -1e-7 and value < best[0]:
                best[:] = [value, a.copy()]
            return value

        # The speed envelope switches active acceleration/braking constraints.
        # Asking for sub-microsecond time changes wastes iterations at those
        # kinks; geometry is still independently validated at 1e-6 metres.
        objective_tolerance = 1e-7 if stage == "curvature" else 1e-5
        result = minimize(cost, alpha, method="SLSQP", bounds=list(zip(low, high)),
                          constraints={"type": "ineq", "fun": lambda a: evaluate(a)[4]},
                          options={"maxiter": max_iterations, "ftol": objective_tolerance, "disp": False})
        cost(result.x)
        solver_info.append({"stage": stage, "success": bool(result.success),
                            "message": str(result.message), "iterations": int(result.nit),
                            "objective_tolerance": objective_tolerance})
        if best[1] is None:
            raise RuntimeError(f"No feasible racing line: {result.message}. Check vehicle dimensions and corridor.")
        alpha = best[1]
        if stage == "curvature":
            _, seed_k, seed_ds, _, _, _ = evaluate(alpha)
            _, seed_v = speed_profile(seed_k, seed_ds, vehicle)
            curvature_seed_time = float(np.sum(2 * seed_ds / (seed_v + np.roll(seed_v, -1))))
        if not result.success:
            warnings.warn(f"{stage}: {result.message}; returning the best validated feasible candidate", RuntimeWarning)

    p, k, ds, guard, constraints, _ = evaluate(alpha)
    lateral, velocity = speed_profile(k, ds, vehicle)
    report = metrics(p, k, ds, velocity, track, vehicle, guard)
    if constraints.min() < -1e-6 or report["max_friction_ellipse"] > 1 + 1e-6:
        raise RuntimeError("Final trajectory failed validation")
    _check_simple(p, "racing line")
    report["objective"] = objective
    report["global_optimum_guaranteed"] = False
    report["vehicle_geometry"] = "circumscribed disk" if vehicle.length else "width-based disk"
    report["samples"] = samples
    report["control_points"] = control_points
    report["max_iterations_per_stage"] = max_iterations
    report["curvature_seed_lap_time_s"] = curvature_seed_time
    control = left + alpha[:, None] * (right - left)
    return Plan(p, k, ds, velocity, lateral, alpha, report, solver_info, control)


def get_optimal_line(x_left, y_left, x_right, y_right, N=None, *, vehicle=None, **kwargs):
    """Convenience wrapper; third return is alpha at the NEW control stations.

    Unlike the old implementation these stations are reconstructed normal
    cross-sections. Do not use alpha to interpolate the input boundary arrays.
    Prefer plan_racing_line with original cone polygons for full validation.
    """
    l, r = np.column_stack((x_left, y_left)), np.column_stack((x_right, y_right))
    if l.shape != r.shape or (N is not None and N != len(l)):
        raise ValueError("Boundary lengths and optional N must match")
    kwargs.setdefault("samples", len(l))
    kwargs.setdefault("control_points", min(48, len(l) // 4))
    plan = plan_racing_line(Track(l, r), vehicle, **kwargs)
    return plan.points[:, 0], plan.points[:, 1], plan.alpha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / "fsd_racetrack_dataset-main" / "dataset")
    parser.add_argument("--track", type=int, default=1)
    parser.add_argument("--objective", choices=("lap_time", "curvature"), default="lap_time")
    parser.add_argument("--controls", type=int, default=48)
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--config", type=Path, help="JSON object with Vehicle field names")
    parser.add_argument("--output", type=Path, help="Output prefix for CSV, JSON and PNG")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    settings = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    vehicle = Vehicle(**settings)
    track = load_track(args.data_dir, args.track)
    plan = plan_racing_line(track, vehicle, objective=args.objective, control_points=args.controls,
                            samples=args.samples, max_iterations=args.max_iterations)
    report = {**plan.metrics, "track_id": args.track, "data_dir": str(args.data_dir.resolve()),
              "vehicle": asdict(vehicle), "solver": plan.solver,
              "path_control_points_m": plan.control_points.tolist(),
              "path_knot_parameter": "uniform in [0,1]; append first control point; periodic cubic spline"}
    print(json.dumps({key: value for key, value in report.items() if not key.startswith("path_")},
                     indent=2, ensure_ascii=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        s = np.r_[0., np.cumsum(plan.ds[:-1])]
        np.savetxt(args.output.with_suffix(".csv"), np.column_stack((s, plan.points, plan.curvature, plan.velocity, plan.ds)),
                   delimiter=",", header="s_m,x_m,y_m,signed_curvature_per_m,speed_mps,segment_length_m", comments="")
    if args.output or not args.no_show:
        import matplotlib
        if args.no_show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for boundary, color in ((track.left, "#b89b00"), (track.right, "#2474bd")):
            b = np.vstack((boundary, boundary[0]))
            axes[0].plot(b[:, 0], b[:, 1], ".-", color=color, markersize=3)
        scatter = axes[0].scatter(*plan.points.T, c=plan.velocity, cmap="viridis", s=8)
        fig.colorbar(scatter, ax=axes[0], label="Speed (m/s)")
        axes[0].set(aspect="equal", title=f"Track {args.track} | estimated lap {plan.metrics['lap_time_s']:.2f} s", xlabel="x (m)", ylabel="y (m)")
        s = np.r_[0., np.cumsum(plan.ds)]
        axes[1].plot(s, np.r_[plan.lateral_velocity, plan.lateral_velocity[0]], "--", label="Lateral limit")
        axes[1].plot(s, np.r_[plan.velocity, plan.velocity[0]], label="Coupled periodic speed")
        axes[1].set(xlabel="Distance (m)", ylabel="Speed (m/s)", title="Flying lap; includes closing segment")
        axes[1].legend()
        for ax in axes:
            ax.grid(alpha=.25)
        fig.tight_layout()
        if args.output:
            fig.savefig(args.output.with_suffix(".png"), dpi=160)
        if not args.no_show:
            plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
