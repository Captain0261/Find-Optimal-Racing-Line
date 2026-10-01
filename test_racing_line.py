"""Regression tests for geometry, periodicity and physical constraints."""
import unittest
import numpy as np
from scipy.interpolate import CubicSpline
from racing_line import (Vehicle, Track, periodic_spline, speed_profile,
                         plan_racing_line, generate_velocity_profile, load_track)
from pathlib import Path


def ring(n=96, inner=8., outer=12.):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    unit = np.column_stack((np.cos(t), np.sin(t)))
    return Track(inner * unit, outer * unit)


class PeriodicTests(unittest.TestCase):
    def test_periodic_spline_retains_last_point_and_derivatives(self):
        t = np.linspace(0, 2 * np.pi, 23, endpoint=False)
        p = np.column_stack((10 * np.cos(t), 7 * np.sin(t)))
        spline, s = periodic_spline(p)
        np.testing.assert_allclose(spline(s), p, atol=1e-12)
        for derivative in (0, 1, 2):
            np.testing.assert_allclose(spline(spline.x[0], derivative), spline(spline.x[-1], derivative), atol=1e-12)
        closed, _ = periodic_spline(np.vstack((p, p[0])))
        np.testing.assert_allclose(spline(s), closed(s), atol=1e-12)

    def test_closing_edge_obeys_acceleration(self):
        # Previously the final 0.1 m edge could jump from 1 m/s to ~8 m/s.
        k = np.array([.01, .01, .01, .01, .3])
        ds = np.array([1., 1., 1., .9, .1])
        vehicle = Vehicle()
        _, v = speed_profile(k, ds, vehicle)
        a = (np.roll(v, -1) ** 2 - v ** 2) / (2 * ds)
        self.assertLessEqual(a.max(), vehicle.a_acc + 1e-8)
        self.assertGreaterEqual(a.min(), -vehicle.a_dec - 1e-8)

    def test_shifted_start_gives_same_speed(self):
        rng = np.random.default_rng(12)
        k = rng.uniform(-.2, .2, 71)
        ds = rng.uniform(.1, 1, 71)
        _, expected = speed_profile(k, ds)
        for shift in (1, 17, 70):
            _, actual = speed_profile(np.roll(k, shift), np.roll(ds, shift))
            np.testing.assert_allclose(actual, np.roll(expected, shift), atol=1e-9)

    def test_friction_ellipse_and_steering_rate(self):
        vehicle = Vehicle(max_steer_rate=.5)
        t = np.linspace(0, 2 * np.pi, 120, endpoint=False)
        k, ds = .2 * np.sin(t), np.full(len(t), .2)
        _, v = speed_profile(k, ds, vehicle)
        vn = np.roll(v, -1)
        a = (vn ** 2 - v ** 2) / (2 * ds)
        lat = np.maximum(v ** 2, vn ** 2) * np.maximum(np.abs(k), np.abs(np.roll(k, -1)))
        ellipse = (lat / vehicle.a_lat_max) ** 2 + (a / np.where(a >= 0, vehicle.a_acc, vehicle.a_dec)) ** 2
        self.assertLessEqual(ellipse.max(), 1 + 1e-8)
        steer = np.arctan(vehicle.wheelbase * k)
        rate = np.abs(np.roll(steer, -1) - steer) / ds * np.maximum(v, vn)
        self.assertLessEqual(rate.max(), vehicle.max_steer_rate + 1e-8)

    def test_circle_has_analytical_speed(self):
        k, ds = np.full(80, .1), np.full(80, 2 * np.pi * 10 / 80)
        _, v = speed_profile(k, ds)
        np.testing.assert_allclose(v, 10., atol=1e-8)

    def test_curvature_sign(self):
        t = np.linspace(0, 2 * np.pi, 80, endpoint=False)
        x, y = 10 * np.cos(t), 10 * np.sin(t)
        _, _, _, k = generate_velocity_profile(x, y)
        self.assertTrue(np.all(k > 0))
        _, _, _, reversed_k = generate_velocity_profile(x[::-1], y[::-1])
        self.assertTrue(np.all(reversed_k < 0))

    def test_invalid_input_is_rejected(self):
        for kwargs in ({"width": 0}, {"safety_margin": -1}, {"a_acc": float("nan")}):
            with self.assertRaises(ValueError):
                Vehicle(**kwargs)
        with self.assertRaises(ValueError):
            speed_profile(np.ones(5), np.ones(5))
        with self.assertRaises(ValueError):
            speed_profile(np.zeros(5), np.zeros(5))
        with self.assertRaises(ValueError):
            periodic_spline([[0, 0], [1, 0], [1, 0], [0, 1]])


class GeometryTests(unittest.TestCase):
    def test_normal_correspondence_with_reversed_shifted_boundary(self):
        track = ring()
        track = Track(track.left, np.roll(track.right[::-1], 29, axis=0))
        left, right = track.cross_sections(24)
        np.testing.assert_allclose(np.linalg.norm(right - left, axis=1), 4., atol=.03)
        self.assertTrue(np.all(track.clearance((left + right) / 2) > 1.9))

    def test_narrow_corridor_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "too narrow"):
            plan_racing_line(ring(inner=8, outer=9), control_points=16, samples=96)

    def test_crossing_boundaries_are_rejected(self):
        with self.assertRaises(ValueError):
            Track([[0, 0], [3, 3], [0, 3], [3, 0]], [[-1, -1], [4, -1], [4, 4], [-1, 4]])

    def test_vehicle_length_uses_conservative_footprint(self):
        self.assertAlmostEqual(Vehicle(width=1.2, length=2., safety_margin=.2).radius, np.hypot(1.2, 2.) / 2 + .2)

    def test_all_dataset_corridors(self):
        data = Path(__file__).resolve().parent.parent / "fsd_racetrack_dataset-main" / "dataset"
        if not data.exists():
            self.skipTest("External dataset is not installed")
        for i in range(1, 10):
            with self.subTest(track=i):
                track = load_track(data, i)
                left, right = track.cross_sections(48)
                self.assertTrue(np.all(track.clearance((left + right) / 2) > 0))


class OptimizationTests(unittest.TestCase):
    def test_circle_objectives_and_continuous_clearance(self):
        track = ring()
        curved = plan_racing_line(track, objective="curvature", control_points=16, samples=128, max_iterations=60)
        timed = plan_racing_line(track, objective="lap_time", control_points=16, samples=128, max_iterations=60)
        # Actual curvature minimization moves OUT on a circle; time minimization
        # moves IN under a lateral-acceleration-only circle model.
        self.assertGreater(np.mean(np.linalg.norm(curved.points, axis=1)), 10.3)
        self.assertLess(np.mean(np.linalg.norm(timed.points, axis=1)), 9.7)
        self.assertLess(timed.metrics["lap_time_s"], curved.metrics["lap_time_s"])
        for plan in (curved, timed):
            self.assertGreaterEqual(plan.metrics["min_clearance_lower_bound_m"], .8 - 1e-6)
            self.assertLessEqual(plan.metrics["max_friction_ellipse"], 1 + 1e-6)
            self.assertTrue(all(stage["success"] for stage in plan.solver))
            # Independent dense check of the continuous geometry between the
            # optimization samples, using the exported control points.
            controls = plan.control_points
            curve = CubicSpline(np.linspace(0, 1, len(controls) + 1),
                                np.vstack((controls, controls[0])), bc_type="periodic")
            dense = curve(np.linspace(0, 1, 4096, endpoint=False))
            self.assertGreaterEqual(track.clearance(dense).min(),
                                    plan.metrics["min_clearance_lower_bound_m"] - 1e-6)


if __name__ == "__main__":
    unittest.main()
