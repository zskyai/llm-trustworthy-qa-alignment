import unittest

from src.selective_calibration import CoverageRiskCalibrator, expected_calibration_error


class SelectiveCalibrationTests(unittest.TestCase):
    def test_threshold_meets_validation_risk(self):
        calibrator = CoverageRiskCalibrator(target_risk=0.66)
        threshold = calibrator.fit([0.95, 0.8, 0.6, 0.2], [True, True, False, False])
        self.assertGreaterEqual(threshold, 0.8)
        summary = calibrator.summarize([0.95, 0.8, 0.6, 0.2], [True, True, False, False])
        self.assertLessEqual(summary.selective_risk, 0.2)
        self.assertLessEqual(summary.risk_upper_bound, 0.66)

    def test_threshold_maximizes_coverage_under_risk_target(self):
        calibrator = CoverageRiskCalibrator(target_risk=0.8)
        threshold = calibrator.fit([0.95, 0.8, 0.6], [True, True, False])
        self.assertEqual(threshold, 0.0)
        summary = calibrator.summarize([0.95, 0.8, 0.6], [True, True, False])
        self.assertAlmostEqual(summary.coverage, 1.0)

    def test_rejects_all_when_validation_risk_target_is_infeasible(self):
        calibrator = CoverageRiskCalibrator(target_risk=0.0)
        calibrator.fit([1.0, 0.9], [False, False])
        self.assertFalse(calibrator.predict(1.0))
        self.assertEqual(calibrator.summarize([1.0, 0.9], [False, False]).coverage, 0.0)

    def test_ece_is_zero_for_perfect_bins(self):
        self.assertAlmostEqual(expected_calibration_error([1.0, 0.0], [True, False]), 0.0)


if __name__ == "__main__":
    unittest.main()
