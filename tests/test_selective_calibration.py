import unittest

from src.selective_calibration import CoverageRiskCalibrator, expected_calibration_error


class SelectiveCalibrationTests(unittest.TestCase):
    def test_threshold_meets_validation_risk(self):
        calibrator = CoverageRiskCalibrator(target_risk=0.2)
        threshold = calibrator.fit([0.95, 0.8, 0.6, 0.2], [True, True, False, False])
        self.assertGreaterEqual(threshold, 0.8)
        summary = calibrator.summarize([0.95, 0.8, 0.6, 0.2], [True, True, False, False])
        self.assertLessEqual(summary.selective_risk, 0.2)

    def test_ece_is_zero_for_perfect_bins(self):
        self.assertAlmostEqual(expected_calibration_error([1.0, 0.0], [True, False]), 0.0)


if __name__ == "__main__":
    unittest.main()
