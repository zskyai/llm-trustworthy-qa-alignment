import math
import unittest

import torch

from src.losses import (
    completion_logps_from_logits,
    dpo_loss,
    orpo_loss,
    simpo_loss,
)


class CompletionLogProbTests(unittest.TestCase):
    def test_completion_mask_and_length_normalization(self):
        logits = torch.zeros(1, 5, 4)
        labels = torch.tensor([[0, 1, 2, 3, 1]])
        completion_mask = torch.tensor([[0, 0, 0, 1, 1]], dtype=torch.bool)

        summed = completion_logps_from_logits(
            logits, labels, completion_mask, normalization="sum"
        )
        averaged = completion_logps_from_logits(
            logits, labels, completion_mask, normalization="mean"
        )

        self.assertEqual(summed.token_counts.tolist(), [2])
        self.assertAlmostEqual(summed.values.item(), -2.0 * math.log(4.0), places=6)
        self.assertAlmostEqual(averaged.values.item(), -math.log(4.0), places=6)

    def test_empty_completion_is_rejected(self):
        logits = torch.zeros(1, 3, 2)
        labels = torch.tensor([[0, 1, 0]])
        completion_mask = torch.zeros_like(labels, dtype=torch.bool)

        with self.assertRaises(ValueError):
            completion_logps_from_logits(logits, labels, completion_mask)


class DpoLossTests(unittest.TestCase):
    def test_policy_equal_to_reference_has_log_two_loss(self):
        chosen = torch.tensor([-3.0, -5.0])
        rejected = torch.tensor([-4.0, -6.0])
        output = dpo_loss(chosen, rejected, chosen, rejected, reduction="none")

        expected = torch.full((2,), math.log(2.0))
        self.assertTrue(torch.allclose(output.loss, expected, atol=1e-7))

    def test_gradient_increases_chosen_and_decreases_rejected(self):
        chosen = torch.tensor([-2.0], requires_grad=True)
        rejected = torch.tensor([-2.0], requires_grad=True)
        reference = torch.tensor([-2.0])

        output = dpo_loss(chosen, rejected, reference, reference, beta=0.5)
        output.loss.backward()

        self.assertLess(chosen.grad.item(), 0.0)
        self.assertGreater(rejected.grad.item(), 0.0)

    def test_extreme_log_probabilities_remain_finite(self):
        output = dpo_loss(
            torch.tensor([-1.0e4]),
            torch.tensor([-2.0e4]),
            torch.tensor([-1.0e4]),
            torch.tensor([-2.0e4]),
        )
        self.assertTrue(torch.isfinite(output.loss).item())

    def test_reference_logps_are_frozen_by_the_loss(self):
        chosen = torch.tensor([-2.0], requires_grad=True)
        rejected = torch.tensor([-3.0], requires_grad=True)
        reference_chosen = torch.tensor([-2.5], requires_grad=True)
        reference_rejected = torch.tensor([-3.0], requires_grad=True)

        output = dpo_loss(
            chosen,
            rejected,
            reference_chosen,
            reference_rejected,
        )
        output.loss.backward()

        self.assertIsNone(reference_chosen.grad)
        self.assertIsNone(reference_rejected.grad)


class SimpoLossTests(unittest.TestCase):
    def test_target_margin_has_log_two_loss(self):
        chosen = torch.tensor([-1.5])
        rejected = torch.tensor([-2.0])
        output = simpo_loss(
            chosen,
            rejected,
            beta=2.0,
            gamma_beta_ratio=0.5,
        )

        self.assertAlmostEqual(output.loss.item(), math.log(2.0), places=6)

    def test_sft_anchor_strengthens_chosen_gradient(self):
        chosen_no_anchor = torch.tensor([-2.0], requires_grad=True)
        rejected_no_anchor = torch.tensor([-2.0], requires_grad=True)
        no_anchor = simpo_loss(
            chosen_no_anchor,
            rejected_no_anchor,
            beta=1.0,
            gamma_beta_ratio=0.0,
        )
        no_anchor.loss.backward()
        self.assertLess(chosen_no_anchor.grad.item(), 0.0)
        self.assertGreater(rejected_no_anchor.grad.item(), 0.0)

        chosen_anchor = torch.tensor([-2.0], requires_grad=True)
        rejected_anchor = torch.tensor([-2.0], requires_grad=True)
        with_anchor = simpo_loss(
            chosen_anchor,
            rejected_anchor,
            beta=1.0,
            gamma_beta_ratio=0.0,
            sft_weight=0.25,
        )
        with_anchor.loss.backward()

        self.assertLess(chosen_anchor.grad.item(), chosen_no_anchor.grad.item())
        self.assertAlmostEqual(
            rejected_anchor.grad.item(), rejected_no_anchor.grad.item(), places=7
        )


class OrpoLossTests(unittest.TestCase):
    def test_near_zero_and_large_negative_logps_are_finite(self):
        output = orpo_loss(
            torch.tensor([0.0, -1.0e-12]),
            torch.tensor([0.0, -1.0e4]),
            reduction="none",
        )

        self.assertTrue(torch.isfinite(output.loss).all().item())
        self.assertTrue(torch.isfinite(output.preference_margin).all().item())

    def test_gradient_increases_chosen_and_decreases_rejected(self):
        chosen = torch.tensor([-2.0], requires_grad=True)
        rejected = torch.tensor([-2.0], requires_grad=True)

        output = orpo_loss(chosen, rejected, beta=0.2)
        output.loss.backward()

        self.assertLess(chosen.grad.item(), 0.0)
        self.assertGreater(rejected.grad.item(), 0.0)

    def test_invalid_hyperparameter_is_rejected(self):
        with self.assertRaises(ValueError):
            orpo_loss(torch.tensor([-2.0]), torch.tensor([-3.0]), beta=0.0)


if __name__ == "__main__":
    unittest.main()
