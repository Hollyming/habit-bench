import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from benchmarks.locomo.evaluator import LoCoMoEvaluator


class _FakeAgentManager:
    def __init__(self, reset_error=None):
        self.reset_calls = 0
        self.reset_error = reset_error

    def reset(self):
        self.reset_calls += 1
        if self.reset_error is not None:
            raise self.reset_error


def _bare_evaluator(manager, *, run_error=None):
    evaluator = LoCoMoEvaluator.__new__(LoCoMoEvaluator)
    evaluator.agent_manager = None
    evaluator._init_dataset = lambda: setattr(evaluator, "agent_manager", manager)

    def run_loop():
        if run_error is not None:
            raise run_error

    evaluator._run_evaluation_loop = run_loop
    report = SimpleNamespace(summary={})
    evaluator._generate_report = lambda *_args: report
    evaluator._log = Mock()
    return evaluator, report


class LoCoMoFinalCleanupTests(unittest.TestCase):
    @patch("benchmarks.locomo.evaluator.get_usage_tracker")
    def test_success_resets_and_releases_final_agent(self, usage_tracker):
        manager = _FakeAgentManager()
        evaluator, expected_report = _bare_evaluator(manager)

        result = evaluator.evaluate()

        self.assertIs(result, expected_report)
        self.assertEqual(manager.reset_calls, 1)
        self.assertIsNone(evaluator.agent_manager)
        usage_tracker.return_value.reset.assert_called_once_with()

    @patch("benchmarks.locomo.evaluator.get_usage_tracker")
    def test_cleanup_failure_does_not_mask_evaluation_failure(self, _usage_tracker):
        manager = _FakeAgentManager(reset_error=RuntimeError("cleanup failed"))
        evaluator, _ = _bare_evaluator(
            manager,
            run_error=ValueError("evaluation failed"),
        )

        with self.assertRaisesRegex(ValueError, "evaluation failed"):
            evaluator.evaluate()

        self.assertEqual(manager.reset_calls, 1)
        self.assertIsNone(evaluator.agent_manager)
        evaluator._log.assert_called_once()


if __name__ == "__main__":
    unittest.main()
