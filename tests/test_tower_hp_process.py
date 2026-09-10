import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tower_hp_process import TowerHPProcess


class TowerHPProcessTests(unittest.TestCase):
    def setUp(self):
        fixture = Path(__file__).parent / "fixtures" / "ocr_worker"
        self.environment = patch.dict(os.environ, {"PYTHONPATH": str(fixture)})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def start(self, **kwargs):
        worker = TowerHPProcess(model_name="test", startup_timeout=15, **kwargs)
        self.addCleanup(worker.close)
        return worker

    def predict(self, worker, *values):
        return worker.predict(input=[np.full((4, 8, 3), v, np.uint8) for v in values],
                              batch_size=len(values))

    def test_persistent_process_preserves_order_and_closes(self):
        worker = self.start()
        self.assertNotEqual(worker.pid, os.getpid())
        self.assertEqual([r["rec_text"] for r in self.predict(worker, 123, 45)], ["123", "45"])
        self.assertEqual(self.predict(worker, 67)[0]["rec_text"], "67")
        self.assertIsNone(worker._process.poll())
        worker.close()
        worker.close()
        self.assertIsNotNone(worker._process.poll())
        self.assertNotIn("paddle", sys.modules)

    def test_batch_error_is_reported_and_worker_can_continue(self):
        worker = self.start()
        with self.assertRaisesRegex(RuntimeError, "test batch failure"):
            self.predict(worker, 249)
        self.assertEqual(self.predict(worker, 123)[0]["rec_text"], "123")

    def test_startup_error(self):
        with self.assertRaisesRegex(RuntimeError, "test initialization failure"):
            TowerHPProcess(model_name="fail", startup_timeout=15)

    def test_timeout_kills_worker_and_rejects_late_results(self):
        worker = self.start(request_timeout=.05)
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            self.predict(worker, 250)
        self.assertIsNotNone(worker._process.poll())
        with self.assertRaisesRegex(RuntimeError, "not running"):
            self.predict(worker, 123)

    def test_crash_does_not_hang_parent(self):
        worker = self.start()
        with self.assertRaisesRegex(RuntimeError, "exited"):
            self.predict(worker, 251)


if __name__ == "__main__":
    unittest.main()
