import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from .s3_experiment_logger import S3ExperimentLogger, _parse_s3_uri


class FakeS3Client:
    def __init__(self):
        self.uploads = []

    def upload_file(self, filename, bucket, key):
        self.uploads.append((Path(filename).name, bucket, key))


class FlakyS3Client(FakeS3Client):
    """Fake client that starts failing once `fail` is set."""

    def __init__(self):
        super().__init__()
        self.fail = False

    def upload_file(self, filename, bucket, key):
        if self.fail:
            raise RuntimeError("simulated S3 outage")
        super().upload_file(filename, bucket, key)


class S3ExperimentLoggerTest(unittest.TestCase):
    def test_writes_training_job_contract_and_uploads(self):
        environment = {
            "BT_TRAINING_JOB_ID": "job-123",
            "BT_TRAINING_JOB_NAME": "qwen-sft",
            "BT_TRAINING_PROJECT_ID": "project-456",
            "BT_TRAINING_PROJECT_NAME": "customer-sft",
            "BT_NODE_RANK": "0",
            "RANK": "0",
        }
        client = FakeS3Client()

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, environment, clear=True):
                logger = S3ExperimentLogger(
                    {"learning_rate": 3e-5, "batch_size": 32},
                    s3_uri="s3://experiment-bucket/training/runs",
                    local_root=temp_dir,
                    sync_every_seconds=3600,
                    s3_client=client,
                )
                logger.log(1, train_loss=2.4, learning_rate=3e-5)
                logger.close()

                run_dir = Path(temp_dir) / "job-123"
                meta = json.loads((run_dir / "meta.json").read_text())
                metrics = [
                    json.loads(line)
                    for line in (run_dir / "metrics.jsonl").read_text().splitlines()
                ]

        self.assertEqual(meta["source"], "baseten-training-jobs-v1")
        self.assertEqual(meta["training_job_id"], "job-123")
        self.assertEqual(meta["training_project_id"], "project-456")
        self.assertEqual(metrics[0]["step"], 1)
        self.assertEqual(metrics[0]["train_loss"], 2.4)
        self.assertIn(
            (
                "metrics.jsonl",
                "experiment-bucket",
                "training/runs/job-123/metrics.jsonl",
            ),
            client.uploads,
        )

    def test_non_primary_process_does_not_write(self):
        client = FakeS3Client()
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(
                os.environ,
                {"BT_TRAINING_JOB_ID": "job-123", "RANK": "1"},
                clear=True,
            ):
                logger = S3ExperimentLogger(
                    {},
                    s3_uri="s3://experiment-bucket/runs",
                    local_root=temp_dir,
                    s3_client=client,
                )
                logger.log(1, train_loss=2.4)
                logger.close()
                self.assertFalse((Path(temp_dir) / "job-123").exists())
        self.assertEqual(client.uploads, [])

    def test_rejects_non_s3_uri(self):
        with self.assertRaises(ValueError):
            _parse_s3_uri("https://example.com/runs")

    def _make_logger(self, environment, temp_dir, client, **kwargs):
        with patch.dict(os.environ, environment, clear=True):
            return S3ExperimentLogger(
                {"learning_rate": 3e-5},
                s3_uri="s3://experiment-bucket/runs",
                local_root=temp_dir,
                s3_client=client,
                **kwargs,
            )

    def test_bt_node_rank_zero_is_primary_without_rank(self):
        client = FakeS3Client()
        environment = {"BT_TRAINING_JOB_ID": "job-123", "BT_NODE_RANK": "0"}
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = self._make_logger(environment, temp_dir, client)
            self.assertTrue(logger.enabled)
            logger.log(1, train_loss=2.4)
            logger.close()
            self.assertTrue((Path(temp_dir) / "job-123" / "metrics.jsonl").exists())
        self.assertGreater(len(client.uploads), 0)

    def test_bt_node_rank_nonzero_is_not_primary(self):
        client = FakeS3Client()
        environment = {"BT_TRAINING_JOB_ID": "job-123", "BT_NODE_RANK": "1"}
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = self._make_logger(environment, temp_dir, client)
            self.assertFalse(logger.enabled)
            logger.log(1, train_loss=2.4)
            logger.close()
            self.assertFalse((Path(temp_dir) / "job-123").exists())
        self.assertEqual(client.uploads, [])

    def test_empty_rank_falls_through_to_bt_node_rank(self):
        client = FakeS3Client()
        environment = {
            "BT_TRAINING_JOB_ID": "job-123",
            "RANK": "",
            "BT_NODE_RANK": "0",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = self._make_logger(environment, temp_dir, client)
            self.assertTrue(logger.enabled)
            logger.close()

    def test_upload_failure_strict_raises(self):
        client = FlakyS3Client()
        environment = {"BT_TRAINING_JOB_ID": "job-123", "RANK": "0"}
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = self._make_logger(
                environment, temp_dir, client, strict_uploads=True
            )
            client.fail = True
            with self.assertRaises(RuntimeError):
                logger.sync()
            logger.closed = True  # keep the atexit close() from re-raising at exit

    def test_upload_failure_non_strict_returns_false_and_continues(self):
        client = FlakyS3Client()
        environment = {"BT_TRAINING_JOB_ID": "job-123", "RANK": "0"}
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = self._make_logger(environment, temp_dir, client)
            client.fail = True
            self.assertFalse(logger.sync())
            logger.log(1, train_loss=2.4)  # must not raise
            logger.close()
            metrics = (
                (Path(temp_dir) / "job-123" / "metrics.jsonl").read_text().splitlines()
            )
            self.assertEqual(len(metrics), 1)

    def test_sync_interval_gates_uploads_and_resyncs_only_metrics(self):
        client = FakeS3Client()
        environment = {"BT_TRAINING_JOB_ID": "job-123", "RANK": "0"}
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, environment, clear=True):
                logger = S3ExperimentLogger(
                    {"learning_rate": 3e-5},
                    s3_uri="s3://experiment-bucket/runs",
                    local_root=temp_dir,
                    sync_every_seconds=60,
                    s3_client=client,
                )
                uploads_after_init = len(client.uploads)
                self.assertEqual(uploads_after_init, 3)  # full pass at init

                with patch.object(
                    time, "monotonic", return_value=logger._last_sync + 10
                ):
                    logger.log(1, train_loss=2.4)
                self.assertEqual(len(client.uploads), 3)  # interval not elapsed

                with patch.object(
                    time, "monotonic", return_value=logger._last_sync + 61
                ):
                    logger.log(2, train_loss=2.3)
                # interval elapsed: exactly one more upload, metrics.jsonl only
                self.assertEqual(len(client.uploads), 4)
                self.assertEqual(client.uploads[-1][0], "metrics.jsonl")
                logger.closed = True  # skip close() to keep upload counts exact

    def test_log_survives_non_serializable_value(self):
        client = FakeS3Client()
        environment = {"BT_TRAINING_JOB_ID": "job-123", "RANK": "0"}

        class FakeTensor:
            def __repr__(self):
                return "tensor(2.4000)"

        with tempfile.TemporaryDirectory() as temp_dir:
            logger = self._make_logger(environment, temp_dir, client)
            logger.log(1, train_loss=FakeTensor(), learning_rate=3e-5)  # must not raise
            logger.close()
            metrics = [
                json.loads(line)
                for line in (Path(temp_dir) / "job-123" / "metrics.jsonl")
                .read_text()
                .splitlines()
            ]
        self.assertEqual(metrics[0]["train_loss"], "tensor(2.4000)")
        self.assertEqual(metrics[0]["learning_rate"], 3e-5)


if __name__ == "__main__":
    unittest.main()
