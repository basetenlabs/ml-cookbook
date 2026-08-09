import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from .s3_experiment_logger import S3ExperimentLogger, _parse_s3_uri


class FakeS3Client:
    def __init__(self):
        self.uploads = []

    def upload_file(self, filename, bucket, key):
        self.uploads.append((Path(filename).name, bucket, key))


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


if __name__ == "__main__":
    unittest.main()
