import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from . import s3_experiment_logger
from .s3_experiment_logger import S3ExperimentLogger, _parse_s3_uri


def read_metrics(run_dir):
    """All metrics records in a run dir, chunk files concatenated in order."""
    return [
        json.loads(line)
        for path in sorted(Path(run_dir).glob("metrics*.jsonl"))
        for line in path.read_text().splitlines()
    ]


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
                metrics = read_metrics(run_dir)

        self.assertEqual(meta["source"], "baseten-training-jobs-v1")
        self.assertEqual(meta["training_job_id"], "job-123")
        self.assertEqual(meta["training_project_id"], "project-456")
        self.assertEqual(metrics[0]["step"], 1)
        self.assertEqual(metrics[0]["train_loss"], 2.4)
        self.assertIn(
            (
                "metrics-00001.jsonl",
                "experiment-bucket",
                "training/runs/job-123/metrics-00001.jsonl",
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
            self.assertTrue(
                (Path(temp_dir) / "job-123" / "metrics-00001.jsonl").exists()
            )
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
            self.assertEqual(len(read_metrics(Path(temp_dir) / "job-123")), 1)

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
                # interval elapsed: exactly one more upload, the metrics chunk
                self.assertEqual(len(client.uploads), 4)
                self.assertEqual(client.uploads[-1][0], "metrics-00001.jsonl")
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
            metrics = read_metrics(Path(temp_dir) / "job-123")
        self.assertEqual(metrics[0]["train_loss"], "tensor(2.4000)")
        self.assertEqual(metrics[0]["learning_rate"], 3e-5)

    def test_chunk_rolls_at_size_threshold(self):
        client = FakeS3Client()
        environment = {"BT_TRAINING_JOB_ID": "job-123", "RANK": "0"}
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(s3_experiment_logger, "METRICS_CHUNK_MAX_BYTES", 120):
                logger = self._make_logger(environment, temp_dir, client)
                for step in range(1, 6):  # each record is ~55 bytes
                    logger.log(step, train_loss=2.4)
                logger.closed = True  # keep atexit close() out of the counts

            run_dir = Path(temp_dir) / "job-123"
            chunks = sorted(p.name for p in run_dir.glob("metrics*.jsonl"))
            self.assertGreater(len(chunks), 1)
            self.assertEqual(chunks[0], "metrics-00001.jsonl")
            # the record that crossed the threshold seals the chunk
            self.assertGreaterEqual(
                (run_dir / "metrics-00001.jsonl").stat().st_size, 120
            )
            # concatenating chunks in name order preserves step order
            self.assertEqual(
                [record["step"] for record in read_metrics(run_dir)], [1, 2, 3, 4, 5]
            )

    def test_periodic_sync_uploads_only_changed_chunks(self):
        client = FakeS3Client()
        environment = {"BT_TRAINING_JOB_ID": "job-123", "RANK": "0"}
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(s3_experiment_logger, "METRICS_CHUNK_MAX_BYTES", 120):
                logger = self._make_logger(
                    environment, temp_dir, client, sync_every_seconds=60
                )
                self.assertEqual(len(client.uploads), 3)  # full pass at init

                def log_and_sync(step, **metrics):
                    with patch.object(
                        time, "monotonic", return_value=logger._last_sync + 61
                    ):
                        logger.log(step, **metrics)

                log_and_sync(1, train_loss=2.4)
                # active chunk re-uploaded, static files skipped
                self.assertEqual(client.uploads[3:], [_upload("metrics-00001.jsonl")])

                log_and_sync(2, train_loss=2.3, filler="x" * 80)  # forces a roll
                # sealed chunk 1 uploads once more with its final bytes...
                self.assertEqual(client.uploads[4:], [_upload("metrics-00001.jsonl")])

                log_and_sync(3, train_loss=2.2)
                # ...then only the new chunk uploads; chunk 1 is never re-PUT
                self.assertEqual(client.uploads[5:], [_upload("metrics-00002.jsonl")])

                names = [name for name, _, _ in client.uploads]
                self.assertEqual(names.count("hyperparams.json"), 1)
                self.assertEqual(names.count("meta.json"), 1)
                self.assertEqual(names.count("metrics-00001.jsonl"), 3)
                logger.closed = True  # keep atexit close() out of the counts

    def test_close_does_final_full_pass(self):
        client = FakeS3Client()
        environment = {"BT_TRAINING_JOB_ID": "job-123", "RANK": "0"}
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = self._make_logger(
                environment, temp_dir, client, sync_every_seconds=3600
            )
            logger.log(1, train_loss=2.4)  # interval never elapses: no upload
            self.assertEqual(len(client.uploads), 3)
            logger.close()
            self.assertEqual(
                client.uploads[3:],
                [
                    _upload("hyperparams.json"),
                    _upload("meta.json"),
                    _upload("metrics-00001.jsonl"),
                ],
            )


def _upload(filename):
    return (filename, "experiment-bucket", f"runs/job-123/{filename}")


class ManifestAndDashboardCompatibilityTest(unittest.TestCase):
    """The readers must handle both chunked and legacy single-file layouts."""

    @classmethod
    def setUpClass(cls):
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import build_manifests
        import dashboard

        cls.build_manifests = build_manifests
        cls.dashboard = dashboard

    def _write_run(self, run_dir, metrics_by_file):
        run_dir.mkdir(parents=True)
        for filename, steps in metrics_by_file.items():
            (run_dir / filename).write_text(
                "".join(
                    json.dumps({"step": s, "time": float(s), "train_loss": 3.0 - s / 10})
                    + "\n"
                    for s in steps
                )
            )
        (run_dir / "hyperparams.json").write_text('{"learning_rate": 3e-05}\n')
        (run_dir / "meta.json").write_text('{"source": "test"}\n')

    def test_chunked_run_indexes_with_glob_and_ordered_records(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "job-chunked"
            self._write_run(
                run_dir,
                {
                    "metrics-00001.jsonl": [1, 2, 3],
                    "metrics-00002.jsonl": [4, 5, 6],
                    "metrics-00010.jsonl": [7, 8],  # non-consecutive still sorts
                },
            )
            run_json, summary, _ = self.build_manifests.detect_run(run_dir)

            self.assertEqual(run_json["metrics"]["file"], "metrics*.jsonl")
            self.assertIn("train_loss", run_json["metrics"]["series"])
            self.assertEqual(summary["last_step"], 8)

            records = self.dashboard.load_metrics_records(
                run_json["metrics"], run_dir
            )
            self.assertEqual([r["step"] for r in records], [1, 2, 3, 4, 5, 6, 7, 8])

    def test_legacy_single_file_run_still_indexes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "job-legacy"
            self._write_run(run_dir, {"metrics.jsonl": [1, 2, 3]})
            run_json, summary, _ = self.build_manifests.detect_run(run_dir)

            self.assertEqual(summary["last_step"], 3)
            # the new glob manifest resolves the legacy layout...
            records = self.dashboard.load_metrics_records(
                run_json["metrics"], run_dir
            )
            self.assertEqual([r["step"] for r in records], [1, 2, 3])
            # ...and a pre-existing run.json with the literal filename still works
            legacy_cfg = {"file": "metrics.jsonl", "x_axis": "step"}
            records = self.dashboard.load_metrics_records(legacy_cfg, run_dir)
            self.assertEqual([r["step"] for r in records], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
