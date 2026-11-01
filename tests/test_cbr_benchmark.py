"""Hardware-free checks for the CBR rerun's measurement and failure handling."""
import contextlib
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import zlib

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cbr_benchmark", ROOT / "experiments/benchmark_cbr.py")
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)
RETAINED = ROOT / BENCH.RETAINED_PATH


class AccountingTests(unittest.TestCase):
    def fixture(self, folder):
        start = 100_000_000_000
        benchmark = {"codec": "h264", "target_mbps": 12, "warmup_frames": 1,
                     "measured_frames": 3, "start_mono_ns": start,
                     "end_mono_ns": start + 2_500_000_000,
                     "elapsed_seconds": 2.5, "fps": 1.2}
        rows = []
        for fid, offset in enumerate((-1_000_000_000, 400_000_000, 1_300_000_000, 2_100_000_000)):
            rows.append({"frame_id": fid, "qp_bytes": 32768, "qp_crc32": zlib.crc32(bytes(32768)),
                         "target_bitrate_bps": 12_000_000, "applied_bitrate_bps": 12_000_000,
                         "encoded_bytes": 20, "encode_begin_mono_ns": start + offset - 100,
                         "encode_end_mono_ns": start + offset - 1, "completion_mono_ns": start + offset})
        path = Path(folder) / "events.csv"
        self.write_rows(path, rows)
        return path, benchmark, rows

    def write_rows(self, path, rows):
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_windows_retain_all_frames_and_mark_partial_interval(self):
        with tempfile.TemporaryDirectory() as folder:
            path, benchmark, _ = self.fixture(folder)
            audit, measured = BENCH.audit_events(path, benchmark)
            self.assertEqual(audit["measured_frames"], 3)
            output = Path(folder) / "windows.csv"
            BENCH.write_windows(output, benchmark, measured)
            with output.open(newline="") as stream:
                windows = list(csv.DictReader(stream))
            self.assertEqual([w["full_second"] for w in windows], ["True", "True", "False"])
            self.assertEqual(sum(int(w["completed_frames"]) for w in windows), 3)
            self.assertEqual([float(w["fps"]) for w in windows], [1, 1, 2])

    def test_rejects_missing_frame_nonzero_qp_wrong_rate_and_wrong_timing(self):
        for field, value in [("frame_id", 8), ("qp_crc32", 9), ("applied_bitrate_bps", 35_000_000), ("completion_mono_ns", 99_000_000_000)]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as folder:
                path, benchmark, rows = self.fixture(folder)
                rows[1][field] = value
                self.write_rows(path, rows)
                with self.assertRaises(ValueError):
                    BENCH.audit_events(path, benchmark)
        with tempfile.TemporaryDirectory() as folder:
            path, benchmark, _ = self.fixture(folder)
            benchmark["fps"] = 60
            with self.assertRaises(ValueError):
                BENCH.audit_events(path, benchmark)

    def test_failed_conditions_remain_in_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = [{"run": f"r{i}", "video": "ballet", "codec": "av1", "target_mbps": 12, "repeat": 1} for i in range(3)]
            verdict = BENCH.summarize(root, plan, {"r0": {"status": "VALID", "fps": 30.0}, "r1": {"status": "ERROR", "error": "failed"}})
            with (root / "SUMMARY.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(verdict["valid_runs"], 1)
            self.assertEqual([r["status"] for r in rows], ["VALID", "ERROR", "NOT_RUN"])
            self.assertEqual(rows[0]["fps"], "30.0")


class PipelineTests(unittest.TestCase):
    def run_pipeline(self, corrupt=False):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            code = root / "code"
            code.mkdir()
            BENCH.materialize(code, RETAINED)
            support = (code / "cbr_support.py").read_text()
            self.assertNotIn("live_model", support)
            self.assertNotIn("import numpy", support)
            self.assertNotIn("class Inference", support)
            fake = root / "bin"
            fake.mkdir()
            ffmpeg = fake / "ffmpeg"
            ffmpeg.write_text("#!/usr/bin/env python3\nimport sys\nfor i in range(5):\n sys.stdout.buffer.write(bytes([i])*12)\nsys.stdout.buffer.flush()\n")
            ffmpeg.chmod(0o755)
            encoder = fake / "encoder"
            encoder.write_text('''#!/usr/bin/env python3
import os,struct,sys,time
inp=struct.Struct("<4sQIII"); out=struct.Struct("<4sQQQII")
def read(n):
    result=b""
    while len(result)<n:
        part=sys.stdin.buffer.read(n-len(result))
        if not part: raise EOFError()
        result+=part
    return result
sys.stdout.buffer.write(b"RDY1"); sys.stdout.buffer.flush()
while True:
    try: magic,fid,raw_size,qp_size,rate=inp.unpack(read(inp.size))
    except EOFError: break
    assert magic==b"FRM2" and raw_size==12
    assert read(qp_size)==bytes(qp_size)
    assert os.pread(int(sys.argv[3]),raw_size,0)==bytes([fid])*12
    begin=time.monotonic_ns(); end=time.monotonic_ns()
    oid=fid+1 if os.environ.get("CBR_TEST_CORRUPT_ID")=="1" else fid
    sys.stdout.buffer.write(out.pack(b"AU02",oid,begin,end,1,rate)+b"x")
    sys.stdout.buffer.flush()
''')
            encoder.chmod(0o755)
            output = root / "output"
            old_path = list(sys.path)
            sys.path.insert(0, str(code))
            saved_modules = {key: sys.modules.pop(key, None) for key in ("cbr_support", "shared_frame")}
            try:
                import cbr_support
                cbr_support.W, cbr_support.H, cbr_support.FRAME_BYTES = 4, 2, 12
                argv = ["--video", str(root / "fake.mp4"), "--encoder", str(encoder), "--codec", "h264",
                        "--bitrate", "12", "--output", str(output), "--frames", "3", "--warmup", "2"]
                with mock.patch.dict(os.environ, {"PATH": str(fake) + os.pathsep + os.environ["PATH"], "CBR_TEST_CORRUPT_ID": "1" if corrupt else "0"}), contextlib.redirect_stdout(io.StringIO()):
                    if corrupt:
                        with self.assertRaisesRegex(RuntimeError, "completed NVENC frame"):
                            BENCH.worker_main(argv)
                        self.assertFalse((output / "benchmark.json").exists())
                    else:
                        self.assertEqual(BENCH.worker_main(argv), 0)
                        result = json.loads((output / "benchmark.json").read_text())
                        self.assertEqual(result["measured_inference_calls"], 0)
                        with (output / "producer_events.csv").open(newline="") as stream:
                            rows = list(csv.DictReader(stream))
                        self.assertEqual([int(r["frame_id"]) for r in rows], list(range(5)))
                        self.assertLessEqual(int(rows[1]["completion_mono_ns"]), result["start_mono_ns"])
                        self.assertLess(result["start_mono_ns"], int(rows[2]["completion_mono_ns"]))
                        self.assertLessEqual(int(rows[-1]["completion_mono_ns"]), result["end_mono_ns"])
            finally:
                sys.path[:] = old_path
                for key, value in saved_modules.items():
                    sys.modules.pop(key, None)
                    if value is not None:
                        sys.modules[key] = value

    @unittest.skipUnless(hasattr(os, "memfd_create"), "Linux shared-frame test")
    def test_pipeline_preserves_frame_identity_and_warmup_gate(self):
        self.run_pipeline()

    @unittest.skipUnless(hasattr(os, "memfd_create"), "Linux shared-frame test")
    def test_pipeline_rejects_incorrect_native_frame_identity(self):
        self.run_pipeline(corrupt=True)


if __name__ == "__main__":
    unittest.main()
