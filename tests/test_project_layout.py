"""Check input paths after placing src, tests, and data at the project root."""

import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import entanglement_disturbance as rq1
import multi_mutation_realexec_eval as multi
import rq3_eval
import rq3_manifest
import scalability_v2
import utilities


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ProjectLayoutTests(unittest.TestCase):
    def test_circuit_paths_stay_inside_this_project(self):
        expected = PROJECT_ROOT / "quantum_circuits"
        paths = {
            "shared utilities": Path(utilities.quantum_path),
            "RQ1 and RQ2": rq1.DEFAULT_QASM_DIR,
            "RQ3 manifest": rq3_manifest.CIRCUIT_DIR,
            "multiple mutations": multi.CIRCUIT_DIR,
            "scalability": scalability_v2.DEFAULT_QASM_DIR,
        }
        for name, path in paths.items():
            with self.subTest(module=name):
                self.assertEqual(path.resolve(), expected)

    def test_input_paths_stay_inside_this_project(self):
        inputs = PROJECT_ROOT / "inputs"
        paths = [
            (rq1.DEFAULT_MANIFEST, "certified_circuits.txt"),
            (scalability_v2.DEFAULT_MANIFEST, "certified_circuits.txt"),
            (rq3_eval.DEFAULT_CIRCUIT_LIST, "certified_circuits.txt"),
            (rq3_manifest.CANONICAL_KEYFILE, "rq3_circuits.txt"),
            (multi.CANONICAL_KEYFILE, "rq3_circuits.txt"),
            (rq3_manifest.DEFAULT_MANIFEST, "rq3_single_error_manifest.json"),
        ]
        for path, name in paths:
            with self.subTest(path=str(path)):
                self.assertEqual(path.resolve(), inputs / name)
                self.assertTrue(path.is_file())

    def test_fixed_inputs_remain_readable(self):
        frame = rq1.resolve_circuit_corpus()
        self.assertEqual(Path(frame["qasm_dir"]), PROJECT_ROOT / "quantum_circuits")
        self.assertEqual(len(rq3_manifest.canonical_keys()), 238)
        manifest = rq3_manifest.load_manifest(verify_runtime=False)
        self.assertEqual(len(manifest["circuits"]), 238)
        self.assertEqual(len(manifest["slots"]), 3570)

    def test_validation_helper_can_be_loaded_from_source_module(self):
        with contextlib.redirect_stdout(io.StringIO()):
            rq3_eval.validate_qmon_mechanism([])

    def test_batch_builder_still_emits_every_partition(self):
        from test_exact_execution import check_batched_instrumentation_partitions

        with contextlib.redirect_stdout(io.StringIO()):
            check_batched_instrumentation_partitions()

    def test_exported_project_does_not_depend_on_parent_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            exported = Path(temporary).resolve() / "export with spaces"
            shutil.copytree(PROJECT_ROOT / "src", exported / "src")
            shutil.copytree(PROJECT_ROOT / "tests", exported / "tests")
            shutil.copytree(PROJECT_ROOT / "inputs", exported / "inputs")
            shutil.copytree(PROJECT_ROOT / "quantum_circuits", exported / "quantum_circuits")
            # A sibling corpus must not mask a root-calculation error.
            (Path(temporary) / "quantum_circuits").mkdir()
            script = """
import contextlib, io, json, sys
from pathlib import Path
import entanglement_disturbance as rq1
import multi_mutation_realexec_eval as multi
import rq3_eval, rq3_manifest, scalability_v2, utilities
root = Path(sys.argv[1]).resolve()
for path in (rq1.DEFAULT_QASM_DIR, multi.CIRCUIT_DIR,
             rq3_manifest.CIRCUIT_DIR, scalability_v2.DEFAULT_QASM_DIR,
             Path(utilities.quantum_path)):
    assert path.resolve() == root / 'quantum_circuits', path
frame = rq1.resolve_circuit_corpus()
assert Path(frame['qasm_dir']) == root / 'quantum_circuits'
assert len(rq3_manifest.canonical_keys()) == 238
assert len(rq3_manifest.load_manifest(verify_runtime=False)['slots']) == 3570
with contextlib.redirect_stdout(io.StringIO()):
    rq3_eval.validate_qmon_mechanism([])
print(json.dumps({'root': str(root), 'ok': True}))
"""
            environment = dict(os.environ, PYTHONPATH=str(exported / "src"),
                               PYTHONDONTWRITEBYTECODE="1")
            for cwd in (exported, exported / "src", Path(temporary)):
                with self.subTest(cwd=str(cwd)):
                    result = subprocess.run(
                        [sys.executable, "-c", script, str(exported)],
                        cwd=cwd, env=environment, capture_output=True, text=True,
                        timeout=90,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(json.loads(result.stdout),
                                     {"root": str(exported), "ok": True})


if __name__ == "__main__":
    unittest.main()
