import copy
import unittest
from unittest import mock

import rq3_manifest


class ManifestRuntimeValidationTests(unittest.TestCase):
    def setUp(self):
        self.manifest = rq3_manifest.load_manifest(verify_runtime=False)
        self.stored = copy.deepcopy(self.manifest["fingerprints"])
        self.saved_cache = set(rq3_manifest._REPLAY_VERIFIED_HASHES)
        rq3_manifest._REPLAY_VERIFIED_HASHES.clear()

    def tearDown(self):
        rq3_manifest._REPLAY_VERIFIED_HASHES.clear()
        rq3_manifest._REPLAY_VERIFIED_HASHES.update(self.saved_cache)

    def _runtime_with_source(self, source_hash):
        current = copy.deepcopy(self.stored)
        current["sources"]["sha256"] = source_hash
        return current

    def test_source_fix_is_admissible_only_after_complete_replay(self):
        current = self._runtime_with_source("0" * 64)
        with mock.patch.object(
            rq3_manifest, "runtime_fingerprints", return_value=current
        ), mock.patch.object(
            rq3_manifest, "assert_manifest_replay", return_value=True
        ) as replay:
            validated = rq3_manifest.validate_manifest(
                self.manifest, verify_runtime=True
            )
        self.assertIs(validated, self.manifest)
        replay.assert_called_once_with(self.manifest)

    def test_source_fix_without_replay_fails_closed(self):
        current = self._runtime_with_source("0" * 64)
        with mock.patch.object(
            rq3_manifest, "runtime_fingerprints", return_value=current
        ):
            with self.assertRaisesRegex(ValueError, "requires complete"):
                rq3_manifest.validate_manifest(
                    self.manifest,
                    verify_runtime=True,
                    verify_replay=False,
                )

    def test_immutable_runtime_inputs_still_reject_drift(self):
        for field in ("qasm_corpus", "dependencies", "protocol"):
            with self.subTest(field=field):
                current = copy.deepcopy(self.stored)
                current[field]["sha256"] = "0" * 64
                with mock.patch.object(
                    rq3_manifest, "runtime_fingerprints", return_value=current
                ), mock.patch.object(
                    rq3_manifest, "assert_manifest_replay"
                ) as replay:
                    with self.assertRaisesRegex(
                        ValueError, f"{field} checksum mismatch"
                    ):
                        rq3_manifest.validate_manifest(
                            self.manifest, verify_runtime=True
                        )
                    replay.assert_not_called()

    def test_replay_cache_is_bound_to_current_source_hash(self):
        runtime_a = self._runtime_with_source("0" * 64)
        runtime_b = self._runtime_with_source("1" * 64)
        with mock.patch.object(
            rq3_manifest,
            "runtime_fingerprints",
            side_effect=[runtime_a, runtime_a, runtime_b],
        ), mock.patch.object(
            rq3_manifest, "assert_manifest_replay", return_value=True
        ) as replay:
            rq3_manifest.validate_manifest(
                self.manifest, verify_runtime=True
            )
            rq3_manifest.validate_manifest(
                self.manifest, verify_runtime=True
            )
            rq3_manifest.validate_manifest(
                self.manifest, verify_runtime=True
            )
        self.assertEqual(replay.call_count, 2)


if __name__ == "__main__":
    unittest.main()
