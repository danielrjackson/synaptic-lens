"""Offline contract tests for the standalone Hub downloader."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "download_models", Path(__file__).resolve().parents[2] / "scripts" / "download_models.py"
)
downloads = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(downloads)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.calls = []
        self.payload = b"numeric model payload"
        self.manifest = {
            "exports": [{"file": "weights/model.npz", "bytes": len(self.payload),
                         "sha256": hashlib.sha256(self.payload).hexdigest()}],
            "model_license": {"file": "LICENSE-MODEL.md", "sha256": hashlib.sha256(b"license").hexdigest()},
            "documentation_license": {"file": "LICENSE-SOFTWARE.md", "sha256": hashlib.sha256(b"license").hexdigest()},
        }
        self.put("weights/model.npz", self.payload)
        self.put("README.md", b"Model card and provenance")
        self.put("LICENSE-MODEL.md", b"license")
        self.put("LICENSE-SOFTWARE.md", b"license")
        self.release = {"repository": "owner/models", "revision": "a" * 40,
                        "manifest_file": "export-manifest.json"}
        self.refresh_manifest()
        self.mock = patch.object(downloads, "catalog", return_value={"test": self.release})
        self.mock.start()
        self.addCleanup(self.mock.stop)

    def put(self, name, content):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
        return p

    def refresh_manifest(self):
        content = json.dumps(self.manifest).encode()
        self.put("export-manifest.json", content)
        self.release["manifest_sha256"] = hashlib.sha256(content).hexdigest()

    def fetch(self, **kwargs):
        self.calls.append(kwargs)
        return self.root / kwargs["filename"]

    def run_download(self, filenames=None, **kwargs):
        return downloads.download_release("test", filenames, downloader=self.fetch, **kwargs)

    def test_selected_file_and_pins(self):
        result = self.run_download(["weights/model.npz"], cache_dir=self.root, local_files_only=True)
        self.assertEqual(Path(result["files"]["weights/model.npz"]["path"]).read_bytes(), self.payload)
        for call in self.calls:
            self.assertEqual(call["revision"], "a" * 40)
            self.assertEqual(call["repo_id"], "owner/models")
            self.assertIs(call["token"], False)
            self.assertTrue(call["local_files_only"])
            self.assertEqual(call["cache_dir"], str(self.root))

    def test_empty_selection_lists_without_weights(self):
        result = self.run_download()
        self.assertEqual(result["files"], {})
        self.assertEqual(result["available_files"][0]["file"], "weights/model.npz")
        self.assertNotIn("weights/model.npz", [c["filename"] for c in self.calls])

    def test_all_explicit(self):
        self.assertIn("weights/model.npz", self.run_download(all_files=True)["files"])

    def test_conflicting_selection(self):
        with self.assertRaises(ValueError):
            self.run_download(["weights/model.npz"], all_files=True)
        self.assertEqual(self.calls, [])

    def test_manifest_corruption_stops_before_payload(self):
        self.put("export-manifest.json", b"corrupt")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.run_download(all_files=True)
        self.assertEqual(len(self.calls), 1)

    def test_payload_corruption_is_preserved(self):
        damaged = b"x" * len(self.payload)
        self.put("weights/model.npz", damaged)
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.run_download(all_files=True)
        self.assertEqual((self.root / "weights/model.npz").read_bytes(), damaged)

    def test_snapshot_symlink_returns_regular_blob(self):
        link = self.root / "snapshot-link"
        try:
            link.symlink_to(self.root / "weights/model.npz")
        except OSError:
            self.skipTest("Symlink creation unavailable")
        result = downloads.verify(link, hashlib.sha256(self.payload).hexdigest())
        self.assertFalse(result.is_symlink())
        self.assertEqual(result.read_bytes(), self.payload)

    def test_wrong_size(self):
        self.put("weights/model.npz", b"short")
        with self.assertRaisesRegex(ValueError, "Size mismatch"):
            self.run_download(all_files=True)

    def test_unknown_file(self):
        with self.assertRaisesRegex(ValueError, "absent"):
            self.run_download(["unknown.npz"])
        self.assertEqual(len(self.calls), 1)

    def test_paths_are_relative(self):
        for name in ("../model", "/model", "C:/model", "a\\b", "a//b", "./a"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.run_download([name])

    def test_duplicate_manifest_rejected(self):
        self.manifest["exports"] *= 2
        self.refresh_manifest()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.run_download(all_files=True)

    def test_revision_must_be_pinned(self):
        self.release["revision"] = "main"
        with self.assertRaisesRegex(ValueError, "full commit"):
            self.run_download()
        self.assertEqual(self.calls, [])

    def test_license_corruption_rejected(self):
        self.put("LICENSE-MODEL.md", b"changed terms")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.run_download()


if __name__ == "__main__":
    unittest.main()
