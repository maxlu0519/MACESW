"""Static contract tests for the Phase I source generator."""

import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from phasei_ml import source


class SourceContractTests(unittest.TestCase):
    def test_branch_contract_matches_mustard_reader(self):
        self.assertEqual(
            set(source.BRANCH_CONTRACT),
            {"w", "t", "x", "y", "z", "pdgID", "E", "px", "py", "pz"},
        )
        self.assertEqual(source.SCHEMA_VERSION, 1)

    def test_geometry_rejects_alternate_center_or_dimensions(self):
        geometry, _, _ = source._load_geometry(None)
        self.assertEqual(geometry, source.DEFAULT_GEOMETRY)
        with tempfile.TemporaryDirectory() as directory:
            geometry_file = Path(directory) / "geometry.yaml"
            geometry_file.write_text(
                "shape: cylinder\nradius_mm: 20\nthickness_mm: 60\ncenter_mm: [0, 0, 0]\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                source._load_geometry(geometry_file)

    def test_default_geometry_uses_checked_in_phasei_yaml(self):
        geometry, digest, path = source._load_geometry(None)
        geometry_file = source.DEFAULT_GEOMETRY_PATH
        self.assertEqual(path, str(geometry_file))
        self.assertEqual(digest, hashlib.sha256(geometry_file.read_bytes()).hexdigest())
        self.assertEqual(geometry["shape"], "cylinder")
        self.assertEqual(geometry["radius_mm"], 30.0)
        self.assertEqual(geometry["thickness_mm"], 60.0)
        self.assertEqual(geometry["center_mm"], [0.0, 0.0, 0.0])

    def test_existing_root_or_manifest_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root_file = Path(directory) / "source.root"
            manifest_file = Path(str(root_file) + ".json")
            source._require_outputs_available(root_file, manifest_file)
            manifest_file.touch()
            with self.assertRaises(FileExistsError):
                source._require_outputs_available(root_file, manifest_file)
            manifest_file.unlink()
            root_file.touch()
            with self.assertRaises(FileExistsError):
                source._require_outputs_available(root_file, manifest_file)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is optional")
    def test_chunk_has_one_particle_and_on_shell_positron(self):
        _, chunk = source._generate_chunk(0, 16, 7, "e+")
        self.assertEqual(len(chunk["pdgID"]), 16)
        self.assertTrue((chunk["pdgID"] == -11).all())
        momentum2 = chunk["px"] ** 2 + chunk["py"] ** 2 + chunk["pz"] ** 2
        mass = source.PARTICLES["e+"][1]
        energy = chunk["E"]
        self.assertTrue(((energy**2 - momentum2 - mass**2).__abs__() < 2e-3).all())

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is optional")
    def test_chunk_seed_depends_on_chunk_id(self):
        first = source._generate_chunk(0, 8, 11, "gamma")[1]
        repeat = source._generate_chunk(0, 8, 11, "gamma")[1]
        other = source._generate_chunk(1, 8, 11, "gamma")[1]
        self.assertTrue((first["E"] == repeat["E"]).all())
        self.assertFalse((first["E"] == other["E"]).all())

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is optional")
    def test_ordered_stream_is_independent_of_worker_count(self):
        import numpy as np

        def collect(workers):
            chunks = list(
                source._ordered_chunks(
                    events=11,
                    seed=27,
                    particle="gamma",
                    workers=workers,
                    chunk_size=3,
                )
            )
            return {
                name: np.concatenate([chunk[name] for chunk in chunks])
                for name in chunks[0]
            }

        single_worker = collect(1)
        multiple_workers = collect(2)
        self.assertEqual(set(single_worker), set(multiple_workers))
        for name in single_worker:
            np.testing.assert_array_equal(single_worker[name], multiple_workers[name])

    @unittest.skipUnless(
        importlib.util.find_spec("numpy") and importlib.util.find_spec("ROOT"),
        "NumPy and PyROOT are required for the ROOT integration check",
    )
    def test_generated_root_tree_schema_values_and_manifest(self):
        import ROOT

        with tempfile.TemporaryDirectory() as directory:
            root_path = Path(directory) / "positrons.root"
            manifest = source.generate_source(
                particle="e+",
                events=4,
                seed=123,
                output=root_path,
                tree="PhaseISource",
                workers=1,
                chunk_size=2,
            )
            root_file = ROOT.TFile.Open(str(root_path), "READ")
            self.assertTrue(root_file and not root_file.IsZombie())
            try:
                tree = root_file.Get("PhaseISource")
                self.assertTrue(tree)
                self.assertEqual(tree.GetEntries(), 4)
                expected_branches = {"w", "t", "x", "y", "z", "pdgID", "E", "px", "py", "pz"}
                actual_branches = {branch.GetName() for branch in tree.GetListOfBranches()}
                self.assertEqual(actual_branches, expected_branches)

                for name in ("w", "x", "y", "z"):
                    self.assertEqual(tree.GetBranch(name).GetLeaf(name).GetTypeName(), "Float_t")
                self.assertEqual(tree.GetBranch("t").GetLeaf("t").GetTypeName(), "Double_t")
                for name, expected in (("pdgID", "vector<int>"), ("E", "vector<float>"),
                                       ("px", "vector<float>"), ("py", "vector<float>"),
                                       ("pz", "vector<float>")):
                    actual = tree.GetBranch(name).GetClassName().replace("std::", "").replace(" ", "")
                    self.assertTrue(actual.startswith(expected), f"{name}: {actual}")

                mass = source.PARTICLES["e+"][1]
                for entry_index in range(tree.GetEntries()):
                    tree.GetEntry(entry_index)
                    self.assertEqual(int(tree.pdgID.size()), 1)
                    self.assertEqual(int(tree.E.size()), 1)
                    self.assertEqual(int(tree.px.size()), 1)
                    self.assertEqual(int(tree.py.size()), 1)
                    self.assertEqual(int(tree.pz.size()), 1)
                    self.assertEqual(int(tree.pdgID[0]), -11)
                    x, y, z = float(tree.x), float(tree.y), float(tree.z)
                    self.assertLessEqual(x * x + y * y, 30.0**2 + 1e-3)
                    self.assertLessEqual(abs(z), 30.0 + 1e-4)
                    energy = float(tree.E[0])
                    kinetic = energy - mass
                    self.assertGreaterEqual(kinetic, 0.001 - 1e-5)
                    self.assertLessEqual(kinetic, 60.0 + 1e-5)
                    momentum2 = sum(float(getattr(tree, name)[0]) ** 2 for name in ("px", "py", "pz"))
                    self.assertAlmostEqual(energy**2 - momentum2, mass**2, delta=2e-3)
            finally:
                root_file.Close()

            manifest_path = Path(str(root_path) + ".json")
            stored_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected_sha = hashlib.sha256(root_path.read_bytes()).hexdigest()
            self.assertEqual(manifest["sha256"], expected_sha)
            self.assertEqual(stored_manifest["sha256"], expected_sha)
            self.assertEqual(manifest["source_file"], str(root_path.resolve()))
            self.assertEqual(manifest["count"], 4)


if __name__ == "__main__":
    unittest.main()
