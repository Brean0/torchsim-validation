"""Exercise the real snapshot consumer; no native engines are needed."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_force_comparison.py"
SNAPSHOTS = ROOT / "evidence" / "force-doubling"


class ForceComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.data = {
            name: json.loads((SNAPSHOTS / f"{name}.json").read_text())
            for name in ("torchsim", "ase", "lammps", "jaxmd", "torchmd")
        }

    def consume(self):
        for name, data in self.data.items():
            (self.directory / f"{name}.json").write_text(json.dumps(data))
        return subprocess.run(
            [sys.executable, str(RUNNER), "--evidence-dir", str(self.directory)],
            text=True, capture_output=True, check=False,
        )

    def assert_rejected(self):
        result = self.consume()
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("validation failed", result.stderr.lower())

    def test_saved_native_evidence_validates(self):
        result = self.consume()
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["default_pairforces_factor"], 2.0)
        self.assertEqual(report["distances"], [1.1, 1.3, 1.5])
        self.assertEqual(report["external_reference_count"], 4)
        self.assertFalse(report["fresh_engine_runs"])

    def test_external_reference_disagreement_is_rejected(self):
        self.data["lammps"]["cases"][0]["forces"][0][0] += 0.1
        self.assert_rejected()

    def test_nonfinite_reference_is_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                self.data["ase"]["cases"][0]["forces"][0][0] = value
                self.assert_rejected()

    def test_malformed_reference_is_rejected(self):
        for bad in ([[1, 2, 3]], [[1, 2], [3, 4]], [["bad", 0, 0], [0, 0, 0]]):
            with self.subTest(forces=bad):
                self.data["ase"]["cases"][0]["forces"] = bad
                self.assert_rejected()

    def test_missing_or_duplicate_distance_is_rejected(self):
        original = copy.deepcopy(self.data["ase"]["cases"])
        self.data["ase"]["cases"] = original[:2]
        self.assert_rejected()
        self.data["ase"]["cases"] = original + [original[0]]
        self.assert_rejected()

    def test_wrong_doubling_factor_is_rejected(self):
        self.data["torchsim"]["cases"][0]["forces"]["PairForcesModel_full_default/default"][0][0] *= 1.5
        self.assert_rejected()

    def test_both_controls_and_neighbor_backends_are_checked(self):
        original = copy.deepcopy(self.data["torchsim"])
        for backend in ("default", "torch_nl_n2"):
            for label in ("PairForcesModel_half", "PairPotentialModel", "PairForcesModel_full_default"):
                with self.subTest(label=label, backend=backend):
                    self.data["torchsim"] = copy.deepcopy(original)
                    self.data["torchsim"]["cases"][1]["forces"][f"{label}/{backend}"][0][0] += 0.1
                    self.assert_rejected()

    def test_original_soft_sphere_controls_are_checked(self):
        self.data["torchsim"]["cases"][-1]["forces"]["PairForcesModel_half/torch_nl_n2"][0][0] = -1.0
        self.assert_rejected()

    def test_torchmd_physical_unit_conversion_is_checked(self):
        self.data["torchmd"]["physical_epsilon_1ev_cases"][0]["forces_ev_angstrom"][0][0] = -9.0
        self.assert_rejected()

    def test_configuration_and_source_commit_are_checked(self):
        self.data["ase"]["configuration"]["pbc"] = True
        self.assert_rejected()
        self.data["ase"]["configuration"]["pbc"] = False
        self.data["torchsim"]["commit"] = "0" * 40
        self.assert_rejected()

    def test_vacuous_all_zero_force_agreement_is_rejected(self):
        zeros = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        for name in ("ase", "lammps", "jaxmd", "torchmd"):
            for case in self.data[name]["cases"]:
                case["forces"] = copy.deepcopy(zeros)
        for case in self.data["torchsim"]["cases"]:
            if case["potential"] == "lennard_jones":
                case["forces"] = {key: copy.deepcopy(zeros) for key in case["forces"]}
        for case in self.data["torchmd"]["physical_epsilon_1ev_cases"]:
            case["forces_ev_angstrom"] = copy.deepcopy(zeros)
        self.assert_rejected()

    def test_unpinned_engine_version_is_rejected(self):
        self.data["ase"]["version"] = "0.0.0"
        self.assert_rejected()

    def test_coherent_noncentral_and_momentum_invalid_lj_vectors_are_rejected(self):
        original = copy.deepcopy(self.data)
        for mutation in ("transverse", "net_force"):
            with self.subTest(mutation=mutation):
                self.data = copy.deepcopy(original)
                for index in range(3):
                    force = copy.deepcopy(self.data["ase"]["cases"][index]["forces"])
                    if mutation == "transverse":
                        force[0][1], force[1][1] = 0.2, -0.2
                    else:
                        force[1][0] += 0.2
                    for name in ("ase", "lammps", "jaxmd", "torchmd"):
                        self.data[name]["cases"][index]["forces"] = copy.deepcopy(force)
                    for label in self.data["torchsim"]["cases"][index]["forces"]:
                        factor = 2.0 if label.startswith("PairForcesModel_full_default/") else 1.0
                        self.data["torchsim"]["cases"][index]["forces"][label] = [
                            [factor * value for value in row] for row in force
                        ]
                    self.data["torchmd"]["physical_epsilon_1ev_cases"][index]["forces_ev_angstrom"] = copy.deepcopy(force)
                self.assert_rejected()

    def test_coherent_invalid_soft_sphere_vectors_are_rejected(self):
        original = copy.deepcopy(self.data)
        for force in (
            [[0, 0, 0], [0, 0, 0]],
            [[-0.25, 0, 0], [0.25, 0, 0]],
            [[0.5, 0, 0], [-0.5, 0, 0]],
            [[-0.5, 0.2, 0], [0.5, -0.2, 0]],
            [[-0.5, 0, 0], [0.4, 0, 0]],
        ):
            with self.subTest(forces=force):
                self.data = copy.deepcopy(original)
                self.data["jaxmd"]["original_soft_sphere_control"]["forces"] = force
                soft = self.data["torchsim"]["cases"][-1]
                for label in soft["forces"]:
                    factor = 2.0 if label.startswith("PairForcesModel_full_default/") else 1.0
                    soft["forces"][label] = [[factor * value for value in row] for row in force]
                self.assert_rejected()

    def test_live_resolves_relative_interpreters_before_using_temporary_cwd(self):
        # Native computation is the external boundary. Executable fixture
        # interpreters exercise the real CLI, temporary cwd, output and validator.
        fixture_dir = self.directory / "fixtures"
        fixture_dir.mkdir()
        wrapper = (
            f"#!{sys.executable}\n"
            "import os, pathlib, shutil, sys\n"
            "engine = pathlib.Path(sys.argv[1]).stem.removeprefix('native_')\n"
            "shutil.copyfile(pathlib.Path(os.environ['FORCE_TEST_FIXTURES']) / (engine + '.json'), sys.argv[2])\n"
        )
        for name, data in self.data.items():
            (fixture_dir / f"{name}.json").write_text(json.dumps(data))
            executable = self.directory / ".venvs" / name / "bin/python"
            executable.parent.mkdir(parents=True)
            executable.write_text(wrapper)
            executable.chmod(0o755)
        for mode in ("env_dir", "flags", "environment"):
            with self.subTest(mode=mode):
                environment = os.environ.copy()
                environment["FORCE_TEST_FIXTURES"] = str(fixture_dir)
                command = [sys.executable, str(RUNNER), "--live", "--evidence-dir", f"output-{mode}"]
                for name in self.data:
                    environment.pop(name.upper() + "_PYTHON", None)
                if mode == "env_dir":
                    command.extend(["--env-dir", ".venvs"])
                elif mode == "flags":
                    for name in self.data:
                        command.extend([f"--{name}-python", f".venvs/{name}/bin/python"])
                else:
                    environment.update({name.upper() + "_PYTHON": f".venvs/{name}/bin/python" for name in self.data})
                result = subprocess.run(command, cwd=self.directory, env=environment, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(json.loads(result.stdout)["fresh_engine_runs"])
                self.assertTrue((self.directory / f"output-{mode}" / "comparison.json").exists())

    def test_noneditable_vcs_provenance_ignores_enclosing_unrelated_git(self):
        # A wheel installed inside this repository's .venvs belongs to its
        # native source commit, not to the evidence repository's ancestor Git.
        package = self.directory / "site" / "audit_fake_native"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        metadata = package.parent / "audit_fake_native-1.0.dist-info"
        metadata.mkdir()
        (metadata / "METADATA").write_text("Metadata-Version: 2.1\nName: audit-fake-native\nVersion: 1.0\n")
        (metadata / "direct_url.json").write_text(json.dumps({
            "url": "https://example.com/native.git",
            "vcs_info": {"vcs": "git", "commit_id": "a" * 40},
        }))
        # An unrelated/broken ancestor .git must never be inspected first.
        (self.directory / ".git").mkdir()
        code = (
            "import sys; sys.path[:0] = sys.argv[1:3]; "
            "from common import source_provenance; "
            "import audit_fake_native; "
            "print(source_provenance(audit_fake_native.__file__, 'audit-fake-native', 'a'*40)['commit'])"
        )
        result = subprocess.run([sys.executable, "-c", code, str(ROOT / "references"), str(package.parent)],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "a" * 40)


if __name__ == "__main__":
    unittest.main()
