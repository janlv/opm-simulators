import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


class TestTimeSteppingAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_dir = Path(os.path.dirname(__file__))
        cls.data_dir_bo = test_dir.parent.joinpath("test_data/SPE1CASE1a")

    def test_blackoil_report_and_converged_timestep_api(self):
        with tempfile.TemporaryDirectory() as output_dir:
            script = textwrap.dedent(f"""
                from opm.simulators import BlackOilSimulator

                sim = BlackOilSimulator(
                    "SPE1CASE1.DATA",
                    args=["--linear-solver=ilu0", "--output-dir={output_dir}"],
                )
                sim.setup_mpi(init=True, finalize=True)
                sim.step_init()

                assert hasattr(sim, "advance_report_timestep")
                assert hasattr(sim, "last_report_timestep_seconds")
                assert hasattr(sim, "advance_converged_timestep")
                assert hasattr(sim, "last_converged_timestep_seconds")

                assert sim.advance_report_timestep()
                report_dt = sim.last_report_timestep_seconds()
                assert report_dt > 0.0
                assert abs(report_dt - sim.get_dt()) < 1.0e-7

                assert sim.advance_converged_timestep()
                assert sim.last_converged_timestep_seconds() > 0.0
                sim.step_cleanup()
            """)
            subprocess.run(
                [sys.executable, "-c", script],
                cwd=self.data_dir_bo,
                env=os.environ.copy(),
                check=True,
                text=True,
                capture_output=True,
            )


    def test_blackoil_live_reservoir_payloads(self):
        with tempfile.TemporaryDirectory() as output_dir:
            script = textwrap.dedent(f"""
                import numpy as np
                from opm.simulators import BlackOilSimulator

                sim = BlackOilSimulator(
                    "SPE1CASE1.DATA",
                    args=["--linear-solver=ilu0", "--output-dir={output_dir}"],
                )
                sim.setup_mpi(init=True, finalize=True)
                sim.step_init()

                initial_names = set(sim.available_live_cell_arrays())
                for name in ("SWAT", "RPORV", "1OVERBW"):
                    assert name in initial_names, name
                initial = sim.get_live_cell_arrays(("SWAT", "RPORV", "1OVERBW"))
                cell_count = np.asarray(initial["SWAT"]).size
                assert cell_count > 0
                assert np.asarray(initial["RPORV"]).size == cell_count
                assert np.asarray(initial["1OVERBW"]).size == cell_count
                assert np.all(np.isfinite(initial["RPORV"]))
                assert np.all(np.asarray(initial["RPORV"]) > 0.0)
                assert np.all(np.asarray(initial["1OVERBW"]) > 0.0)

                assert sim.advance_converged_timestep()
                step_names = set(sim.available_live_cell_arrays())
                face_names = (
                    "WATER_RESERVOIR_FACE_RATE_I+",
                    "WATER_RESERVOIR_FACE_RATE_J+",
                    "WATER_RESERVOIR_FACE_RATE_K+",
                    "WATER_SURFACE_FACE_RATE_I+",
                    "WATER_SURFACE_FACE_RATE_J+",
                    "WATER_SURFACE_FACE_RATE_K+",
                )
                for name in face_names:
                    assert name in step_names, name
                faces = sim.get_live_cell_arrays(face_names)
                for name in face_names:
                    values = np.asarray(faces[name], dtype=float)
                    assert values.size == cell_count
                    assert np.all(np.isfinite(values))

                topology = sim.get_live_well_topology()
                rates = sim.get_live_well_rates()
                perf_count = np.asarray(topology["perf_well_index"]).size
                assert len(topology["well_names"]) > 0
                for name in ("perf_i", "perf_j", "perf_k"):
                    assert np.asarray(topology[name]).size == perf_count
                for name in ("perf_water_reservoir_rate", "perf_water_surface_rate"):
                    values = np.asarray(rates[name], dtype=float)
                    assert values.size == perf_count
                    assert np.all(np.isfinite(values))

                nnc_topology = sim.get_live_nnc_topology()
                nnc_rates = sim.get_live_nnc_rates()
                nnc_count = np.asarray(nnc_topology["nnc1"]).size
                assert np.asarray(nnc_topology["nnc2"]).size == nnc_count
                for name in ("water_reservoir_rate", "water_surface_rate"):
                    values = np.asarray(nnc_rates[name], dtype=float)
                    assert values.size == nnc_count
                    assert np.all(np.isfinite(values))

                sim.step_cleanup()
            """)
            subprocess.run(
                [sys.executable, "-c", script],
                cwd=self.data_dir_bo,
                env=os.environ.copy(),
                check=True,
                text=True,
                capture_output=True,
            )


    def test_blackoil_static_tranz_after_step_init(self):
        with tempfile.TemporaryDirectory() as output_dir:
            script = textwrap.dedent(f"""
                import numpy as np
                from opm.simulators import BlackOilSimulator

                sim = BlackOilSimulator(
                    "SPE1CASE1.DATA",
                    args=["--linear-solver=ilu0", "--output-dir={output_dir}"],
                )
                sim.setup_mpi(init=True, finalize=True)
                sim.step_init()

                # TRANZ is static: listed and readable before any converged timestep.
                assert "TRANZ" in sim.available_live_cell_arrays()
                tranz = np.asarray(sim.get_live_cell_array("TRANZ"), dtype=float)
                batched = np.asarray(sim.get_live_cell_arrays(("TRANZ",))["TRANZ"], dtype=float)
                cells = np.asarray(sim.get_active_to_global_cells())
                assert tranz.size == cells.size > 0
                assert np.array_equal(tranz, batched)
                assert np.all(np.isfinite(tranz))
                assert np.all(tranz >= 0.0)
                # SPE1CASE1 is a fully active 10 x 10 x 3 grid: every cell above the bottom
                # layer has a K+ connection, and the bottom layer has none.
                layer = cells // 100
                assert np.all(tranz[layer < 2] > 0.0)
                assert np.all(tranz[layer == 2] == 0.0)

                # Static array: unchanged after a converged timestep.
                assert sim.advance_converged_timestep()
                assert np.array_equal(np.asarray(sim.get_live_cell_array("TRANZ")), tranz)
                sim.step_cleanup()
            """)
            subprocess.run(
                [sys.executable, "-c", script],
                cwd=self.data_dir_bo,
                env=os.environ.copy(),
                check=True,
                text=True,
                capture_output=True,
            )


    def test_blackoil_step_cleanup_before_first_timestep(self):
        with tempfile.TemporaryDirectory() as output_dir:
            script = textwrap.dedent(f"""
                from opm.simulators import BlackOilSimulator

                sim = BlackOilSimulator(
                    "SPE1CASE1.DATA",
                    args=["--linear-solver=ilu0", "--output-dir={output_dir}"],
                )
                sim.setup_mpi(init=True, finalize=True)
                sim.step_init()

                # A caller whose own set-up fails after step_init() closes the session
                # without having advanced it. No report step has run, so the nonlinear
                # solver does not exist yet, and the cleanup must not need it.
                assert sim.step_cleanup() == 0
                print("step_cleanup returned")
            """)
            # A crash inside step_cleanup() ends the child with a signal, which check=True
            # turns into a failure here.
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=self.data_dir_bo,
                env=os.environ.copy(),
                check=True,
                text=True,
                capture_output=True,
            )
            self.assertIn("step_cleanup returned", result.stdout)


if __name__ == "__main__":
    unittest.main()
