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


if __name__ == "__main__":
    unittest.main()
