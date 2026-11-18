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


    def test_blackoil_fault_faces_only_in_nnc_rates(self):
        # The live API reports the water rate of each grid connection in exactly one place
        # (see the comment above ensureWaterFluxCache in
        # opm/simulators/flow/python/PyLiveReservoirData_impl.hpp): a pair on OPM's output
        # NNC list only in get_live_nnc_rates() (positive from nnc1 to nnc2), any other pair
        # of direct neighbours in the I+/J+/K+ arrays of the cell with the smaller Cartesian
        # index (positive out of that cell). A face across a fault must not also land in the
        # I+/J+ slot its direction id points to; OPM's own restart flows (FLRWATI+) still do
        # that at the cell2 end of a pair, which is the case this deck is built around.
        data_dir = Path(os.path.dirname(__file__)).parent.joinpath("test_data/FAULT_NNC")
        with tempfile.TemporaryDirectory() as output_dir:
            script = textwrap.dedent(f"""
                import numpy as np
                from opm.simulators import BlackOilSimulator

                # Tight nonlinear tolerances, strict in every iteration, so that the per-cell
                # water balance below closes to far less than the face rates it checks.
                # With the defaults (CNV 1e-2, MB 1e-7) the largest residual was 2.3e-4 m3.
                sim = BlackOilSimulator(
                    "FAULT_NNC.DATA",
                    args=[
                        "--linear-solver=ilu0",
                        "--output-dir={output_dir}",
                        "--tolerance-cnv=1e-6",
                        "--tolerance-cnv-relaxed=1e-6",
                        "--tolerance-mb=1e-10",
                        "--tolerance-mb-relaxed=1e-10",
                    ],
                )
                sim.setup_mpi(init=True, finalize=True)
                sim.step_init()

                # FAULT_NNC is a fully active 6 x 4 x 3 grid. Columns i = 2, 3 (zero-based)
                # lie one layer deeper (I fault, full throw); rows j = 2, 3 lie half a layer
                # deeper (J fault).
                nx, ny, nz = 6, 4, 3
                cells = np.asarray(sim.get_active_to_global_cells())  # zero-based Cartesian
                assert cells.size == nx * ny * nz
                active_of = np.full(nx * ny * nz, -1)
                active_of[cells] = np.arange(cells.size)

                def ijk(g):
                    return int(g % nx), int((g // nx) % ny), int(g // (nx * ny))

                def cart(i, j, k):
                    return i + nx * (j + ny * k)

                # nnc1/nnc2 are one-based Cartesian indices with nnc1 < nnc2.
                topology = sim.get_live_nnc_topology()
                nnc1 = np.asarray(topology["nnc1"]) - 1
                nnc2 = np.asarray(topology["nnc2"]) - 1
                assert np.all(nnc1 < nnc2)
                pairs = [(ijk(a), ijk(b)) for a, b in zip(nnc1, nnc2)]
                pair_index = {{(int(a), int(b)): n for n, (a, b) in enumerate(zip(nnc1, nnc2))}}

                # (a) The geometry the test relies on, as the simulator reports it.
                # Low-I edge of the deep block, the cell2 end: (1, j, k) touches only
                # (2, j, k-1), which has the smaller Cartesian index and is nnc1.
                cell2_end = [(a, b) for a, b in pairs
                             if a[0] == b[0] + 1 and a[1] == b[1] and a[2] == b[2] - 1]
                # High-I edge: (3, j, k) touches (4, j, k+1); nnc1 is the cell with lower i.
                lower_i_first = [(a, b) for a, b in pairs
                                 if b[0] == a[0] + 1 and a[1] == b[1] and b[2] == a[2] + 1]
                # J fault, half throw: (i, 1, k) touches (i, 2, k-1) (nnc1) beside its
                # regular half face to the Cartesian J+ neighbour (i, 2, k).
                j_fault = [(a, b) for a, b in pairs
                           if a[0] == b[0] and a[1] == b[1] + 1 and a[2] == b[2] - 1]
                assert sorted(b for _, b in cell2_end) == sorted(
                    (1, j, k) for j in range(ny) for k in range(1, nz)), cell2_end
                assert sorted(a for a, _ in lower_i_first) == sorted(
                    (3, j, k) for j in range(ny) for k in range(nz - 1)), lower_i_first
                assert sorted(b for _, b in j_fault) == sorted(
                    (i, 1, k) for i in range(nx) for k in range(1, nz)), j_fault
                assert len(pairs) == len(cell2_end) + len(lower_i_first) + len(j_fault), pairs

                before = sim.get_live_cell_arrays(("RPORV", "SWAT", "1OVERBW"))
                water_before = (np.asarray(before["RPORV"]) * np.asarray(before["SWAT"])
                                * np.asarray(before["1OVERBW"]))

                assert sim.advance_converged_timestep()
                dt = sim.last_converged_timestep_seconds()
                after = sim.get_live_cell_arrays(("RPORV", "SWAT", "1OVERBW"))
                water_after = (np.asarray(after["RPORV"]) * np.asarray(after["SWAT"])
                               * np.asarray(after["1OVERBW"]))
                face_names = (
                    "WATER_RESERVOIR_FACE_RATE_I+",
                    "WATER_RESERVOIR_FACE_RATE_J+",
                    "WATER_RESERVOIR_FACE_RATE_K+",
                    "WATER_SURFACE_FACE_RATE_I+",
                    "WATER_SURFACE_FACE_RATE_J+",
                    "WATER_SURFACE_FACE_RATE_K+",
                )
                faces = {{name: np.asarray(values, dtype=float)
                         for name, values in sim.get_live_cell_arrays(face_names).items()}}
                nnc_rates = sim.get_live_nnc_rates()
                nnc_res = np.asarray(nnc_rates["water_reservoir_rate"], dtype=float)
                nnc_surf = np.asarray(nnc_rates["water_surface_rate"], dtype=float)

                # (b) Water crosses the I fault. Measured on the first converged step (dt one
                # day): |rate| 1.1e-4 to 2.7e-4 m3/s at the I-fault pairs, both bases. The
                # threshold 1e-6 m3/s is a hundred times below the smallest of them, so it
                # only fails if the flow has stopped, and it lies far above the balance
                # residual of (d) divided by dt (about 1e-13 m3/s), so a nonzero NNC rate
                # here cannot be solver noise.
                rate_threshold = 1.0e-6
                for a, b in cell2_end + lower_i_first:
                    n = pair_index[(cart(*a), cart(*b))]
                    assert abs(nnc_res[n]) > rate_threshold, (a, b, nnc_res[n])
                    assert abs(nnc_surf[n]) > rate_threshold, (a, b, nnc_surf[n])
                # The J-fault cells keep their regular J+ face to the Cartesian neighbour.
                for _, b in j_fault:
                    assert abs(faces["WATER_SURFACE_FACE_RATE_J+"][active_of[cart(*b)]]) \\
                        > rate_threshold, b

                # (c) Every cell whose Cartesian I+ neighbour lies across the full-throw fault
                # (i = 1 and i = 3) shares no face with that neighbour, so both I+ arrays are
                # exactly zero there, while the NNC of the pair on the same side carries the
                # flow. Before the face rule the cell2 end (1, j, k) held the NNC rate in I+.
                fault_pair_of = {{b: a for a, b in cell2_end}}
                fault_pair_of.update({{a: b for a, b in lower_i_first}})
                for i in (1, 3):
                    for j in range(ny):
                        for k in range(nz):
                            n_act = active_of[cart(i, j, k)]
                            assert faces["WATER_RESERVOIR_FACE_RATE_I+"][n_act] == 0.0, (i, j, k)
                            assert faces["WATER_SURFACE_FACE_RATE_I+"][n_act] == 0.0, (i, j, k)
                            other = fault_pair_of.get((i, j, k))
                            if other is None:
                                continue  # (1, j, 0) and (3, j, nz-1) touch nothing across
                            key = tuple(sorted((cart(i, j, k), cart(*other))))
                            assert abs(nnc_surf[pair_index[key]]) > rate_threshold, (i, j, k)

                # (d) Per-cell water balance on the surface basis over the step:
                #   water_after - water_before = -dt * (net outflow),
                # with each face rate out of the cell holding it and into its Cartesian +
                # neighbour, and each NNC rate out of nnc1 and into nnc2. Cells with a well
                # connection are left out (their well rate is not in the sum).
                outflow = np.zeros(cells.size)
                for name, offset in (("WATER_SURFACE_FACE_RATE_I+", 1),
                                     ("WATER_SURFACE_FACE_RATE_J+", nx),
                                     ("WATER_SURFACE_FACE_RATE_K+", nx * ny)):
                    rate = faces[name]
                    holders = np.flatnonzero(rate)
                    np.add.at(outflow, holders, rate[holders])
                    np.add.at(outflow, active_of[cells[holders] + offset], -rate[holders])
                np.add.at(outflow, active_of[nnc1], nnc_surf)
                np.add.at(outflow, active_of[nnc2], -nnc_surf)
                residual = (water_after - water_before) + dt * outflow

                wells = sim.get_live_well_topology()
                well_cells = active_of[[cart(i, j, k) for i, j, k in zip(
                    wells["perf_i"], wells["perf_j"], wells["perf_k"])]]
                no_well = np.ones(cells.size, dtype=bool)
                no_well[well_cells] = False
                # Measured: dt * max|NNC surface rate| = 86400 s * 2.7e-4 m3/s = 23.6 m3, and
                # the largest residual 8.9e-9 m3 with the tolerances above. The tolerance
                # 1e-5 m3 is six orders below the first and three orders above the second.
                # A rate counted twice or not at all would leave a residual of the order of
                # dt times an NNC rate (10 to 24 m3) in two cells.
                balance_tolerance = 1.0e-5
                assert dt * np.max(np.abs(nnc_surf)) > 100.0 * balance_tolerance
                worst = np.max(np.abs(residual[no_well]))
                assert worst < balance_tolerance, worst
                print("largest balance residual", worst)

                sim.step_cleanup()
            """)
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=data_dir,
                env=os.environ.copy(),
                check=True,
                text=True,
                capture_output=True,
            )
            self.assertIn("largest balance residual", result.stdout)


if __name__ == "__main__":
    unittest.main()
