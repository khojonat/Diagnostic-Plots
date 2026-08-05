import os
import tempfile
import h5py
import numpy as np

from scripts.Tully_Fisher import plot_tully_fisher
from scripts.helpers import compute_rotation_curve_and_save
from tests.generate_test_galaxy import create_test_galaxy_snapshot


def test_compute_rotation_curve_and_save_writes_vrot_gas():
    with tempfile.TemporaryDirectory() as tmpdir:
        snap_path = create_test_galaxy_snapshot(output_dir=tmpdir, filename="test_galaxy.hdf5")

        with h5py.File(snap_path, "r") as f:
            header = dict(f["Header"].attrs.items())
            particle_data = {
                "header": header,
                "halo_pos": np.zeros(3, dtype=float),
                "particles": {
                    0: {
                        "Coordinates": f["PartType0"]["Coordinates"][:],
                        "Velocities": f["PartType0"]["Velocities"][:],
                        "Masses": f["PartType0"]["Masses"][:],
                    },
                    1: {
                        "Coordinates": f["PartType1"]["Coordinates"][:],
                        "Velocities": f["PartType1"]["Velocities"][:],
                        "Masses": f["PartType1"]["Masses"][:],
                    },
                    4: {
                        "Coordinates": f["PartType4"]["Coordinates"][:],
                        "Velocities": f["PartType4"]["Velocities"][:],
                        "Masses": f["PartType4"]["Masses"][:],
                    },
                },
                "run_num": 0,
                "snapnum": "test",
                "target": 0,
            }

        outpath = compute_rotation_curve_and_save(
            run_num=0,
            snapnum="test",
            target=0,
            output_dir=tmpdir,
            particle_data=particle_data,
            filename_suffix="_test",
        )

        assert os.path.exists(outpath)

        with h5py.File(outpath, "r") as f_out:
            assert "vrot_gas" in f_out
            vrot_gas = float(f_out["vrot_gas"][()])
            assert np.isfinite(vrot_gas)
            assert vrot_gas > 0.0


def test_plot_tully_fisher_reads_vrot_gas():
    with tempfile.TemporaryDirectory() as tmpdir:
        snap_path = create_test_galaxy_snapshot(output_dir=tmpdir, filename="test_galaxy.hdf5")

        with h5py.File(snap_path, "r") as f:
            header = dict(f["Header"].attrs.items())
            particle_data = {
                "header": header,
                "halo_pos": np.zeros(3, dtype=float),
                "particles": {
                    0: {
                        "Coordinates": f["PartType0"]["Coordinates"][:],
                        "Velocities": f["PartType0"]["Velocities"][:],
                        "Masses": f["PartType0"]["Masses"][:],
                    },
                    1: {
                        "Coordinates": f["PartType1"]["Coordinates"][:],
                        "Velocities": f["PartType1"]["Velocities"][:],
                        "Masses": f["PartType1"]["Masses"][:],
                    },
                    4: {
                        "Coordinates": f["PartType4"]["Coordinates"][:],
                        "Velocities": f["PartType4"]["Velocities"][:],
                        "Masses": f["PartType4"]["Masses"][:],
                    },
                },
                "run_num": 0,
                "snapnum": "test",
                "target": 0,
            }

        outpath = compute_rotation_curve_and_save(
            run_num=0,
            snapnum="test",
            target=0,
            output_dir=tmpdir,
            particle_data=particle_data,
            filename_suffix="_test",
        )

        png_path = plot_tully_fisher(
            rot_curve_file=outpath,
            run_num=0,
            output_dir=tmpdir,
            filename_suffix="_test",
        )

        assert png_path is not None
        assert os.path.exists(png_path)
