from pathlib import Path

from oceanicospy.models.sfincspy.execution.run_case import CaseRunner


def _dict_folders(root):
    return {
        "input": f"{root}/input/",
        "pros": f"{root}/pros/",
        "run": f"{root}/run/",
        "output": f"{root}/output/",
    }


class TestFillSlurmFile:
    def test_copies_template_and_returns_path(self, tmp_path):
        (tmp_path / "run").mkdir()
        (tmp_path / "output").mkdir()
        runner = CaseRunner(_dict_folders(str(tmp_path)))

        out_path = runner.fill_slurm_file(case_name="CasoE1", ntasks=60)

        assert Path(out_path).is_file()
        assert Path(out_path).name == "launcher_sfincs.slurm"
        assert Path(out_path).parent == tmp_path / "run"

    def test_fills_expected_placeholders(self, tmp_path):
        (tmp_path / "run").mkdir()
        (tmp_path / "output").mkdir()
        runner = CaseRunner(_dict_folders(str(tmp_path)))

        out_path = runner.fill_slurm_file(
            case_name="CasoE1", ntasks=60, image_name="sfincs-cpu_sfincs-v2.3.0-mt-Faber-Release.sif",
        )
        text = Path(out_path).read_text()

        assert '#SBATCH --job-name="CasoE1"' in text
        assert "#SBATCH --ntasks-per-node=60" in text
        assert f'--output={tmp_path}/output/%j.out' in text
        assert f'--error={tmp_path}/output/%j.err' in text
        assert "singularity exec /localapps/sfincs-cpu_sfincs-v2.3.0-mt-Faber-Release.sif sfincs" in text
        assert f"cd {tmp_path}/run/" in text

    def test_does_not_touch_native_bash_variables(self, tmp_path):
        # $HOSTNAME, $SLURM_JOB_ID, etc. are resolved by the shell at run
        # time, not by our templating - they must survive untouched.
        (tmp_path / "run").mkdir()
        (tmp_path / "output").mkdir()
        runner = CaseRunner(_dict_folders(str(tmp_path)))

        out_path = runner.fill_slurm_file(case_name="CasoE1", ntasks=60)
        text = Path(out_path).read_text()

        for bash_var in ["$HOSTNAME", "$SLURM_JOB_ACCOUNT", "$SLURM_JOB_USER", "$SLURM_JOB_ID"]:
            assert bash_var in text

    def test_no_mpi_launcher_and_no_result_file_relocation(self, tmp_path):
        """Regression guard: SFINCS runs OpenMP-only inside the container
        (confirmed against the real slurm_launch.sh files in this project's
        SFINCS backup - no MPI launcher command is used, unlike the XBeach
        template) and keeps its own output files (sfincs_map.nc /
        sfincs_his.nc / sfincs.log) in run/ rather than moving them to
        output/ like the XBeach template does."""
        (tmp_path / "run").mkdir()
        (tmp_path / "output").mkdir()
        runner = CaseRunner(_dict_folders(str(tmp_path)))

        out_path = runner.fill_slurm_file(case_name="CasoE1", ntasks=60)
        text = Path(out_path).read_text()

        mpi_launcher_token = "mpi" + "run"  # split to avoid a pytest assertion-diff quirk with this token
        assert mpi_launcher_token not in text
        assert "mv " not in text

    def test_uses_the_production_default_image(self, tmp_path):
        (tmp_path / "run").mkdir()
        (tmp_path / "output").mkdir()
        runner = CaseRunner(_dict_folders(str(tmp_path)))

        out_path = runner.fill_slurm_file(case_name="CasoE1", ntasks=60)
        text = Path(out_path).read_text()

        assert "sfincs-cpu_sfincs-v2.3.0-mt-Faber-Release.sif" in text
