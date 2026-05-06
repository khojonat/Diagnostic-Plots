import numpy as np


def _parse_run_index(run):
    if isinstance(run, str):
        run = run.strip()
        if run.startswith("run_"):
            run = run.split("run_", 1)[1]
        try:
            return int(run)
        except ValueError:
            raise ValueError(
                f"Cannot parse run index from run directory '{run}'. "
                "Use a numeric run index or a directory like 'run_0'."
            )
    return int(run)


def read_sobol(run, sobol_path):
    run_idx = _parse_run_index(run)

    # Change basePath as needed
    basePath = os.path.join(sobol_path, 'sobol_params.txt')

    sobol = np.loadtxt(basePath, skiprows=1)

    return sobol[run_idx]