# oceanicospy

**`oceanicospy` is an open-source, user-friendly Python library that serves as a framework for automating and standardizing the scientific workflow in numerical wave and hydrodynamic modelling at the coastal scale**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15725746.svg)](https://doi.org/10.5281/zenodo.15725746)
![Version](https://img.shields.io/badge/version-0.1.0-blue)
![Status](https://img.shields.io/badge/status-stable-green)
![License](https://img.shields.io/badge/license-GPLv3-green)

---

## Features

- **Observations** — read data from RBR, AQUAlogger, AWAC, CTD, weather stations, and more.
- **Analysis** — temporal and spectral techniques, including `WaveSpectralAnalyzer` and tidal analysis.
- **Models** — preprocessing automation for numerical models (SWAN, WW3, XBeach).
- **Retrievals** — automated downloads from ERA5, Copernicus Marine (CMEMS), and UHSLC real-time data.
- **GIS** — geospatial utilities for coastal data (shapefiles, XYZ grids, projections).
- **Plots** — quick, publication-ready visualizations of oceanographic variables.
- **Utils** — shared helper functions used across all subpackages.

---

## Installation

> **oceanicospy** is currently in beta. The pre-release flag is required to install the latest version.

### Local environment

Create a Python environment (conda, venv, or similar) and install from PyPI:

```bash
pip install oceanicospy
```

Verify the installation:

```bash
pip show oceanicospy
```

Expected output:

```
Name: oceanicospy
Version: 0.1.0
Summary: Python library that serves as a framework for automating and standardizing the scientific workflow in numerical wave and hydrodynamic modelling at the coastal scale.
Home-page: https://github.com/oceanicos-dev-org/oceanicospy
Author: 
Author-email: OCEANICOS developer team <oceanicos_med@unal.edu.co>
License: GNU GENERAL PUBLIC LICENSE
```

### Google Colab

Install the latest stable version in a Colab notebook:

```python
!pip install oceanicospy
```

A runtime restart may be required after installation due to dependency conflicts with Colab's pre-installed packages.

---

## Package structure

```
oceanicospy/
├── analysis/      # temporal and spectral analysis (WaveSpectralAnalyzer, tidal analysis, …)
├── gis/           # geospatial utilities (shapefiles, XYZ data, projections)
├── models/        # numerical model preprocessing (SWAN, WW3, XBeach)
├── observations/  # instrument readers (RBR, AQUAlogger, AWAC, CTD, …)
├── plots/         # visualization utilities
├── retrievals/    # automated data retrieval (ERA5, CMEMS, UHSLC)
└── utils/         # shared helpers
```

---

## Quick start

Import the full package:

```python
import oceanicospy
```

Or import only what you need:

```python
from oceanicospy.observations.pressure_sensors import RBR
from oceanicospy.analysis import WaveSpectralAnalyzer
```

> Wildcard imports (`from oceanicospy.analysis import *`) are convenient for exploration but can shadow names from other libraries. Prefer explicit imports in scripts.

---

## Contributing

Contributions are welcome. The workflow follows the standard GitHub fork-and-pull-request model.

### 1. Fork the repository

Go to [github.com/oceanicos-dev-org/oceanicospy](https://github.com/oceanicos-dev-org/oceanicospy) and click **Fork**.

### 2. Set up SSH authentication

Follow the [GitHub SSH guide](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account) if you haven't already.

### 3. Clone your fork

```bash
git clone git@github.com:YOUR-USERNAME/oceanicospy.git
cd oceanicospy
```

Optionally add the upstream remote to stay in sync:

```bash
git remote add upstream git@github.com:oceanicos-dev-org/oceanicospy.git
```

### 4. Install in editable mode

```bash
pip install -e .
```

Install optional dependency groups for documentation or development:

```bash
pip install -e ".[docs]"   # sphinx, sphinx-book-theme, nbsphinx, myst-nb
pip install -e ".[dev]"    # pytest, build, twine
```

### 5. Create a feature branch

Never commit directly to `main` or `integration`:

```bash
git checkout -b YOUR-USERNAME/my-feature
```

### 6. Commit and push

```bash
git add path/to/changed_file.py
git commit -m "add some particular feature to certain module"
git push origin YOUR-USERNAME/my-feature
```

### 7. Open a pull request

On GitHub, navigate to your fork and click **Contribute → Open a pull request**. Provide a clear title and description of what changed and why.

---

## License

Distributed under the **GNU General Public License v3 (GPLv3)**. See [`LICENSE`](LICENSE) for details.

---

## Contact

OCEANICOS developer team — [oceanicos_med@unal.edu.co](mailto:oceanicos_med@unal.edu.co)
