# ComFree-Sim: GPU-Parallelized Analytical Contact Physics Engine

![Teaser](teaser.jpg)

ComFree-Sim is a GPU-parallelized analytical contact physics engine designed for scalable contact-rich robotics simulation and control. This engine provides efficient simulation of complex interaction dynamics while exploiting modern GPU hardware for significant computational speedup.

## Overview

ComFree-Sim enables fast and accurate simulation of robots interacting with their environment through contacts. The engine supports large-scale parallel simulations, making it ideal for:

- Contact-rich robotics tasks (manipulation, locomotion, etc.)
- Multi-environment parallel simulation
- GPU-accelerated physics simulation
- Scalable simulation pipelines for learning and control

## Resources

- **Project Website**: https://irislab.tech/comfree-sim/
- **Documentation**: https://irislab.tech/comfree-doc/intro.html
- **Paper (arXiv)**: https://arxiv.org/abs/2603.12185

## Installation

Install the package using pip:

```bash
pip install .
```

Or with UV package manager:

```bash
uv sync
```

## Citation

If you use ComFree-Sim in your research, please cite:

```bibtex
@article{borse2026comfree,
  title={ComFree-Sim: A GPU-Parallelized Analytical Contact Physics Engine for Scalable Contact-Rich Robotics Simulation and Control},
  author={Borse, Chetan and Xie, Zhixian and Huang, Wei-Cheng and Jin, Wanxin},
  journal={arXiv preprint arXiv:2603.12185},
  year={2026}
}
```

## Quick Start

### Basic Simulation with Viewer

Run an interactive simulation with the native MuJoCo viewer:

```bash
python tests_local_viewer/test_viewer.py
```

This script loads a test scene and displays the simulation in real-time using the built-in MuJoCo viewer. You can modify the `engine` variable (0=MJC, 1=MJWARP, 2=COMFREE_WARP) to compare different simulation backends.

### Headless Simulation

Run a headless simulation without visualization:

```bash
python test_headless.py
```

This script runs simulation without viewer. It supports streaming the simulation state to a remote visualization server. For detailed options and configuration, refer to the [documentation](https://irislab.tech/comfree-doc/intro.html).

### Throughput Benchmarking

Run a throughput benchmark with parallel hand simulation:

```bash
python tests_local_viewer/test_throuput_hand.py
```

This script evaluates the performance of different engines with parallel environments. It benchmarks:
- Mujoco in Warp
- ComFree-Sim contact physics in Warp

Results include throughput metrics and step time statistics across multiple parallel environments.

### Rolling Friction Test

Run the rolling friction viewer test:

```bash
python tests_local_viewer/test_rolling.py
```

This script runs a cylinder rolling test (`benchmark/test_data/cylinder_rolling.xml`) and logs the linear velocity decay under different rolling friction settings.

### Z-Axis Rotation (Torsional Friction) Test

Run the torsional friction viewer test:

```bash
python tests_local_viewer/test_z_rotate.py
```

This script runs a ball rotation test (`benchmark/test_data/ball_rotation.xml`) and logs angular velocity decay under different torsional friction settings.

### Python API

```python
import comfree_warp as cf_warp

# Create your simulation environment
# See documentation for detailed examples
```

For comprehensive examples and tutorials, visit the [documentation](https://irislab.tech/comfree-doc/intro.html).

## License

See LICENSE file for details.
