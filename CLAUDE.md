# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

2D Gaussian Splatting (2DGS) for geometrically accurate radiance fields. Represents scenes with 2D oriented disks (surfels) instead of 3D Gaussians, using perspective-correct differentiable rasterization. Paper: https://arxiv.org/pdf/2403.17888

## Setup

```bash
git clone https://github.com/hbb1/2d-gaussian-splatting.git --recursive
conda env create --file environment.yml
conda activate surfel_splatting
```

Requires Python 3.8, PyTorch 2.0, CUDA. Two custom CUDA submodules are built during install:
- `submodules/diff-surfel-rasterization` — perspective-correct surfel rasterizer
- `submodules/simple-knn` — k-NN for densification

## Key Commands

```bash
# Training
python train.py -s <dataset_path>
python train.py -s <dataset_path> --lambda_normal 0.05 --lambda_distortion 0.1 --depth_ratio 1

# Rendering + mesh extraction
python render.py -m <model_path> -s <dataset_path>                          # bounded
python render.py -m <model_path> -s <dataset_path> --mesh_res 1024 --unbounded  # unbounded

# COLMAP data prep
python convert.py -s <source_path> --resize

# Interactive viewer
python view.py -s <dataset_path> -m <model_path>

# Metrics
python metrics.py -m <model_path>
```

## Architecture

**Entry points:** `train.py` (training loop), `render.py` (inference + mesh extraction), `view.py` (viewer), `convert.py` (data prep)

**Pipeline flow:**
1. **Data loading** — `scene/dataset_readers.py` + `scene/colmap_loader.py` parse COLMAP or Blender format
2. **Gaussian model** — `scene/gaussian_model.py` defines surfel parameters: position, 2D scale, rotation (quaternion), opacity, SH coefficients (degree 0-3)
3. **Rasterization** — `gaussian_renderer/__init__.py` calls the CUDA `diff_surfel_rasterization` operator, returns RGB, depth, normals, distortion maps
4. **Training** — L1 + SSIM loss, normal consistency loss (after iter 7000), depth distortion loss (after iter 3000). Densification via split/clone runs iters 500-15000
5. **Mesh extraction** — `utils/mesh_utils.py` implements TSDF fusion from rendered depth maps (bounded and unbounded modes)

**Arguments:** `arguments/__init__.py` defines three groups: `ModelParams`, `PipelineParams`, `OptimizationParams`. Key params:
- `depth_ratio`: 0 = mean depth (unbounded), 1 = median depth (bounded)
- `lambda_normal`, `lambda_dist`: regularization weights
- Training runs 30,000 iterations by default

**Supported data formats:** COLMAP (PINHOLE/SIMPLE_PINHOLE only, detected via `sparse/` dir) and Blender (detected via `transforms_train.json`)

**Evaluation scripts:** `scripts/m360_eval.py`, `scripts/dtu_eval.py`, `scripts/tnt_eval.py` run full benchmark pipelines

## Known Constraints

- Only ideal pinhole cameras supported — principal point must be at image center
- Bounded mesh extraction requires tuning `depth_trunc`; unbounded mode auto-computes parameters
- Custom CUDA rasterizer requires NVIDIA GPU with compatible CUDA toolkit
