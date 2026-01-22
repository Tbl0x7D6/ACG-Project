# Advanced Computer Graphics Project

A physics-based simulation framework for cloth, fluid dynamics, and rigid body interactions, built with [Taichi](https://taichi-lang.cn/).

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Taichi](https://img.shields.io/badge/Taichi-1.7.4-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

https://github.com/user-attachments/assets/bcc62bbd-2327-4c81-9f00-52e7cb95cb3f

## Features

- **Cloth Simulation**: Mass-spring model with collision handling
- **Fluid Simulation**: Level-set based fluid solver with volume conservation
- **Rigid Body Dynamics**: Collision detection using BVH acceleration
- **Fluid-Structure Interaction (FSI)**: Strong coupling between fluid and rigid bodies
- **Mesh Export**: Convert simulation results to mesh files (PLY format) for Blender rendering

## Project Structure

```
ACG-Project/
├── cloth.py                                        # Cloth draping on a fixed sphere
├── cloth_bunny.py                                  # Cloth draping on a rotatable bunny mesh
├── cloth_bunny_copy.py                             # Cloth draping on a fixed bunny mesh (for Blender rendering)
├── cloth_moving_bunny.py                           # A bunny mesh falling onto a cloth
├── cloth_trivial_rigid.py                          # A sphere falling onto cloth
├── fluid_2d.py                                     # 2D fluid simulation
├── fluid_3d.py                                     # 3D fluid simulation
├── fluid_2d_FSI.py                                 # 2D fluid with rigid body interaction
├── fluid_3d_FSI.py                                 # 3D fluid with rigid body interaction
├── rigid.py                                        # A rotating rigid stick simulation
├── reinit_levelset.py                              # Level-set reinitialization utility
├── materials/                                      # Core simulation modules
│   ├── cloth.py                                    # Cloth class with mass-spring model
│   ├── rigid_body.py                               # RigidBody class with BVH collision
│   └── load_mesh.py                                # OBJ mesh loader
├── render/                                         # Blender rendering files
│   ├── Wood/                                       # Wood texture
│   ├── fluid.blend                                 # Blender file for fluid rendering
│   └── kloofendal_48d_partly_cloudy_puresky_4k.exr # HDRI environment map
├── utils/                                          # Utility modules
│   ├── transfer.py                                 # Data transfer utilities
│   ├── Cloth2Mesh/                                 # Cloth to mesh conversion
│   │   ├── clothtomesh.py                          # Export cloth particles to PLY
│   │   ├── blender_scripts.py                      # Blender automation scripts
│   │   └── change_scene.py                         # Link up rendering scene
│   └── SDF2Mesh/                                   # SDF to mesh conversion
│       ├── tomesh.py                               # Main mesh extraction pipeline
│       ├── all2mesh.py                             # Convert all frames to mesh
│       ├── run.py                                  # Run mesh conversion with multi-threading
│       ├── marching_cubes.py                       # Marching Cubes algorithm
│       ├── taubin.py                               # Taubin mesh smoothing
│       ├── gaussian.py                             # Gaussian smoothing
│       ├── laplacian.py                            # Laplacian smoothing
│       ├── subdivision.py                          # Loop subdivision
│       ├── downsampling.py                         # Mesh downsampling
│       └── blender_scripts.py                      # Blender automation scripts for fluid mesh
├── objects/                                        # 3D model files (OBJ format)
├── pyproject.toml                                  # Project configuration
└── README.md
```

## Prerequisites

- **uv**: 0.9.8
- **Vulkan SDK**: 1.4.313.0.
- **GPU Driver / CUDA**: 535.274.02, CUDA 12.2.
- **Blender**: 4.0+ (optional, for mesh rendering)
- **FFmpeg**: (optional, for video generation from frames)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Tbl0x7D6/ACG-Project.git
cd ACG-Project
```

### 2. Install Dependencies

```bash
uv sync
```

## Usage

Only part of the simulation scripts support real-time rendering by taichi GUI, and others that require afterwards rendering in Blender are marked accordingly.

### Rigid Body Simulation

(*Real-time*) Simple rotating rigid stick simulation:
```bash
uv run rigid.py
```

### Cloth Simulation

(*Real-time*) Simulation of a cloth draping over a fixed sphere:
```bash
uv run cloth.py
```

(*Real-time*) Simulation of a sphere falling onto a cloth:
```bash
uv run cloth_trivial_rigid.py
```

(*Real-time*) Cloth draping on a bunny mesh (supporting real-time interaction by rotating the bunny with the mouse):
```bash
uv run cloth_bunny.py
```

(*Real-time*) Simulation of cloth draping on a fixed bunny mesh:
```bash
uv run cloth_bunny_copy.py
```

(*Real-time*) Simulation of a bunny mesh falling onto a cloth:
```bash
uv run cloth_moving_bunny.py
```

### Fluid Simulation

(*Real-time*) 2D fluid simulation. Screenshot of each frame will be saved to `output/` folder:
```bash
uv run fluid_2d.py
```

(*Not real-time*) 3D fluid simulation. Fluid levelset data will be saved to `levelset/` folder:
```bash
uv run fluid_3d.py
```

### Fluid-Structure Interaction

(*Real-time*) 2D FSI with a moving bunny. Screenshot of each frame will be saved to `output/` folder:
```bash
uv run fluid_2d_FSI.py
```

(*Not real-time*) 3D FSI simulation with a bunny mesh. Fluid levelset data will be saved to `levelset/` folder and rigid body states will be saved to `rigid_states/` folder:
```bash
uv run fluid_3d_FSI.py
```

### Mesh Export

The simulation can export mesh files for rendering:

1. **Fluid Mesh**: use the provided utility to convert levelset data to mesh files. Run the following command to convert all frames of levelset data to mesh files (0 and 100 are the start and end frame indices as an example):
```bash
uv run utils/SDF2Mesh/all2mesh.py levelset plys 0 100
# This step is extremely time-consuming. Please wait patiently for the process to complete.
```
2. **Rigid Body Mesh**: this step only aggregates rigid body state into a single file, and blender scripts are used to load the position and orientation data for rendering afterwards. Run the following command to aggregate rigid body states (output aggregated file will be `rigid_states/final.txt`, 0 and 100 are the start and end frame indices as an example):
```bash
uv run utils/transfer.py 0 100
```

### Rendering \& Video Generation

After obtaining the mesh files, you can use Blender to render the scenes and use FFmpeg to generate videos from the rendered frames.
1. Follow the instructions in [Stop-motion-OBJ](https://github.com/neverhood311/Stop-motion-OBJ) to set up the Blender add-on for sequentially importing OBJ/PLY files.
2. Open the provided Blender files [fluid.blend](render/fluid.blend) and set the start and end frames number according to the number of mesh files.
3. Load all the mesh files following the above add-on instructions. Then select the bunny object in the scene.
4. Go to the `Scripting` tab in Blender, and run the provided script to automate setting up materials and positions. (This step is memory-intensive; ensure your system has sufficient resources.)
5. Press `Ctrl + F12` to render the animation to frames. Rendered frames will be saved in `render/cache/` folder by default.
6. Use FFmpeg to convert the rendered frames to a video:
```bash
ffmpeg -framerate 60 -i render/cache/frame_%04d.png -c:v libx265 -crf 20 -preset medium -pix_fmt yuv420p output.mp4
```

Congratulations! You have successfully completed the whole simulation and rendering pipeline.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Taichi Graphics](https://taichi.graphics/) for the amazing parallel computing framework
- [Blender](https://www.blender.org/) for the powerful 3D software
- Course instructors and TAs for guidance and support
