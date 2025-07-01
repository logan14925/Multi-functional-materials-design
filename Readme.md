# Multi-functional Material Inverse Design

A repository for inverse designing multi-functional materials based on RCGAN, containing Abaqus simulation, post-processing, experimental analysis, and machine learning.

## Author

- **Name:** Xue Jiacheng  
- **Email:** jiachengxue2001@gmail.com

## Project Structure

Below is the project file tree, with a brief introduction to each main folder:

```
Code_Project/
├── Abaqus/         # Scripts and configs for finite element modeling and simulation (Abaqus)
│   ├── code/           # Python scripts for geometry, meshing, and post-processing
│   ├── FEM_config.json # Configuration file for simulation parameters
│   └── parameters.csv  # Example parameter set for batch simulations
├── Exp/            # Experimental data and analysis scripts
│   ├── calib/          # Calibration scripts and data for experiments
│   └── possion_dic/    # Scripts/data for Poisson's ratio calculation from experiments
├── ML/             # Machine learning models and utilities
│   ├── dataset/        # Datasets for training and evaluation
│   ├── forward/        # Forward prediction model scripts
│   ├── modles.py       # Model definitions
│   └── rcgan/          # RCGAN-based inverse design scripts
└── Readme.md       # Project introduction and documentation
```

## Workflow Overview

1. **Simulation**: Use the scripts in `Abaqus/` to generate and simulate material structures under various parameter settings.
2. **Experiment**: Analyze experimental data in `Exp/` to extract material properties for model validation.
3. **Machine Learning**: Train and apply models in `ML/` to predict or design material structures with desired properties.

This integrated approach accelerates the discovery and optimization of advanced materials by combining simulation, experiment, and AI.