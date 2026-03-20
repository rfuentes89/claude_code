# MRI Research Project

This repository contains research code for MRI sequence development and reconstruction.

## Languages and roles

- MATLAB:
  - MRI sequences (IDEA)
  - Pulseq for MRI sequences design
  - Raw reconstruction

- Python:
  - Data processing
  - Pypulseq for sequence design
  - Pipelines
  - Visualization
  - Integration layer

- Julia:
  - High-performance numerical routines
  - Optimization and reconstruction experiments
  - KomaMRI for MRI simulations

## Guidelines

- Use Python as the main orchestration language
- Do not rewrite MATLAB sequence logic unless necessary
- Prefer Julia only for performance-critical parts
- Keep functions modular and well-documented

## Data

- Do not modify raw data
- Outputs should be reproducible

## Goal

Build reproducible MRI reconstruction pipelines combining MATLAB, Python, and Julia
