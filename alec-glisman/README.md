# ReWoTes: ML Property Predict

Alec Glisman

## Overview

This directory contains files for the ML Property Predict project for Mat3ra.com.

Input data is accessed from the Materials Project and the data is cleaned into Pandas Dataframes inside `data/data_load.py`.
The input data source to the machine learning model can be augmented with additional Materials Project data with the `MaterialData` init method and external data can also be merged using its respective `add_data_columns` method.
The cleaned data is archived using Pandas in conjunction with HDF5 to lower runtime costs for model development.

## Usage

A Conda environment file has been provided (`requirements.yml`) to set up a Python environment called `ml-band-gaps` with the following command

```[bash]
conda env create -f requirements.yml
```

The overall project can then be run with

```[bash]
python3 main.py
```

Note that the data is sourced from the Materials Project, which requires an API key to access it.
I have added my API key to the `.gitignore` for security reasons, so users will need to generate their own and add it to an `api_key.txt` file.
