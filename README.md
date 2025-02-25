# Utilizing EarthFormer Transformer for Radar Data Extraction from Satellite Imagery Using the SEVIR Dataset (NI-MVI semestral project)

## About
This project aims to generate synthetic radar data from satellite imagery from the [SEVIR dataset](http://sevir.mit.edu/sevir-dataset) 
using the [Earthformer](https://arxiv.org/abs/2207.05833) transformer and U-Net convolutional network.

The implementation was adapted from or inspired by the following sources:
* [implementation of U-Net for satellite-to-radar translation by the authors of the SEVIR dataset](https://github.com/MIT-AI-Accelerator/neurips-2020-sevir)
* [official implementation of the Earthformer model](https://github.com/amazon-science/earth-forecasting-transformer)
* [implementation of U-Net in PyTorch for Carvana Image Masking Challenge](https://github.com/milesial/Pytorch-UNet)
* [my bachelor's thesis focused on precipitation nowcasting with neural networks](https://gitlab.fit.cvut.cz/miskafil/bi-bap)

The rest of the code is my own work.

## Assignment
The SEVIR (Storm EVent ImagRy) dataset provides diverse data modalities such as radar, satellite imagery, and lightning, making it a valuable resource for meteorological analysis. The EarthFormer transformer, modeled after the Transformer architecture, shows promise in handling Earth data, thus apt for satellite imagery.

This work aims to employ the EarthFormer transformer to extract and interpret radar data from the SEVIR dataset. The proposed work is structured in three segments:

Data Preparation and Pre-processing:
Preparing the radar data within SEVIR for analysis, addressing data quality issues, and structuring the data for the EarthFormer transformer.

Radar Data Extraction using EarthFormer:
Utilizing EarthFormer transformer for mining satellite data to model radar data and use other machine learning models (UNET) to evaluate EarthFormer's effectiveness.

Evaluation and Future Implications:
Evaluating the performance of EarthFormer in radar data extraction from SEVIR and discussing the broader implications of this exploration for real-time meteorological analysis using satellite imagery.

This work aims to demonstrate how the EarthFormer transformer, coupled with the SEVIR dataset, can enhance meteorological analysis and predictions.


## Get started
### Install packages
For full use (training, testing, prediction) create a conda environment with:
```shell
conda env create -n <NAME> -f environment.yaml
conda activate <NAME>
```

If you only want to run a "lightweight" version of the project (good for quick debugging, running notebooks etc.) you can install the core packages with:
```shell
pip install -r requirements.txt
```

### Download data
The SEVIR dataset can be downloaded from [AWS](https://registry.opendata.aws/sevir/). Since the dataset is very large (~ 1 TB),
this project uses only subset of the dataset (specifically visible spectrum satellite images are not used). To download only the files of this subset, run:

```shell
python scripts/download_sevir.py --dir <DOWNLOAD_DIRECTORY>
```

After downloading the data, change the `dataset/sevir_root_dir` field
in the particular config file to the directory with the downloaded data.

Note: All the downloaded files will take up about 220 GB of space.

## Training
### Training the models
Train the model:
```shell
python train.py --config <CONFIG_FILE> --gpu <GPU_NUMBER>
```

### Testing the models
To test the model a path to a PyTorch Lightning checkpoint with the trained model must be provided.

Test the model:
```shell
python test.py --config <CONFIG_FILE> --gpu <GPU_NUMBER> --checkpoint <CKPT_PATH>
```

### Other information
To show all the argument options, run:
```shell
python train.py --help
```

## Predicting
Example of inference and visualization of the predictions is given in `prediction.ipynb`.

## Report

A full report of the project is available in the `report_latex/report.pdf` file.
