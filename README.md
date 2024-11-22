# Series2Vec for EEG

Course project at KTH Royal Institute of Technology under the supervision of Erik Fransén and Gonzalo Uribarri. In this work we try to apply the self-supervised learning framework Series2Vec to a neurodegenerative diseases EEG classification task.

This repository was cloned from https://github.com/Navidfoumani/Series2Vec, a Pytorch implementation of Series2Vec from Foumani et al. \[1\]. 

We used a dataset published by Miltiadous et al. of resting-state EEG recordings from subjects diagnosed with Alzheimer's disease or Frontotemporal dementia and healthy participants \[2\].

## Datasets

### EEG dataset
Please download dataset files (we only use the already preprocessed "derivatives" file) from [here](https://openneuro.org/datasets/ds004504/versions/1.0.6) and place them in Dataset/EEG/EEG/derivatives.


### Original datasets: Get data from UEA Archive and HAR and Ford Challenge
For the datasets of the original article.
Download dataset files and place them into the specified folder
UEA: [Here](https://www.timeseriesclassification.com/aeon-toolkit/Archives/Multivariate2018_ts.zip)

Copy the datasets folder to: Datasets/UEA/

The Datasets of the Benchmarks should be downloaded by running `downloader.py`.

## Setup
Updated from the original repository.
only use `Python 3.8`, Linux

`pip install -r requirements.txt`

## Run
Run main.py with the desired arguments. Those are defined in utils/args.py.

## References

\[1\] Foumani, N.M., Tan, C.W., Webb, G.I. et al. Series2vec: similarity-based self-supervised representation learning for time series classification. Data Min Knowl Disc 38, 2520–2544 (2024). https://doi.org/10.1007/s10618-024-01043-w

\[2\] Andreas Miltiadous and Katerina D. Tzimourta and Theodora Afrantou and Panagiotis Ioannidis and Nikolaos Grigoriadis and Dimitrios G. Tsalikakis and Pantelis Angelidis and Markos G. Tsipouras and Evripidis Glavas and Nikolaos Giannakeas and Alexandros T. Tzallas (2023). A dataset of EEG recordings from: Alzheimer's disease, Frontotemporal dementia and Healthy subjects. OpenNeuro. \[Dataset\] doi: doi:10.18112/openneuro.ds004504.v1.0.6

