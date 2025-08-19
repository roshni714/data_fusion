# Data Fusion for High-Resolution Estimation

This repository contains the code to replicate the results from Data Fusion for High-Resolution Estimation.


## Data
The data for this paper is publicly available [here](https://drive.google.com/drive/folders/1ALGzHTkK1k4X5HJIPtzwmwJE1EApMMbki). The data from the ``Cleaned`` can be saved to a directory called ``data`` in this folder to replicate experiments.

## Environment
To setup the environment for running the code, edit the prefixfield of the environment.yml file to point to the directory of your conda environments. Then, run

```
conda env create -n new_env  --file environment.yml
conda activate new_env
```

## Reproduction of Paper Figures
To reproduce figures from paper, run the following command to launch all experiments.

```
cd data_fusion
chmod +x run_scripts.sh
./run_scripts.sh
```
After the experiments are complete, generate figures by running the following command.
```
python figures.py --indicator all
```

 
