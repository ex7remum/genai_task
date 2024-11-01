# Controllable GenAI (AIRI) and BayesGroup (HSE) test task 2024-2025
 
In this repository we implement DDPM with classifier-free guidance to generate 64x64 images based on Food-101 dataset.

We trained our model on A100 80GB with cuda version 11.4 and with Python 3.8.10.

All libraries and versions used for training can be found in file `requirements.txt`. 

To train model use command:
 
```bash
python3 train.py exp.config_path=path/to/train/config.yaml
```
 
To test trained model use:
 
```bash
python3 inference.py exp.config_path=path/to/inference/config.yaml train.checkpoint_path="path/to/ckpt"
```

Result metrics would be saved in file `result_metrics.json`.

Our final trained checkpoint: [link](https://drive.google.com/file/d/1z7Pp_oqV3slGnh6NxWstrN014ejpjrXZ/view?usp=sharing) (FID: 25.84).

Final report: [link](https://api.wandb.ai/links/extremum/fvmcqruq).
