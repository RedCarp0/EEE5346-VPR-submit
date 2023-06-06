# EEE5346-VPR-submit

## Environment
* Conda environment: see `py39_VPR.yaml`

## How to Run
* Training: Run `run_experiment.py` directly
* Val output: See `load-checkpoint-valdataset-night.ipynb`

## Note
* We have various versions of code, the submitted code is one of them.

## File System
* To prevent possible issues while running the code, we demonstrate the original construction of the package.
```
EEE5346-VPR/
|-- load-checkpoint-valdataset-night.ipynb
|-- LCD_DIY_1 -> /home/rcvlab/private_datasets/LCD_DIY_1
|-- README.md
|-- __init__.py
|-- dataset_txt
|   |-- Autumn_mini_val_query.txt
|   |-- Autumn_val_stereo_centre.txt
|   |-- DIY1_day.txt
|   |-- DIY1_night.txt
|   |-- Night_mini_val_ref.txt
|   |-- Night_val_stereo_centre.txt
|   |-- Suncloud_mini_val_ref.txt
|   `-- Suncloud_val_stereo_centre.txt
|-- draft.py
|-- ee5346_dataset -> /home/rcvlab/public_datasets/ee5346_dataset
|-- experiment_config
|   |-- config1.yaml
|   |-- config2.yaml
|   |-- config3.yaml
|   |-- config4.yaml
|   |-- config5.yaml
|   |-- config6.yaml
|   |-- config7.yaml
|   |-- config8.yaml
|-- experiment_output
|   |-- experiment1
|   |-- experiment2
|   |-- experiment3
|   |-- experiment4
|   |-- experiment5
|   |-- experiment6
|   `-- experiment8
|-- loop_closure_dataset
|   |-- DIY1_groundtruth_20.txt
|   |-- DIY1_groundtruth_25.txt
|   |-- robotcar_qAutumn_dbNight_diff_final.txt
|   |-- robotcar_qAutumn_dbNight_easy_final.txt
|   |-- robotcar_qAutumn_dbNight_val_final.txt
|   |-- robotcar_qAutumn_dbSunCloud_diff_final.txt
|   |-- robotcar_qAutumn_dbSunCloud_easy_final.txt
|   `-- robotcar_qAutumn_dbSunCloud_val_final.txt
|-- model.py
|-- output_autumn_night.txt
|-- output_autumn_suncloud.txt
|-- py39_VPR.yaml
|-- run_experiment.py
`-- utils.py
```