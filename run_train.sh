#!/bin/bash
#SBATCH -p gpi.compute
#SBATCH -w gpic11
#SBATCH --time 8:00:00
#SBATCH --mem 16G
#SBATCH --gres gpu:1,gpumem:8G

python Step3_WSI_classification_ACMIL.py --seed 200 --split $SLURM_ARRAY_TASK_ID --name "518x518" --level "lvl1" --arch ga --n_token 5 --data_dir "/mnt/work/users/lauren.jimenez/carlos_conference/features/tcga_level_1_cvpr_crc2sk.h5" --n_masked_patch 10 --mask_drop 0.6 --wandb_mode online --config "/home/usuaris/imatge/lauren.jimenez/Documents/ACMIL_to_conference/ACMIL/config/skin.yml"
