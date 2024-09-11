import numpy as np
import os
import math
import time
import sys
import pandas as pd
import argparse
import textwrap

'''
# recursive copying result files from source to dest: 
# rsync -av -e ssh --include '*/' --include 'out_cubes*.txt' --exclude '*' tk2737@greene.hpc.nyu.edu:/scratch/tk2737/DeepFDR/data/sim/ .
# rsync -av --include '*/' --include 'out_cubes0.1_lr_0.01.txt' --exclude '*' /path/to/source/ /path/to/destination/

THESE ARE MY HYPERPAREMETERS FOR SIMULATION:
10% cube: p-value proportion = 0.2
        python -m crfasrnn_hpc.train --datapath /Users/taehyo/Dropbox/NYU/Research/Research/Data/deepfdr/data/mu/mu_n4_2/data0.1.npy --labelpath /Users/taehyo/Dropbox/NYU/Research/Research/Data/deepfdr/data/cubes0.1.npy --savepath /Users/taehyo/Downloads/TEMP --e 5 --lr 1e-2 --replications 10
        python -m crfasrnn_hpc.train --datapath /Users/taehyo/Dropbox/NYU/Research/Research/Data/deepfdr/data_crfrnn/mu/mu_n2_2/data1.npy --labelpath /Users/taehyo/Dropbox/NYU/Research/Research/Data/deepfdr/data_crfrnn/cube1.npy --savepath /Users/taehyo/Dropbox/NYU/Research/Research/Data/deepfdr/data_crfrnn/mu/mu_n2_2 --e 5 --lr 1e-2 --replications 50

        self.alpha = std_value
        self.beta = std_position
        self.gamma = std_position
        self.spatial_ker_weight = 0.6
        self.bilateral_ker_weight = 1.4
20% cube: p-value proportion = 0.1
        python -m crfasrnn_hpc.train --datapath /Users/taehyo/Dropbox/NYU/Research/Research/Data/deepfdr/data/mu/mu_n2_2/data0.2.npy --labelpath /Users/taehyo/Dropbox/NYU/Research/Research/Data/deepfdr/data/cubes0.2.npy --savepath /Users/taehyo/Downloads/TEMP --e 10 --replications 10 --lr 5e-3
        self.alpha = std_value
        self.beta = std_position
        self.gamma = std_position
        self.spatial_ker_weight = 0.1
        self.bilateral_ker_weight = 1.4
30% cube: p-value proportion = 0.1  
        python -m crfasrnn_hpc.train --datapath /Users/taehyo/Dropbox/NYU/Research/Research/Data/deepfdr/data/mu/mu_n2_2/data0.3.npy --labelpath /Users/taehyo/Dropbox/NYU/Research/Research/Data/deepfdr/data/cubes0.3.npy --savepath /Users/taehyo/Downloads/TEMP --e 10 --lr 5e-3 --replications 10
        self.alpha = std_value
        self.beta = std_position
        self.gamma = std_position
        self.spatial_ker_weight = 0.1
        self.bilateral_ker_weight = 1.4

For 20% and 30% cubes' mu_1 = -4,-3.5,-3.0: 
        python -m crfasrnn_hpc.train --datapath /Users/taehyo/Dropbox/NYU/Research/Research/Data/deepfdr/data/mu/mu_n2_2/data0.2.npy --labelpath /Users/taehyo/Dropbox/NYU/Research/Research/Data/deepfdr/data/cubes0.2.npy --savepath /Users/taehyo/Downloads/TEMP --e 10 --replications 10 --lr 1e-3
        self.alpha = std_value
        self.beta = std_position
        self.gamma = std_position
        self.spatial_ker_weight = 0.1
        self.bilateral_ker_weight = 1.4
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test_Statistics Generation')
    parser.add_argument('--simulation_path', default="./", type=str)
    args = parser.parse_args()

    savepath_list = []
    mu_list = ['mu_n4_2', 'mu_n35_2', 'mu_n3_2', 'mu_n25_2', 'mu_n2_2', 'mu_n15_2', 'mu_n1_2', 'mu_n05_2', 'mu_n0_2'] #,
    sigma_list = ['sigma_125_1', 'sigma_25_1', 'sigma_5_1', 'sigma_1_1', 'sigma_2_1', 'sigma_4_1', 'sigma_8_1'] # 
    root = '/scratch/tk2737/DeepFDR/data/sim'

    for sig in sigma_list:
        path = root + '/' + 'sigma/' + sig
        savepath_list.append(path)

    for mu in mu_list:
        path = root + '/' + 'mu/' + mu
        savepath_list.append(path)
    
    for path in savepath_list:
        savepath = path
        datapath = os.path.join(savepath, 'data0.1.npy')
        labelpath = '/scratch/tk2737/DeepFDR/data/sim/cubes0.1.npy'
        filepath = os.path.join(args.simulation_path, "cube10", os.path.basename(os.path.normpath(savepath)))
        
        if not os.path.isdir(filepath):
            os.makedirs(filepath)
            
        script = textwrap.dedent(f'''#!/bin/bash
#
#SBATCH --job-name=GTG
#SBATCH --nodes=1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=20
#SBATCH --time=1:00:00

function cleanup_tmp_dir()
{{
if [[ "$TMPDIR" != "" ]] && [[ -d $TMPDIR ]]; then
    rm -rf $TMPDIR
fi
}}

trap cleanup_tmp_dir SIGKILL EXIT

export TMPDIR=/vast/tk2737/tmp/tmp-job-$SLURM_JOB_ID
mkdir -p $TMPDIR

module purge
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

singularity exec --nv --overlay /scratch/tk2737/singularity/crfrnn/overlay-50G-10M.ext3:ro \\
/scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "source /ext3/env.sh; cd \\
/scratch/tk2737/crfrnn; python -m crfasrnn_hpc.train --savepath {savepath} --datapath \\
{datapath} --labelpath {labelpath} --e 5 --lr 1e-2 --replications 50"''')
        with open(os.path.join(filepath, 'run_crfrnn.sbatch'), 'w') as outfile:
            outfile.write(script)










