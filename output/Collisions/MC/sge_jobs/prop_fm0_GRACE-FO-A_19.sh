#!/bin/bash -l
#$ -l h_rt=2:0:0
#$ -l mem=2G
#$ -l tmpfs=15G
#$ -N Prop_fm0_GRACE-FO-A_19
#$ -wd /home/$USER/Scratch/MCCollisions/GRACE-FO-A/propagation_fm0
cd $TMPDIR
cp /home/$USER/path/to/propagate_and_calculate.py $TMPDIR
cp /home/$USER/Scratch/MCCollisions/MC/interpolated_MC_ephems/GRACE-FO-A/nominal_collision.csv $TMPDIR
cp output/Collisions/MC/interpolated_MC_ephems/GRACE-FO-A/GRACE-FO-A_fm0_perturbed_states.csv $TMPDIR
module load python/3.7
python propagate_and_calculate.py GRACE-FO-A 0 19
cp * /home/$USER/Scratch/MCCollisions/GRACE-FO-A/propagation_fm0/
