#!/bin/bash
BASE_DIR="/home/zcesccc/Scratch/MCCollisions"
PYTHON_SCRIPT="/lustre/home/zcesccc/mc_collisions/ERP_tools/source/GetStatsFromMCJobs.py"

for sat_name in $(ls $BASE_DIR); do
  if [ -d "${BASE_DIR}/${sat_name}" ]; then
    echo "Processing $sat_name"
    nohup python $PYTHON_SCRIPT $sat_name > "output_${sat_name}.log" 2>&1 &
  fi
done
