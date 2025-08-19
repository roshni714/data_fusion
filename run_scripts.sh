#!/bin/bash


python automate_config.py --indicator "synthetic" --methods "state_adjusted" --balancing-features "X" --n-constraints 1 --slack 0.01 --evaluation-group "X" --moment-group "X"
python automate_config.py --indicator "synthetic" --methods "model_free" --evaluation-group "X" --moment-group "X"
python automate_config.py --indicator "synthetic" --methods "ground_truth" --evaluation-group "X"
python automate_config.py --indicator "synthetic" --methods "state_unadjusted" --evaluation-group "X"
python automate_config.py --indicator "synthetic" --methods "national" --evaluation-group "X" --moment-group "X"

indicators=("medicaid_ins" "snap" "RECVDVACC")
for indicator in "${indicators[@]}";
do
  python automate_config.py --indicator "$indicator" --methods "ground_truth" --evaluation-group "state_name"
  python automate_config.py --indicator "$indicator" --methods "state_adjusted" --balancing-features "intercept" --n-constraints 1 --slack 0.01 --evaluation-group "state_name" --moment-group "state_name"
  python automate_config.py --indicator "$indicator" --methods "model_free" --evaluation-group "state_name" --moment-group "state_name"
  python automate_config.py --indicator "$indicator" --methods "national" --evaluation-group "state_name" --moment-group "state_name"
  python automate_config.py --indicator "$indicator" --methods "state_unadjusted" --evaluation-group "state_name"
done


mkdir scripts
python generate_sbatches.py

for experiment in scripts/*.sh
do
    echo $experiment
    chmod u+x $experiment
    $experiment
    sleep 1
done
