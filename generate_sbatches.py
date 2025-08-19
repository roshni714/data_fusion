import os


GPU_SBATCH_PREFACE = """#!/bin/bash
#SBATCH -J train-gpu
#SBATCH -p gpu
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -t 1-             # limit of 1 day runtime
#SBATCH -G 1              # limit of 2 GPU's per user
#SBATCH -o train-gpu-%j.out
#SBATCH --job-name="{}.sh"
#SBATCH --error="{}/{}_err.log"
#SBATCH --output="{}/{}_out.log"\n
"""


SBATCH_PREFACE = """#!/bin/bash
#SBATCH -t 5:00:00
#SBATCH -c 1
#SBATCH --mem 20GB
#SBATCH -p normal
#SBATCH --ntasks-per-node=1
#SBATCH --job-name="{}.sh"
#SBATCH --error="{}/{}_err.log"
#SBATCH --output="{}/{}_out.log"\n
"""

OUTPUT_PATH = "/home/users/rsahoo/zfs/gsb/intermediate-yens/rsahoo/survey/scripts"


def generate_learn_run(indicator):
    configs = os.listdir(f"configs/{indicator}")
    # "state_adjusted_balancing_republican_pct_age_cat_lamb=1.0.yaml",
    # "state_adjusted_balancing_republican_pct_age_cat_education_lamb=1.0.yaml"]
    for config in configs:
        exp_id = indicator + "_" + config
        script_fn = os.path.join(OUTPUT_PATH, "{}.sh".format(exp_id))
        with open(script_fn, "w") as f:
            print(
                SBATCH_PREFACE.format(exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id),
                file=f,
            )
            base_cmd = f"python main.py main --config configs/{indicator}/{config}"
            print(base_cmd, file=f)
            print("sleep 1", file=f)


indicators = ["medicaid_ins", "snap", "RECVDVACC"]
# indicators=["RECVDVACC"]
for indicator in indicators:
    generate_learn_run(indicator)
