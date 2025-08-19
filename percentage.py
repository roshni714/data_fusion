import pandas as pd

INDICATORS = ["medicaid_ins", "RECVDVACC", "snap"]

for indicator in INDICATORS:
    print(f"Processing {indicator}...")
    df = pd.read_csv(f"results/{indicator}.csv")
    res_df = df.groupby("method")["mae"].agg("mean").reset_index()

    for method in ["state_unadjusted", "national"]:
        improvement = (
            res_df[res_df.method == "state_adjusted"]["mae"].item()
            - res_df[res_df.method == method]["mae"].item()
        ) / res_df[res_df.method == method]["mae"].item()
        print(f"{indicator} {method} improvement: {improvement:.2%}")
    print("------\n")
