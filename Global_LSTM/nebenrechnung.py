import pandas as pd

datasplit_df = pd.read_csv("./data/datasplit.csv", usecols=["ALPAKAS_ID", "calibration_start", "validation_start"])

datasplit_df = datasplit_df.rename(columns={
    "calibration_start": "training_start",
    "validation_start": "testing_start"
})

datasplit_df["training_start"] = pd.to_datetime(datasplit_df["training_start"])
datasplit_df["testing_start"] = pd.to_datetime(datasplit_df["testing_start"])

datasplit_df.to_csv("./data/datasplit.csv")