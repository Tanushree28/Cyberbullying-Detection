import pandas as pd


def df_prep(x, y=None, labels=None):
    if y is not None:
        df = pd.DataFrame(y, columns=labels)
        df["x"] = x
    else:
        df = pd.DataFrame(x, columns=labels)
        df["x"] = df.index
    df = df.melt(id_vars="x", value_name="y", var_name="series")
    return df
