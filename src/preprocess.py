import numpy as np
import pandas as pd


def ensure_columns(df, required_columns):
    """
    如果缺少某些列，就补成空列
    这样后续代码不会因为列不存在而报错
    """
    df = df.copy()
    for col in required_columns:
        if col not in df.columns:
            df[col] = np.nan
    return df


def basic_clean(df):
    """
    基础数据清洗
    只做非常简单的处理：
    1. 明显不合理的负数改成空值
    2. 固含量限制在 0~1
    """
    df = df.copy()

    positive_cols = [
        "density",
        "viscosity",
        "surface_tension",
        "tank_diameter",
        "liquid_height",
        "impeller_diameter",
        "speed_rps",
        "clearance",
        "mixing_time",
        "power",
    ]

    for col in positive_cols:
        if col in df.columns:
            df.loc[df[col] <= 0, col] = np.nan

    if "solid_fraction" in df.columns:
        df["solid_fraction"] = df["solid_fraction"].clip(0, 1)

    return df