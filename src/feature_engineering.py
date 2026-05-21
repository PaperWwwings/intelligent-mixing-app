import numpy as np
import pandas as pd


def add_features(df):
    """
    增加无量纲特征：
    D/T (桨径比), H/T (液深比), C/T (离底间隙比),
    S/T (层间距比), Re (雷诺数), Fr (弗劳德数), We (韦伯数)
    """
    df = df.copy()
    eps = 1e-8  # 防止除以0

    # 如果有 rpm 没有 rps，则自动换算
    if "speed_rpm" in df.columns and "speed_rps" not in df.columns:
        df["speed_rps"] = df["speed_rpm"] / 60.0

    # ---- 几何比 ----
    if {"impeller_diameter", "tank_diameter"}.issubset(df.columns):
        df["D_T"] = df["impeller_diameter"] / (df["tank_diameter"] + eps)

    if {"liquid_height", "tank_diameter"}.issubset(df.columns):
        df["H_T"] = df["liquid_height"] / (df["tank_diameter"] + eps)

    if {"clearance", "tank_diameter"}.issubset(df.columns):
        df["C_T"] = df["clearance"] / (df["tank_diameter"] + eps)

    # 层间距比 S/T
    if {"impeller_spacing", "tank_diameter"}.issubset(df.columns):
        df["S_T"] = df["impeller_spacing"] / (df["tank_diameter"] + eps)

    # ---- 无量纲数 ----
    # 雷诺数 Re = rho * N * D^2 / mu
    if {"density", "speed_rps", "impeller_diameter", "viscosity"}.issubset(df.columns):
        df["Re"] = (
            df["density"] * df["speed_rps"] * (df["impeller_diameter"] ** 2)
            / (df["viscosity"] + eps)
        )

    # 弗劳德数 Fr = N^2 * D / g
    if {"speed_rps", "impeller_diameter"}.issubset(df.columns):
        g = 9.81
        df["Fr"] = (df["speed_rps"] ** 2) * df["impeller_diameter"] / g

    # 韦伯数 We = rho * N^2 * D^3 / sigma
    if {"density", "speed_rps", "impeller_diameter", "surface_tension"}.issubset(df.columns):
        df["We"] = (
            df["density"] * (df["speed_rps"] ** 2) * (df["impeller_diameter"] ** 3)
            / (df["surface_tension"] + eps)
        )

    return df