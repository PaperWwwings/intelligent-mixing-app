import numpy as np
import pandas as pd
from src.config import RAW_DATA_PATH, ensure_dirs


def generate_scientific_mock_data(n_samples=2500, random_state=42):
    """
    基于真实搅拌选型工程经验数据生成科学仿真数据集。
    多层桨的支持：
    - 不同桨型和工况对应不同的典型层数
    - 多层桨的功率采用层数修正系数
    - 多层桨的混合时间采用幂律衰减模型
    - 多层桨对固液悬浮有增益效果
    """
    rng = np.random.default_rng(random_state)
    rows = []

    process_goals = ["mixing_low_visc", "mixing_high_visc", "dispersion", "suspension"]

    for _ in range(n_samples):
        # 1. 随机选择工艺目标
        goal = rng.choice(process_goals, p=[0.3, 0.2, 0.25, 0.25])

        # 2. 生成流体物性
        density = rng.uniform(900, 1300)
        surface_tension = rng.uniform(0.040, 0.072)

        if goal == "mixing_high_visc":
            viscosity = rng.uniform(1.0, 50.0)
            solid_fraction = 0.0
        elif goal == "suspension":
            viscosity = rng.uniform(0.001, 0.05)
            solid_fraction = rng.uniform(0.05, 0.30)
        else:
            viscosity = rng.uniform(0.001, 0.1)
            solid_fraction = 0.0 if goal != "dispersion" else rng.uniform(0.0, 0.05)

        # 3. 反应釜几何尺寸
        tank_diameter = rng.uniform(0.5, 3.0)
        liquid_height = rng.uniform(0.9, 1.5) * tank_diameter
        H_T = liquid_height / tank_diameter
        baffle_num = 4 if viscosity < 5.0 else 0

        # =====================================================================
        # 4. 桨型选择、D/T比、转速、桨层数
        # =====================================================================
        if goal == "mixing_low_visc":
            impeller_type = rng.choice(["propeller", "pitched_blade"], p=[0.6, 0.4])
            if impeller_type == "propeller":
                D_T = rng.uniform(0.15, 0.30)
                rpm = rng.uniform(150, 300)
            else:
                D_T = rng.uniform(0.30, 0.40)
                rpm = rng.uniform(100, 155)
            # 低粘度混合：液位较高时倾向于多层
            if H_T > 1.2:
                num_impellers = rng.choice([1, 2, 3], p=[0.2, 0.5, 0.3])
            else:
                num_impellers = rng.choice([1, 2], p=[0.6, 0.4])

        elif goal == "mixing_high_visc":
            impeller_type = "anchor"
            D_T = rng.uniform(0.85, 0.95)
            rpm = rng.uniform(10, 50)
            # 锚式桨结构特殊，工业上几乎都是单层
            num_impellers = 1

        elif goal == "dispersion":
            impeller_type = "rushton"
            D_T = rng.uniform(0.30, 0.40)
            rpm = rng.uniform(115, 250)
            # 分散体系：液位高时常用多层涡轮
            if H_T > 1.1:
                num_impellers = rng.choice([1, 2, 3, 4], p=[0.15, 0.35, 0.35, 0.15])
            else:
                num_impellers = rng.choice([1, 2, 3], p=[0.3, 0.4, 0.3])

        elif goal == "suspension":
            impeller_type = rng.choice(["pitched_blade", "propeller"], p=[0.7, 0.3])
            D_T = rng.uniform(0.25, 0.45)
            rpm = rng.uniform(150, 300)
            # 悬浮体系：多层桨有助于全釜悬浮
            if H_T > 1.2:
                num_impellers = rng.choice([1, 2, 3], p=[0.15, 0.45, 0.40])
            else:
                num_impellers = rng.choice([1, 2], p=[0.4, 0.6])

        # 换算物理量
        impeller_diameter = D_T * tank_diameter
        speed_rps = rpm / 60.0
        if impeller_type == "anchor":
            clearance = 0.05 * tank_diameter
        else:
            clearance = rng.uniform(0.15, 0.30) * tank_diameter

        # 层间距计算：多层桨等间距布置
        if num_impellers > 1:
            available_height = liquid_height - clearance
            impeller_spacing = available_height / num_impellers
            # 加入少量随机扰动，模拟非完全等间距
            impeller_spacing *= rng.uniform(0.90, 1.10)
        else:
            impeller_spacing = 0.0

        # =====================================================================
        # 5. 物理量科学计算
        # =====================================================================

        # (1) 单层桨功率 P_single = Np * rho * N^3 * D^5
        Np_map = {"propeller": 0.5, "pitched_blade": 1.3, "rushton": 5.0, "anchor": 0.5}
        Np = Np_map[impeller_type]
        Re = density * speed_rps * (impeller_diameter ** 2) / viscosity
        if Re < 1000:
            Np = Np + (300.0 / (Re + 1e-3))

        power_single = Np * density * (speed_rps ** 3) * (impeller_diameter ** 5)

        # 多层桨功率修正：P_total = P_single * num * interaction_factor
        # 层间相互干涉导致每层效率略降（约 0.85~0.95）
        interaction_factor = rng.uniform(0.85, 0.95)
        power = power_single * num_impellers * interaction_factor
        power *= (1 + 0.5 * solid_fraction)
        power = power * rng.normal(1.0, 0.05)

        # (2) 混合时间
        N_tm_const = {"propeller": 30, "pitched_blade": 35, "rushton": 50, "anchor": 80}
        base_tm = N_tm_const[impeller_type] / (speed_rps + 1e-5)
        mixing_time_single = base_tm * ((viscosity / 0.001) ** 0.1) * (tank_diameter / 1.0)

        # 多层桨混合时间修正：t_multi = t_single / (num ^ alpha)
        # alpha 约 0.3~0.5（文献经验值），多层桨显著缩短混合时间
        mixing_time = mixing_time_single / (num_impellers ** 0.4)
        mixing_time = mixing_time * rng.normal(1.0, 0.08)

        # (3) 悬浮评分
        if goal == "suspension":
            suspension_index = (speed_rps ** 2) * impeller_diameter / (tank_diameter + 1e-3)
            suspension_score = 1.0 - np.exp(-1.5 * suspension_index)
            # 多层桨对悬浮的增益
            suspension_score *= (1.0 + 0.12 * (num_impellers - 1))
        else:
            suspension_score = 0.95 if viscosity < 1.0 else 0.5

        suspension_score = np.clip(suspension_score + rng.normal(0, 0.02), 0.0, 1.0)

        # 6. 工艺要求约束
        target_mixing_time_req = mixing_time * rng.uniform(0.8, 1.5)
        max_power_req = power * rng.uniform(1.1, 2.0)
        operation_mode = rng.choice(["batch", "continuous"], p=[0.8, 0.2])

        if goal in ("mixing_low_visc", "mixing_high_visc"):
            out_goal = "mixing"
        else:
            out_goal = goal

        row = {
            "density": float(density),
            "viscosity": float(viscosity),
            "surface_tension": float(surface_tension),
            "tank_diameter": float(tank_diameter),
            "liquid_height": float(liquid_height),
            "baffle_num": int(baffle_num),
            "solid_fraction": float(solid_fraction),
            "process_goal": out_goal,
            "target_mixing_time_req": float(target_mixing_time_req),
            "max_power_req": float(max_power_req),
            "operation_mode": operation_mode,
            "impeller_type": impeller_type,
            "impeller_diameter": float(impeller_diameter),
            "speed_rps": float(speed_rps),
            "clearance": float(clearance),
            "num_impellers": int(num_impellers),
            "impeller_spacing": float(impeller_spacing),
            "mixing_time": float(max(mixing_time, 1.0)),
            "power": float(max(power, 1.0)),
            "suspension_score": float(suspension_score),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    ensure_dirs()
    print("[INFO] 正在基于工程学定律生成含多层桨的科学流体搅拌数据...")
    df = generate_scientific_mock_data(n_samples=2500, random_state=42)

    df.to_csv(RAW_DATA_PATH, index=False, encoding="utf-8-sig")
    print(f"[INFO] 生成完毕, 共 {len(df)} 条数据, 已保存到: {RAW_DATA_PATH}")

    print("\n--- 数据预览 ---")
    print(df[['process_goal', 'impeller_type', 'viscosity', 'speed_rps', 'num_impellers']].head(10))

    print("\n--- 各桨型的平均层数 ---")
    print(df.groupby('impeller_type')['num_impellers'].mean().round(2))

    print("\n--- 各桨型的 P/V (kW/m3) ---")
    df['P_V'] = (df['power'] / 1000) / ((np.pi / 4) * df['tank_diameter'] ** 2 * df['liquid_height'])
    print(df.groupby('impeller_type')['P_V'].mean().round(2))


if __name__ == "__main__":
    main()