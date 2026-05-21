"""
参数敏感性分析脚本 (最终修正版)
包含四组实验，用于验证智能搅拌选型系统的工程合理性。
修复了负号字体显示问题，并修正了实验四中的功率放大约束。
"""

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import FIGURES_DIR, ensure_dirs
from src.recommender import MixingRecommender

# =====================================================================
# 全局图表设置 (修复负号与中文乱码)
# =====================================================================
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False  # 关键修复：正常显示负号


def get_base_case():
    """返回基准工况，后续各实验在此基础上做控制变量"""
    return {
        "density": 1050.0,
        "viscosity": 0.05,
        "surface_tension": 0.060,
        "tank_diameter": 1.2,
        "liquid_height": 1.3,
        "baffle_num": 4,
        "solid_fraction": 0.05,
        "process_goal": "mixing",
        "target_mixing_time_req": 120.0,
        "max_power_req": 2000.0,
        "operation_mode": "batch",
    }


# =====================================================================
# 实验 1：黏度对推荐桨型的影响
# =====================================================================
def experiment_viscosity(recommender):
    print("\n[INFO] 开始运行实验 1：黏度对推荐桨型的影响...")

    viscosities = np.logspace(-3, 1.7, 30)  # 0.001 ~ 50 Pa.s
    results = []

    for mu in viscosities:
        case = get_base_case()
        case["viscosity"] = float(mu)
        case["process_goal"] = "mixing"
        
        recs = recommender.recommend(case, top_k=1)
        r = recs[0]
        results.append({
            "viscosity": mu,
            "impeller_type": r["impeller_type"],
            "speed_rpm": r["speed_rpm"],
            "num_impellers": r["num_impellers"],
            "pred_power": r["pred_power"],
            "pred_mixing_time": r["pred_mixing_time"],
            "fitness_score": r["fitness_score"],
        })

    df = pd.DataFrame(results)

    type_color = {
        "propeller": "royalblue",
        "pitched_blade": "forestgreen",
        "rushton": "darkorange",
        "anchor": "crimson",
    }
    type_label = {
        "propeller": "Propeller",
        "pitched_blade": "Pitched Blade",
        "rushton": "Rushton",
        "anchor": "Anchor",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 图 1a
    ax = axes[0, 0]
    for itype in df["impeller_type"].unique():
        mask = df["impeller_type"] == itype
        ax.scatter(
            df.loc[mask, "viscosity"],
            [1] * mask.sum(),
            c=type_color.get(itype, "gray"),
            label=type_label.get(itype, itype),
            s=100, edgecolors="black", linewidth=0.5, zorder=5
        )
    ax.set_xscale("log")
    ax.set_xlabel("Viscosity (Pa.s)")
    ax.set_ylabel("Recommended Type")
    ax.set_title("(a) Viscosity vs Recommended Impeller Type")
    ax.set_yticks([])
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)

    # 图 1b
    ax = axes[0, 1]
    for itype in df["impeller_type"].unique():
        mask = df["impeller_type"] == itype
        ax.scatter(
            df.loc[mask, "viscosity"],
            df.loc[mask, "speed_rpm"],
            c=type_color.get(itype, "gray"),
            label=type_label.get(itype, itype),
            s=60, edgecolors="black", linewidth=0.5
        )
    ax.set_xscale("log")
    ax.set_xlabel("Viscosity (Pa.s)")
    ax.set_ylabel("Speed (RPM)")
    ax.set_title("(b) Viscosity vs Recommended Speed")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)

    # 图 1c
    ax = axes[1, 0]
    ax.plot(df["viscosity"], df["pred_power"], "o-", color="darkorange", markersize=4)
    ax.set_xscale("log")
    ax.set_xlabel("Viscosity (Pa.s)")
    ax.set_ylabel("Predicted Power (W)")
    ax.set_title("(c) Viscosity vs Predicted Power")
    ax.grid(True, linestyle=":", alpha=0.5)

    # 图 1d
    ax = axes[1, 1]
    for itype in df["impeller_type"].unique():
        mask = df["impeller_type"] == itype
        ax.scatter(
            df.loc[mask, "viscosity"],
            df.loc[mask, "num_impellers"],
            c=type_color.get(itype, "gray"),
            label=type_label.get(itype, itype),
            s=60, edgecolors="black", linewidth=0.5
        )
    ax.set_xscale("log")
    ax.set_xlabel("Viscosity (Pa.s)")
    ax.set_ylabel("Number of Impellers")
    ax.set_title("(d) Viscosity vs Recommended Impeller Layers")
    ax.set_yticks([1, 2, 3, 4])
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    path = FIGURES_DIR / "sensitivity_exp1_viscosity.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  [SUCCESS] 图表已保存: {path}")


# =====================================================================
# 实验 2：遗传算法收敛曲线
# =====================================================================
def experiment_ga_convergence(recommender):
    print("\n[INFO] 开始运行实验 2：遗传算法收敛曲线...")

    case = get_base_case()
    case["process_goal"] = "suspension"
    case["solid_fraction"] = 0.10
    case["target_mixing_time_req"] = 80.0

    optimizer = recommender.optimizer

    fig, ax = plt.subplots(figsize=(8, 5))

    from src.recommender import fill_user_defaults, predict_topk_impellers
    from src.preprocess import ensure_columns
    from src.feature_engineering import add_features
    from src.config import CLASSIFIER_FEATURES

    user_case = fill_user_defaults(case)
    df_input = pd.DataFrame([user_case])
    df_input = ensure_columns(df_input, CLASSIFIER_FEATURES)
    df_input = add_features(df_input)

    classifier_bundle = recommender.classifier_bundle
    candidates = predict_topk_impellers(classifier_bundle, df_input, k=3)

    colors = ["royalblue", "forestgreen", "darkorange", "crimson"]

    for idx, (itype, prob) in enumerate(candidates):
        optimizer.optimize_for_impeller(user_case, itype)
        history = optimizer.last_history

        generations = list(range(1, len(history) + 1))
        ax.plot(
            generations, history,
            "o-", color=colors[idx % len(colors)],
            label=f"{itype.upper()} (Prob={prob:.2f})",
            markersize=4
        )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness Score (0-100)")
    ax.set_title("GA Convergence Curve")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    path = FIGURES_DIR / "sensitivity_exp2_ga_convergence.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  [SUCCESS] 图表已保存: {path}")


# =====================================================================
# 实验 3：固含量对悬浮工艺的影响
# =====================================================================
def experiment_solid_fraction(recommender):
    print("\n[INFO] 开始运行实验 3：固含量对悬浮工艺的影响...")

    fractions = np.linspace(0.01, 0.30, 20)
    results = []

    for sf in fractions:
        case = get_base_case()
        case["process_goal"] = "suspension"
        case["solid_fraction"] = float(sf)
        case["target_mixing_time_req"] = 100.0

        recs = recommender.recommend(case, top_k=1)
        r = recs[0]
        results.append({
            "solid_fraction": sf,
            "speed_rpm": r["speed_rpm"],
            "num_impellers": r["num_impellers"],
            "pred_power": r["pred_power"],
            "pred_suspension_score": r["pred_suspension_score"],
        })

    df = pd.DataFrame(results)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(df["solid_fraction"], df["speed_rpm"], "o-", color="royalblue", markersize=4)
    ax.set_xlabel("Solid Fraction")
    ax.set_ylabel("Recommended Speed (RPM)")
    ax.set_title("(a) Solid Fraction vs Speed")
    ax.grid(True, linestyle=":", alpha=0.5)

    ax = axes[0, 1]
    ax.plot(df["solid_fraction"], df["pred_suspension_score"], "s-", color="forestgreen", markersize=4)
    ax.set_xlabel("Solid Fraction")
    ax.set_ylabel("Predicted Suspension Score")
    ax.set_title("(b) Solid Fraction vs Suspension Score")
    ax.grid(True, linestyle=":", alpha=0.5)

    ax = axes[1, 0]
    ax.plot(df["solid_fraction"], df["pred_power"], "^-", color="darkorange", markersize=4)
    ax.set_xlabel("Solid Fraction")
    ax.set_ylabel("Predicted Power (W)")
    ax.set_title("(c) Solid Fraction vs Power")
    ax.grid(True, linestyle=":", alpha=0.5)

    ax = axes[1, 1]
    ax.plot(df["solid_fraction"], df["num_impellers"], "D-", color="crimson", markersize=4)
    ax.set_xlabel("Solid Fraction")
    ax.set_ylabel("Number of Impellers")
    ax.set_title("(d) Solid Fraction vs Impeller Layers")
    ax.set_yticks([1, 2, 3, 4])
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    path = FIGURES_DIR / "sensitivity_exp3_solid_fraction.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  [SUCCESS] 图表已保存: {path}")


# =====================================================================
# 实验 4：釜径放大效应 (已应用体积放大功率约束修复)
# =====================================================================
def experiment_scale_up(recommender):
    print("\n[INFO] 开始运行实验 4：釜径放大效应...")

    diameters = np.linspace(0.5, 3.0, 20)
    results = []

    for T in diameters:
        case = get_base_case()
        case["tank_diameter"] = float(T)
        case["liquid_height"] = float(T * 1.1)  
        case["target_mixing_time_req"] = 120.0
        
        # 【关键修复】：等体积功率放大准则，允许大釜具备更大的功率约束上限
        case["max_power_req"] = 2000.0 * ((T / 1.2) ** 3)

        recs = recommender.recommend(case, top_k=1)
        r = recs[0]
        results.append({
            "tank_diameter": T,
            "impeller_diameter": r["impeller_diameter"],
            "speed_rpm": r["speed_rpm"],
            "num_impellers": r["num_impellers"],
            "pred_power": r["pred_power"],
        })

    df = pd.DataFrame(results)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(df["tank_diameter"], df["impeller_diameter"], "o-", color="royalblue", markersize=5)
    ax.set_xlabel("Tank Diameter (m)")
    ax.set_ylabel("Recommended Impeller Dia. (m)")
    ax.set_title("(a) Scale-up: Tank Dia. vs Impeller Dia.")
    ax.grid(True, linestyle=":", alpha=0.5)

    ax = axes[0, 1]
    ax.plot(df["tank_diameter"], df["speed_rpm"], "s-", color="forestgreen", markersize=5)
    ax.set_xlabel("Tank Diameter (m)")
    ax.set_ylabel("Recommended Speed (RPM)")
    ax.set_title("(b) Scale-up: Tank Dia. vs Speed")
    ax.grid(True, linestyle=":", alpha=0.5)

    ax = axes[1, 0]
    ax.plot(df["tank_diameter"], df["pred_power"], "^-", color="darkorange", markersize=5)
    ax.set_xlabel("Tank Diameter (m)")
    ax.set_ylabel("Predicted Power (W)")
    ax.set_title("(c) Scale-up: Tank Dia. vs Power")
    ax.grid(True, linestyle=":", alpha=0.5)

    ax = axes[1, 1]
    ax.plot(df["tank_diameter"], df["num_impellers"], "D-", color="crimson", markersize=5)
    ax.set_xlabel("Tank Diameter (m)")
    ax.set_ylabel("Number of Impellers")
    ax.set_title("(d) Scale-up: Tank Dia. vs Impeller Layers")
    ax.set_yticks([1, 2, 3, 4])
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    path = FIGURES_DIR / "sensitivity_exp4_scale_up.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  [SUCCESS] 图表已保存: {path}")


# =====================================================================
# 主运行入口
# =====================================================================
def main():
    ensure_dirs()

    print("-" * 50)
    print(" 智能搅拌选型系统 - 参数敏感性分析")
    print("-" * 50)

    print("[INFO] 正在加载已训练的机器学习模型与寻优引擎...")
    recommender = MixingRecommender()
    print("[SUCCESS] 模型加载完毕\n")

    # 依次运行所有实验
    experiment_viscosity(recommender)
    experiment_ga_convergence(recommender)
    experiment_solid_fraction(recommender)
    experiment_scale_up(recommender)

    print("\n" + "-" * 50)
    print(f"[SUCCESS] 全部实验完成！所有高清图表已保存至: {FIGURES_DIR}")
    print("-" * 50)


if __name__ == "__main__":
    main()