"""
参数敏感性分析
四组实验，用于验证智能搅拌选型系统的工程合理性。
"""

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import FIGURES_DIR, ensure_dirs
from src.recommender import MixingRecommender

# 中文
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams["axes.unicode_minus"] = False


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
    """
    固定其他参数，将黏度从 0.001 Pa.s 变化到 50 Pa.s，
    观察分类模型推荐的桨型如何从低粘度桨过渡到高粘度桨。
    """
    print("\n[实验 1] 黏度对推荐桨型的影响...")

    viscosities = np.logspace(-3, 1.7, 30)  # 0.001 ~ 50 Pa.s
    results = []

    for mu in viscosities:
        case = get_base_case()
        case["viscosity"] = float(mu)
        # 高粘度时自动切到 mixing 目标
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

    # 为桨型分配颜色和标记
    type_color = {
        "propeller": "royalblue",
        "pitched_blade": "forestgreen",
        "rushton": "darkorange",
        "anchor": "crimson",
    }
    type_label = {
        "propeller": "Propeller (推进式)",
        "pitched_blade": "Pitched Blade (斜叶桨)",
        "rushton": "Rushton (涡轮式)",
        "anchor": "Anchor (锚式)",
    }

    # --- 图 1a：黏度 vs 推荐桨型 ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    for itype in df["impeller_type"].unique():
        mask = df["impeller_type"] == itype
        ax.scatter(
            df.loc[mask, "viscosity"],
            [1] * mask.sum(),  # 占位 y 值
            c=type_color.get(itype, "gray"),
            label=type_label.get(itype, itype),
            s=100, edgecolors="black", linewidth=0.5, zorder=5
        )
    ax.set_xscale("log")
    ax.set_xlabel("Viscosity (Pa.s)")
    ax.set_ylabel("Recommended Type")
    ax.set_title("(a) Viscosity vs Recommended Impeller Type")
    ax.set_yticks([])
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.5)

    # --- 图 1b：黏度 vs 推荐转速 ---
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
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.5)

    # --- 图 1c：黏度 vs 预测功耗 ---
    ax = axes[1, 0]
    ax.plot(df["viscosity"], df["pred_power"], "o-", color="darkorange", markersize=4)
    ax.set_xscale("log")
    ax.set_xlabel("Viscosity (Pa.s)")
    ax.set_ylabel("Predicted Power (W)")
    ax.set_title("(c) Viscosity vs Predicted Power")
    ax.grid(True, linestyle=":", alpha=0.5)

    # --- 图 1d：黏度 vs 推荐层数 ---
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
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    path = FIGURES_DIR / "sensitivity_exp1_viscosity.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  [INFO] 图表已保存: {path}")


# =====================================================================
# 实验 2：遗传算法收敛曲线
# =====================================================================
def experiment_ga_convergence(recommender):
    """
    给定一个明确的工况，运行遗传算法，
    记录每一代的最优适应度得分，绘制收敛曲线。
    """
    print("\n[实验 2] 遗传算法收敛曲线...")

    case = get_base_case()
    case["process_goal"] = "suspension"
    case["solid_fraction"] = 0.10
    case["target_mixing_time_req"] = 80.0

    recs = recommender.recommend(case, top_k=3)

    # 获取每种候选桨型的收敛历史
    # 由于 recommender 内部对每个桨型调用一次 GA，
    # 手动调用 optimizer 
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
            label=f"{itype.upper()} (p={prob:.2f})",
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
    print(f"  [INFO] 图表已保存: {path}")


# =====================================================================
# 实验 3：固含量对悬浮工艺的影响
# =====================================================================

def experiment_solid_fraction(recommender):
    """
    固定悬浮工艺，将固含量从 0.01 变化到 0.30，
    观察推荐转速、悬浮评分、功耗和层数的变化趋势。
    """
    print("\n[实验 3] 固含量对悬浮工艺的影响...")
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
            "fitness_score": r["fitness_score"],
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
    print(f"  [INFO] 图表已保存: {path}")

# =====================================================================
# 实验 4：釜径放大效应
# =====================================================================

def experiment_scale_up(recommender):
    """
    固定物性条件，将釜径从 0.5m 放大到 3.0m，
    观察推荐桨径、转速、功耗和层数如何随放大而变化。
    """
    print("\n[实验 4] 釜径放大效应...")
    diameters = np.linspace(0.5, 3.0, 20)
    results = []
    for T in diameters:
        case = get_base_case()
        case["tank_diameter"] = float(T)
        case["liquid_height"] = float(T * 1.1)  # 保持 H/T 不变
        case["target_mixing_time_req"] = 120.0
        case["max_power_req"] = 5000.0  # 放大后允许更大功率
        recs = recommender.recommend(case, top_k=1)
        r = recs[0]
        results.append({
            "tank_diameter": T,
            "impeller_diameter": r["impeller_diameter"],
            "speed_rpm": r["speed_rpm"],
            "num_impellers": r["num_impellers"],
            "pred_power": r["pred_power"],
            "pred_mixing_time": r["pred_mixing_time"],
            "fitness_score": r["fitness_score"],
        })
    df = pd.DataFrame(results)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    ax.plot(df["tank_diameter"], df["impeller_diameter"], "o-", color="royalblue", markersize=4)
    ax.set_xlabel("Tank Diameter (m)")
    ax.set_ylabel("Recommended Impeller Diameter (m)")
    ax.set_title("(a) Scale-up: Tank Dia. vs Impeller Dia.")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax = axes[0, 1]
    ax.plot(df["tank_diameter"], df["speed_rpm"], "s-", color="forestgreen", markersize=4)
    ax.set_xlabel("Tank Diameter (m)")
    ax.set_ylabel("Recommended Speed (RPM)")
    ax.set_title("(b) Scale-up: Tank Dia. vs Speed")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax = axes[1, 0]
    ax.plot(df["tank_diameter"], df["pred_power"], "^-", color="darkorange", markersize=4)
    ax.set_xlabel("Tank Diameter (m)")
    ax.set_ylabel("Predicted Power (W)")
    ax.set_title("(c) Scale-up: Tank Dia. vs Power")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax = axes[1, 1]
    ax.plot(df["tank_diameter"], df["num_impellers"], "D-", color="crimson", markersize=4)
    ax.set_xlabel("Tank Diameter (m)")
    ax.set_ylabel("Number of Impellers")
    ax.set_title("(d) Scale-up: Tank Dia. vs Impeller Layers")
    ax.set_yticks([1, 2, 3, 4])
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    path = FIGURES_DIR / "sensitivity_exp4_scale_up.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  [INFO] 图表已保存: {path}")

# =====================================================================
# 主函数
# =====================================================================
def main():
    ensure_dirs()

    print("=" * 60)
    print(" 参数敏感性分析")
    print(" 共 4 组控制变量实验")
    print("=" * 60)

    print("\n[INFO] 正在加载已训练的模型...")
    recommender = MixingRecommender()
    print("[INFO] 模型加载完毕\n")

    experiment_viscosity(recommender)
    experiment_ga_convergence(recommender)
    experiment_solid_fraction(recommender)
    experiment_scale_up(recommender)

    print("\n" + "=" * 60)
    print(" 全部实验完成")
    print(f" 图表已保存至: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()