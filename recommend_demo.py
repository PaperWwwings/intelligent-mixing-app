import pandas as pd
from src.recommender import MixingRecommender
from src.config import REPORTS_DIR


def main():
    user_case = {
        "density": 1050.0,
        "viscosity": 0.08,
        "surface_tension": 0.060,
        "tank_diameter": 1.2,
        "liquid_height": 1.3,
        "baffle_num": 4,
        "solid_fraction": 0.08,
        "process_goal": "suspension",
        "target_mixing_time_req": 100.0,
        "max_power_req": 1500.0,
        "operation_mode": "batch",
    }

    print("[INFO] AI 搅拌选型引擎正在进行智能寻优，请稍候...")
    recommender = MixingRecommender()
    results = recommender.recommend(user_case, top_k=3)

    output_rows = []

    print("\n" + "-" * 60)
    print(" 智能选型推荐方案计算结果")
    print("-" * 60)

    for i, r in enumerate(results, start=1):
        print(f"\n[方案 {i}] 综合推荐指数: {r['fitness_score']:.1f} / 100.0")
        print(f"  > 桨型: {r['impeller_type'].upper()} (匹配度: {r['classifier_probability']:.2%})")
        print(f"  > 桨径: {r['impeller_diameter']:.3f} m | 离底高度: {r['clearance']:.3f} m")
        print(f"  > 桨层数: {r['num_impellers']} 层 | 层间距: {r['impeller_spacing']:.3f} m")
        print(f"  > 转速: {r['speed_rpm']:.1f} RPM ({r['speed_rps']:.2f} rps)")
        print(f"  > 预测功耗: {r['pred_power']:.1f} W | 混合时间: {r['pred_mixing_time']:.1f} s | 悬浮评分: {r['pred_suspension_score']:.2f}")

        output_rows.append({
            "【输入】工艺目标": user_case["process_goal"],
            "【输入】流体密度(kg/m3)": user_case["density"],
            "【输入】流体黏度(Pa.s)": user_case["viscosity"],
            "【输入】反应釜内径(m)": user_case["tank_diameter"],
            "【输入】液位高度(m)": user_case["liquid_height"],
            "【输入】固含量": user_case["solid_fraction"],
            "【输入】期望混合时间上限(s)": user_case["target_mixing_time_req"],
            "【输入】设备功率上限(W)": user_case["max_power_req"],

            "【推荐】方案排名": i,
            "【推荐】综合推荐指数(0-100)": round(r["fitness_score"], 1),
            "【推荐】桨型": r["impeller_type"].upper(),
            "【推荐】分类匹配概率": f"{r['classifier_probability']*100:.1f}%",
            "【推荐】桨径 D(m)": round(r["impeller_diameter"], 3),
            "【推荐】离底高度 C(m)": round(r["clearance"], 3),
            "【推荐】桨层数": r["num_impellers"],
            "【推荐】层间距(m)": round(r["impeller_spacing"], 3),
            "【推荐】转速 N(RPM)": round(r["speed_rpm"], 1),

            "【预测】轴功率(W)": round(r["pred_power"], 1),
            "【预测】混合时间(s)": round(r["pred_mixing_time"], 1),
            "【预测】悬浮评分(0-1)": round(r["pred_suspension_score"], 3),
        })

    print("\n" + "-" * 60)

    df_output = pd.DataFrame(output_rows)
    base_name = "智能推荐方案表"
    ext = ".xlsx"
    counter = 1
    while True:
        excel_path = REPORTS_DIR / f"{base_name}_{counter}{ext}"
        if not excel_path.exists():
            break
        counter += 1

    df_output.to_excel(excel_path, index=False)
    print(f"[INFO] 详细推荐报告已成功导出至: {excel_path}\n")


if __name__ == "__main__":
    main()