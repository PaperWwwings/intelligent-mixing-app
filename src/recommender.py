import joblib
import numpy as np
import pandas as pd

from src.config import (
    CLASSIFIER_MODEL_PATH,
    REGRESSOR_MODEL_PATH,
    CLASSIFIER_FEATURES,
)
from src.preprocess import ensure_columns
from src.feature_engineering import add_features
from src.optimizer import SimpleGAOptimizer


def fill_user_defaults(user_case):
    """
    给用户输入补默认值，防止用户少输参数导致报错
    """
    default_case = {
        "density": 1000.0,
        "viscosity": 0.01,
        "surface_tension": 0.072,
        "tank_diameter": 1.0,
        "liquid_height": 1.0,
        "baffle_num": 4,
        "solid_fraction": 0.0,
        "process_goal": "mixing",
        "target_mixing_time_req": 120.0,
        "max_power_req": 2000.0,
        "operation_mode": "batch",
    }

    new_case = default_case.copy()
    new_case.update(user_case)
    return new_case


def predict_topk_impellers(classifier_bundle, df_one_row, k=3):
    """
    预测前 k 个最可能的搅拌器类型
    """
    model = classifier_bundle["model"]
    feature_cols = classifier_bundle["feature_cols"]

    X = df_one_row[feature_cols]
    probs = model.predict_proba(X)[0]

    # 从 pipeline 最后一步里取类别名称
    classes = model.named_steps["model"].classes_

    k = min(k, len(classes))
    idx = np.argsort(probs)[::-1][:k]

    results = []
    for i in idx:
        results.append((classes[i], float(probs[i])))

    return results


class MixingRecommender:
    """
    智能搅拌推荐器主类
    """

    def __init__(self):
        self.classifier_bundle = joblib.load(CLASSIFIER_MODEL_PATH)
        self.regressor_bundle = joblib.load(REGRESSOR_MODEL_PATH)
        self.optimizer = SimpleGAOptimizer(self.regressor_bundle)

    def recommend(self, user_case, top_k=3):
        """
        主推荐函数
        输入：用户工况
        输出：智能推荐方案
        """
        user_case = fill_user_defaults(user_case)

        # 先做分类初筛：推荐合适的搅拌器类型
        df = pd.DataFrame([user_case])
        df = ensure_columns(df, CLASSIFIER_FEATURES)
        df = add_features(df)

        candidate_impellers = predict_topk_impellers(
            self.classifier_bundle,
            df,
            k=top_k
        )

        # 针对每种候选搅拌器做遗传算法参数寻优
        all_results = []
        for impeller_type, prob in candidate_impellers:
            best_solution = self.optimizer.optimize_for_impeller(user_case, impeller_type)
            best_solution["classifier_probability"] = prob
            all_results.append(best_solution)

        # === 核心修改在这里：改用 fitness_score 排序，且分数越高越好 (reverse=True) ===
        all_results = sorted(all_results, key=lambda x: x["fitness_score"], reverse=True)

        # 只返回前 Top K 个最佳方案
        return all_results[:top_k]