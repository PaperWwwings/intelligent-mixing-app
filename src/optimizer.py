import numpy as np
import pandas as pd

from src.config import REGRESSOR_FEATURES
from src.preprocess import ensure_columns
from src.feature_engineering import add_features
from src.model_utils import predict_performance


class SimpleGAOptimizer:
    """
    智能搅拌遗传算法寻优器 (向量化加速)
    """

    def __init__(self, regressor_bundle, pop_size=40, generations=30,
                 mutation_rate=0.2, random_state=42):
        self.regressor_bundle = regressor_bundle
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.rng = np.random.default_rng(random_state)
        self.last_history = []

    def _get_ranges(self, impeller_type):
        if impeller_type == "propeller":
            return {"D_T": (0.15, 0.45), "speed_rps": (1.5, 8.0), "C_T": (0.10, 0.35), "num_impellers": (1.0, 3.0)}
        elif impeller_type == "pitched_blade":
            return {"D_T": (0.25, 0.50), "speed_rps": (1.0, 6.0), "C_T": (0.10, 0.35), "num_impellers": (1.0, 3.0)}
        elif impeller_type == "rushton":
            return {"D_T": (0.30, 0.45), "speed_rps": (1.5, 7.0), "C_T": (0.10, 0.30), "num_impellers": (1.0, 4.0)}
        elif impeller_type == "anchor":
            return {"D_T": (0.75, 0.95), "speed_rps": (0.05, 1.0), "C_T": (0.02, 0.10), "num_impellers": (1.0, 1.0)}
        else:
            return {"D_T": (0.25, 0.50), "speed_rps": (1.0, 6.0), "C_T": (0.10, 0.35), "num_impellers": (1.0, 3.0)}

    def _random_gene(self, impeller_type):
        ranges = self._get_ranges(impeller_type)
        return np.array([
            self.rng.uniform(*ranges["D_T"]),
            self.rng.uniform(*ranges["speed_rps"]),
            self.rng.uniform(*ranges["C_T"]),
            self.rng.uniform(*ranges["num_impellers"]),
        ])

    def _clip_gene(self, gene, impeller_type):
        ranges = self._get_ranges(impeller_type)
        gene[0] = np.clip(gene[0], *ranges["D_T"])
        gene[1] = np.clip(gene[1], *ranges["speed_rps"])
        gene[2] = np.clip(gene[2], *ranges["C_T"])
        gene[3] = np.clip(gene[3], *ranges["num_impellers"])
        return gene

    def _build_cases_batch(self, base_case, impeller_type, population):
        """【加速核心】将一整代群体（如40个）一次性打包为DataFrame"""
        tank_diameter = float(base_case["tank_diameter"])
        liquid_height = float(base_case["liquid_height"])
        
        # 复制出 40 份基准工况
        cases = [base_case.copy() for _ in range(len(population))]
        
        for i, gene in enumerate(population):
            cases[i]["impeller_type"] = impeller_type
            cases[i]["impeller_diameter"] = float(gene[0] * tank_diameter)
            cases[i]["speed_rps"] = float(gene[1])
            cases[i]["clearance"] = float(gene[2] * tank_diameter)
            
            num_impellers = max(1, min(int(round(gene[3])), 4))
            cases[i]["num_impellers"] = num_impellers
            
            if num_impellers > 1:
                available = liquid_height - cases[i]["clearance"]
                cases[i]["impeller_spacing"] = float(max(available / num_impellers, 0.01))
            else:
                cases[i]["impeller_spacing"] = 0.0
                
        return cases

    def _score_population_batch(self, base_case, impeller_type, population):
        """【加速核心】一次性对整代种群进行特征工程、预测与打分"""
        cases = self._build_cases_batch(base_case, impeller_type, population)
        df = pd.DataFrame(cases)
        
        # 批量特征工程
        df = ensure_columns(df, REGRESSOR_FEATURES)
        df = add_features(df)
        
        # === 修复核心开始：绕过只取单值的旧函数，直接执行批量矩阵预测 ===
        X = df[self.regressor_bundle["feature_cols"]]
        models = self.regressor_bundle["models"]
        
        # 一次性预测出 40 个结果的数组
        mixing_times = models["mixing_time"].predict(X)
        powers = models["power"].predict(X)
        suspension_scores = models["suspension_score"].predict(X)
        
        # 装入字典供最终提取
        preds = {
            "mixing_time": mixing_times,
            "power": powers,
            "suspension_score": suspension_scores
        }
        # === 修复核心结束 ===
        
        req_time = base_case.get("target_mixing_time_req", 120.0)
        req_power = base_case.get("max_power_req", 2000.0)
        is_suspension = (base_case.get("process_goal") == "suspension")

        # 批量计算惩罚与奖励 (利用 NumPy 矩阵运算)
        penalty = np.zeros(len(population))
        penalty += np.maximum(0, (mixing_times - req_time) / req_time) * 200.0
        penalty += np.maximum(0, (powers - req_power) / req_power) * 300.0
        if is_suspension:
            penalty += np.maximum(0, 0.8 - suspension_scores) * 500.0

        power_scores = np.maximum(0, 40.0 * (1.0 - (powers / req_power)))
        time_scores = np.maximum(0, 40.0 * (1.0 - (mixing_times / req_time)))
        suspension_bonuses = suspension_scores * 20.0

        final_scores = power_scores + time_scores + suspension_bonuses - penalty
        final_scores = np.clip(final_scores, 0.0, 100.0)
        costs = -final_scores  # GA 最小化代价
        
        return costs, preds, cases, final_scores

    def optimize_for_impeller(self, base_case, impeller_type):
        self.last_history = []
        population = [self._random_gene(impeller_type) for _ in range(self.pop_size)]

        for gen in range(self.generations):
            # 【一键批量打分】
            costs, _, _, _ = self._score_population_batch(base_case, impeller_type, population)
            
            # 组装排序
            scored = [(costs[i], population[i]) for i in range(self.pop_size)]
            scored.sort(key=lambda x: x[0])

            # 记录历史最优
            best_score_this_gen = -scored[0][0]
            self.last_history.append(best_score_this_gen)

            # 遗传繁衍...
            elite_num = max(2, self.pop_size // 4)
            elites = [item[1] for item in scored[:elite_num]]
            parent_pool = [item[1] for item in scored[:max(4, self.pop_size // 2)]]

            new_population = elites.copy()
            while len(new_population) < self.pop_size:
                p1 = parent_pool[self.rng.integers(0, len(parent_pool))]
                p2 = parent_pool[self.rng.integers(0, len(parent_pool))]
                alpha = self.rng.uniform(0, 1)
                child = alpha * p1 + (1 - alpha) * p2

                if self.rng.uniform(0, 1) < self.mutation_rate:
                    idx = self.rng.integers(0, 4)
                    ranges = self._get_ranges(impeller_type)
                    keys = ["D_T", "speed_rps", "C_T", "num_impellers"]
                    r = ranges[keys[idx]]
                    scale = 0.08 * (r[1] - r[0])
                    child[idx] += self.rng.normal(0, scale)

                child = self._clip_gene(child, impeller_type)
                new_population.append(child)
            population = new_population

        # 最后一代决出胜负
        costs, preds, cases, final_scores = self._score_population_batch(base_case, impeller_type, population)
        best_idx = np.argmin(costs)
        
        # 组装结果
        result = {
            "impeller_type": cases[best_idx]["impeller_type"],
            "impeller_diameter": cases[best_idx]["impeller_diameter"],
            "speed_rps": cases[best_idx]["speed_rps"],
            "speed_rpm": cases[best_idx]["speed_rps"] * 60.0,
            "clearance": cases[best_idx]["clearance"],
            "num_impellers": cases[best_idx]["num_impellers"],
            "impeller_spacing": cases[best_idx]["impeller_spacing"],
            "pred_mixing_time": preds["mixing_time"][best_idx],
            "pred_power": preds["power"][best_idx],
            "pred_suspension_score": preds["suspension_score"][best_idx],
            "fitness_score": final_scores[best_idx],
        }
        return result