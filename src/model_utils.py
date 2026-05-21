import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import mean_absolute_error, r2_score

# 导入项目配置
from src.config import (
    CLASSIFIER_FEATURES,
    CLASSIFIER_NUMERIC_FEATURES,
    CLASSIFIER_CATEGORICAL_FEATURES,
    REGRESSOR_FEATURES,
    REGRESSOR_NUMERIC_FEATURES,
    REGRESSOR_CATEGORICAL_FEATURES,
    REGRESSOR_TARGETS,
    CLASSIFIER_MODEL_PATH,
    REGRESSOR_MODEL_PATH,
)
from src.preprocess import ensure_columns

# 导入新增的画图工具函数
from src.plotting import (
    plot_feature_importance, 
    plot_actual_vs_predicted, 
    plot_confusion_matrix_custom
)


def build_preprocessor(numeric_cols, categorical_cols):
    """
    构造预处理器：
    - 数值列：遇到缺失值用中位数填补
    - 类别列：遇到缺失值用众数填补，然后做 OneHot (独热) 编码转换成数字
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    return preprocessor


def train_impeller_classifier(df):
    """
    训练搅拌器类型分类模型，并在训练完成后自动画图保存
    """
    print("\n=============================================")
    print("开始训练：搅拌器类型分类模型 (用于初筛候选桨型)...")
    print("=============================================")

    # 1. 提取需要的列并去掉目标值为空的脏数据
    df = ensure_columns(df, CLASSIFIER_FEATURES + ["impeller_type"])
    df = df.dropna(subset=["impeller_type"]).copy()

    X = df[CLASSIFIER_FEATURES]
    y = df["impeller_type"]

    # 2. 划分训练集和测试集 (80%用来训练，20%用来测试)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y, # 保证各种搅拌器类型的比例在训练和测试集里一致
    )

    # 3. 构造预处理器和随机森林分类器
    preprocessor = build_preprocessor(
        CLASSIFIER_NUMERIC_FEATURES,
        CLASSIFIER_CATEGORICAL_FEATURES,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=100,        # 树的数量减半，足够用了
                max_depth=15,            # 限制树的最大深度为15层
                min_samples_leaf=2,      # 限制叶子节点最小样本数
                random_state=42,
                n_jobs=-1
            )),
        ]
    )

    # 4. 开始训练并预测
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 5. 打印准确率报告
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ 分类模型准确率 Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("\n详细分类报告：")
    print(classification_report(y_test, y_pred))

    # ====================保存论文需要的图表 ====================
    print("正在生成分类模型图表...")
    
    # 提取有哪些搅拌器类型
    classes = model.named_steps['model'].classes_
    
    # 画混淆矩阵
    plot_confusion_matrix_custom(y_test, y_pred, classes, "classifier_confusion_matrix.png")
    
    # 画特征重要性图
    plot_feature_importance(
        model, 
        CLASSIFIER_NUMERIC_FEATURES, 
        CLASSIFIER_CATEGORICAL_FEATURES, 
        "搅拌器选型分类", 
        "classifier_feature_importance.png"
    )
    print("✅ 分类图表已保存至 results/figures/ 目录")
    # ===============================================================

    # 6. 保存模型到文件
    bundle = {
        "model": model,
        "feature_cols": CLASSIFIER_FEATURES,
    }

    joblib.dump(bundle, CLASSIFIER_MODEL_PATH, compress=3)
    print(f"✅ 分类模型已保存到：{CLASSIFIER_MODEL_PATH}")


def train_performance_regressors(df):
    """
    训练多个回归模型 (每个预测目标：功耗、混合时间等，训练一个专门的模型)
    """
    print("\n=============================================")
    print("开始训练：搅拌性能回归模型 (用于预测核心物理量)...")
    print("=============================================")

    models = {}

    for target in REGRESSOR_TARGETS:
        print(f"\n>> 正在训练目标物理量：【{target}】")

        # 1. 准备数据
        df_work = ensure_columns(df, REGRESSOR_FEATURES + [target])
        df_work = df_work.dropna(subset=[target]).copy()

        X = df_work[REGRESSOR_FEATURES]
        y = df_work[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )

        # 2. 构建管道
        preprocessor = build_preprocessor(
            REGRESSOR_NUMERIC_FEATURES,
            REGRESSOR_CATEGORICAL_FEATURES,
        )

        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", RandomForestRegressor(
                    n_estimators=100,        # 树的数量减半
                    max_depth=15,            # 限制树的深度，阻止无限细分
                    min_samples_leaf=2,      # 强行压缩模型体积的核心
                    random_state=42,
                    n_jobs=-1
                )),
            ]
        )

        # 3. 训练并预测
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 4. 计算指标: MAE(平均绝对误差)，R2(决定系数，越接近1越好)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"   [结果] 误差 MAE: {mae:.4f}, 拟合优度 R2: {r2:.4f}")

        # ==================== 保存回归散点图和特征重要性 ====================
        plot_actual_vs_predicted(
            y_test, 
            y_pred, 
            target, 
            f"regressor_{target}_actual_vs_pred.png"
        )
        
        plot_feature_importance(
            model, 
            REGRESSOR_NUMERIC_FEATURES, 
            REGRESSOR_CATEGORICAL_FEATURES, 
            f"{target} 预测", 
            f"regressor_{target}_feature_importance.png"
        )
        # =========================================================================

        # 5. 存入字典
        models[target] = model

    # 6. 打包保存所有回归模型
    bundle = {
        "models": models,
        "feature_cols": REGRESSOR_FEATURES,
        "target_cols": REGRESSOR_TARGETS,
    }

    joblib.dump(bundle, REGRESSOR_MODEL_PATH, compress=3)
    print("\n✅ 所有图表已保存至 results/figures/ 目录")
    print(f"✅ 回归模型已打包保存到：{REGRESSOR_MODEL_PATH}")


def predict_performance(regressor_bundle, df_one_row):
    """
    提供给遗传算法调用的快速预测接口
    用打包好的所有回归模型，一次性预测出一组参数对应的所有物理性能
    """
    result = {}

    feature_cols = regressor_bundle["feature_cols"]
    models = regressor_bundle["models"]

    X = df_one_row[feature_cols]

    for target, model in models.items():
        pred_value = model.predict(X)[0]
        result[target] = float(pred_value)

    return result