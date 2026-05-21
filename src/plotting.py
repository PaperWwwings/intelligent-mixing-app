import matplotlib
# 【关键修复】强制使用 'Agg' 后端，纯后台画图，不弹出GUI窗口，解决主线程报错
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from src.config import FIGURES_DIR
# 设置中文字体，防止图片中的中文变成方块（适配Windows/Mac）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
import numpy as np
import pandas as pd
from src.config import FIGURES_DIR

# 设置中文字体，防止图片中的中文变成方块
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def plot_feature_importance(model_pipeline, numeric_cols, cat_cols, title, filename):
    """
    画特征重要性条形图（论文中用于解释哪些物理量最重要）
    """
    # 提取特征名称
    preprocessor = model_pipeline.named_steps['preprocessor']
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_names = cat_encoder.get_feature_names_out(cat_cols)
    feature_names = numeric_cols + list(cat_names)
    
    # 获取随机森林的特征重要性
    importances = model_pipeline.named_steps['model'].feature_importances_
    
    # 组合并排序
    df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    df_imp = df_imp.sort_values(by='Importance', ascending=False).head(15) # 只取前15个重要的
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=df_imp, palette='viridis')
    plt.title(f"{title} - Top 15 特征重要性")
    plt.xlabel("重要性占比")
    plt.ylabel("特征")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300)  # 存为高清图用于论文
    plt.close()

def plot_actual_vs_predicted(y_true, y_pred, target_name, filename):
    """
    回归模型：画真实值与预测值的散点对比图（越靠近对角线越准）
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, color='royalblue')
    
    # 画一条 y=x 的标准线
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    plt.title(f"{target_name} - 真实值 vs 预测值")
    plt.xlabel("真实值")
    plt.ylabel("预测值")
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close()

def plot_confusion_matrix_custom(y_true, y_pred, classes, filename):
    """
    分类模型：画混淆矩阵热力图（对角线颜色越深越准）
    """
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title("搅拌器分类模型 - 混淆矩阵")
    plt.xlabel("预测搅拌器类型")
    plt.ylabel("真实搅拌器类型")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close()