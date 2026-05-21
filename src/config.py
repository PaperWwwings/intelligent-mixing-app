from pathlib import Path

# ==================== 路径配置 ====================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORTS_DIR = RESULTS_DIR / "reports"

RAW_DATA_PATH = RAW_DIR / "mock_mixing_data.csv"
CLASSIFIER_MODEL_PATH = MODEL_DIR / "impeller_classifier.joblib"
REGRESSOR_MODEL_PATH = MODEL_DIR / "performance_regressors.joblib"

# ==================== 原始数据字段 ====================
REQUIRED_RAW_COLUMNS = [
    "density",
    "viscosity",
    "surface_tension",
    "tank_diameter",
    "liquid_height",
    "baffle_num",
    "solid_fraction",
    "process_goal",
    "operation_mode",
    "impeller_type",
    "impeller_diameter",
    "speed_rps",
    "clearance",
    "num_impellers",
    "impeller_spacing",
    "mixing_time",
    "power",
    "suspension_score",
]

# ==================== 分类模型特征 ====================
# 设计原则：
# 只保留"用户能够告知系统"的物性与工艺描述信息。
# 工艺约束字段（如 max_power_req）属于"期望边界"而非"物理描述"，
# 放入分类器特征会引入数据泄露，已被剔除。
CLASSIFIER_FEATURES = [
    "density",
    "viscosity",
    "surface_tension",
    "tank_diameter",
    "liquid_height",
    "baffle_num",
    "solid_fraction",
    "process_goal",
    "operation_mode",
    # 衍生的无量纲几何特征：
    # 让分类器感知"设备的形状"而非绝对尺寸，
    # 有助于模型在不同尺度的反应釜之间泛化
    "H_T",
]

CLASSIFIER_NUMERIC_FEATURES = [
    "density",
    "viscosity",
    "surface_tension",
    "tank_diameter",
    "liquid_height",
    "baffle_num",
    "solid_fraction",
    "H_T",
]

CLASSIFIER_CATEGORICAL_FEATURES = [
    "process_goal",
    "operation_mode",
]

# ==================== 回归模型特征 ====================
# 设计原则：
# 回归模型的任务是"给定一个完整的搅拌方案，预测其物理性能"。
# 因此所有输入均为可观测、可计算的物理量或几何参数。
# 严格剔除了以下两类"污染源"：
# (1) 工艺约束字段：max_power_req, target_mixing_time_req
#     这些字段在数据生成时与标签存在比例关系（data leakage），
#     会使模型通过"逆推约束"而非"学习物理规律"来预测目标，
#     极大损害模型的泛化能力与可解释性。
# (2) 冗余的绝对尺寸字段：
#     在引入了无量纲特征（如D_T, Re）后，绝对尺寸（如tank_diameter）
#     对模型的信息贡献度大幅降低，可适当精简以降低维度诅咒的风险。
REGRESSOR_FEATURES = [
    # 流体物性（决定阻力与流态的根本原因）
    "density",
    "viscosity",
    "surface_tension",
    "solid_fraction",
    # 设备几何（保留釜径以供无量纲数计算兜底）
    "tank_diameter",
    "liquid_height",
    "baffle_num",
    # 类别特征
    "process_goal",
    "operation_mode",
    "impeller_type",
    # 搅拌器操作参数（直接决定流场的可控变量）
    "impeller_diameter",
    "speed_rps",
    "clearance",
    "num_impellers",
    "impeller_spacing",
    # 无量纲几何比（屏蔽绝对尺寸，提升跨尺度泛化能力）
    "D_T",
    "H_T",
    "C_T",
    "S_T",
    # 无量纲动力学数（核心流态判据，是模型学习物理机理的关键桥梁）
    # Re = rho*N*D^2/mu：判断层流/湍流的决定性指标，直接影响 Np 取值
    # Fr = N^2*D/g：衡量惯性力与重力之比，影响自由液面漩涡
    # We = rho*N^2*D^3/sigma：液液/气液分散体系的核心控制参数
    "Re",
    "Fr",
    "We",
]

REGRESSOR_NUMERIC_FEATURES = [
    "density",
    "viscosity",
    "surface_tension",
    "solid_fraction",
    "tank_diameter",
    "liquid_height",
    "baffle_num",
    "impeller_diameter",
    "speed_rps",
    "clearance",
    "num_impellers",
    "impeller_spacing",
    "D_T",
    "H_T",
    "C_T",
    "S_T",
    "Re",
    "Fr",
    "We",
]

REGRESSOR_CATEGORICAL_FEATURES = [
    "process_goal",
    "operation_mode",
    "impeller_type",
]

# ==================== 回归目标 ====================
REGRESSOR_TARGETS = [
    "mixing_time",
    "power",
    "suspension_score",
]


def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)