from src.config import RAW_DATA_PATH, REQUIRED_RAW_COLUMNS, ensure_dirs

from src.data_loader import load_csv_data

from src.preprocess import ensure_columns, basic_clean

from src.feature_engineering import add_features

from src.model_utils import train_impeller_classifier, train_performance_regressors





def main():

    ensure_dirs()



    print("读取数据中...")

    df = load_csv_data(RAW_DATA_PATH)



    print(f"原始数据形状：{df.shape}")



    # 补齐列名

    df = ensure_columns(df, REQUIRED_RAW_COLUMNS)



    # 基础清洗

    df = basic_clean(df)



    # 增加特征

    df = add_features(df)



    print(f"特征工程后数据形状：{df.shape}")



    # 训练分类模型

    train_impeller_classifier(df)



    # 训练回归模型

    train_performance_regressors(df)



    print("\n训练完成！")





if __name__ == "__main__":

    main()
