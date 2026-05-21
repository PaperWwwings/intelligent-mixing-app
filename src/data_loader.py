import pandas as pd


def load_csv_data(file_path):
    """
    读取 CSV 文件
    """
    df = pd.read_csv(file_path)
    return df