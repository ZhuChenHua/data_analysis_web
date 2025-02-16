import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer


class DataProcessor:
    def __init__(self, df):
        self.raw_df = df
        self.processed_df = df.copy()

    def apply_operations(self, operations):
        """执行数据清洗操作"""
        try:
            if operations.get("drop_na"):
                self.processed_df = self.processed_df.dropna()

            if operations.get("drop_duplicates"):
                self.processed_df = self.processed_df.drop_duplicates()

            if operations.get("fill_na"):
                self._fill_missing_values(operations)

            if operations.get("standardization"):
                self._standardize_columns(operations["standardization_cols"])

            if operations.get("discretization"):
                self._discretize_columns(
                    operations["discretization_cols"], operations["n_bins"]
                )

            return self.processed_df
        except Exception as e:
            raise ValueError(f"数据处理错误: {str(e)}")

    def _fill_missing_values(self, operations):
        """处理缺失值"""
        col = operations["fill_na_col"]
        method = operations["fill_method"]

        if method == "mean":
            self.processed_df[col] = self.processed_df[col].fillna(
                self.processed_df[col].mean()
            )
        elif method == "median":
            self.processed_df[col] = self.processed_df[col].fillna(
                self.processed_df[col].median()
            )
        elif method == "mode":
            self.processed_df[col] = self.processed_df[col].fillna(
                self.processed_df[col].mode()[0]
            )
        elif method == "ffill":
            self.processed_df[col] = self.processed_df[col].ffill()
        elif method == "custom":
            self.processed_df[col] = self.processed_df[col].fillna(
                operations["custom_value"]
            )

    def _standardize_columns(self, columns):
        """标准化处理"""
        scaler = StandardScaler()
        self.processed_df[columns] = scaler.fit_transform(self.processed_df[columns])

    def _discretize_columns(self, columns, n_bins=3):
        """分箱离散化"""
        discretizer = KBinsDiscretizer(
            n_bins=n_bins, encode="ordinal", strategy="uniform"
        )
        self.processed_df[columns] = discretizer.fit_transform(
            self.processed_df[columns]
        )

    def calculate_statistics(self, operation, column=None):
        """数据计算功能"""
        if operation == "describe":
            return self.processed_df.describe()

        if column is None:
            raise ValueError("需要选择计算列")

        if operation == "sum":
            return self.processed_df[column].sum()
        elif operation == "mean":
            return self.processed_df[column].mean()
        elif operation == "median":
            return self.processed_df[column].median()
        elif operation == "std":
            return self.processed_df[column].std()
        elif operation == "value_counts":
            return self.processed_df[column].value_counts()
