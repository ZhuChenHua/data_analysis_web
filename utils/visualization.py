import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd


class DataVisualizer:
    def __init__(self, df):
        self.df = df

    def create_plot(self, plot_type, **kwargs):
        """创建可视化图表"""
        plot_method = getattr(self, f"_{plot_type}", None)
        if not plot_method:
            raise ValueError(f"不支持的图表类型: {plot_type}")
        return plot_method(**kwargs)

    def _scatter(self, x, y, color=None):
        return px.scatter(self.df, x=x, y=y, color=color)

    def _line(self, x, y, color=None):
        return px.line(self.df, x=x, y=y, color=color)

    def _bar(self, x, y, color=None):
        return px.bar(self.df, x=x, y=y, color=color)

    def _histogram(self, x, nbins=20):
        return px.histogram(self.df, x=x, nbins=nbins)

    def _box(self, x, y):
        return px.box(self.df, x=x, y=y)

    def _pie(self, names, values):
        return px.pie(self.df, names=names, values=values)

    def _heatmap(self, columns):
        corr = self.df[columns].corr()
        return px.imshow(
            corr,
            labels=dict(x="特征", y="特征", color="相关系数"),
            x=columns,
            y=columns,
        )

    def _density_contour(self, x, y):
        return px.density_contour(self.df, x=x, y=y)

    def _violin(self, x, y):
        return px.violin(self.df, x=x, y=y)

    def _area(self, x, y):
        return px.area(self.df, x=x, y=y)

    def _pairplot(self, columns):
        return px.scatter_matrix(self.df[columns])
