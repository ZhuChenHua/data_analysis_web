import streamlit as st
import pandas as pd
import numpy as np
from utils.data_processing import DataProcessor
from utils.visualization import DataVisualizer
import asyncio
from io import BytesIO

# 页面配置
st.set_page_config(
    page_title="数据分析平台",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 自定义样式
st.markdown(
    """
<style>
.stSelectbox div {flex: 1;}
.stDownloadButton button {width: 100%;}
.st-bb {overflow: auto;}
.css-18e3th9 {padding: 2rem 1rem 10rem;}
</style>
""",
    unsafe_allow_html=True,
)


# 初始化会话状态
def init_session():
    session_defaults = {"raw_data": None, "processor": None, "current_df": None}
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session()


# 异步装饰器
def async_task(func):
    async def wrapper(*args, **kwargs):
        return await asyncio.get_event_loop().run_in_executor(
            None, func, *args, **kwargs
        )

    return wrapper


# 文件上传组件
def file_uploader():
    with st.sidebar:
        st.header("数据上传")
        uploaded_file = st.file_uploader("选择CSV/Excel文件", type=["csv", "xlsx"])

        if uploaded_file and st.session_state.raw_data is None:
            try:
                if uploaded_file.name.endswith("csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                st.session_state.raw_data = df
                st.session_state.processor = DataProcessor(df)
                st.session_state.current_df = df.copy()
                st.success("数据加载成功!")
            except Exception as e:
                st.error(f"文件读取错误: {str(e)}")


# 数据清洗页面
def show_data_cleaning():
    st.header("📥 数据清洗与预处理")

    with st.expander("高级数据清洗选项", expanded=True):
        with st.form("cleaning_form"):
            col1, col2 = st.columns([1, 2])

            with col1:
                st.subheader("基础操作")
                drop_na = st.checkbox("删除空值")
                drop_duplicates = st.checkbox("删除重复值")
                fill_na = st.checkbox("填充缺失值")
                standardization = st.checkbox("标准化处理")
                discretization = st.checkbox("分箱离散化")

            with col2:
                st.subheader("参数设置")
                operations = {}

                if fill_na:
                    fill_col = st.selectbox(
                        "选择填充列", st.session_state.current_df.columns
                    )
                    fill_method = st.selectbox(
                        "填充方法",
                        ["均值填充", "中位数填充", "众数填充", "前向填充", "自定义值"],
                    )
                    operations.update(
                        {
                            "fill_na": True,
                            "fill_na_col": fill_col,
                            "fill_method": fill_method.replace("填充", "").lower(),
                            "custom_value": (
                                st.number_input("自定义值")
                                if fill_method == "自定义值"
                                else None
                            ),
                        }
                    )

                if standardization:
                    std_cols = st.multiselect(
                        "选择标准化列",
                        st.session_state.current_df.select_dtypes(
                            include=np.number
                        ).columns,
                    )
                    operations["standardization_cols"] = std_cols

                if discretization:
                    disc_cols = st.multiselect(
                        "选择分箱列",
                        st.session_state.current_df.select_dtypes(
                            include=np.number
                        ).columns,
                    )
                    n_bins = st.slider("分箱数量", 2, 10, 3)
                    operations.update(
                        {"discretization_cols": disc_cols, "n_bins": n_bins}
                    )

            if st.form_submit_button("执行处理"):
                try:
                    operations.update(
                        {"drop_na": drop_na, "drop_duplicates": drop_duplicates}
                    )

                    with st.spinner("正在处理数据..."):
                        st.session_state.current_df = (
                            st.session_state.processor.apply_operations(operations)
                        )
                        st.success("数据处理完成!")
                except Exception as e:
                    st.error(str(e))

    # 结果显示与下载
    if st.session_state.current_df is not None:
        st.subheader("处理结果")
        st.dataframe(st.session_state.current_df)

        csv = st.session_state.current_df.to_csv(index=False).encode()
        st.download_button(
            label="下载处理结果",
            data=csv,
            file_name="processed_data.csv",
            mime="text/csv",
        )


# 数据可视化页面
def show_visualization():
    st.header("📊 数据可视化")

    with st.expander("可视化配置", expanded=True):
        col1, col2 = st.columns([1, 3])

        with col1:
            plot_type = st.selectbox(
                "选择图表类型",
                [
                    "散点图",
                    "折线图",
                    "柱状图",
                    "直方图",
                    "箱线图",
                    "饼图",
                    "热力图",
                    "密度图",
                    "小提琴图",
                    "面积图",
                    "矩阵散点图",
                ],
            )

        with col2:
            vis = DataVisualizer(st.session_state.current_df)
            params = {}

            if plot_type in ["散点图", "折线图", "柱状图"]:
                params["x"] = st.selectbox("X轴", st.session_state.current_df.columns)
                params["y"] = st.selectbox("Y轴", st.session_state.current_df.columns)
                params["color"] = st.selectbox(
                    "颜色分组", [None] + list(st.session_state.current_df.columns)
                )

            elif plot_type == "直方图":
                params["x"] = st.selectbox(
                    "选择数值列",
                    st.session_state.current_df.select_dtypes(
                        include=np.number
                    ).columns,
                )
                params["nbins"] = st.slider("分箱数量", 5, 100, 20)

            elif plot_type == "箱线图":
                params["x"] = st.selectbox(
                    "分类列",
                    st.session_state.current_df.select_dtypes(
                        exclude=np.number
                    ).columns,
                )
                params["y"] = st.selectbox(
                    "数值列",
                    st.session_state.current_df.select_dtypes(
                        include=np.number
                    ).columns,
                )

            elif plot_type == "饼图":
                params["names"] = st.selectbox(
                    "分类列",
                    st.session_state.current_df.select_dtypes(
                        exclude=np.number
                    ).columns,
                )
                params["values"] = st.selectbox(
                    "数值列",
                    st.session_state.current_df.select_dtypes(
                        include=np.number
                    ).columns,
                )

            elif plot_type == "热力图":
                params["columns"] = st.multiselect(
                    "选择数值列",
                    st.session_state.current_df.select_dtypes(
                        include=np.number
                    ).columns,
                    default=st.session_state.current_df.select_dtypes(
                        include=np.number
                    ).columns.tolist()[:5],
                )

            elif plot_type == "矩阵散点图":
                params["columns"] = st.multiselect(
                    "选择分析列",
                    st.session_state.current_df.columns,
                    default=st.session_state.current_df.columns.tolist()[:5],
                )

            if st.button("生成图表"):
                try:
                    plot_method_map = {
                        "散点图": "scatter",
                        "折线图": "line",
                        "柱状图": "bar",
                        "直方图": "histogram",
                        "箱线图": "box",
                        "饼图": "pie",
                        "热力图": "heatmap",
                        "密度图": "density_contour",
                        "小提琴图": "violin",
                        "面积图": "area",
                        "矩阵散点图": "pairplot",
                    }

                    fig = vis.create_plot(plot_method_map[plot_type], **params)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"图表生成错误: {str(e)}")


# 主界面逻辑
file_uploader()

if st.session_state.raw_data is not None:
    page = st.sidebar.selectbox(
        "功能导航", ["数据概览", "数据清洗", "数据分析", "数据可视化"]
    )

    if page == "数据概览":
        st.header("📋 数据概览")
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("原始数据样本")
            st.dataframe(st.session_state.raw_data.head(10), height=400)

        with col2:
            st.subheader("数据摘要")
            st.write(f"总行数: {len(st.session_state.raw_data)}")
            st.write(f"总列数: {len(st.session_state.raw_data.columns)}")
            st.write("数据类型分布:")
            st.write(st.session_state.raw_data.dtypes.value_counts())

    elif page == "数据清洗":
        show_data_cleaning()

    elif page == "数据分析":
        st.header("🧮 数据分析")
        analysis_type = st.selectbox(
            "选择分析类型", ["描述统计", "数据分布", "自定义计算"]
        )

        if analysis_type == "描述统计":
            st.write(st.session_state.processor.calculate_statistics("describe"))

        elif analysis_type == "数据分布":
            selected_col = st.selectbox(
                "选择分析列", st.session_state.current_df.columns
            )
            st.write(
                st.session_state.processor.calculate_statistics(
                    "value_counts", selected_col
                )
            )

        elif analysis_type == "自定义计算":
            with st.form("calculation_form"):
                col1, col2 = st.columns(2)
                with col1:
                    operation = st.selectbox(
                        "运算类型", ["求和", "平均值", "中位数", "标准差"]
                    )
                with col2:
                    selected_col = st.selectbox(
                        "选择计算列",
                        st.session_state.current_df.select_dtypes(
                            include=np.number
                        ).columns,
                    )

                if st.form_submit_button("执行计算"):
                    try:
                        result = st.session_state.processor.calculate_statistics(
                            operation.lower(), selected_col
                        )
                        st.success(f"{selected_col}列{operation}结果: {result:.2f}")
                    except Exception as e:
                        st.error(str(e))

    elif page == "数据可视化":
        show_visualization()

else:
    st.info("👈 请从左侧上传数据文件开始分析")
