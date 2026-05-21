import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.recommender import MixingRecommender
from src.config import ensure_dirs


# ==================== 页面基础配置 ====================
st.set_page_config(
    page_title="流体搅拌设备选型系统",
    layout="wide",
    initial_sidebar_state="expanded"
)

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# ==================== 系统初始化 ====================
@st.cache_resource
def load_model():
    ensure_dirs()
    return MixingRecommender()

recommender = load_model()


# ==================== 左侧控制面板 (输入参数) ====================
st.sidebar.title("边界条件与工况设置")
st.sidebar.markdown("---")

st.sidebar.markdown("### 1. 物性参数")
density = st.sidebar.number_input("流体密度 (kg/m³)", 500.0, 2000.0, 1050.0, 10.0)
viscosity = st.sidebar.number_input("动力黏度 (Pa·s)", 0.001, 100.0, 0.08, 0.001, format="%.4f")
surface_tension = st.sidebar.number_input("表面张力 (N/m)", 0.01, 0.1, 0.06, 0.001)

st.sidebar.markdown("### 2. 设备几何约束")
tank_diameter = st.sidebar.slider("反应釜内径 (m)", 0.3, 5.0, 1.2, 0.1)
liquid_height = st.sidebar.slider("有效液位高度 (m)", 0.5, 8.0, 1.3, 0.1)
baffle_num = st.sidebar.selectbox("挡板配置 (数量)", [0, 2, 4], 2)

st.sidebar.markdown("### 3. 工艺目标与限值")
goal_map = {"mixing": "均相混合", "suspension": "固液悬浮", "dispersion": "多相分散"}
process_goal = st.sidebar.selectbox("核心工艺目标", list(goal_map.keys()), format_func=lambda x: goal_map[x])
target_mixing_time = st.sidebar.number_input("容许混合时间上限 (s)", 10.0, 600.0, 100.0, 10.0)
max_power = st.sidebar.number_input("装机功率上限 (W)", 100.0, 10000.0, 1500.0, 100.0)
solid_fraction = st.sidebar.slider("固相体积分数", 0.0, 0.5, 0.08, 0.01)

st.sidebar.markdown("### 4. 操作模式")
mode_map = {"batch": "间歇操作 (Batch)", "continuous": "连续操作 (Continuous)"}
operation_mode = st.sidebar.radio("运行状态", list(mode_map.keys()), format_func=lambda x: mode_map[x])


# ==================== 主界面区 ====================
st.title("流体搅拌设备选型与性能评价系统")
st.markdown("<span style='color:#6c757d; font-size:14px;'>山东大学机械工程学院智能制造 | 计算平台 V1.0</span>", unsafe_allow_html=True)
st.markdown("---")

user_case = {
    "density": density,
    "viscosity": viscosity,
    "surface_tension": surface_tension,
    "tank_diameter": tank_diameter,
    "liquid_height": liquid_height,
    "baffle_num": baffle_num,
    "solid_fraction": solid_fraction,
    "process_goal": process_goal,
    "target_mixing_time_req": target_mixing_time,
    "max_power_req": max_power,
    "operation_mode": operation_mode,
}

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    run = st.button("提交计算 (Submit Computation)", use_container_width=True, type="primary")

if run:
    with st.spinner("系统正在执行参数空间搜索与代理模型评估..."):
        try:
            results = recommender.recommend(user_case, top_k=3)
            st.session_state['results'] = results
            st.session_state['user_case'] = user_case
        except Exception as e:
            st.error(f"计算域异常: {str(e)}")
            st.stop()


# ==================== 结果呈现区 ====================
if 'results' in st.session_state:
    results = st.session_state['results']
    
    st.subheader("选型结果概览")
    
    cols = st.columns(3)
    rank_labels = ["最优推荐方案", "备选方案 A", "备选方案 B"]
    
    for idx, (col, r) in enumerate(zip(cols, results)):
        with col:
            score = r['fitness_score']
            # 取消大红大绿，采用深蓝(主方案)和钢灰(备选方案)的工业配色
            border_color = "#2c3e50" if idx == 0 else "#6c757d"
            bg_color = "#f8f9fa" if idx == 0 else "#ffffff"
            
            st.markdown(f"**{rank_labels[idx]}**")
            
            st.markdown(f"""
                <div style='padding: 15px; border-top: 4px solid {border_color}; border-bottom: 1px solid #dee2e6; border-left: 1px solid #dee2e6; border-right: 1px solid #dee2e6; background-color: {bg_color}; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);'>
                <div style='margin-bottom: 10px;'>
                    <span style='font-size: 18px; font-weight: bold; color: #343a40;'>{r['impeller_type'].upper()}</span><br>
                    <span style='font-size: 12px; color: #6c757d;'>综合评价指数: {score:.1f} / 100.0</span> | 
                    <span style='font-size: 12px; color: #6c757d;'>分类匹配度: {r['classifier_probability']*100:.1f}%</span>
                </div>
                <div style='font-size: 13px; color: #495057; line-height: 1.6;'>
                    <b>[ 结构参数 ]</b><br>
                    • 桨叶直径: {r['impeller_diameter']:.3f} m<br>
                    • 离底间隙: {r['clearance']:.3f} m<br>
                    • 桨叶配置: {r['num_impellers']} 层 (层间距 {r['impeller_spacing']:.3f} m)<br>
                    <br>
                    <b>[ 操作参数 ]</b><br>
                    • 额定转速: {r['speed_rpm']:.1f} RPM<br>
                    <br>
                    <b>[ 性能预测 ]</b><br>
                    • 预测轴功率: {r['pred_power']:.1f} W<br>
                    • 宏观混合时间: {r['pred_mixing_time']:.1f} s<br>
                    • 悬浮状态系数: {r['pred_suspension_score']:.3f}
                </div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("多维度性能对比分析")
    
    df_compare = pd.DataFrame([
        {
            "方案": f"{rank_labels[i]}\n({r['impeller_type'].upper()})",
            "综合评价指数": r['fitness_score'],
            "预测轴功率(W)": r['pred_power'],
            "混合时间(s)": r['pred_mixing_time'],
            "悬浮状态系数(%)": r['pred_suspension_score'] * 100,
            "额定转速(RPM)": r['speed_rpm'],
        }
        for i, r in enumerate(results)
    ])
    
    colors = ['#4e79a7', '#a0cbe8', '#f28e2b']
    
    fig_radar = go.Figure()
    for i, r in enumerate(results):
        power_score = max(0, 100 - r['pred_power']/20)
        time_score = max(0, 100 - r['pred_mixing_time']/2)
        
        fig_radar.add_trace(go.Scatterpolar(
            r=[r['fitness_score'], r['pred_suspension_score']*100, power_score, time_score],
            theta=['综合评价指数', '悬浮状态系数', '能耗效率表现', '宏观混合效率'],
            fill='toself',
            name=rank_labels[i],
            line_color=colors[i]
        ))
    
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        title="候选方案特征分布",
        font=dict(size=12)
    )
    
    col_chart1, col_chart2 = st.columns([2, 1])
    with col_chart1:
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with col_chart2:
        fig_bar = px.bar(
            df_compare.melt(id_vars=['方案'], value_vars=['综合评价指数', '预测轴功率(W)', '混合时间(s)']),
            x='方案', y='value', color='variable',
            barmode='group', title="关键数据对比", height=400,
            color_discrete_sequence=['#4e79a7', '#e15759', '#76b7b2']
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.subheader("数据报表导出")
    
    report_data = []
    for i, r in enumerate(results):
        report_data.append({
            "方案类别": rank_labels[i],
            "工艺目标": user_case['process_goal'],
            "流体密度(kg/m³)": user_case['density'],
            "流体黏度(Pa·s)": user_case['viscosity'],
            "反应釜内径(m)": user_case['tank_diameter'],
            "液位高度(m)": user_case['liquid_height'],
            "固相体积分数": user_case['solid_fraction'],
            "边界约束_混合时间上限(s)": user_case['target_mixing_time_req'],
            "边界约束_轴功率上限(W)": user_case['max_power_req'],
            "综合评价指数": round(r['fitness_score'], 2),
            "推荐桨型": r['impeller_type'].upper(),
            "桨叶直径(m)": round(r['impeller_diameter'], 3),
            "离底间隙(m)": round(r['clearance'], 3),
            "桨叶层数": r['num_impellers'],
            "层间距(m)": round(r['impeller_spacing'], 3),
            "额定转速(RPM)": round(r['speed_rpm'], 1),
            "预测_轴功率(W)": round(r['pred_power'], 2),
            "预测_混合时间(s)": round(r['pred_mixing_time'], 2),
            "预测_悬浮状态系数": round(r['pred_suspension_score'], 4),
        })
    
    df_report = pd.DataFrame(report_data)
    st.dataframe(df_report, use_container_width=True, height=150)
    
    csv = df_report.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="生成并导出工程选型数据表 (.csv)",
        data=csv,
        file_name=f"选型计算报表_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
        mime='text/csv'
    )

    with st.expander("系统计算内核说明"):
        st.markdown("""
            **技术架构**：
            本系统底层评价引擎基于 `RandomForest` 多目标回归代理模型，融合 `NSGA` 类演化算法思想进行全局参数搜索。
            
            **底层物理机理关联式约束**：
            - **搅拌轴功率**：$P = N_p \\cdot \\rho \\cdot n^3 \\cdot d^5$ （引入层流区非线性惩罚与层间干涉修正矩阵）。
            - **宏观混合时间**：依从无量纲混合数原理 $N \\cdot t_m = C$，并在深釜结构下计入多层桨的幂律衰减效应。
            - **固液悬浮状态**：基于 Zwietering 临界悬浮动能演化理论进行归一化评估。
            
            *免责声明：本计算结果基于数学代理模型输出，涉及高危敏感物料时请结合中试数据验证。*
        """)

else:
    st.info("系统就绪。请设定左侧边界条件及工况参数后，点击「提交计算」以生成选型方案。")
