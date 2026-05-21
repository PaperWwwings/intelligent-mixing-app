import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.recommender import MixingRecommender
from src.config import ensure_dirs


# ==================== 页面配置 ====================
st.set_page_config(
    page_title="智能流体搅拌选型系统",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================== 初始化 ====================
@st.cache_resource
def load_model():
    """缓存模型，避免每次刷新都重新加载"""
    ensure_dirs()
    return MixingRecommender()


recommender = load_model()


# ==================== 侧边栏：输入参数 ====================
st.sidebar.title("工况参数输入")

st.sidebar.markdown("### 1. 流体性质")
density = st.sidebar.number_input("密度 (kg/m³)", min_value=500.0, max_value=2000.0, value=1050.0, step=10.0)
viscosity = st.sidebar.number_input("黏度 (Pa·s)", min_value=0.001, max_value=100.0, value=0.08, step=0.001, format="%.4f")
surface_tension = st.sidebar.number_input("表面张力 (N/m)", min_value=0.01, max_value=0.1, value=0.06, step=0.001)

st.sidebar.markdown("### 2. 设备几何")
tank_diameter = st.sidebar.slider("反应釜内径 (m)", min_value=0.3, max_value=5.0, value=1.2, step=0.1)
liquid_height = st.sidebar.slider("液位高度 (m)", min_value=0.5, max_value=8.0, value=1.3, step=0.1)
baffle_num = st.sidebar.selectbox("挡板数量", options=[0, 2, 4], index=2)

st.sidebar.markdown("### 3. 工艺要求")
process_goal = st.sidebar.selectbox(
    "工艺目标",
    options=["mixing", "suspension", "dispersion"],
    format_func=lambda x: {"mixing": "混合", "suspension": "固液悬浮", "dispersion": "分散"}[x]
)
target_mixing_time = st.sidebar.number_input("期望混合时间上限 (s)", min_value=10.0, max_value=600.0, value=100.0, step=10.0)
max_power = st.sidebar.number_input("设备功率上限 (W)", min_value=100.0, max_value=10000.0, value=1500.0, step=100.0)
solid_fraction = st.sidebar.slider("固含量 (体积分数)", min_value=0.0, max_value=0.5, value=0.08, step=0.01)

st.sidebar.markdown("### 4. 操作条件")
operation_mode = st.sidebar.radio("操作模式", options=["batch", "continuous"], format_func=lambda x: "间歇" if x == "batch" else "连续")


# ==================== 主界面 ====================
st.title("智能流体搅拌选型系统")
st.markdown("""
    基于机器学习与遗传算法的智能搅拌设备选型平台  
    **开发者：** 山东大学机械工程学院  
    **技术路线：** 分类初筛 → 代理模型预测 → GA寻优 → 多目标评价
""")

# 构建用户输入字典
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

# 运行按钮
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    run_button = st.button("🚀 启动智能选型计算", use_container_width=True, type="primary")

if run_button:
    with st.spinner("正在进行多目标寻优，请稍候..."):
        try:
            results = recommender.recommend(user_case, top_k=3)
            st.session_state['results'] = results
            st.session_state['user_case'] = user_case
            st.success("✅ 选型计算完成！")
        except Exception as e:
            st.error(f"❌ 计算出错：{str(e)}")
            st.stop()

# 展示结果
if 'results' in st.session_state:
    results = st.session_state['results']
    
    # ==================== 方案概览卡片 ====================
    st.markdown("---")
    st.subheader("推荐方案概览")
    
    cols = st.columns(3)
    for idx, (col, r) in enumerate(zip(cols, results)):
        with col:
            # 根据分数给不同颜色
            score = r['fitness_score']
            if score >= 90:
                color = "green"
                emoji = "🥇"
            elif score >= 75:
                color = "orange"
                emoji = "🥈"
            else:
                color = "gray"
                emoji = "🥉"
            
            st.metric(
                label=f"{emoji} 方案 {idx+1}: {r['impeller_type'].upper()}",
                value=f"{score:.1f} 分",
                delta=f"匹配度 {r['classifier_probability']*100:.1f}%",
                delta_color="normal" if r['classifier_probability'] > 0.5 else "inverse"
            )
            
            st.markdown(f"""
                <div style='padding: 10px; border-left: 5px solid {color}; background-color: #f0f0f0;'>
                <b>结构参数：</b><br>
                桨径: {r['impeller_diameter']:.3f} m<br>
                离底高度: {r['clearance']:.3f} m<br>
                桨层数: <b>{r['num_impellers']} 层</b><br>
                层间距: {r['impeller_spacing']:.3f} m<br><br>
                <b>操作参数：</b><br>
                转速: {r['speed_rpm']:.1f} RPM<br><br>
                <b>预测性能：</b><br>
                功耗: {r['pred_power']:.1f} W<br>
                混合时间: {r['pred_mixing_time']:.1f} s<br>
                悬浮评分: {r['pred_suspension_score']:.2f}
                </div>
            """, unsafe_allow_html=True)

    # ==================== 可视化对比 ====================
    st.markdown("---")
    st.subheader("方案对比分析")
    
    # 准备对比数据
    df_compare = pd.DataFrame([
        {
            "方案": f"方案{i+1}\n({r['impeller_type'].upper()})",
            "综合得分": r['fitness_score'],
            "功耗(W)": r['pred_power'],
            "混合时间(s)": r['pred_mixing_time'],
            "悬浮评分": r['pred_suspension_score'] * 100,  # 放大到0-100便于对比
            "转速(RPM)": r['speed_rpm'],
            "桨层数": r['num_impellers'],
        }
        for i, r in enumerate(results)
    ])
    
    # 雷达图
    categories = ['综合得分', '悬浮评分', '转速(RPM)']
    # 反转功耗和混合时间（越小越好，需要取倒数或反向）
    
    fig_radar = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, r in enumerate(results):
        # 标准化到 0-100 范围用于雷达图展示
        power_score = max(0, 100 - r['pred_power']/20)  # 功耗越小分越高
        time_score = max(0, 100 - r['pred_mixing_time']/2)  # 时间越短分越高
        
        fig_radar.add_trace(go.Scatterpolar(
            r=[r['fitness_score'], r['pred_suspension_score']*100, power_score, time_score],
            theta=['综合得分', '悬浮效果', '能耗效率', '混合效率'],
            fill='toself',
            name=f"方案{i+1}: {r['impeller_type'].upper()}",
            line_color=colors[i]
        ))
    
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        title="多维度性能雷达图"
    )
    
    col_chart1, col_chart2 = st.columns([2, 1])
    with col_chart1:
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with col_chart2:
        # 条形图对比
        fig_bar = px.bar(
            df_compare.melt(id_vars=['方案'], value_vars=['综合得分', '功耗(W)', '混合时间(s)']),
            x='方案',
            y='value',
            color='variable',
            barmode='group',
            title="关键指标对比",
            height=400
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ==================== 详细数据表与导出 ====================
    st.markdown("---")
    st.subheader("详细数据与导出")
    
    # 构建包含输入条件的完整报告表
    report_data = []
    for i, r in enumerate(results):
        report_data.append({
            # 输入条件
            "工艺目标": user_case['process_goal'],
            "流体密度(kg/m³)": user_case['density'],
            "流体黏度(Pa·s)": user_case['viscosity'],
            "反应釜内径(m)": user_case['tank_diameter'],
            "液位高度(m)": user_case['liquid_height'],
            "固含量": user_case['solid_fraction'],
            "期望混合时间上限(s)": user_case['target_mixing_time_req'],
            "设备功率上限(W)": user_case['max_power_req'],
            
            # 推荐结果
            "方案排名": i + 1,
            "综合推荐指数(0-100)": round(r['fitness_score'], 1),
            "推荐桨型": r['impeller_type'].upper(),
            "分类器匹配概率": f"{r['classifier_probability']*100:.1f}%",
            "设计桨径(m)": round(r['impeller_diameter'], 3),
            "设计离底高度(m)": round(r['clearance'], 3),
            "桨层数": r['num_impellers'],
            "层间距(m)": round(r['impeller_spacing'], 3),
            "操作转速(RPM)": round(r['speed_rpm'], 1),
            
            # 预测性能
            "预测轴功率(W)": round(r['pred_power'], 1),
            "预测混合时间(s)": round(r['pred_mixing_time'], 1),
            "预测悬浮评分": round(r['pred_suspension_score'], 3),
        })
    
    df_report = pd.DataFrame(report_data)
    st.dataframe(df_report, use_container_width=True, height=200)
    
    # 下载按钮
    csv = df_report.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="下载 CSV 报告",
        data=csv,
        file_name=f"搅拌选型报告_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime='text/csv',
    )

    # 技术说明
    with st.expander("🔍 查看技术详情"):
        st.markdown("""
            **算法流程说明：**
            1. **分类初筛**：RandomForest 分类器根据流体物性推荐 1-3 种候选桨型
            2. **代理模型**：RandomForest 回归器预测每种方案的性能（功耗、混合时间、悬浮评分）
            3. **遗传寻优**：GA 在每种候选桨型内搜索最优的桨径比、转速、离底高度、层数
            4. **多目标评价**：基于归一化奖励 + 惩罚函数的百分制评分体系（0-100分）
            
            **物理模型依据：**
            - 功率计算：$P = N_p \\cdot \\rho \\cdot n^3 \\cdot d^5$，含层间干涉修正
            - 混合时间：基于 $N \\cdot t_m = \\text{const}$ 经验关联式
            - 多层桨修正：功率×层数×0.9，混合时间÷层数^0.4
        """)

else:
    # 初始状态提示
    st.info("请在左侧侧边栏输入工况参数，然后点击上方的【启动智能选型计算】按钮")
    
    # 展示示例图片或说明
    st.markdown("""
        ### 系统功能简介
        
        本系统基于 **机器学习 + 遗传算法** 实现智能搅拌设备选型，可自动推荐：
        
        - **桨型选择**：推进式、斜叶桨、Rushton涡轮、锚式等
        - **结构参数**：桨径、离底高度、桨层数、层间距
        - **操作参数**：最优转速
        - **性能预测**：功耗、混合时间、悬浮效果
        
        ### 适用场景
        
        | 工艺类型 | 典型应用 |
        |---------|---------|
        | 混合 (Mixing) | 低/高粘度液体调和 |
        | 固液悬浮 (Suspension) | 催化剂悬浮、结晶操作 |
        | 分散 (Dispersion) | 气液分散、液液乳化 |
    """)