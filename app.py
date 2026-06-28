import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="转废为安智能决策平台",
    page_icon="⛏️",
    layout="wide"
)

# =========================
# 一、基础计算函数
# =========================
def predict_comsol_results(material_type, od, ca, urea, pressure, curing_days):
    """
    第一版代理模型：用于模拟 COMSOL 输出。
    后期可替换为真实 COMSOL 导出的 CSV 数据或拟合模型。
    输出：CaCO3生成量、孔隙率、相对渗透率、初始孔隙率
    """
    material_params = {
        "尾砂": {"phi0": 0.385, "k0": 1.00, "alpha": 0.00135, "cmax": 85},
        "废石": {"phi0": 0.450, "k0": 1.50, "alpha": 0.00105, "cmax": 70},
        "煤矸石": {"phi0": 0.420, "k0": 1.30, "alpha": 0.00115, "cmax": 75},
        "混合固废": {"phi0": 0.400, "k0": 1.10, "alpha": 0.00125, "cmax": 80},
    }
    p = material_params[material_type]
    phi0 = p["phi0"]
    alpha = p["alpha"]
    cmax = p["cmax"]

    t_h = max(curing_days, 0.01) * 24

    f_bac = od / (1 + od / 1.5)
    balance = min(ca, urea) / max(ca, urea)
    f_ca = min(ca / 500, 1.5)
    f_urea = min(urea / 500, 1.5)

    if pressure < 1000:
        f_p = 0.60
    elif pressure <= 5000:
        f_p = 1.00
    elif pressure <= 8000:
        f_p = 0.85
    else:
        f_p = 0.65

    rate = 0.0042 * f_bac * f_ca * f_urea * balance * f_p
    caco3 = cmax * (1 - np.exp(-rate * t_h))
    caco3 = min(caco3, cmax)

    porosity = max(phi0 - alpha * caco3, 0.28)

    k_rel = (porosity / phi0) ** 5
    k_rel = max(min(k_rel, 1.0), 0.05)

    return caco3, porosity, k_rel, phi0


def calc_flac3d_params(caco3, porosity, k_rel, phi0):
    """
    将 COMSOL 输出换算为 FLAC3D 可用的力学参数。
    """
    i_c = min(caco3 / 85, 1.0)
    i_phi = min(max((phi0 - porosity) / max(phi0 - 0.28, 0.001), 0), 1.0)
    i_k = min(max((1 - k_rel) / 0.70, 0), 1.0)

    i_m = 0.45 * i_c + 0.35 * i_phi + 0.20 * i_k

    e0, emax = 1.0e8, 3.0e8
    c0, cmax_mc = 1.5e5, 4.05e5
    phi0_mc, phimax_mc = 25.0, 30.0
    t0, tmax = 5.0e4, 1.0e5

    elastic = e0 + (emax - e0) * i_m
    cohesion = c0 + (cmax_mc - c0) * i_m
    friction = phi0_mc + (phimax_mc - phi0_mc) * i_m
    tension = t0 + (tmax - t0) * i_m

    fr_rad = np.deg2rad(friction)
    ucs = 2 * cohesion * np.cos(fr_rad) / (1 - np.sin(fr_rad))
    ucs_mpa = ucs / 1e6

    return i_m, elastic, cohesion, friction, tension, ucs_mpa


def safety_evaluation(ucs_mpa, k_rel, porosity, displacement, fos):
    """
    安全评分：强度、抗渗、孔隙、位移、安全系数五类指标。
    """
    score = 0

    if ucs_mpa >= 1.5:
        score += 35
    elif ucs_mpa >= 1.0:
        score += 28
    elif ucs_mpa >= 0.5:
        score += 18
    else:
        score += 8

    if k_rel <= 0.45:
        score += 25
    elif k_rel <= 0.60:
        score += 18
    elif k_rel <= 0.80:
        score += 10
    else:
        score += 5

    if porosity <= 0.33:
        score += 20
    elif porosity <= 0.35:
        score += 15
    elif porosity <= 0.37:
        score += 10
    else:
        score += 5

    if displacement <= 0.02:
        score += 10
    elif displacement <= 0.05:
        score += 6
    else:
        score += 2

    if fos >= 1.5:
        score += 10
    elif fos >= 1.2:
        score += 6
    else:
        score += 2

    if score >= 85:
        grade, color = "安全", "#16a34a"
    elif score >= 70:
        grade, color = "较安全", "#2563eb"
    elif score >= 55:
        grade, color = "预警", "#f59e0b"
    else:
        grade, color = "危险", "#dc2626"

    return score, grade, color


def risk_warning(od, ca, urea, pressure, curing_days, ucs_mpa, k_rel, porosity, displacement, fos):
    warnings = []

    if od < 0.5:
        warnings.append("菌液 OD 偏低，可能导致矿化反应不足，建议提高菌液活性或延长养护时间。")
    if od > 1.8:
        warnings.append("菌液 OD 偏高，可能造成入口局部沉淀和堵塞，建议降低菌液浓度或分段注浆。")
    if abs(ca - urea) / max(ca, urea) > 0.3:
        warnings.append("Ca²⁺与尿素浓度匹配度不足，建议尽量接近 1:1。")
    if pressure < 1000:
        warnings.append("注浆压力偏低，可能导致反应液深部迁移不足。")
    if pressure > 8000:
        warnings.append("注浆压力偏高，可能导致胶结液快速穿透，形成不均匀矿化。")
    if curing_days < 7:
        warnings.append("养护时间偏短，早期强度可能不足，建议至少进行 7 d 以上养护。")
    if ucs_mpa < 1.0:
        warnings.append("预测 UCS 低于 1.0 MPa，承载能力不足，建议提高矿化强度或延长养护。")
    if k_rel > 0.6:
        warnings.append("相对渗透率偏高，抗渗强化效果不足，建议优化沉淀量与孔隙填充效果。")
    if porosity > 0.35:
        warnings.append("孔隙率偏高，固废结构仍偏疏松，建议增强胶结反应。")
    if displacement > 0.05:
        warnings.append("工程位移偏大，存在变形风险，建议提高支护或强化等级。")
    if fos < 1.2:
        warnings.append("安全系数偏低，建议重新优化参数或提高设计裕度。")

    if len(warnings) == 0:
        warnings.append("当前参数组合风险较低，胶结强化效果较好。")

    return warnings


def carbon_calculation(volume, cement_dosage, ef_cement, micp_emission):
    co2_cement = volume * cement_dosage * ef_cement
    co2_micp = micp_emission
    reduction = co2_cement - co2_micp
    rate = reduction / co2_cement * 100 if co2_cement > 0 else 0
    return co2_cement, co2_micp, reduction, rate


def make_curve(material_type, od, ca, urea, pressure, curing_days):
    times = np.linspace(0, curing_days, 100)
    rows = []
    for d in times:
        c, p, k, _ = predict_comsol_results(material_type, od, ca, urea, pressure, d)
        rows.append({
            "养护时间/d": d,
            "CaCO3生成量/(kg/m3)": c,
            "孔隙率": p,
            "相对渗透率": k,
        })
    return pd.DataFrame(rows)


def build_line_chart(df, x_col, y_col, title, y_title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode="lines"))
    fig.update_layout(
        title=title,
        xaxis_title="养护时间 / d",
        yaxis_title=y_title,
        height=360,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


# =========================
# 二、页面标题与说明
# =========================
st.title("转废为安——矿山固废胶结强化安全评估与智能决策平台")
st.caption("Python + Streamlit 自主平台第一版：仿真结果预测、FLAC3D 参数换算、安全评价、风险预警、智能推荐与报告导出")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem;}
    .small-card {
        padding: 14px 16px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        background: #f9fafb;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# 三、侧边栏输入
# =========================
st.sidebar.header("基础参数输入")
material_type = st.sidebar.selectbox("矿山固废类型", ["尾砂", "废石", "煤矸石", "混合固废"])
scene_type = st.sidebar.selectbox("应用场景", ["采空区胶结强化", "边坡胶结强化", "固废回填胶结强化"])
od = st.sidebar.slider("菌液 OD600", 0.1, 2.0, 1.0, 0.1)
ca = st.sidebar.slider("Ca²⁺浓度 / mol·m⁻³", 100, 1000, 500, 50)
urea = st.sidebar.slider("尿素浓度 / mol·m⁻³", 100, 1000, 500, 50)
pressure = st.sidebar.slider("注浆压力 / Pa", 500, 10000, 3000, 500)
curing_days = st.sidebar.slider("养护时间 / d", 1, 28, 14, 1)

st.sidebar.header("FLAC3D 工程指标")
displacement = st.sidebar.number_input("顶板/边坡最大位移 / m", value=0.03, min_value=0.00, step=0.01)
fos = st.sidebar.number_input("安全系数 FOS", value=1.30, min_value=0.10, step=0.05)

st.sidebar.header("碳排放核算参数")
volume = st.sidebar.number_input("工程体积 / m³", value=1000.0, min_value=0.0, step=100.0)
cement_dosage = st.sidebar.number_input("传统水泥用量 / kg·m⁻³", value=100.0, min_value=0.0, step=10.0)
ef_cement = st.sidebar.number_input("水泥碳排放因子 / kgCO₂·kg⁻¹", value=0.85, min_value=0.0, step=0.05)
micp_emission = st.sidebar.number_input("MICP方案估算碳排放 / kgCO₂", value=30000.0, min_value=0.0, step=1000.0)

# =========================
# 四、可选读取仿真CSV
# =========================
st.sidebar.header("仿真数据耦合")
uploaded_file = st.sidebar.file_uploader("可选：上传 COMSOL/FLAC3D CSV", type=["csv"])
use_csv = False
csv_message = "当前使用内置代理模型预测。"

if uploaded_file is not None:
    try:
        sim_df = pd.read_csv(uploaded_file)
        required_cols = {"time_d", "caco3_kg_m3", "porosity", "k_rel"}
        if required_cols.issubset(set(sim_df.columns)):
            last = sim_df.sort_values("time_d").iloc[-1]
            caco3 = float(last["caco3_kg_m3"])
            porosity = float(last["porosity"])
            k_rel = float(last["k_rel"])
            phi0 = float(sim_df["porosity"].iloc[0])
            use_csv = True
            csv_message = "已读取上传 CSV，并使用最后一行作为当前评价结果。"
        else:
            csv_message = "CSV列名不符合要求，已自动切换为内置代理模型。需要列：time_d, caco3_kg_m3, porosity, k_rel。"
    except Exception as e:
        csv_message = f"CSV读取失败，已自动切换为内置代理模型。错误信息：{e}"

if not use_csv:
    caco3, porosity, k_rel, phi0 = predict_comsol_results(material_type, od, ca, urea, pressure, curing_days)

# =========================
# 五、核心计算
# =========================
i_m, elastic, cohesion, friction, tension, ucs_mpa = calc_flac3d_params(caco3, porosity, k_rel, phi0)
score, grade, color = safety_evaluation(ucs_mpa, k_rel, porosity, displacement, fos)
warnings = risk_warning(od, ca, urea, pressure, curing_days, ucs_mpa, k_rel, porosity, displacement, fos)
co2_cement, co2_micp, co2_reduce, co2_rate = carbon_calculation(volume, cement_dosage, ef_cement, micp_emission)

# =========================
# 六、页面展示
# =========================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "安全总览", "仿真预测", "力学换算", "风险预警", "智能推荐", "报告导出"
])

with tab1:
    st.subheader("一、安全等级评价")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("安全评分", f"{score}/100")
    col2.metric("安全等级", grade)
    col3.metric("预测 UCS", f"{ucs_mpa:.3f} MPa")
    col4.metric("矿化强化指数", f"{i_m:.3f}")

    st.markdown(
        f"""
        <div style='background:{color};padding:22px;border-radius:14px;color:white;text-align:center;font-size:28px;font-weight:700;'>
        当前方案安全等级：{grade}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(csv_message)

    result_df = pd.DataFrame({
        "指标": [
            "应用场景", "固废类型", "CaCO3平均生成量", "平均孔隙率", "相对渗透率",
            "弹性模量 E", "黏聚力 c", "内摩擦角 φ", "抗拉强度 tension", "安全系数 FOS"
        ],
        "结果": [
            scene_type, material_type, f"{caco3:.3f} kg/m³", f"{porosity:.3f}", f"{k_rel:.3f}",
            f"{elastic:.2e} Pa", f"{cohesion:.2e} Pa", f"{friction:.2f}°", f"{tension:.2e} Pa", f"{fos:.2f}"
        ]
    })
    st.table(result_df)

    st.subheader("碳排放对比")
    carbon_df = pd.DataFrame({
        "方案": ["传统水泥胶结方案", "MICP胶结强化方案"],
        "碳排放/kgCO2": [co2_cement, co2_micp]
    })
    fig_carbon = go.Figure()
    fig_carbon.add_trace(go.Bar(x=carbon_df["方案"], y=carbon_df["碳排放/kgCO2"]))
    fig_carbon.update_layout(title="传统水泥方案与MICP方案碳排放对比", yaxis_title="kgCO2", height=360)
    st.plotly_chart(fig_carbon, use_container_width=True)
    st.success(f"预计减排量：{co2_reduce:.2f} kgCO₂；预计减排率：{co2_rate:.2f}%")

with tab2:
    st.subheader("二、COMSOL 生化—渗流—孔隙演化结果预测")
    st.write("第一版平台采用代理模型预测；后期可以上传 COMSOL 导出的 CSV，实现基于仿真数据库的评价。")

    curve_df = make_curve(material_type, od, ca, urea, pressure, curing_days)
    c1, c2, c3 = st.columns(3)
    c1.plotly_chart(build_line_chart(curve_df, "养护时间/d", "CaCO3生成量/(kg/m3)", "CaCO3生成量—时间曲线", "kg/m³"), use_container_width=True)
    c2.plotly_chart(build_line_chart(curve_df, "养护时间/d", "孔隙率", "孔隙率—时间曲线", "孔隙率"), use_container_width=True)
    c3.plotly_chart(build_line_chart(curve_df, "养护时间/d", "相对渗透率", "相对渗透率—时间曲线", "相对渗透率"), use_container_width=True)

    st.dataframe(curve_df, use_container_width=True)

    template_df = pd.DataFrame({
        "time_d": [0, 7, 14, 21, 28],
        "caco3_kg_m3": [0, 35, 58, 72, 80],
        "porosity": [0.385, 0.355, 0.335, 0.320, 0.310],
        "k_rel": [1.000, 0.680, 0.500, 0.400, 0.330]
    })
    st.download_button(
        "下载CSV模板",
        data=template_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="comsol_flac3d_template.csv",
        mime="text/csv"
    )

with tab3:
    st.subheader("三、COMSOL → FLAC3D 力学参数换算")
    st.write("平台根据 CaCO3生成量、孔隙率和相对渗透率，计算矿化强化指数，并换算为 FLAC3D 可使用的力学参数。")

    mech_df = pd.DataFrame({
        "FLAC3D参数": ["弹性模量 E", "黏聚力 c", "内摩擦角 φ", "抗拉强度 tension", "无侧限抗压强度 UCS"],
        "数值": [f"{elastic:.3e}", f"{cohesion:.3e}", f"{friction:.3f}", f"{tension:.3e}", f"{ucs_mpa:.3f}"],
        "单位": ["Pa", "Pa", "°", "Pa", "MPa"]
    })
    st.table(mech_df)

    st.markdown(
        f"""
        <div class='small-card'>
        <b>换算逻辑：</b> CaCO3生成量越高、孔隙率越低、相对渗透率越低，说明矿化胶结越充分；平台据此提高 E、c、φ、tension 和 UCS，用于 FLAC3D 对比仿真。
        </div>
        <div class='small-card'>
        <b>当前矿化强化指数：</b> {i_m:.3f}<br>
        <b>当前预测UCS：</b> {ucs_mpa:.3f} MPa
        </div>
        """,
        unsafe_allow_html=True
    )

with tab4:
    st.subheader("四、风险预警")
    for w in warnings:
        if "风险较低" in w:
            st.success(w)
        else:
            st.warning(w)

    st.subheader("主要风险指标解释")
    explain_df = pd.DataFrame({
        "指标": ["UCS", "相对渗透率", "孔隙率", "工程位移", "安全系数"],
        "当前值": [f"{ucs_mpa:.3f} MPa", f"{k_rel:.3f}", f"{porosity:.3f}", f"{displacement:.3f} m", f"{fos:.2f}"],
        "评价含义": [
            "反映胶结体承载能力",
            "反映抗渗与孔隙堵塞效果",
            "反映固废结构致密程度",
            "反映工程变形控制效果",
            "反映整体稳定裕度"
        ]
    })
    st.table(explain_df)

with tab5:
    st.subheader("五、智能参数推荐")
    target_ucs = st.number_input("目标 UCS / MPa", value=1.2, min_value=0.1, step=0.1)
    target_k = st.number_input("目标相对渗透率上限", value=0.45, min_value=0.05, max_value=1.0, step=0.05)

    if st.button("生成智能推荐方案"):
        recs = []
        for od_i in np.arange(0.6, 1.81, 0.2):
            for ca_i in [300, 400, 500, 600, 700]:
                for urea_i in [300, 400, 500, 600, 700]:
                    for p_i in [2000, 3000, 4000, 5000]:
                        for d_i in [7, 14, 21, 28]:
                            c_i, eps_i, k_i, phi_i = predict_comsol_results(material_type, od_i, ca_i, urea_i, p_i, d_i)
                            im_i, e_i, coh_i, fr_i, ten_i, ucs_i = calc_flac3d_params(c_i, eps_i, k_i, phi_i)
                            score_i, grade_i, _ = safety_evaluation(ucs_i, k_i, eps_i, displacement, fos)

                            if ucs_i >= target_ucs and k_i <= target_k:
                                cost_index = od_i * 10 + ca_i / 100 + urea_i / 100 + p_i / 1000 + d_i / 10
                                recs.append({
                                    "菌液OD": round(float(od_i), 2),
                                    "Ca²⁺浓度": ca_i,
                                    "尿素浓度": urea_i,
                                    "注浆压力/Pa": p_i,
                                    "养护时间/d": d_i,
                                    "预测UCS/MPa": round(float(ucs_i), 3),
                                    "相对渗透率": round(float(k_i), 3),
                                    "安全评分": score_i,
                                    "安全等级": grade_i,
                                    "成本指数": round(float(cost_index), 2)
                                })

        if len(recs) == 0:
            st.error("未找到满足要求的方案。可以适当降低目标 UCS，或放宽相对渗透率限制。")
        else:
            rec_df = pd.DataFrame(recs).sort_values(by=["成本指数", "安全评分"], ascending=[True, False])
            st.success("已生成推荐方案，默认按成本指数从低到高排序。")
            st.dataframe(rec_df.head(10), use_container_width=True)
            st.download_button(
                "下载推荐方案CSV",
                data=rec_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="智能推荐方案.csv",
                mime="text/csv"
            )

with tab6:
    st.subheader("六、评价报告导出")
    report_text = f"""
《转废为安》矿山固废胶结强化安全评价报告

一、基础信息
应用场景：{scene_type}
固废类型：{material_type}
菌液 OD600：{od}
Ca²⁺浓度：{ca} mol/m³
尿素浓度：{urea} mol/m³
注浆压力：{pressure} Pa
养护时间：{curing_days} d

二、COMSOL 预测结果
CaCO3平均生成量：{caco3:.3f} kg/m³
平均孔隙率：{porosity:.3f}
相对渗透率：{k_rel:.3f}
数据来源说明：{csv_message}

三、FLAC3D 力学参数换算结果
弹性模量 E：{elastic:.3e} Pa
黏聚力 c：{cohesion:.3e} Pa
内摩擦角 φ：{friction:.3f}°
抗拉强度 tension：{tension:.3e} Pa
预测 UCS：{ucs_mpa:.3f} MPa
矿化强化指数：{i_m:.3f}

四、安全评价结果
安全评分：{score}/100
安全等级：{grade}
最大位移：{displacement:.3f} m
安全系数 FOS：{fos:.2f}

五、风险预警
{chr(10).join(["- " + w for w in warnings])}

六、碳排放估算
传统水泥方案碳排放：{co2_cement:.2f} kgCO₂
MICP方案碳排放：{co2_micp:.2f} kgCO₂
预计减排量：{co2_reduce:.2f} kgCO₂
预计减排率：{co2_rate:.2f}%

七、综合建议
建议结合现场固废粒径、初始孔隙率、渗透率、菌液活性和单轴抗压试验结果，对平台参数进行进一步标定。
在工程应用中，应优先选择安全评分较高、相对渗透率较低、成本指数较低的 MICP 胶结强化方案。
""".strip()

    st.text_area("自动生成评价报告", report_text, height=520)
    st.download_button(
        label="下载安全评价报告TXT",
        data=report_text.encode("utf-8-sig"),
        file_name="转废为安安全评价报告.txt",
        mime="text/plain"
    )