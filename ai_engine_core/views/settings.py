import streamlit as st

from utils.ui import render_page_header

def render_settings():
    render_page_header(
        "系统设置",
        "检测与可解释性的运行控制",
        icon="settings",
    )

    if "settings" not in st.session_state:
        st.session_state.settings = {
            "threshold": 0.5,
            "enable_shap": True,
            "enable_explain": True,
            "enable_zh_translation": True,
        }

    s = st.session_state.settings

    st.markdown("""
    <div style="background-color: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
        <h3 style="margin: 0 0 1rem 0; border: none; font-size: 1.1rem; color: var(--igp-text);">检测配置</h3>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        s["threshold"] = st.slider("判定阈值 (真新闻)", 0.05, 0.95, float(s.get("threshold", 0.5)), 0.01)
    with c2:
        s["enable_shap"] = st.toggle("启用 SHAP 分析", value=bool(s.get("enable_shap", True)))
    with c3:
        s["enable_explain"] = st.toggle("启用可视化解释", value=bool(s.get("enable_explain", True)))
    with c4:
        s["enable_zh_translation"] = st.toggle("中英翻译网桥", value=bool(s.get("enable_zh_translation", True)))

    st.markdown("""
    <div style="background-color: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-top: 1.5rem;">
        <h3 style="margin: 0 0 1rem 0; border: none; font-size: 1.1rem; color: var(--igp-text);">说明</h3>
        <div style="color: var(--igp-muted); font-size: 0.95rem; line-height: 1.6;">
            阈值用于控制 P(Real) 的真假判定边界。启用 SHAP 分析和可视化解释 (热力图) 会显著增加计算时间。
        </div>
    </div>
    """, unsafe_allow_html=True)
