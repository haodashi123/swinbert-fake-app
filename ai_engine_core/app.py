import streamlit as st
from views import detection, analytics, batch, report, settings, history, cases
from core.model_engine import RealTimeDetector
from utils.ui import inject_global_styles
from utils.icons import icon_path, inline_svg

# Page Configuration
st.set_page_config(
    page_title="多模态虚假新闻检测平台",
    page_icon=icon_path("shield"),
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_global_styles()

# Initialize detector globally if not present
if 'detector' not in st.session_state:
    st.session_state.detector = RealTimeDetector()

# Sidebar
with st.sidebar:
    st.markdown(
        f"""
        <div class="igp-brand">
          <div class="igp-brand__icon">{inline_svg("shield", size_px=18, color="var(--igp-primary)")}</div>
          <div>
            <div class="igp-brand__name">信息治理平台 <span class="igp-badge">v2.1</span></div>
            <div class="igp-brand__sub">先进的数字内容验证与审计平台</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("---")
    
    if "page" not in st.session_state:
        st.session_state.page = "detection"

    nav_items = [
        ("detection", "search", "内容检测"),
        ("analytics", "chart", "数据分析"),
        ("cases", "folder", "案例库"),
        ("batch", "box", "批量处理"),
        ("report", "report", "生成报告"),
        ("history", "clock", "历史记录"),
        ("settings", "settings", "系统设置"),
    ]
    for key, icon_name, label in nav_items:
        is_active = st.session_state.page == key
        icon_color = "var(--igp-primary)" if is_active else "var(--igp-muted)"

        c_icon, c_btn = st.columns([0.18, 0.82], gap="small", vertical_alignment="center")
        with c_icon:
            st.markdown(
                f"<div class='igp-nav-icon'>{inline_svg(icon_name, size_px=18, color=icon_color)}</div>",
                unsafe_allow_html=True,
            )
        with c_btn:
            btn_type = "primary" if is_active else "secondary"
            if st.button(label, use_container_width=True, type=btn_type, key=f"nav_{key}"):
                st.session_state.page = key
                st.rerun()
    
    st.markdown("---")
    st.caption("技术支持：PyTorch + SwinBERT")

page = st.session_state.page
if page == "analytics":
    analytics.render_analytics()
elif page == "batch":
    batch.render_batch()
elif page == "report":
    report.render_report()
elif page == "history":
    history.render_history()
elif page == "settings":
    settings.render_settings()
elif page == "cases":
    cases.render_cases()
else:
    detection.render_detection()
