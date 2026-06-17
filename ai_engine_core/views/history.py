import streamlit as st
import pandas as pd

from utils.ui import render_page_header

def render_history():
    render_page_header(
        "历史记录",
        "回顾本会话中的历史分析任务",
        icon="clock",
    )

    items = st.session_state.get("history", [])
    if not items:
        st.info("暂无历史记录。请先在内容检测页面进行分析。")
        return

    df = pd.DataFrame([{
        "时间": x.get("ts", ""),
        "结论": x.get("label", ""),
        "置信度(真)": x.get("prob_real", None),
        "文本内容": (x.get("text", "")[:120] + "…") if len(x.get("text", "")) > 120 else x.get("text", ""),
    } for x in items])

    st.dataframe(df, use_container_width=True, height=420)

    idx = st.number_input("查看索引", min_value=0, max_value=max(0, len(items) - 1), value=max(0, len(items) - 1), step=1)
    item = items[int(idx)]

    st.markdown("""
    <div style="background-color: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-top: 1.5rem;">
        <h3 style="margin: 0 0 1rem 0; border: none; font-size: 1.1rem; color: var(--igp-text);">详情</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"**时间戳**: {item.get('ts','')}")
    st.markdown(f"**预测结论**: {item.get('label','')}  |  **P(Real)**: {item.get('prob_real',0):.3f}")
    st.text_area("文本", value=item.get("text", ""), height=140, disabled=True)
    if item.get("image") is not None:
        st.image(item["image"], use_container_width=True)

    if st.button("加载到内容检测页面", type="primary", use_container_width=True):
        st.session_state.prefill_text = item.get("text", "")
        st.session_state.prefill_image = item.get("image", None)
        st.session_state.page = "detection"
        st.rerun()
