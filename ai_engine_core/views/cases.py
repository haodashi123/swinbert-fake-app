import streamlit as st
from datetime import datetime

from core.case_store import CaseStore
from utils.ui import render_page_header


def render_cases():
    render_page_header(
        "案例库",
        "标注与治理工作流",
        icon="folder",
    )

    store = CaseStore.default()
    cases = store.list_cases()
    if not cases:
        st.info("暂无案例。请从内容检测页面保存案例。")
        return

    st.markdown("""
    <div style="background-color: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 1.5rem;">
        <h3 style="margin: 0; border: none; font-size: 1.1rem; color: var(--igp-text);">案例列表</h3>
    </div>
    """, unsafe_allow_html=True)

    status_options = ["待处理", "审核中", "已结案"]
    decision_options = ["未标记", "真新闻", "假新闻", "不确定"]

    status_filter = st.multiselect("状态筛选", options=status_options, default=status_options)
    decision_filter = st.multiselect("结论筛选", options=decision_options, default=decision_options)
    query = st.text_input("搜索 (文本 / 标签 / 负责人)", value="")

    def match(c):
        if c.get("status") not in status_filter:
            return False
        if c.get("decision") not in decision_filter:
            return False
        if not query.strip():
            return True
        q = query.strip().lower()
        t = ((c.get("input") or {}).get("text") or "").lower()
        tags = ",".join(c.get("tags") or []).lower()
        assignee = (c.get("assignee") or "").lower()
        return q in t or q in tags or q in assignee

    filtered = [c for c in cases if match(c)]
    if not filtered:
        st.warning("没有匹配筛选条件的案例。")
        return

    rows = []
    for i, c in enumerate(filtered):
        pred = c.get("prediction") or {}
        rows.append({
            "序号": i,
            "ID": c.get("id", "")[:10],
            "状态": c.get("status", ""),
            "结论": c.get("decision", ""),
            "预测": pred.get("label", ""),
            "置信度(真)": float(pred.get("prob_real", 0.5)),
            "负责人": c.get("assignee", ""),
            "更新时间": c.get("updated_at", c.get("created_at", "")),
        })

    st.dataframe(rows, use_container_width=True, height=320)
    selected_idx = st.number_input("查看索引", min_value=0, max_value=max(0, len(filtered) - 1), value=0, step=1)
    case = filtered[int(selected_idx)]

    if "case_open_id" in st.session_state:
        open_id = st.session_state.pop("case_open_id")
        for i, c in enumerate(filtered):
            if c.get("id") == open_id:
                case = c
                selected_idx = i
                break

    st.markdown("""
    <div style="background-color: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-top: 1.5rem;">
        <h3 style="margin: 0 0 1rem 0; border: none; font-size: 1.1rem; color: var(--igp-text);">案例详情</h3>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        status = st.radio("状态", options=status_options, horizontal=True, index=status_options.index(case.get("status", "待处理")) if case.get("status") in status_options else 0)
    with c2:
        decision = st.radio("人工结论", options=decision_options, horizontal=True, index=decision_options.index(case.get("decision", "未标记")) if case.get("decision") in decision_options else 0)
    with c3:
        assignee = st.text_input("负责人", value=case.get("assignee", ""))

    tags_str = st.text_input("标签 (逗号分隔)", value=",".join(case.get("tags") or []))
    notes = st.text_area("备注", value=case.get("notes", ""), height=120)

    pred = case.get("prediction") or {}
    st.markdown(f"**预测结果**: {pred.get('label','')}  |  **置信度(真)**: {float(pred.get('prob_real',0.5)):.3f}  |  **判定阈值**: {float(pred.get('threshold',0.5)):.2f}")

    input_text = (case.get("input") or {}).get("text", "")
    st.text_area("输入文本", value=input_text, height=160, disabled=True)

    img_path = (case.get("input") or {}).get("image_path")
    img_url = (case.get("input") or {}).get("image_url")
    if img_path:
        st.image(img_path, use_container_width=True)
    elif img_url:
        st.image(img_url, use_container_width=True)

    heatmap_path = (case.get("artifacts") or {}).get("heatmap_path")
    if heatmap_path:
        st.markdown("**注意力热力图**")
        st.image(heatmap_path, use_container_width=True)

    c_save, c_export = st.columns([1, 1])
    with c_save:
        if st.button("保存更改", type="primary", use_container_width=True):
            case["status"] = status
            case["decision"] = decision
            case["assignee"] = assignee
            case["tags"] = [t.strip() for t in tags_str.split(",") if t.strip()]
            case["notes"] = notes
            case["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            store.upsert_case(case)
            st.success("已保存。")
            st.rerun()

    with c_export:
        pkg = store.export_case_package(case.get("id", ""))
        if pkg:
            st.download_button(
                "下载案例压缩包 (ZIP)",
                data=pkg,
                file_name=f"case_{case.get('id','')[:10]}.zip",
                mime="application/zip",
                use_container_width=True,
            )

    st.markdown("**附件管理**")
    upload = st.file_uploader("上传附件", type=None)
    if upload is not None:
        saved = store.add_attachment(case.get("id", ""), upload)
        if saved:
            st.success("附件已保存。")
            st.rerun()
