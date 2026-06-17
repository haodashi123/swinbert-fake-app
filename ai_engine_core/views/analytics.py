import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from core.data_factory import MockDataLoader
from utils.ui import render_page_header


def render_analytics():
    render_page_header(
        "模型效能分析",
        "基于 Fakeddit 样本的快速基准测试 (Swin + BERT)",
        icon="chart",
    )

    detector = st.session_state.get("detector")
    if detector is None:
        st.error("检测器未初始化。")
        return

    st.markdown("""
    <div style="background-color: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 1.5rem;">
        <h3 style="margin: 0; border: none; font-size: 1.1rem; color: var(--igp-text);">基准测试设置</h3>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        n_samples = st.slider("样本数量", min_value=10, max_value=200, value=50, step=10)
    with c2:
        seed = st.number_input("随机种子", min_value=0, max_value=999999, value=2026, step=1)
    with c3:
        include_images = st.toggle("启用图像", value=True)

    run = st.button("开始测试", type="primary", use_container_width=True)
    if not run:
        return

    samples = MockDataLoader.load_fakeddit_sample(n=int(n_samples))
    if not samples:
        st.warning("未加载到样本。")
        return

    y_true = []
    y_prob = []
    rows = []

    progress = st.progress(0)
    total = len(samples)
    for i, s in enumerate(samples):
        text = str(s.get("clean_title", ""))
        img = s.get("image_url") if include_images else None
        gt = int(s.get("label", 0))
        try:
            translate_zh = bool((st.session_state.get("settings") or {}).get("enable_zh_translation", True))
            r = detector.predict_all(text, img, with_shap=False, translate_zh=translate_zh)
            prob = float(r["model_c"]["prob"])
        except Exception:
            prob = 0.5
        pred = 1 if prob > 0.5 else 0

        y_true.append(gt)
        y_prob.append(prob)
        rows.append({
            "序号": i,
            "文本": text[:160],
            "实际标签": "真新闻" if gt == 1 else "假新闻",
            "预测标签": "真新闻" if pred == 1 else "假新闻",
            "置信度(真)": prob,
            "是否正确": pred == gt,
        })
        progress.progress(int((i + 1) / total * 100))
    progress.empty()

    y_true_np = np.array(y_true, dtype=int)
    y_pred_np = (np.array(y_prob) > 0.5).astype(int)

    tp = int(((y_true_np == 1) & (y_pred_np == 1)).sum())
    tn = int(((y_true_np == 0) & (y_pred_np == 0)).sum())
    fp = int(((y_true_np == 0) & (y_pred_np == 1)).sum())
    fn = int(((y_true_np == 1) & (y_pred_np == 0)).sum())

    acc = (tp + tn) / max(1, len(y_true_np))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, (prec + rec))

    st.markdown("""
    <div style="background-color: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 1.5rem;">
        <h3 style="margin: 0 0 1rem 0; border: none; font-size: 1.1rem; color: var(--igp-text);">核心指标</h3>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("准确率 (Accuracy)", f"{acc:.1%}")
    m2.metric("精确率 (Precision)", f"{prec:.1%}")
    m3.metric("召回率 (Recall)", f"{rec:.1%}")
    m4.metric("F1 值", f"{f1:.3f}")

    cm = np.array([[tn, fp], [fn, tp]])
    cm_df = pd.DataFrame(cm, index=["实际假", "实际真"], columns=["预测假", "预测真"])
    fig_cm = px.imshow(
        cm_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale=[(0.0, "#F5F7FB"), (1.0, "#1260A3")],
    )
    fig_cm.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320)

    st.markdown("""
    <div style="background-color: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 1.5rem;">
        <h3 style="margin: 0 0 1rem 0; border: none; font-size: 1.1rem; color: var(--igp-text);">混淆矩阵 (Confusion Matrix)</h3>
    </div>
    """, unsafe_allow_html=True)
    st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar": False})

    df = pd.DataFrame(rows)
    df_err = df[df["是否正确"] == False].copy()
    df_err["错误强度"] = (df_err["置信度(真)"] - 0.5).abs()
    df_err = df_err.sort_values("错误强度", ascending=False).head(50)

    st.markdown("""
    <div style="background-color: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
        <h3 style="margin: 0 0 1rem 0; border: none; font-size: 1.1rem; color: var(--igp-text);">主要错误案例</h3>
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(df_err[["序号", "实际标签", "预测标签", "置信度(真)", "文本"]], use_container_width=True, height=320)
