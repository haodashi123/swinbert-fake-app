import streamlit as st
import pandas as pd
import numpy as np

from core.data_factory import MockDataLoader
from utils.ui import render_page_header


def _guess_sep(name: str) -> str:
    if name.lower().endswith(".tsv"):
        return "\t"
    return ","


def render_batch():
    render_page_header(
        "批量处理",
        "对文件运行 Swin + BERT 模型并导出结果",
        icon="box",
    )

    detector = st.session_state.get("detector")
    if detector is None:
        st.error("检测器未初始化。")
        return

    if "settings" not in st.session_state:
        st.session_state.settings = {"threshold": 0.5, "enable_shap": True, "enable_explain": True, "enable_zh_translation": True}
    threshold = float(st.session_state.settings.get("threshold", 0.5))

    st.markdown("""
    <div style="background-color: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 1.5rem;">
        <h3 style="margin: 0 0 1rem 0; border: none; font-size: 1.1rem; color: var(--igp-text);">数据输入</h3>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1.2, 1])
    with c1:
        file = st.file_uploader("上传 CSV/TSV 文件", type=["csv", "tsv"])
    with c2:
        use_images = st.toggle("启用图像列", value=True)

    df = None
    if file is not None:
        sep = _guess_sep(file.name)
        try:
            df = pd.read_csv(file, sep=sep)
        except Exception:
            try:
                df = pd.read_csv(file)
            except Exception as e:
                st.error(f"文件读取失败: {e}")
                df = None

    if df is None:
        with st.expander("使用内置样本数据集", expanded=False):
            if st.button("加载样本数据 (50行)", type="secondary", use_container_width=True):
                samples = MockDataLoader.load_fakeddit_sample(n=50)
                df = pd.DataFrame(samples)
                st.session_state.batch_df = df
        df = st.session_state.get("batch_df")

    if df is None or df.empty:
        return

    st.markdown("**数据预览**")
    st.dataframe(df.head(20), use_container_width=True, height=240)

    cols = list(df.columns)
    default_text = "clean_title" if "clean_title" in cols else cols[0]
    default_img = "image_url" if "image_url" in cols else (cols[1] if len(cols) > 1 else cols[0])

    c3, c4, c5 = st.columns([1.2, 1.2, 1])
    with c3:
        text_col = st.selectbox("文本列 (Text)", options=cols, index=cols.index(default_text) if default_text in cols else 0)
    with c4:
        img_col = st.selectbox("图像列 (Image)", options=cols, index=cols.index(default_img) if default_img in cols else 0, disabled=not use_images)
    with c5:
        max_rows = st.number_input("最大处理行数", min_value=1, max_value=min(5000, len(df)), value=min(200, len(df)), step=1)

    run = st.button("开始批量推理", type="primary", use_container_width=True)
    if not run:
        return

    out_rows = []
    progress = st.progress(0)
    n = int(max_rows)

    for i in range(n):
        row = df.iloc[i]
        text = str(row.get(text_col, ""))
        img = None
        if use_images:
            img = row.get(img_col, None)
            if isinstance(img, float) and np.isnan(img):
                img = None
        try:
            r = detector.predict_all(text, img, with_shap=False, translate_zh=bool(st.session_state.settings.get("enable_zh_translation", True)))
            prob = float(r["model_c"]["prob"])
        except Exception:
            prob = 0.5
        pred = "真新闻" if prob >= threshold else "假新闻"
        out_rows.append({
            "行号": i,
            "置信度(真)": prob,
            "预测结论": pred,
            "文本内容": text[:300],
        })
        progress.progress(int((i + 1) / n * 100))

    progress.empty()
    out_df = pd.DataFrame(out_rows)

    st.markdown("""
    <div style="background-color: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-top: 1.5rem;">
        <h3 style="margin: 0 0 1rem 0; border: none; font-size: 1.1rem; color: var(--igp-text);">处理结果</h3>
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(out_df, use_container_width=True, height=420)
    st.download_button(
        "下载结果 (CSV)",
        data=out_df.to_csv(index=False).encode("utf-8"),
        file_name="batch_results.csv",
        mime="text/csv",
        use_container_width=True,
    )
