import streamlit as st
import pandas as pd
import time
from PIL import Image
from core.model_engine import RealTimeDetector
from core.data_factory import MockDataLoader
from utils.viz_helper import plot_text_heatmap
import urllib.request
from io import BytesIO
import os
import plotly.express as px
import plotly.graph_objects as go
from html import escape as html_escape
from core.case_store import CaseStore
from core.source_connectors import fetch_url_payload
from utils.ui import render_page_header
from utils.icons import inline_svg

# Initialize detector once
if 'detector' not in st.session_state:
    # Force fresh initialization
    st.session_state.detector = RealTimeDetector()

def render_detection():
    render_page_header(
        "多模态验证系统",
        "用于数字信息治理的三流融合架构",
        icon="search",
    )
    
    if "settings" not in st.session_state:
        st.session_state.settings = {"threshold": 0.5, "enable_shap": True, "enable_explain": True, "enable_zh_translation": True}
    if "history" not in st.session_state:
        st.session_state.history = []

    # Initialize variables
    selected_sample = None
    text_input = ""
    uploaded_file = None
    image_url = None
    source_url = None
    source_meta = None
    
    # Session State for Analysis Persistence
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    if 'explanation_result' not in st.session_state:
        st.session_state.explanation_result = None
    
    # Auto-clear stale explanation results from previous version
    if st.session_state.explanation_result and 'heatmap_swin' not in st.session_state.explanation_result:
        st.session_state.explanation_result = None

    # --- 1. Input Section (Unified) ---
    st.markdown(f"""
    <div style="background-color: white; padding: 2rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 2rem;">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.5rem;">
            <div style="display: flex; align-items: center;">
                <span style="margin-right: 0.5rem;">{inline_svg("pencil", size_px=22, color="var(--igp-primary)", title="Input")}</span>
                <h3 style="margin: 0; border: none; font-size: 1.25rem; color: var(--igp-text);">内容输入</h3>
            </div>
            <div style="display: flex; gap: 0.5rem;">
                <span style="color: var(--igp-muted); font-size: 0.85rem;">支持：手动录入 / 网页抓取 / 样本库</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    c_in_1, c_in_2 = st.columns([1.5, 1], gap="large")
    
    with c_in_1:
        # Load samples (Cached)
        if 'dataset_samples' not in st.session_state:
            with st.spinner("正在访问 Fakeddit 仓库..."):
                st.session_state.dataset_samples = MockDataLoader.load_fakeddit_sample()
        
        # UI for loading sample or URL
        c_src_1, c_src_2 = st.columns([1, 1])
        with c_src_1:
            c_i, c_b = st.columns([0.18, 0.82], gap="small", vertical_alignment="center")
            with c_i:
                st.markdown(
                    f"<div class='igp-nav-icon'>{inline_svg('dice', size_px=18, color='var(--igp-muted)', title='Random sample')}</div>",
                    unsafe_allow_html=True,
                )
            with c_b:
                if st.button("随机样本", use_container_width=True, type="secondary"):
                    import random
                    sample = random.choice(st.session_state.dataset_samples)
                    st.session_state.manual_text = sample['clean_title']
                    st.session_state.temp_image_url = sample['image_url']
                    st.session_state.temp_image_bytes = None
                    st.session_state.temp_source_url = None
                    st.session_state.temp_source_kind = "dataset"
                    st.rerun()
        with c_src_2:
            c_i, c_p = st.columns([0.18, 0.82], gap="small", vertical_alignment="center")
            with c_i:
                st.markdown(
                    f"<div class='igp-nav-icon'>{inline_svg('globe', size_px=18, color='var(--igp-muted)', title='Web fetch')}</div>",
                    unsafe_allow_html=True,
                )
            with c_p:
                with st.popover("网页抓取", use_container_width=True):
                    url_val = st.text_input("粘贴 URL", placeholder="https://...")
                    fill_mode = st.selectbox("填充内容", options=["标题+正文", "仅标题", "仅正文", "仅摘要"], index=0)
                    if st.button("开始抓取并填充", use_container_width=True, type="primary"):
                        if url_val.strip():
                            with st.spinner("正在抓取内容..."):
                                payload = fetch_url_payload(url_val.strip())
                                if payload and payload.get("ok"):
                                    title = payload.get("title") or ""
                                    body = payload.get("body_text") or ""
                                    desc = payload.get("description") or ""
                                    if fill_mode == "仅标题":
                                        text_fill = title or payload.get("text") or ""
                                    elif fill_mode == "仅正文":
                                        text_fill = body or payload.get("body_snippet") or payload.get("text") or ""
                                    elif fill_mode == "仅摘要":
                                        text_fill = desc or payload.get("text") or ""
                                    else:
                                        if title and body:
                                            text_fill = f"{title}\n\n{body}"
                                        else:
                                            text_fill = payload.get("text") or ""

                                    st.session_state.manual_text = text_fill
                                    st.session_state.temp_image_url = payload.get("image_url")
                                    st.session_state.temp_image_bytes = payload.get("image_bytes")
                                    st.session_state.temp_source_url = payload.get("final_url") or payload.get("url")
                                    st.session_state.temp_source_kind = "url"
                                    st.session_state.url_payload = payload
                                    st.rerun()
                                else:
                                    st.error(f"抓取失败：{(payload or {}).get('error') or 'unknown_error'}")

        # Main Text Input
        text_input = st.text_area(
            "文本内容",
            height=160,
            placeholder="请输入待验证的文本内容...",
            label_visibility="collapsed",
            key="manual_text",
        )
        
        uploaded_file = st.file_uploader("上传本地图像 (可选)", type=["png", "jpg", "jpeg"])
        if uploaded_file:
            st.session_state.temp_uploaded_file = uploaded_file
            st.session_state.temp_image_url = None
            st.session_state.temp_image_bytes = None
            st.session_state.temp_source_kind = "manual"

    with c_in_2:
        url_payload = st.session_state.get("url_payload") if isinstance(st.session_state.get("url_payload"), dict) else {}

        def _pick_url_image(u: str):
            st.session_state.temp_image_url = u
            st.session_state.temp_image_bytes = None

        if st.session_state.get("temp_source_kind") == "url":
            imgs = url_payload.get("images") or []
            if isinstance(imgs, list) and len(imgs) > 1:
                urls = [x.get("url") for x in imgs if isinstance(x, dict) and x.get("url")]
                current_url = st.session_state.get("temp_image_url")
                if current_url and current_url not in urls:
                    urls = [current_url] + urls

                if urls:
                    with st.expander("候选图片预览（点选用）", expanded=False):
                        preview_imgs = [x for x in imgs if isinstance(x, dict) and x.get("url")]
                        selected_u = st.session_state.get("temp_image_url")

                        if not preview_imgs:
                            st.caption("暂无可预览的候选图片。")
                        else:
                            preview_imgs = preview_imgs[:20]
                            with st.container():
                                st.markdown('<div class="igp-url-img-strip-sentinel"></div>', unsafe_allow_html=True)
                                st.markdown(
                                    """
                                    <style>
                                      .igp-url-img-strip-sentinel { display: none; }
                                      .igp-thumb-scroll {
                                        overflow-x: auto !important;
                                        overflow-y: hidden !important;
                                        white-space: nowrap;
                                        padding: 2px 2px 12px 2px !important;
                                        -webkit-overflow-scrolling: touch;
                                      }
                                      .igp-thumb-card {
                                        display: inline-block;
                                        vertical-align: top;
                                        width: 280px !important;
                                        min-width: 280px !important;
                                        max-width: 280px !important;
                                        margin-right: 12px;
                                      }
                                      .igp-thumb-img{
                                        border:1px solid var(--igp-border);
                                        border-radius:8px;
                                        overflow:hidden;
                                        background:#fff;
                                        margin-bottom: 8px;
                                        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
                                      }
                                      .igp-thumb-img.selected{
                                        border:2px solid var(--igp-primary);
                                        box-shadow: 0 0 0 2px rgba(18, 96, 163, 0.2);
                                      }
                                      .igp-thumb-img img{
                                        width:100%;
                                        height:180px;
                                        object-fit:cover;
                                        display:block;
                                      }
                                      .igp-thumb-meta{
                                        font-size:13px;
                                        color:var(--igp-muted);
                                        margin-bottom:8px;
                                        line-height:1.4;
                                        white-space:nowrap;
                                        overflow:hidden;
                                        text-overflow:ellipsis;
                                        padding: 0 4px;
                                      }
                                    </style>
                                    """,
                                    unsafe_allow_html=True,
                                )
                                cards_html = []
                                valid_items = []
                                for i, x in enumerate(preview_imgs):
                                    u = str(x.get("url") or "").strip()
                                    if not u:
                                        continue
                                    src = str(x.get("source") or "img")
                                    valid_items.append((len(valid_items), i + 1, u, src))
                                if valid_items:
                                    for _, n, u, s in valid_items:
                                        selected_cls = "selected" if u == selected_u else ""
                                        cards_html.append(
                                            f'<div class="igp-thumb-card"><div class="igp-thumb-img {selected_cls}"><img src="{html_escape(u)}" alt="img" /></div><div class="igp-thumb-meta">#{n} {html_escape(s)}</div></div>'
                                        )
                                    st.markdown(
                                        f'<div class="igp-thumb-scroll">{"".join(cards_html)}</div>',
                                        unsafe_allow_html=True,
                                    )
                                    pick_cols = st.columns(len(valid_items), gap="small")
                                    for idx, (_, n, u, _) in enumerate(valid_items):
                                        with pick_cols[idx]:
                                            st.button(
                                                f"选#{n}",
                                                key=f"igp_thumb_pick_{idx}",
                                                use_container_width=True,
                                                disabled=(u == selected_u),
                                                on_click=_pick_url_image,
                                                args=(u,),
                                            )
                            st.caption("横向滑动查看；点下方对应按钮即可选用（不会丢失已填文本）。")

        # Resolve what image to display
        display_img = None
        source_url = st.session_state.get("temp_source_url")
        image_url = st.session_state.get("temp_image_url")
        image_bytes = st.session_state.get("temp_image_bytes")
        
        if uploaded_file:
            display_img = uploaded_file
        elif image_bytes:
            display_img = image_bytes
        elif image_url:
            display_img = image_url
        
        if display_img:
             st.image(display_img, use_container_width=True)
             if image_url and not uploaded_file:
                 st.caption(f"预览源: {image_url[:50]}...")
        else:
             st.markdown(f"""
             <div style="height: 100%; display: flex; align-items: center; justify-content: center; background: var(--igp-surface-2); border-radius: 8px; color: var(--igp-muted); min-height: 250px; border: 2px dashed var(--igp-border);">
                <div style="text-align: center;">
                    <div style="margin-bottom: 0.5rem;">{inline_svg("image", size_px=34, color="var(--igp-muted)", title="No image")}</div>
                    <div style="font-size: 0.9rem;">暂无视觉输入</div>
                    <div style="font-size: 0.75rem; margin-top: 0.25rem;">上传本地图片或通过网页抓取</div>
                </div>
             </div>
             """, unsafe_allow_html=True)
             
    # Primary Action Button (Centered)
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    c_btn_1, c_btn_2, c_btn_3 = st.columns([1, 1.2, 1])
    with c_btn_2:
        st.markdown(
            f"<div style='display:flex; justify-content:center; margin-bottom: 0.35rem;'>{inline_svg('rocket', size_px=22, color='var(--igp-primary)', title='Run')}</div>",
            unsafe_allow_html=True,
        )
        predict_btn = st.button("开始分析", use_container_width=True, type="primary")
    
    st.markdown("</div>", unsafe_allow_html=True) # End Input Card

    if predict_btn and text_input:
        # Clear previous results
        st.session_state.analysis_result = None
        st.session_state.explanation_result = None
        
        # Inference
        try:
            image_bytes = st.session_state.get("temp_image_bytes")
            img_source = uploaded_file if uploaded_file else (BytesIO(image_bytes) if image_bytes else image_url)
            result = st.session_state.detector.predict_all(
                text_input,
                img_source,
                with_shap=bool(st.session_state.settings.get("enable_shap", True)),
                translate_zh=bool(st.session_state.settings.get("enable_zh_translation", True)),
            )
            st.session_state.analysis_result = result
            st.session_state.source_url = source_url
            st.session_state.source_meta = st.session_state.get("url_payload")
            
            # Explanation
            if bool(st.session_state.settings.get("enable_explain", True)):
                with st.spinner("正在生成注意力可视化..."):
                    heatmap_swin, tokens = st.session_state.detector.explain(
                        text_input,
                        img_source,
                        translate_zh=bool(st.session_state.settings.get("enable_zh_translation", True)),
                    )
                    st.session_state.explanation_result = {
                        'heatmap_swin': heatmap_swin,
                        'tokens': tokens
                    }
            
            # History
            try:
                from datetime import datetime
                prob = float(result["model_c"]["prob"])
                threshold = float(st.session_state.settings.get("threshold", 0.5))
                label = "真新闻" if prob >= threshold else "假新闻"
                st.session_state.history.append({
                    "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "text": text_input,
                    "image": img_source,
                    "prob_real": prob,
                    "label": label,
                })
            except: pass
                
        except Exception as e:
            st.error(f"分析失败: {e}")
            st.session_state.analysis_result = RealTimeDetector.simulate_prediction()
            st.session_state.explanation_result = None

    # --- 2. Results Section (Vertical Layout) ---
    if st.session_state.analysis_result:
        result = st.session_state.analysis_result
        prob = result['model_c']['prob']
        threshold = float(st.session_state.settings.get("threshold", 0.5))
        label = "真新闻" if float(prob) >= threshold else "假新闻"
        conf = prob if label == "真新闻" else 1 - prob
        
        # 2.1 Core Result (Compact Banner)
        # Colors: Real = Green, Fake = Red
        bg_color = "rgba(18, 96, 163, 0.10)" if label == "真新闻" else "rgba(198, 40, 40, 0.10)"
        text_color = "#1260A3" if label == "真新闻" else "#c62828"
        icon_name = "check" if label == "真新闻" else "warning"
        icon_svg = inline_svg(icon_name, size_px=46, color=text_color, title=label)
        
        st.markdown(f"""
        <div style="background-color: {bg_color}; border: 2px solid {text_color}; padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem; display: flex; align-items: center; justify-content: space-between;">
            <div style="display: flex; align-items: center;">
                <div style="margin-right: 1.25rem;">{icon_svg}</div>
                <h1 style="color: {text_color}; font-size: 3.5rem; margin: 0; line-height: 1;">{label}</h1>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.9rem; color: var(--igp-muted); text-transform: uppercase; letter-spacing: 0.05em;">置信度</div>
                <div style="font-family: 'Inter', sans-serif; font-size: 2.5rem; font-weight: 700; color: {text_color};">{conf:.1%}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        c_case_a, c_case_b, c_case_c = st.columns([1, 1.2, 1])
        with c_case_b:
            if st.button("保存为案例", use_container_width=True, type="secondary"):
                try:
                    store = CaseStore.default()
                    image_url = st.session_state.get("temp_image_url")
                    image_bytes = st.session_state.get("temp_image_bytes")
                    img_source = uploaded_file if uploaded_file else (BytesIO(image_bytes) if image_bytes else (image_url or None))
                    source = st.session_state.get("temp_source_kind") or ("url" if st.session_state.get("source_url") else "manual")
                    created = store.create_case_from_analysis(
                        text=text_input,
                        image_source=img_source,
                        analysis_result=st.session_state.analysis_result,
                        explanation_result=st.session_state.explanation_result,
                        threshold=float(st.session_state.settings.get("threshold", 0.5)),
                        source=source,
                        source_url=st.session_state.get("source_url"),
                        source_meta=st.session_state.get("source_meta"),
                    )
                    st.session_state.page = "cases"
                    st.session_state.case_open_id = created.get("id")
                    st.success("已保存至案例库。")
                    st.rerun()
                except Exception as e:
                    st.error(f"保存失败: {e}")

        # 2.2 Deep Analysis (Two Columns)
        c_an_1, c_an_2 = st.columns(2, gap="large")
        
        # Left: Modality Contribution
        with c_an_1:
            st.markdown(f"""
            <div style="background-color: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); height: 100%;">
                <div style="display: flex; align-items: center; margin-bottom: 1.5rem;">
                    <span style="margin-right: 0.5rem;">{inline_svg("scale", size_px=18, color="var(--igp-muted)", title="Modality")}</span>
                    <h4 style="margin: 0; color: var(--igp-text);">模态贡献度分析</h4>
                </div>
            """, unsafe_allow_html=True)
            
            if result.get('shap_scores'):
                scores = result['shap_scores']
                
                # Donut Chart
                labels = ['文本上下文', '视觉内容']
                values = [scores['text_pct'], scores['image_pct']]
                colors = ['#1260A3', '#64748b']
                
                fig = go.Figure(data=[go.Pie(
                    labels=labels, 
                    values=values, 
                    hole=.6,
                    marker=dict(colors=colors),
                    textinfo='label+percent',
                    textfont=dict(size=14),
                    hoverinfo='label+percent+value'
                )])
                
                fig.update_layout(
                    showlegend=False,
                    margin=dict(t=0, b=0, l=0, r=0),
                    height=250,
                )
                
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                
                # Legend (Custom)
                st.markdown(f"""
                <div style="display: flex; justify-content: center; gap: 1rem; margin-top: 1rem;">
                    <div style="display: flex; align-items: center;"><span style="display: inline-block; width: 12px; height: 12px; background-color: {colors[0]}; border-radius: 50%; margin-right: 0.5rem;"></span><span style="font-size: 0.9rem;">文本 ({scores['text_pct']:.1%})</span></div>
                    <div style="display: flex; align-items: center;"><span style="display: inline-block; width: 12px; height: 12px; background-color: {colors[1]}; border-radius: 50%; margin-right: 0.5rem;"></span><span style="font-size: 0.9rem;">视觉 ({scores['image_pct']:.1%})</span></div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("</div>", unsafe_allow_html=True)

        # Right: Interpretability
        with c_an_2:
            st.markdown(f"""
            <div style="background-color: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); height: 100%;">
                <div style="display: flex; align-items: center; margin-bottom: 1.5rem;">
                    <span style="margin-right: 0.5rem;">{inline_svg("search", size_px=18, color="var(--igp-muted)", title="Explainability")}</span>
                    <h4 style="margin: 0; color: var(--igp-text);">可解释性分析</h4>
                </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.explanation_result:
                exp = st.session_state.explanation_result
                heatmap_swin = exp.get('heatmap_swin')
                tokens = exp.get('tokens') or []
                
                # 1. Visual Attention
                st.markdown("<h5 style='margin-bottom: 0.5rem; color: var(--igp-muted);'>视觉注意力可视化</h5>", unsafe_allow_html=True)
                
                if heatmap_swin:
                    # Side-by-Side Visual Comparison
                    c_v1, c_v2 = st.columns(2, gap="small")
                    
                    with c_v1:
                        st.caption("原图")
                        if image_url:
                            st.image(image_url, use_container_width=True)
                        elif uploaded_file:
                            st.image(uploaded_file, use_container_width=True)
                        else:
                            st.markdown("<div style='background:var(--igp-surface-2); height:100%; border-radius:8px; border: 1px dashed var(--igp-border);'></div>", unsafe_allow_html=True)
                            
                    with c_v2:
                        st.caption("注意力热力图")
                        st.image(heatmap_swin, use_container_width=True)
                        
                    # Download Button (Small)
                    buf = BytesIO()
                    heatmap_swin.save(buf, format="PNG")
                    st.download_button("下载热力图", buf.getvalue(), "heatmap.png", "image/png", key="dl_hm", use_container_width=True)
                else:
                    st.info("无视觉注意力热力图。")

                st.markdown("---")

                # 2. Text Attribution
                st.markdown("<h5 style='margin-bottom: 0.5rem; color: var(--igp-muted);'>文本属性归因</h5>", unsafe_allow_html=True)
                
                if tokens:
                    words, scores = zip(*tokens)
                    fig = plot_text_heatmap(words, scores, width=12)
                    st.pyplot(fig, use_container_width=True)
                else:
                    st.info("无文本归因数据。")
            
            else:
                st.markdown("<div style='color:var(--igp-muted); text-align:center; padding: 2rem;'>等待分析数据...</div>", unsafe_allow_html=True)
                
            st.markdown("</div>", unsafe_allow_html=True)
