import base64
from io import BytesIO
import json
import streamlit as st

from utils.ui import render_page_header

def _img_to_base64(pil_img):
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def render_report():
    render_page_header(
        "分析报告",
        "将最近一次分析导出为 HTML/JSON 格式",
        icon="report",
    )

    result = st.session_state.get("analysis_result")
    exp = st.session_state.get("explanation_result")
    if not result:
        st.info("暂无分析数据。请先运行内容检测。")
        return

    s = st.session_state.get("settings", {"threshold": 0.5})
    threshold = float(s.get("threshold", 0.5))
    prob = float(result["model_c"]["prob"])
    label = "真新闻" if prob >= threshold else "假新闻"
    conf = prob if label == "真新闻" else 1 - prob

    heatmap_b64 = None
    if exp and exp.get("heatmap_swin") is not None:
        heatmap_b64 = _img_to_base64(exp["heatmap_swin"])

    shap_scores = result.get("shap_scores") or {}
    text_original = result.get("text_original")
    text_used = result.get("text_used")
    text_meta = result.get("text_meta") or {}
    payload = {
        "prediction": {"prob_real": prob, "label": label, "confidence": conf, "threshold": threshold},
        "shap_scores": shap_scores,
        "text": {"original": text_original, "used": text_used, "meta": text_meta},
        "source": {
            "url": st.session_state.get("source_url"),
            "meta": st.session_state.get("source_meta"),
        },
    }

    html = f"""
    <html>
      <head>
        <meta charset="utf-8"/>
        <style>
          body {{ font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial; background:#F5F7FB; padding: 24px; }}
          .card {{ background:#fff; border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 16px; }}
          .title {{ font-size: 20px; font-weight: 800; color:#0F172A; }}
          .sub {{ color:#64748b; margin-top: 6px; }}
          .banner {{ display:flex; justify-content:space-between; align-items:center; border-radius:12px; padding: 18px; border:2px solid #1260A3; }}
          .label {{ font-size: 42px; font-weight: 800; }}
          .prob {{ font-size: 22px; font-weight: 700; }}
          img {{ max-width:100%; border-radius: 10px; border:1px solid #e5e7eb; }}
          .grid {{ display:grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
        </style>
      </head>
      <body>
        <div class="card">
          <div class="title">多模态验证报告</div>
          <div class="sub">基于 Swin + BERT 架构</div>
        </div>
        <div class="card">
          <div class="title">预测结论</div>
          <div style="margin-top:12px;" class="banner">
            <div class="label">{label}</div>
            <div style="text-align:right;">
              <div class="sub">P(Real): {prob:.3f}</div>
              <div class="prob">置信度: {conf:.1%}</div>
              <div class="sub">判定阈值: {threshold:.2f}</div>
            </div>
          </div>
        </div>
        <div class="card">
          <div class="title">模态贡献度</div>
          <div class="sub">文本权重: {float(shap_scores.get("text_pct", 0.0)):.1%} | 视觉权重: {float(shap_scores.get("image_pct", 0.0)):.1%}</div>
        </div>
    """
    if heatmap_b64:
        html += f"""
        <div class="card">
          <div class="title">注意力热力图</div>
          <div class="sub">基于 Swin EigenCAM 生成</div>
          <div style="margin-top:12px;">
            <img src="data:image/png;base64,{heatmap_b64}" />
          </div>
        </div>
        """
    html += """
      </body>
    </html>
    """

    st.markdown("""
    <div style="background-color: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 1.5rem;">
        <h3 style="margin: 0; border: none; font-size: 1.1rem; color: var(--igp-text);">下载报告</h3>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "下载 HTML 报告",
            data=html.encode("utf-8"),
            file_name="report.html",
            mime="text/html",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "下载 JSON 数据",
            data=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="report.json",
            mime="application/json",
            use_container_width=True,
        )

    with st.expander("预览报告", expanded=False):
        st.components.v1.html(html, height=520, scrolling=True)
