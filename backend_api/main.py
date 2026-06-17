from __future__ import annotations

import base64
import io
import json
import os
import random
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel, Field

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_V2_ROOT = os.path.join(PROJECT_ROOT, "ai_engine_core")
if APP_V2_ROOT not in sys.path:
    sys.path.insert(0, APP_V2_ROOT)

from core.case_store import CaseStore
from core.data_factory import MockDataLoader
from core.model_engine import RealTimeDetector
from core.source_connectors import fetch_url_payload


class FetchUrlRequest(BaseModel):
    url: str = Field(min_length=1)


class TranslateTextRequest(BaseModel):
    text: str = Field(min_length=1)


class PredictRequest(BaseModel):
    text: Optional[str] = ""
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    with_shap: bool = True
    with_explain: bool = False
    translate_zh: bool = True
    threshold: float = 0.5
    source_url: Optional[str] = None
    source_meta: Optional[Dict[str, Any]] = None


class SettingsRequest(BaseModel):
    threshold: float = 0.5
    enable_shap: bool = False
    enable_image_heatmap: bool = False
    enable_text_heatmap: bool = False
    enable_zh_translation: bool = True


class CreateCaseRequest(BaseModel):
    text: str
    text_used: Optional[str] = None
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    analysis_result: Dict[str, Any]
    explanation: Optional[Dict[str, Any]] = None
    threshold: float = 0.5
    source: str = "manual"
    source_url: Optional[str] = None
    source_meta: Optional[Dict[str, Any]] = None


class UpdateCaseRequest(BaseModel):
    status: Optional[str] = None
    decision: Optional[str] = None
    tags: Optional[List[str]] = None
    notes: Optional[str] = None


class BatchRequest(BaseModel):
    rows: List[Dict[str, Any]]
    text_col: str
    image_col: Optional[str] = None
    use_images: bool = True
    max_rows: int = 200
    threshold: float = 0.5
    translate_zh: bool = True


class AnalyticsBenchmarkRequest(BaseModel):
    n_samples: int = 50
    include_images: bool = True
    seed: int = 2026
    translate_zh: bool = True
    custom_data: Optional[str] = None
    custom_data_type: Optional[str] = None
    custom_data_delimiter: Optional[str] = None
    text_col: Optional[str] = None
    image_col: Optional[str] = None
    label_col: Optional[str] = None


app = FastAPI(title="Info Governance API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_detector: Optional[RealTimeDetector] = None
_settings_default: Dict[str, Any] = {
    "threshold": 0.5,
    "enable_shap": False,
    "enable_image_heatmap": False,
    "enable_text_heatmap": False,
    "enable_zh_translation": True,
}
_settings: Dict[str, Any] = dict(_settings_default)
_history: List[Dict[str, Any]] = []
_last_context: Dict[str, Any] = {}
_icons_dir = os.path.join(APP_V2_ROOT, "assets", "icons")
_settings_file = os.path.join(PROJECT_ROOT, "runtime_data", "system_settings.json")
_warmup_done = False
_warmup_error = ""


def get_detector() -> RealTimeDetector:
    global _detector
    if _detector is None:
        _detector = RealTimeDetector()
    return _detector


def _strip_payload_bytes(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(payload)
    raw = out.pop("image_bytes", None)
    if isinstance(raw, (bytes, bytearray)):
        out["image_has_bytes"] = True
    else:
        out["image_has_bytes"] = False
    return out


def _decode_image_base64(data: str) -> io.BytesIO:
    if not data:
        raise ValueError("empty_image_base64")
    buf = data.strip()
    if "," in buf and buf.split(",", 1)[0].startswith("data:"):
        buf = buf.split(",", 1)[1]
    raw = base64.b64decode(buf)
    bio = io.BytesIO(raw)
    bio.seek(0)
    return bio


def _to_png_base64(img) -> str:
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    return base64.b64encode(bio.getvalue()).decode("utf-8")


def _from_png_base64(data: str):
    if not data:
        return None
    raw = _decode_image_base64(data).getvalue()
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _get_store() -> CaseStore:
    return CaseStore.default()


def _file_to_base64(path: str) -> Optional[str]:
    try:
        p = Path(path)
        if not p.exists() or not p.is_file():
            return None
        return base64.b64encode(p.read_bytes()).decode("utf-8")
    except Exception:
        return None


def _load_settings_from_disk() -> None:
    global _settings
    try:
        p = Path(_settings_file)
        if not p.exists():
            return
        raw = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return
        validated = SettingsRequest(**raw).model_dump()
        next_settings = dict(_settings_default)
        next_settings.update(validated)
        _settings = next_settings
    except Exception:
        pass


def _save_settings_to_disk() -> None:
    try:
        p = Path(_settings_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(_settings, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "warmup_done": str(_warmup_done).lower(), "warmup_error": _warmup_error}


def _warmup_models():
    global _warmup_done, _warmup_error
    try:
        detector = get_detector()
        detector.predict_all("warmup text", image_source=None, with_shap=False, translate_zh=False)
        _warmup_done = True
    except Exception as e:
        _warmup_error = str(e)


@app.on_event("startup")
def _startup_warmup():
    _load_settings_from_disk()
    threading.Thread(target=_warmup_models, daemon=True).start()


@app.get("/api/icons/{name}")
def api_get_icon(name: str):
    safe = "".join(ch for ch in str(name) if ch.isalnum() or ch in ("-", "_", "."))
    if not safe:
        raise HTTPException(status_code=400, detail="invalid_icon_name")
    if not safe.endswith(".svg"):
        safe = f"{safe}.svg"
    target = os.path.abspath(os.path.join(_icons_dir, safe))
    if not target.startswith(os.path.abspath(_icons_dir)):
        raise HTTPException(status_code=400, detail="invalid_icon_path")
    if not os.path.exists(target):
        raise HTTPException(status_code=404, detail="icon_not_found")
    data = open(target, "rb").read()
    return Response(content=data, media_type="image/svg+xml")


@app.get("/api/settings")
def api_get_settings() -> Dict[str, Any]:
    return {"ok": True, "settings": _settings}


@app.put("/api/settings")
def api_put_settings(req: SettingsRequest) -> Dict[str, Any]:
    _settings.update(req.model_dump())
    _save_settings_to_disk()
    return {"ok": True, "settings": _settings}


@app.get("/api/history")
def api_get_history() -> Dict[str, Any]:
    # Return history without heavy payload
    items = []
    for h in _history:
        items.append({
            "id": h.get("id"),
            "ts": h.get("ts"),
            "label": h.get("label"),
            "prob_real": h.get("prob_real"),
            "text": h.get("text"),
            "image_url": h.get("image_url"),
        })
    return {"ok": True, "items": items}

@app.get("/api/report/batch")
def api_get_report_batch(ids: str) -> Dict[str, Any]:
    from utils.icons import inline_svg
    id_list = [x.strip() for x in ids.split(",") if x.strip()]
    if not id_list:
        raise HTTPException(status_code=400, detail="no_ids_provided")
        
    histories = []
    for hid in id_list:
        h = next((x for x in _history if str(x.get("id") or "") == str(hid)), None)
        if h:
            histories.append(h)
            
    if not histories:
        raise HTTPException(status_code=404, detail="history_not_found")
        
    reports_html = []
    for h in histories:
        try:
            result = h.get("analysis_result") or {}
            exp = h.get("explanation") or {}
            threshold = float(h.get("threshold", 0.5))
            prob = float((result.get("model_c") or {}).get("prob", 0.5))
            label = "真新闻" if prob >= threshold else "假新闻"
            conf = prob if label == "真新闻" else 1 - prob
            shap_scores = result.get("shap_scores") or {}
            text_original = h.get("text") or ""
            
            heatmap_tag = ""
            if exp.get("heatmap_png_base64"):
                heatmap_tag = f'<img src="data:image/png;base64,{exp.get("heatmap_png_base64")}" />'
                
            token_html = ""
            tokens = exp.get("tokens", [])
            if tokens:
                max_abs = max((abs(float(t[1])) for t in tokens), default=1)
                if max_abs == 0: max_abs = 1
                for word, score in tokens:
                    ratio = min(1.0, abs(float(score)) / max_abs)
                    opacity = 0.08 + ratio * 0.62
                    bg = f"rgba(18,96,163,{opacity})"
                    token_html += f'<span class="token" style="background:{bg};">{word}</span>'
            shap_svg_html = _generate_shap_svg(
                float(shap_scores.get("text_pct", 0)),
                float(shap_scores.get("image_pct", 0))
            )

            bg_color = "rgba(18, 96, 163, 0.10)" if label == "真新闻" else "rgba(198, 40, 40, 0.10)"
            fg_color = "#1260A3" if label == "真新闻" else "#c62828"

            original_img_tag = ""
            if h.get("image_url") and h.get("image_url").startswith("data:image"):
                original_img_tag = f'<div style="margin-top:12px;"><img src="{h.get("image_url")}" style="max-height:300px; border-radius:8px; border:1px solid #e2e8f0;" /></div>'
            elif h.get("image_url"):
                original_img_tag = f'<div style="margin-top:12px;"><img src="{h.get("image_url")}" style="max-height:300px; border-radius:8px; border:1px solid #e2e8f0;" /></div>'

            reports_html.append(f"""
            <div class="report-section">
                <div class="report-top">
                  <div class="report-id">记录编号: {h.get("id","")}</div>
                  <div class="report-id">分析时间: {h.get("ts","")}</div>
                </div>
                <div class="report-banner" style="background:{bg_color};">
                  <div class="report-label" style="color:{fg_color};">{label}</div>
                  <div style="text-align:right;">
                    <div class="report-conf" style="color:{fg_color};">置信度 {conf:.1%}</div>
                    <div class="report-sub" style="font-size:12px; color:{fg_color}; margin-top:4px;">P(Real): {prob:.3f} | 阈值: {threshold:.2f}</div>
                  </div>
                </div>
                <div class="report-section-title">{inline_svg("folder", size_px=16, color="#475569")} 输入文本</div>
                <div class="report-text-block">{text_original}</div>
                {original_img_tag}
                <div class="report-section-title">{inline_svg("report", size_px=16, color="#475569")} 特征贡献度 (SHAP)</div>
                <div style="margin-top:8px;">{shap_svg_html}</div>
                <div class="report-section-title">{inline_svg("search", size_px=16, color="#475569")} 文本归因分布</div>
                <div class="report-sub" style="margin-bottom:8px;">颜色越深影响越大</div>
                <div style="line-height:1.8;">{token_html if token_html else '<div class="report-sub">无</div>'}</div>
                <div class="report-section-title">{inline_svg("image", size_px=16, color="#475569")} 注意力热力图</div>
                <div class="report-sub" style="margin-bottom:8px;">基于 Swin Transformer 视觉激活映射</div>
                <div>{heatmap_tag if heatmap_tag else '<div class="report-sub">无图片或生成失败</div>'}</div>
            </div>
            """)
        except Exception as e:
            print(f"Report generation error for {h.get('id')}: {e}")
            reports_html.append(f'<div class="report-section"><h2>报告生成失败</h2><p>{e}</p></div>')

    final_html = f"""
    <html>
      <head>
        <meta charset="utf-8"/>
        <style>
          body {{ font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial; background:#F5F7FB; padding: 24px; max-width: 900px; margin: 0 auto; }}
          .report-section {{ background:#fff; border-radius: 12px; padding: 32px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 24px; }}
          .report-top {{ display:flex; justify-content:space-between; align-items:center; background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:8px 12px; margin-bottom:16px; }}
          .report-id {{ font-size:12px; color:#64748b; }}
          .report-banner {{ display:flex; justify-content:space-between; align-items:center; border-radius:10px; padding:16px 20px; margin-bottom:20px; }}
          .report-label {{ font-size:22px; font-weight:800; letter-spacing:-0.02em; }}
          .report-conf {{ font-size:16px; font-weight:700; }}
          .report-sub {{ color:#64748b; font-size:13px; }}
          .report-section-title {{ font-size:14px; font-weight:700; color:#374151; border-bottom:1px solid #e2e8f0; padding-bottom:8px; margin:24px 0 12px; display:flex; align-items:center; gap:6px; }}
          .report-section-title:first-of-type {{ margin-top:0; }}
          .report-text-block {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:14px; color:#1e293b; font-size:14px; white-space:pre-wrap; line-height:1.6; }}
          img {{ max-width:100%; border-radius:8px; border:1px solid #e2e8f0; }}
          .token {{ display:inline-block; padding:2px 4px; margin:2px; border-radius:4px; font-family:"Noto Sans SC","PingFang SC","Microsoft YaHei",sans-serif; font-size:14px; color:#1e293b; border:1px solid rgba(0,0,0,0.05); }}
        </style>
      </head>
      <body>
        {"".join(reports_html)}
      </body>
    </html>
    """
    
    return {"ok": True, "html": final_html, "json": {"histories_count": len(histories)}}

@app.get("/api/report/{history_id}")
def api_get_report(history_id: str) -> Dict[str, Any]:
    from utils.icons import inline_svg
    case_data = None
    html = None
    try:
        h = next((x for x in _history if str(x.get("id") or "") == str(history_id)), None)
        if not h:
            raise HTTPException(status_code=404, detail="history_not_found")
    
        result = h.get("analysis_result") or {}
        exp = h.get("explanation") or {}
        threshold = float(h.get("threshold", 0.5))
        prob = float((result.get("model_c") or {}).get("prob", 0.5))
        label = "真新闻" if prob >= threshold else "假新闻"
        conf = prob if label == "真新闻" else 1 - prob
        shap_scores = result.get("shap_scores") or {}
        text_original = h.get("text") or ""
        text_used = result.get("text_used")
        text_meta = result.get("text_meta") or {}

        payload = {
            "prediction": {"prob_real": prob, "label": label, "confidence": conf, "threshold": threshold},
            "shap_scores": shap_scores,
            "text": {"original": text_original, "used": text_used, "meta": text_meta},
            "source": {"url": h.get("source_url"), "meta": h.get("source_meta")},
        }

        heatmap_tag = ""
        if exp.get("heatmap_png_base64"):
            heatmap_tag = f'<img src="data:image/png;base64,{exp.get("heatmap_png_base64")}" />'
            
        token_html = ""
        tokens = exp.get("tokens", [])
        if tokens:
            max_abs = max((abs(float(t[1])) for t in tokens), default=1)
            if max_abs == 0: max_abs = 1
            for word, score in tokens:
                ratio = min(1.0, abs(float(score)) / max_abs)
                opacity = 0.08 + ratio * 0.62
                bg = f"rgba(18,96,163,{opacity})"
                token_html += f'<span class="token" style="background:{bg};">{word}</span>'
        shap_svg_html = _generate_shap_svg(
            float(shap_scores.get("text_pct", 0)),
            float(shap_scores.get("image_pct", 0))
        )

        bg = "rgba(18, 96, 163, 0.10)" if label == "真新闻" else "rgba(198, 40, 40, 0.10)"
        fg = "#1260A3" if label == "真新闻" else "#c62828"

        original_img_tag = ""
        if h.get("image_url") and h.get("image_url").startswith("data:image"):
            original_img_tag = f'<div style="margin-top:12px;"><img src="{h.get("image_url")}" style="max-height:300px; border-radius:8px; border:1px solid #e2e8f0;" /></div>'
        elif h.get("image_url"):
            original_img_tag = f'<div style="margin-top:12px;"><img src="{h.get("image_url")}" style="max-height:300px; border-radius:8px; border:1px solid #e2e8f0;" /></div>'

        html = f"""
        <html>
          <head>
            <meta charset="utf-8"/>
            <style>
              body {{ font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial; background:#F5F7FB; padding: 24px; max-width: 900px; margin: 0 auto; }}
              .report-section {{ background:#fff; border-radius: 12px; padding: 32px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 24px; }}
              .report-top {{ display:flex; justify-content:space-between; align-items:center; background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:8px 12px; margin-bottom:16px; }}
              .report-id {{ font-size:12px; color:#64748b; }}
              .report-banner {{ display:flex; justify-content:space-between; align-items:center; border-radius:10px; padding:16px 20px; margin-bottom:20px; }}
              .report-label {{ font-size:22px; font-weight:800; letter-spacing:-0.02em; }}
              .report-conf {{ font-size:16px; font-weight:700; }}
              .report-sub {{ color:#64748b; font-size:13px; }}
              .report-section-title {{ font-size:14px; font-weight:700; color:#374151; border-bottom:1px solid #e2e8f0; padding-bottom:8px; margin:24px 0 12px; display:flex; align-items:center; gap:6px; }}
              .report-section-title:first-of-type {{ margin-top:0; }}
              .report-text-block {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:14px; color:#1e293b; font-size:14px; white-space:pre-wrap; line-height:1.6; }}
              img {{ max-width:100%; border-radius:8px; border:1px solid #e2e8f0; }}
              .token {{ display:inline-block; padding:2px 4px; margin:2px; border-radius:4px; font-family:"Noto Sans SC","PingFang SC","Microsoft YaHei",sans-serif; font-size:14px; color:#1e293b; border:1px solid rgba(0,0,0,0.05); }}
            </style>
          </head>
          <body>
            <div class="report-section">
                <div class="report-top">
                  <div class="report-id">记录编号: {h.get("id","")}</div>
                  <div class="report-id">分析时间: {h.get("ts","")}</div>
                </div>
                <div class="report-banner" style="background:{bg};">
                  <div class="report-label" style="color:{fg};">{label}</div>
                  <div style="text-align:right;">
                    <div class="report-conf" style="color:{fg};">置信度 {conf:.1%}</div>
                    <div class="report-sub" style="font-size:12px; color:{fg}; margin-top:4px;">P(Real): {prob:.3f} | 阈值: {threshold:.2f}</div>
                  </div>
                </div>
                <div class="report-section-title">{inline_svg("folder", size_px=16, color="#475569")} 输入文本</div>
                <div class="report-text-block">{text_original}</div>
                {original_img_tag}
                <div class="report-section-title">{inline_svg("report", size_px=16, color="#475569")} 特征贡献度 (SHAP)</div>
                <div style="margin-top:8px;">{shap_svg_html}</div>
                <div class="report-section-title">{inline_svg("search", size_px=16, color="#475569")} 文本归因分布</div>
                <div class="report-sub" style="margin-bottom:8px;">颜色越深影响越大</div>
                <div style="line-height:1.8;">{token_html if token_html else '<div class="report-sub">无</div>'}</div>
                <div class="report-section-title">{inline_svg("image", size_px=16, color="#475569")} 注意力热力图</div>
                <div class="report-sub" style="margin-bottom:8px;">基于 Swin Transformer 视觉激活映射</div>
                <div>{heatmap_tag if heatmap_tag else '<div class="report-sub">无图片或生成失败</div>'}</div>
            </div>
          </body>
        </html>
        """
        case_data = payload
    except HTTPException:
        raise
    except Exception as e:
        print(f"Report generation error: {e}")
        html = f"<html><body><h2>报告生成失败</h2><p>{e}</p></body></html>"

    return {"ok": True, "html": html or "", "json": case_data or {}}


@app.delete("/api/history")
def api_clear_history() -> Dict[str, Any]:
    _history.clear()
    return {"ok": True}


@app.post("/api/fetch-url")
def api_fetch_url(req: FetchUrlRequest) -> Dict[str, Any]:
    payload = fetch_url_payload(req.url.strip())
    if not payload or not payload.get("ok"):
        raise HTTPException(status_code=400, detail=(payload or {}).get("error", "fetch_failed"))
    return _strip_payload_bytes(payload)


@app.get("/api/samples/random")
def api_random_sample() -> Dict[str, Any]:
    items = MockDataLoader.load_fakeddit_sample(n=200)
    if not items:
        raise HTTPException(status_code=404, detail="no_samples")
    s = random.choice(items)
    return {"ok": True, "item": s}


@app.post("/api/translate-text")
def api_translate_text(req: TranslateTextRequest) -> Dict[str, Any]:
    detector = get_detector()
    text = req.text.strip()
    text_used, text_meta = detector._normalize_text(text, translate_zh=True)
    return {"ok": True, "text_original": text, "text_used": text_used, "text_meta": text_meta}


@app.get("/api/batch/sample")
def api_batch_sample(n: int = 50) -> Dict[str, Any]:
    n_val = max(1, min(int(n), 200))
    items = MockDataLoader.load_fakeddit_sample(n=n_val)
    return {"ok": True, "items": items}


@app.post("/api/predict")
def api_predict(req: PredictRequest) -> Dict[str, Any]:
    detector = get_detector()
    image_source = None
    if req.image_base64:
        try:
            image_source = _decode_image_base64(req.image_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"invalid_image_base64:{e}")
    elif req.image_url:
        image_source = req.image_url.strip()

    text_for_predict = req.text.strip() if req.text else "placeholder for image-only analysis"
    try:
        result = detector.predict_all(
            text_for_predict,
            image_source=image_source,
            with_shap=bool(req.with_shap),
            translate_zh=bool(req.translate_zh),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"predict_failed:{e}")

    explanation: Dict[str, Any] = {}
    if req.with_explain:
        try:
            explain_text = str((result or {}).get("text_used") or req.text)
            heatmap, tokens = detector.explain(
                explain_text,
                image_source=image_source,
                translate_zh=False,
            )
            if heatmap is not None:
                explanation["heatmap_png_base64"] = _to_png_base64(heatmap)
            explanation["tokens"] = tokens or []
        except Exception as e:
            explanation["error"] = str(e)

    prob = float((result.get("model_c") or {}).get("prob", 0.5))
    label = "真新闻" if prob >= float(req.threshold) else "假新闻"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    saved_image_url = None
    if req.image_url:
        saved_image_url = req.image_url.strip()
    elif req.image_base64:
        saved_image_url = req.image_base64
    _history.append(
        {
            "id": str(uuid.uuid4()),
            "ts": now,
            "label": label,
            "prob_real": prob,
            "text": req.text,
            "image_url": saved_image_url,
            "analysis_result": result,
            "explanation": explanation,
            "threshold": float(req.threshold),
            "source_url": req.source_url,
            "source_meta": req.source_meta,
        }
    )
    if len(_history) > 500:
        del _history[0 : len(_history) - 500]
    _last_context.clear()
    _last_context.update(
        {
            "analysis_result": result,
            "explanation": explanation,
            "threshold": float(req.threshold),
            "text": req.text,
            "source_url": req.source_url,
            "source_meta": req.source_meta,
        }
    )

    return {"ok": True, "result": result, "explanation": explanation}


@app.get("/api/cases")
def api_list_cases() -> Dict[str, Any]:
    store = _get_store()
    return {"ok": True, "items": store.list_cases()}


@app.get("/api/cases/{case_id}")
def api_get_case(case_id: str) -> Dict[str, Any]:
    store = _get_store()
    case = store.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="case_not_found")
    return {"ok": True, "item": case}


@app.get("/api/cases/{case_id}/detail")
def api_case_detail(case_id: str) -> Dict[str, Any]:
    store = _get_store()
    case = store.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="case_not_found")

    inp = case.get("input") or {}
    art = case.get("artifacts") or {}
    case_dir = str(art.get("case_dir") or "")
    attach_dir = Path(case_dir) / "attachments" if case_dir else None
    attachments = []
    if attach_dir and attach_dir.exists():
        for p in sorted(attach_dir.glob("*")):
            if p.is_file():
                attachments.append({"name": p.name, "size": p.stat().st_size})

    return {
        "ok": True,
        "item": case,
        "input_image_base64": _file_to_base64(str(inp.get("image_path") or "")),
        "heatmap_base64": _file_to_base64(str(art.get("heatmap_path") or "")),
        "attachments": attachments,
    }


@app.post("/api/cases")
def api_create_case(req: CreateCaseRequest) -> Dict[str, Any]:
    store = _get_store()
    image_source = None
    if req.image_base64:
        try:
            image_source = _decode_image_base64(req.image_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"invalid_image_base64:{e}")
    elif req.image_url:
        image_source = req.image_url

    explanation_result = None
    if isinstance(req.explanation, dict):
        explanation_result = {}
        if req.explanation.get("heatmap_png_base64"):
            try:
                explanation_result["heatmap_swin"] = _from_png_base64(req.explanation.get("heatmap_png_base64"))
            except Exception:
                explanation_result["heatmap_swin"] = None
        if "tokens" in req.explanation:
            explanation_result["tokens"] = req.explanation.get("tokens") or []

    try:
        created = store.create_case_from_analysis(
            text=req.text,
            text_used=req.text_used,
            image_source=image_source,
            analysis_result=req.analysis_result,
            explanation_result=explanation_result,
            threshold=float(req.threshold),
            source=req.source,
            source_url=req.source_url,
            source_meta=req.source_meta,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"create_case_failed:{e}")

    return {"ok": True, "item": created}


@app.put("/api/cases/{case_id}")
def api_update_case(case_id: str, req: UpdateCaseRequest) -> Dict[str, Any]:
    store = _get_store()
    case = store.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="case_not_found")
    body = req.model_dump(exclude_none=True)
    for k, v in body.items():
        if k == "tags" and isinstance(v, list):
            case[k] = [str(x).strip() for x in v if str(x).strip()]
        else:
            case[k] = v
    case["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    store.upsert_case(case)
    return {"ok": True, "item": case}


@app.get("/api/cases/{case_id}/export")
def api_export_case(case_id: str):
    store = _get_store()
    data = store.export_case_package(case_id)
    if not data:
        raise HTTPException(status_code=404, detail="case_not_found")
    return Response(
        content=data,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="case_{case_id[:10]}.zip"'},
    )


@app.post("/api/cases/{case_id}/attachments")
def api_case_add_attachment(case_id: str, file: UploadFile = File(...)) -> Dict[str, Any]:
    store = _get_store()
    case = store.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="case_not_found")
    data = file.file.read()
    buf = io.BytesIO(data)
    setattr(buf, "name", file.filename or "attachment.bin")
    saved = store.add_attachment(case_id, buf)
    if not saved:
        raise HTTPException(status_code=500, detail="attachment_save_failed")
    return {"ok": True, "path": saved}


@app.delete("/api/cases/{case_id}/attachments/{name}")
def api_case_delete_attachment(case_id: str, name: str):
    store = _get_store()
    case = store.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="case_not_found")
    art = case.get("artifacts") or {}
    case_dir = str(art.get("case_dir") or "")
    if not case_dir:
        raise HTTPException(status_code=404, detail="case_dir_not_found")
    safe = "".join(ch for ch in str(name) if ch.isalnum() or ch in ("-", "_", ".", " "))
    if not safe:
        raise HTTPException(status_code=400, detail="invalid_attachment_name")
    target = Path(case_dir) / "attachments" / safe
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="attachment_not_found")
    target.unlink()
    return {"ok": True}


@app.get("/api/cases/{case_id}/attachments/{name}")
def api_case_download_attachment(case_id: str, name: str):
    store = _get_store()
    case = store.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="case_not_found")
    art = case.get("artifacts") or {}
    case_dir = str(art.get("case_dir") or "")
    if not case_dir:
        raise HTTPException(status_code=404, detail="case_dir_not_found")
    safe = "".join(ch for ch in str(name) if ch.isalnum() or ch in ("-", "_", ".", " "))
    if not safe:
        raise HTTPException(status_code=400, detail="invalid_attachment_name")
    target = Path(case_dir) / "attachments" / safe
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="attachment_not_found")
    data = target.read_bytes()
    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{safe}"'},
    )


@app.post("/api/batch/run")
def api_batch_run(req: BatchRequest) -> Dict[str, Any]:
    detector = get_detector()
    rows = req.rows or []
    n = max(1, min(int(req.max_rows), len(rows)))
    out_rows = []
    for i in range(n):
        row = rows[i] or {}
        text = str(row.get(req.text_col, ""))
        img = None
        if req.use_images and req.image_col:
            img = row.get(req.image_col)
            if img is not None:
                img = str(img)
        try:
            r = detector.predict_all(text, img, with_shap=False, translate_zh=bool(req.translate_zh))
            prob = float(r["model_c"]["prob"])
        except Exception:
            prob = 0.5
        pred = "真新闻" if prob >= float(req.threshold) else "假新闻"
        out_rows.append({"row_index": i, "prob_real": prob, "prediction": pred, "text": text[:300]})
    return {"ok": True, "items": out_rows}


@app.post("/api/analytics/benchmark")
def api_analytics_benchmark(req: AnalyticsBenchmarkRequest) -> Dict[str, Any]:
    detector = get_detector()
    random.seed(int(req.seed))
    n_samples = max(1, min(int(req.n_samples or 50), 200))

    samples = []
    if req.custom_data and req.custom_data_type:
        try:
            text_col = req.text_col or "content"
            image_col = req.image_col or "image_url"
            label_col = req.label_col or "label"
            if req.custom_data_type in ("csv", "tsv"):
                import csv
                import io as csv_io
                delimiter = "\t" if req.custom_data_type == "tsv" else ","
                reader = csv.DictReader(csv_io.StringIO(req.custom_data), delimiter=delimiter)
                for row in reader:
                    label_val = row.get(label_col) or row.get("label") or row.get("2_way_label") or row.get("2_way_label_parsed")
                    gt = int(label_val) if label_val is not None else None
                    text_val = row.get(text_col) or row.get("content") or row.get("clean_title") or ""
                    img_val = row.get(image_col) or row.get("image_url") or ""
                    if gt is not None and text_val:
                        samples.append({
                            "clean_title": text_val,
                            "image_url": img_val,
                            "label": gt,
                        })
            elif req.custom_data_type == "json":
                import json
                items = json.loads(req.custom_data)
                if isinstance(items, list):
                    for row in items:
                        label_val = row.get(label_col) or row.get("label") or row.get("2_way_label") or row.get("2_way_label_parsed")
                        gt = int(label_val) if label_val is not None else None
                        text_val = row.get(text_col) or row.get("content") or row.get("clean_title") or ""
                        img_val = row.get(image_col) or row.get("image_url") or ""
                        if gt is not None and text_val:
                            samples.append({
                                "clean_title": text_val,
                                "image_url": img_val,
                                "label": gt,
                            })
        except Exception as e:
            return {"ok": True, "metrics": {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}, "confusion": [[0, 0], [0, 0]], "errors": [], "error": str(e)}
        if len(samples) > n_samples:
            samples = random.sample(samples, n_samples)
    else:
        samples = MockDataLoader.load_fakeddit_sample(n=n_samples)

    if not samples:
        return {"ok": True, "metrics": {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}, "confusion": [[0, 0], [0, 0]], "errors": []}

    y_true = []
    y_pred = []
    errors = []
    for i, s in enumerate(samples):
        text = str(s.get("clean_title", ""))
        img = s.get("image_url") if req.include_images else None
        gt = int(s.get("label", 0))
        try:
            r = detector.predict_all(text, img, with_shap=False, translate_zh=bool(req.translate_zh))
            prob = float(r["model_c"]["prob"])
        except Exception:
            prob = 0.5
        pred = 1 if prob > 0.5 else 0
        y_true.append(gt)
        y_pred.append(pred)
        if pred != gt:
            errors.append(
                {
                    "index": i,
                    "actual": "真新闻" if gt == 1 else "假新闻",
                    "predicted": "真新闻" if pred == 1 else "假新闻",
                    "prob_real": prob,
                    "text": text[:160],
                    "image_url": img if img else None,
                }
            )

    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    tn = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 0)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
    acc = (tp + tn) / max(1, len(y_true))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, (prec + rec))
    errors = sorted(errors, key=lambda x: abs(float(x["prob_real"]) - 0.5), reverse=True)[:50]

    return {
        "ok": True,
        "metrics": {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1},
        "confusion": [[tn, fp], [fn, tp]],
        "errors": errors,
    }


import math

def _generate_shap_svg(text_pct: float, image_pct: float) -> str:
    shapTextPct = min(1.0, max(0.0, float(text_pct)))
    shapImagePct = min(1.0, max(0.0, float(image_pct)))
    shapChartTextPct = min(0.999, max(0.001, shapTextPct))
    shapChartImagePct = 1.0 - shapChartTextPct
    
    shapCx = 160
    shapCy = 115
    shapRadius = 74
    shapStartDeg = -90
    shapSplitDeg = shapStartDeg + shapChartTextPct * 360
    shapTextMidDeg = shapStartDeg + shapChartTextPct * 180
    shapImageMidDeg = shapSplitDeg + shapChartImagePct * 180
    
    def point_at(deg, r):
        rad = (deg * math.pi) / 180
        return {"x": shapCx + math.cos(rad) * r, "y": shapCy + math.sin(rad) * r}
        
    shapStartPt = point_at(shapStartDeg, shapRadius)
    shapSplitPt = point_at(shapSplitDeg, shapRadius)
    shapTextInnerPt = point_at(shapTextMidDeg, shapRadius)
    shapImageInnerPt = point_at(shapImageMidDeg, shapRadius)
    shapTextMidPt = point_at(shapTextMidDeg, shapRadius + 15)
    shapImageMidPt = point_at(shapImageMidDeg, shapRadius + 15)
    
    shapTextOuterPt_x = shapTextMidPt["x"] + (20 if shapTextMidPt["x"] >= shapCx else -20)
    shapTextOuterPt_y = shapTextMidPt["y"]
    shapImageOuterPt_x = shapImageMidPt["x"] + (20 if shapImageMidPt["x"] >= shapCx else -20)
    shapImageOuterPt_y = shapImageMidPt["y"]
    
    shapTextLabelX = shapTextOuterPt_x + (6 if shapTextOuterPt_x >= shapCx else -6)
    shapImageLabelX = shapImageOuterPt_x + (6 if shapImageOuterPt_x >= shapCx else -6)
    
    shapTextLabelAnchor = "start" if shapTextOuterPt_x >= shapCx else "end"
    shapImageLabelAnchor = "start" if shapImageOuterPt_x >= shapCx else "end"
    
    shapTextLargeArc = 1 if shapChartTextPct > 0.5 else 0
    shapImageLargeArc = 1 if shapChartImagePct > 0.5 else 0
    
    return f"""
    <div style="display:flex; flex-direction:column; align-items:center;">
      <svg viewBox="0 0 320 230" aria-label="shap-pie-chart" style="width:100%; max-width:320px; height:auto; overflow:visible;">
        <defs>
          <filter id="pieShadow" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="0" dy="4" stdDeviation="6" flood-color="#0f172a" flood-opacity="0.15" />
          </filter>
        </defs>
        <g filter="url(#pieShadow)">
          <path d="M {shapCx} {shapCy} L {shapStartPt['x']} {shapStartPt['y']} A {shapRadius} {shapRadius} 0 {shapTextLargeArc} 1 {shapSplitPt['x']} {shapSplitPt['y']} Z" fill="#1260A3"></path>
          <path d="M {shapCx} {shapCy} L {shapSplitPt['x']} {shapSplitPt['y']} A {shapRadius} {shapRadius} 0 {shapImageLargeArc} 1 {shapStartPt['x']} {shapStartPt['y']} Z" fill="#94a3b8"></path>
        </g>
        <polyline points="{shapTextInnerPt['x']},{shapTextInnerPt['y']} {shapTextMidPt['x']},{shapTextMidPt['y']} {shapTextOuterPt_x},{shapTextOuterPt_y}" stroke="#1260A3" stroke-width="1.5" fill="none"></polyline>
        <polyline points="{shapImageInnerPt['x']},{shapImageInnerPt['y']} {shapImageMidPt['x']},{shapImageMidPt['y']} {shapImageOuterPt_x},{shapImageOuterPt_y}" stroke="#94a3b8" stroke-width="1.5" fill="none"></polyline>
        <text x="{shapTextLabelX}" y="{shapTextOuterPt_y + 4}" text-anchor="{shapTextLabelAnchor}" fill="#1260A3" font-size="13px" font-weight="700" font-family="ui-sans-serif, system-ui, sans-serif">{(shapTextPct * 100):.1f}%</text>
        <text x="{shapImageLabelX}" y="{shapImageOuterPt_y + 4}" text-anchor="{shapImageLabelAnchor}" fill="#64748b" font-size="13px" font-weight="700" font-family="ui-sans-serif, system-ui, sans-serif">{(shapImagePct * 100):.1f}%</text>
      </svg>
      <div style="display:flex; gap:20px; align-items:center; justify-content:center; font-size:13px; font-weight:500; color:#475569; margin-top:-10px;">
        <div style="display:inline-flex; align-items:center; gap:8px;">
          <span style="width:12px; height:12px; border-radius:4px; background:#1260A3; box-shadow:0 1px 2px rgba(0,0,0,0.1);"></span>
          <span>文本</span>
        </div>
        <div style="display:inline-flex; align-items:center; gap:8px;">
          <span style="width:12px; height:12px; border-radius:4px; background:#94a3b8; box-shadow:0 1px 2px rgba(0,0,0,0.1);"></span>
          <span>视觉</span>
        </div>
      </div>
    </div>
    """




@app.get("/api/modules")
def api_modules() -> Dict[str, Any]:
    return {
        "ok": True,
        "items": [
            {"key": "detection", "label": "检测"},
            {"key": "cases", "label": "案例库"},
            {"key": "history", "label": "历史"},
            {"key": "report", "label": "报告"},
            {"key": "settings", "label": "设置"},
            {"key": "batch", "label": "批量处理"},
            {"key": "analytics", "label": "模型效能分析"},
        ],
    }
