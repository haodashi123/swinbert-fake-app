import json
import os
import shutil
import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional
import zipfile
from utils.icons import inline_svg


class CaseStore:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.cases_dir = self.base_dir / "cases"
        self.cases_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.base_dir / "cases.json"
        if not self.index_path.exists():
            self._write_index([])

    @staticmethod
    def default():
        app_root = Path(__file__).resolve().parents[1]
        return CaseStore(app_root / "outputs" / "cases")

    def _read_index(self) -> List[Dict[str, Any]]:
        try:
            raw = self.index_path.read_text(encoding="utf-8")
            data = json.loads(raw) if raw.strip() else []
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _write_index(self, cases: List[Dict[str, Any]]):
        tmp = self.index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.index_path)

    def list_cases(self) -> List[Dict[str, Any]]:
        items = self._read_index()
        def key(x: Dict[str, Any]):
            return x.get("updated_at") or x.get("created_at") or ""
        return sorted(items, key=key, reverse=True)

    def get_case(self, case_id: str) -> Optional[Dict[str, Any]]:
        for c in self._read_index():
            if c.get("id") == case_id:
                return c
        return None

    def upsert_case(self, case: Dict[str, Any]):
        items = self._read_index()
        out = []
        found = False
        for c in items:
            if c.get("id") == case.get("id"):
                out.append(case)
                found = True
            else:
                out.append(c)
        if not found:
            out.append(case)
        self._write_index(out)

    def _case_dir(self, case_id: str) -> Path:
        d = self.cases_dir / case_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _save_uploaded_file(self, target_dir: Path, uploaded_file) -> Optional[str]:
        try:
            name = getattr(uploaded_file, "name", None) or "upload"
            safe = "".join(ch for ch in name if ch.isalnum() or ch in "._- ")
            ext_from_name = ""
            if "." in safe:
                ext_candidate = safe.rsplit(".", 1)[1].lower()
                if ext_candidate in {"jpg", "jpeg", "png", "gif", "webp"}:
                    ext_from_name = f".{ext_candidate}"
                    safe = safe.rsplit(".", 1)[0]

            if hasattr(uploaded_file, "seek"):
                uploaded_file.seek(0)
            if hasattr(uploaded_file, "getvalue"):
                content = uploaded_file.getvalue()
            else:
                content = uploaded_file.read()

            if not ext_from_name:
                ext_from_name = self._guess_extension(content)
            path = target_dir / f"{safe}{ext_from_name}"
            path.write_bytes(content)
            return str(path)
        except Exception:
            return None

    def _save_heatmap(self, target_dir: Path, heatmap_img) -> Optional[str]:
        try:
            path = target_dir / "heatmap.png"
            heatmap_img.save(path, format="PNG")
            return str(path)
        except Exception:
            return None

    def _fetch_url_image(self, url: str) -> Optional[bytes]:
        try:
            import requests
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return resp.content
        except Exception:
            return None

    def _guess_extension(self, content: bytes) -> str:
        if content.startswith(b'\xff\xd8\xff'):
            return ".jpg"
        elif content.startswith(b'\x89PNG'):
            return ".png"
        elif content.startswith(b'GIF8'):
            return ".gif"
        elif content.startswith(b'RIFF') and content[8:12] == b'WEBP':
            return ".webp"
        return ".bin"

    def create_case_from_analysis(
        self,
        text: str,
        image_source,
        analysis_result: Dict[str, Any],
        explanation_result: Optional[Dict[str, Any]],
        threshold: float,
        source: str,
        source_url: Optional[str] = None,
        source_meta: Optional[Dict[str, Any]] = None,
        text_used: Optional[str] = None,
    ) -> Dict[str, Any]:
        case_id = uuid.uuid4().hex
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        d = self._case_dir(case_id)
        attachments_dir = d / "attachments"
        attachments_dir.mkdir(parents=True, exist_ok=True)

        input_image_path = None
        input_image_url = None
        if isinstance(image_source, str):
            if image_source.startswith("http"):
                input_image_url = image_source
                img_bytes = self._fetch_url_image(image_source)
                if img_bytes:
                    ext = self._guess_extension(img_bytes)
                    dst = d / f"input_image{ext}"
                    dst.write_bytes(img_bytes)
                    input_image_path = str(dst)
            else:
                try:
                    src_path = Path(image_source)
                    if src_path.exists():
                        dst = d / f"input_image{src_path.suffix or '.png'}"
                        shutil.copyfile(src_path, dst)
                        input_image_path = str(dst)
                    else:
                        input_image_url = image_source
                except Exception:
                    input_image_url = image_source
        elif image_source is not None:
            saved = self._save_uploaded_file(d, image_source)
            if saved:
                input_image_path = saved
            else:
                img_bytes = getattr(image_source, "getvalue", lambda: None)()
                if img_bytes:
                    ext = self._guess_extension(img_bytes)
                    dst = d / f"input_image{ext}"
                    dst.write_bytes(img_bytes)
                    input_image_path = str(dst)

        prob_real = float(analysis_result.get("model_c", {}).get("prob", 0.5))
        pred_label = "真新闻" if prob_real >= float(threshold) else "假新闻"
        shap_scores = analysis_result.get("shap_scores")
        text_used = text_used if text_used is not None else analysis_result.get("text_used")
        text_meta = analysis_result.get("text_meta")

        heatmap_path = None
        if explanation_result and explanation_result.get("heatmap_swin") is not None:
            heatmap_path = self._save_heatmap(d, explanation_result["heatmap_swin"])

        case = {
            "id": case_id,
            "created_at": now,
            "updated_at": now,
            "status": "待处理",
            "decision": "未标记",
            "tags": [],
            "notes": "",
            "source": source,
            "source_url": source_url,
            "source_meta": source_meta,
            "input": {
                "text": text,
                "text_used": text_used,
                "text_meta": text_meta,
                "image_path": input_image_path,
                "image_url": input_image_url,
            },
            "prediction": {
                "prob_real": prob_real,
                "label": pred_label,
                "threshold": float(threshold),
            },
            "shap_scores": shap_scores,
            "artifacts": {
                "heatmap_path": heatmap_path,
                "case_dir": str(d),
            },
        }

        self.upsert_case(case)
        return case

    def add_attachment(self, case_id: str, uploaded_file) -> Optional[str]:
        d = self._case_dir(case_id) / "attachments"
        d.mkdir(parents=True, exist_ok=True)
        return self._save_uploaded_file(d, uploaded_file)

    def export_case_package(self, case_id: str) -> Optional[bytes]:
        case = self.get_case(case_id)
        if not case:
            return None

        d = self._case_dir(case_id)
        heatmap_path = (case.get("artifacts") or {}).get("heatmap_path")
        image_path = (case.get("input") or {}).get("image_path")

        html = self._build_case_report_html(case)
        json_bytes = json.dumps(case, ensure_ascii=False, indent=2).encode("utf-8")

        buf = BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("case.json", json_bytes)
            z.writestr("report.html", html.encode("utf-8"))
            if image_path and Path(image_path).exists():
                z.write(image_path, arcname=f"artifacts/{Path(image_path).name}")

            att_dir = d / "attachments"
            if att_dir.exists():
                for p in att_dir.glob("*"):
                    if p.is_file():
                        z.write(str(p), arcname=f"attachments/{p.name}")

        return buf.getvalue()

    def _build_case_report_html(self, case: Dict[str, Any]) -> str:
        pred = case.get("prediction") or {}
        label = pred.get("label", "")
        prob = float(pred.get("prob_real", 0.5))
        threshold = float(pred.get("threshold", 0.5))
        conf = prob if label == "真新闻" else 1 - prob

        bg = "rgba(18, 96, 163, 0.05)" if label == "真新闻" else "rgba(198, 40, 40, 0.05)"
        fg = "#1e40af" if label == "真新闻" else "#b91c1c"
        border_color = "rgba(18, 96, 163, 0.2)" if label == "真新闻" else "rgba(198, 40, 40, 0.2)"
        icon_svg = inline_svg("check", size_px=18, color="#475569") if label == "真新闻" else inline_svg("warning", size_px=18, color="#475569")

        shap = case.get("shap_scores") or {}
        text_pct = float(shap.get("text_pct", 0.0) or 0.0)
        image_pct = float(shap.get("image_pct", 0.0) or 0.0)

        heatmap_path = ((case.get("artifacts") or {}).get("heatmap_path")) or ""
        heatmap_tag = ""
        if heatmap_path and Path(heatmap_path).exists():
            b64 = self._file_to_base64_png(heatmap_path)
            heatmap_tag = f'<img src="data:image/png;base64,{b64}" />'
        heatmap_html = heatmap_tag if heatmap_tag else "<div class='sub'>无</div>"

        text = (case.get("input") or {}).get("text", "")
        safe_text = (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        text_used = (case.get("input") or {}).get("text_used", "")
        safe_text_used = (text_used or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        image_path = (case.get("input") or {}).get("image_path")
        image_html = ""
        if image_path and Path(image_path).exists():
            img_b64 = self._file_to_base64_png(image_path)
            image_html = f'<div class="data-item"><div class="data-label">附加图像</div><div style="margin-top:8px;"><img src="data:image/png;base64,{img_b64}" /></div></div>'

        decision = case.get("decision", "未标记")
        notes = case.get("notes", "")
        tags = ", ".join(case.get("tags", []))

        return f"""
        <html>
          <head>
            <meta charset="utf-8"/>
            <style>
              body {{ font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial; background:#F5F7FB; padding: 32px; max-width: 900px; margin: 0 auto; }}
              .card {{ background:#fff; border-radius: 12px; padding: 24px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 20px; }}
              .header-card {{ background: linear-gradient(135deg, #1260A3 0%, #0F4F87 100%); color: white; padding: 32px 24px; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(18,96,163,0.2); }}
              .header-title {{ font-size: 24px; font-weight: 800; margin-bottom: 12px; }}
              .header-sub {{ color: rgba(255,255,255,0.8); font-size: 14px; margin-bottom: 4px; }}
              .title {{ font-size: 18px; font-weight: 700; color:#0F172A; border-bottom: 1px solid #E2E8F0; padding-bottom: 12px; margin-bottom: 16px; display: flex; align-items: center; gap: 8px; }}
              .sub {{ color:#475569; line-height: 1.6; font-size: 15px; }}
              .banner {{ display:flex; justify-content:space-between; align-items:center; border-radius:12px; padding: 20px; border:1px solid {border_color}; background:{bg}; }}
              .label {{ font-size: 32px; font-weight: 700; color:{fg}; letter-spacing: -0.01em; }}
              .prob {{ font-size: 18px; font-weight: 600; color:{fg}; margin-top: 4px; }}
              .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
              .data-item {{ margin-bottom: 16px; }}
              .data-label {{ font-size: 13px; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }}
              .data-value {{ font-size: 15px; color: #1e293b; font-weight: 500; }}
              .decision-badge {{ display: inline-block; padding: 6px 16px; border-radius: 999px; font-weight: 700; font-size: 16px; }}
              .decision-真新闻 {{ background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }}
              .decision-假新闻 {{ background: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }}
              .decision-不确定 {{ background: #fef9c3; color: #854d0e; border: 1px solid #fef08a; }}
              .decision-未标记 {{ background: #f1f5f9; color: #475569; border: 1px solid #e2e8f0; }}
              img {{ max-width:100%; border-radius: 8px; border:1px solid #e2e8f0; }}
            </style>
          </head>
          <body>
            <div class="header-card">
              <div class="header-title">信息内容审计报告</div>
              <div class="header-sub">案例编号: {case.get("id","")}</div>
              <div class="header-sub">生成时间: {case.get("created_at","")}</div>
            </div>

            <div class="card">
              <div class="title">{inline_svg("pencil", size_px=18, color="#475569")} 综合审核结论</div>
              <div class="grid-2">
                <div>
                  <div class="data-item">
                    <div class="data-label">人工判定结论</div>
                    <div class="decision-badge decision-{decision}">{decision}</div>
                  </div>
                  <div class="data-item">
                    <div class="data-label">案例标签</div>
                    <div class="data-value">{tags if tags else "—"}</div>
                  </div>
                </div>
                <div>
                  <div class="data-item" style="height: 100%;">
                    <div class="data-label">审核备注说明</div>
                    <div class="data-value" style="background: #f8fafc; padding: 12px; border-radius: 8px; height: calc(100% - 28px); white-space: pre-wrap;">{notes if notes else "无备注"}</div>
                  </div>
                </div>
              </div>
            </div>

            <div class="card">
              <div class="title">{inline_svg("shield", size_px=18, color="#475569")} AI 辅助预测</div>
              <div class="banner">
                <div style="display:flex; align-items:center; gap:16px;">
                  <div style="background: white; border-radius: 50%; padding: 8px; display: flex; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">{icon_svg}</div>
                  <div class="label">{label}</div>
                </div>
                <div style="text-align:right;">
                  <div class="prob">置信度: {conf:.1%}</div>
                  <div class="sub" style="font-size: 13px; margin-top: 8px;">模型得分 P(Real): {prob:.3f} <br/> 判定阈值: {threshold:.2f}</div>
                </div>
              </div>
            </div>

            <div class="card">
              <div class="title">{inline_svg("folder", size_px=18, color="#475569")} 审计内容</div>
              <div class="data-item">
                <div class="data-label">检测文本 (原文)</div>
                <div class="data-value" style="background: #f8fafc; padding: 16px; border-radius: 8px; white-space:pre-wrap; line-height: 1.6; border: 1px solid #e2e8f0;">{safe_text}</div>
              </div>
              {f'''<div class="data-item">
                <div class="data-label">检测文本 (英文翻译)</div>
                <div class="data-value" style="background: #f8fafc; padding: 16px; border-radius: 8px; white-space:pre-wrap; line-height: 1.6; border: 1px solid #e2e8f0; color: #475569;">{safe_text_used}</div>
              </div>''' if safe_text_used and text_used != text else ''}
              {image_html}
            </div>
          </body>
        </html>
        """

    def _file_to_base64_png(self, path: str) -> str:
        data = Path(path).read_bytes()
        import base64
        return base64.b64encode(data).decode("utf-8")
