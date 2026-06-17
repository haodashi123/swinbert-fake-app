import { useEffect, useMemo, useRef, useState } from "react"

const API_BASE = String(import.meta.env.VITE_API_BASE || "").trim().replace(/\/+$/, "")
const apiUrl = (path) => API_BASE ? `${API_BASE}${path}` : path
const iconPath = (name) => `/icons/${name}.svg`

function toBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => {
      const out = String(reader.result || "")
      resolve(out)
    }
    reader.onerror = reject
    reader.readAsDataURL(file)
  })
}

async function postJson(path, body) {
  try {
    const resp = await fetch(apiUrl(path), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    })
    const data = await resp.json()
    if (!resp.ok) {
      throw new Error(data?.detail || "请求失败")
    }
    return data
  } catch (e) {
    if (String(e?.message || "").includes("Failed to fetch")) {
      throw new Error("网络请求失败：请确认后端 API 已启动，并检查 VITE_API_BASE 配置或代理设置")
    }
    throw e
  }
}

async function getJson(path) {
  try {
    const resp = await fetch(apiUrl(path))
    const data = await resp.json()
    if (!resp.ok) {
      throw new Error(data?.detail || "请求失败")
    }
    return data
  } catch (e) {
    if (String(e?.message || "").includes("Failed to fetch")) {
      throw new Error("网络请求失败：请确认后端 API 已启动，并检查 VITE_API_BASE 配置或代理设置")
    }
    throw e
  }
}

async function putJson(path, body) {
  try {
    const resp = await fetch(apiUrl(path), {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    })
    const data = await resp.json()
    if (!resp.ok) {
      throw new Error(data?.detail || "请求失败")
    }
    return data
  } catch (e) {
    if (String(e?.message || "").includes("Failed to fetch")) {
      throw new Error("网络请求失败：请确认后端 API 已启动，并检查 VITE_API_BASE 配置或代理设置")
    }
    throw e
  }
}

async function postForm(path, formData) {
  try {
    const resp = await fetch(apiUrl(path), {
      method: "POST",
      body: formData
    })
    const data = await resp.json()
    if (!resp.ok) {
      throw new Error(data?.detail || "请求失败")
    }
    return data
  } catch (e) {
    if (String(e?.message || "").includes("Failed to fetch")) {
      throw new Error("网络请求失败：请确认后端 API 已启动，并检查 VITE_API_BASE 配置或代理设置")
    }
    throw e
  }
}

async function deleteJson(path) {
  try {
    const resp = await fetch(apiUrl(path), { method: "DELETE" })
    const data = await resp.json()
    if (!resp.ok) {
      throw new Error(data?.detail || "请求失败")
    }
    return data
  } catch (e) {
    if (String(e?.message || "").includes("Failed to fetch")) {
      throw new Error("网络请求失败：请确认后端 API 已启动，并检查 VITE_API_BASE 配置或代理设置")
    }
    throw e
  }
}

const DEFAULT_SETTINGS = {
  threshold: 0.5,
  enable_shap: false,
  enable_image_heatmap: false,
  enable_text_heatmap: false,
  enable_zh_translation: true
}

function normalizeSettings(raw) {
  const next = { ...DEFAULT_SETTINGS, ...(raw || {}) }
  next.threshold = Number(next.threshold || 0.5)
  next.enable_shap = Boolean(next.enable_shap)
  next.enable_image_heatmap = Boolean(next.enable_image_heatmap)
  next.enable_text_heatmap = Boolean(next.enable_text_heatmap)
  next.enable_zh_translation = Boolean(next.enable_zh_translation)
  return next
}

function parseDelimitedText(content, sep) {
  const lines = String(content || "").split(/\r?\n/).filter((x) => x.trim().length > 0)
  if (lines.length === 0) return []
  const headers = lines[0].split(sep).map((x) => x.trim())
  const rows = []
  for (let i = 1; i < lines.length; i += 1) {
    const cols = lines[i].split(sep)
    const row = {}
    headers.forEach((h, idx) => {
      row[h] = (cols[idx] ?? "").trim()
    })
    rows.push(row)
  }
  return rows
}

function downloadBlob(content, fileName, mime) {
  const blob = new Blob([content], { type: mime })
  const url = URL.createObjectURL(blob)
  const a = document.createElement("a")
  a.href = url
  a.download = fileName
  a.click()
  URL.revokeObjectURL(url)
}

export default function App() {
  const caseStatusOptions = ["待处理", "已结案"]
  const caseDecisionOptions = ["真新闻", "假新闻", "不确定"]
  const [page, setPage] = useState("detection")
  const [url, setUrl] = useState("")
  const [fillMode, setFillMode] = useState("仅标题")
  const [text, setText] = useState("")
  const [images, setImages] = useState([])
  const [selectedImageUrl, setSelectedImageUrl] = useState("")
  const [imageFile, setImageFile] = useState(null)
  const [loadingFetch, setLoadingFetch] = useState(false)
  const [loadingPredict, setLoadingPredict] = useState(false)
  const [predictResult, setPredictResult] = useState(null)
  const [explainResult, setExplainResult] = useState(null)
  const [caseSaved, setCaseSaved] = useState(false)
  const [errorMsg, setErrorMsg] = useState("")
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [settings, setSettings] = useState(DEFAULT_SETTINGS)
  const [cases, setCases] = useState([])
  const [caseQuery, setCaseQuery] = useState("")
  const [caseStatusFilters, setCaseStatusFilters] = useState([])
  const [caseDecisionFilters, setCaseDecisionFilters] = useState([])
  const [caseSelectedId, setCaseSelectedId] = useState("")
  const [caseDetail, setCaseDetail] = useState(null)
  const [caseAttachmentFile, setCaseAttachmentFile] = useState(null)
  const [casePage, setCasePage] = useState(1)
  const CASE_PAGE_SIZE = 10
  const [caseSaving, setCaseSaving] = useState(false)
  const [caseDraft, setCaseDraft] = useState(null)
  const [historyItems, setHistoryItems] = useState([])
  const [historySelectedIdx, setHistorySelectedIdx] = useState(0)
  const [previewImageUrl, setPreviewImageUrl] = useState("")
  const [reportData, setReportData] = useState(null)
  const [reportHistoryItems, setReportHistoryItems] = useState([])
  const [reportSelectedIds, setReportSelectedIds] = useState([])
  const [reportPreviewMap, setReportPreviewMap] = useState({})
  const [reportSelectMode, setReportSelectMode] = useState(false)
  const [batchRows, setBatchRows] = useState([])
  const [batchTextCol, setBatchTextCol] = useState("")
  const [batchImageCol, setBatchImageCol] = useState("")
  const [batchUseImages, setBatchUseImages] = useState(true)
  const [batchResults, setBatchResults] = useState([])
  const [batchSampleCount, setBatchSampleCount] = useState(50)
  const [batchMaxRows, setBatchMaxRows] = useState(0)
  const [batchProcessCount, setBatchProcessCount] = useState(0)
  const [batchCustomFile, setBatchCustomFile] = useState(null)
  const [analyticsCfg, setAnalyticsCfg] = useState({ n_samples: 50, seed: 2026, include_images: true })
  const [analyticsData, setAnalyticsData] = useState(null)
  const [analyticsCustomFile, setAnalyticsCustomFile] = useState(null)
  const [analyticsTotalRows, setAnalyticsTotalRows] = useState(0)
  const [analyticsCols, setAnalyticsCols] = useState({ textCol: "", imageCol: "", labelCol: "" })
  const [analyticsRows, setAnalyticsRows] = useState([])
  const [loadingPanel, setLoadingPanel] = useState(false)
  const [toastMsg, setToastMsg] = useState("")
  const [translatedText, setTranslatedText] = useState("")
  const [translationLoading, setTranslationLoading] = useState(false)
  const [customSampleFile, setCustomSampleFile] = useState(null)
  const [customSamples, setCustomSamples] = useState([])
  const [previewDragActive, setPreviewDragActive] = useState(false)
  const previewFileInputRef = useRef(null)
  const updateText = (val) => {
    if (val !== text) {
      setPredictResult(null)
      setExplainResult(null)
    }
    setText(val)
  }

  const updateSelectedImage = (url) => {
    if (url !== selectedImageUrl) {
      setPredictResult(null)
      setExplainResult(null)
    }
    setSelectedImageUrl(url)
  }


  const onPasteFromClipboard = async () => {
    try {
      const items = await navigator.clipboard.read()
      for (const item of items) {
        for (const type of item.types) {
          if (type.startsWith("image/")) {
            const blob = await item.getType(type)
            setImageFile(new File([blob], "clipboard", { type }))
            updateSelectedImage("")
            return
          }
        }
      }
    } catch (e) {
      setErrorMsg("无法从剪切板读取图片，可能需要授权或使用 HTTPS")
    }
  }

  const selectedPreview = useMemo(() => {
    if (imageFile) {
      return URL.createObjectURL(imageFile)
    }
    return selectedImageUrl || ""
  }, [imageFile, selectedImageUrl])
  const hasCjk = (v) => /[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]/.test(String(v || ""))
  const isCjkInput = useMemo(() => hasCjk(text), [text])
  const showTranslationPane = isCjkInput
  const textForPredict = showTranslationPane ? translatedText.trim() : text

  const onFetch = async () => {
    if (!url.trim()) return
    setErrorMsg("")
    setLoadingFetch(true)
    try {
      const payload = await postJson("/api/fetch-url", { url: url.trim() })
      const title = payload.title || ""
      const body = payload.body_text || ""
      const desc = payload.description || ""
      let nextText = ""
      if (fillMode === "仅标题") nextText = title || payload.text || ""
      else if (fillMode === "仅正文") nextText = body || payload.body_snippet || payload.text || ""
      else if (fillMode === "仅摘要") nextText = desc || payload.text || ""
      else nextText = title && body ? `${title}\n\n${body}` : payload.text || ""
      setText(nextText)
      setTranslatedText("")
      setPredictResult(null)
      setExplainResult(null)
      const nextImages = Array.isArray(payload.images) ? payload.images : []
      setImages(nextImages)
      updateSelectedImage(payload.image_url || (nextImages[0]?.url || ""))
      setImageFile(null)
    } catch (e) {
      setErrorMsg(String(e.message || e))
    } finally {
      setLoadingFetch(false)
    }
  }

  const onLoadRandomSample = async () => {
    setErrorMsg("")
    setLoadingFetch(true)
    try {
      if (customSamples.length > 0) {
        const s = customSamples[Math.floor(Math.random() * customSamples.length)]
        setText(String(s.content || s.clean_title || s.title || ""))
        setTranslatedText("")
        setPredictResult(null)
        setExplainResult(null)
        updateSelectedImage(String(s.image_url || ""))
        setImageFile(null)
        setUrl("")
        return
      }
      const d = await getJson("/api/samples/random")
      const s = d?.item || {}
      setText(String(s.clean_title || ""))
      setTranslatedText("")
      setPredictResult(null)
      setExplainResult(null)
      updateSelectedImage(String(s.image_url || ""))
      setImageFile(null)
      setUrl("")
    } catch (e) {
      setErrorMsg(String(e.message || e))
    } finally {
      setLoadingFetch(false)
    }
  }

  const onUploadCustomSampleFile = async (f) => {
    if (!f) return
    setCustomSampleFile(f)
    const name = f.name.toLowerCase()
    try {
      if (name.endsWith(".json")) {
        const text = await f.text()
        const data = JSON.parse(text)
        const items = Array.isArray(data) ? data : (data.items || data.samples || [])
        setCustomSamples(items)
        setToastMsg(`已加载 ${items.length} 条自定义样本`)
      } else {
        const text = await f.text()
        const sep = name.endsWith(".tsv") ? "\t" : ","
        const items = parseDelimitedText(text, sep)
        setCustomSamples(items)
        setToastMsg(`已加载 ${items.length} 条自定义样本`)
      }
    } catch (e) {
      setErrorMsg("文件解析失败：" + String(e.message || e))
    }
  }

  const onPredict = async () => {
    const hasInput = text.trim() || selectedImageUrl || imageFile
    if (!hasInput) {
      setErrorMsg("请输入文本或上传图片后再分析")
      return
    }
    if (showTranslationPane && !textForPredict.trim()) {
      setErrorMsg("未获取到英文翻译，请在右侧手动输入英文后再分析")
      return
    }
    setErrorMsg("")
    setLoadingPredict(true)
    setExplainResult(null)
    setCaseSaved(false)
    try {
      const enableExplain = Boolean(settings.enable_image_heatmap || settings.enable_text_heatmap)
      const body = {
        text: textForPredict,
        image_url: imageFile ? null : selectedImageUrl || null,
        image_base64: imageFile ? await toBase64(imageFile) : null,
        with_shap: Boolean(settings.enable_shap),
        with_explain: enableExplain,
        translate_zh: Boolean(settings.enable_zh_translation),
        threshold: Number(settings.threshold || 0.5),
        source_url: url || null
      }
      const payload = await postJson("/api/predict", body)
      setPredictResult(payload.result || null)
      setExplainResult(payload.explanation || null)
      if (imageFile) {
        setSelectedImageUrl(URL.createObjectURL(imageFile))
      }
      if (showTranslationPane && !translatedText.trim()) {
        const used = String(payload?.result?.text_used || "")
        const applied = Boolean(payload?.result?.text_meta?.translation_applied)
        if (applied && used) setTranslatedText(used)
      }
    } catch (e) {
      setErrorMsg(String(e.message || e))
    } finally {
      setLoadingPredict(false)
    }
  }

  const prob = Number(predictResult?.model_c?.prob || 0)
  const label = prob >= 0.5 ? "真新闻" : "假新闻"
  const confidence = label === "真新闻" ? prob : 1 - prob
  const shap = predictResult?.shap_scores
  const shapTextPct = Math.min(1, Math.max(0, Number(shap?.text_pct || 0)))
  const shapImagePct = Math.min(1, Math.max(0, Number(shap?.image_pct || 0)))
  const shapChartTextPct = Math.min(0.999, Math.max(0.001, shapTextPct))
  const shapChartImagePct = 1 - shapChartTextPct
  const shapCx = 160
  const shapCy = 115
  const shapRadius = 74
  const shapStartDeg = -90
  const shapSplitDeg = shapStartDeg + shapChartTextPct * 360
  const shapTextMidDeg = shapStartDeg + shapChartTextPct * 180
  const shapImageMidDeg = shapSplitDeg + shapChartImagePct * 180
  const shapPointAt = (deg, r) => {
    const rad = (deg * Math.PI) / 180
    return { x: shapCx + Math.cos(rad) * r, y: shapCy + Math.sin(rad) * r }
  }
  const shapStartPt = shapPointAt(shapStartDeg, shapRadius)
  const shapSplitPt = shapPointAt(shapSplitDeg, shapRadius)
  const shapTextInnerPt = shapPointAt(shapTextMidDeg, shapRadius)
  const shapImageInnerPt = shapPointAt(shapImageMidDeg, shapRadius)
  const shapTextMidPt = shapPointAt(shapTextMidDeg, shapRadius + 15)
  const shapImageMidPt = shapPointAt(shapImageMidDeg, shapRadius + 15)
  const shapTextOuterPt = { x: shapTextMidPt.x + (shapTextMidPt.x >= shapCx ? 20 : -20), y: shapTextMidPt.y }
  const shapImageOuterPt = { x: shapImageMidPt.x + (shapImageMidPt.x >= shapCx ? 20 : -20), y: shapImageMidPt.y }
  const shapTextLabelX = shapTextOuterPt.x + (shapTextOuterPt.x >= shapCx ? 6 : -6)
  const shapImageLabelX = shapImageOuterPt.x + (shapImageOuterPt.x >= shapCx ? 6 : -6)
  const shapTextLabelAnchor = shapTextOuterPt.x >= shapCx ? "start" : "end"
  const shapImageLabelAnchor = shapImageOuterPt.x >= shapCx ? "start" : "end"
  const shapTextLargeArc = shapChartTextPct > 0.5 ? 1 : 0
  const shapImageLargeArc = shapChartImagePct > 0.5 ? 1 : 0
  const heatmapSrc = explainResult?.heatmap_png_base64 ? `data:image/png;base64,${explainResult.heatmap_png_base64}` : ""
  const tokenPairs = Array.isArray(explainResult?.tokens) ? explainResult.tokens : []
  const hasVisualInput = Boolean(imageFile || selectedImageUrl)
  const showExplainPanel = Boolean(settings.enable_shap || settings.enable_image_heatmap || settings.enable_text_heatmap)
  const tokenMaxAbs = useMemo(() => {
    const maxVal = tokenPairs.reduce((acc, item) => {
      const score = Math.abs(Number(item?.[1] ?? 0))
      return Math.max(acc, score)
    }, 0)
    return maxVal > 0 ? maxVal : 1
  }, [tokenPairs])
  const navItems = [
    { key: "detection", label: "检测", icon: "search" },
    { key: "cases", label: "案例库", icon: "folder" },
    { key: "history", label: "历史", icon: "clock" },
    { key: "report", label: "报告", icon: "report" },
    { key: "batch", label: "批量处理", icon: "box" },
    { key: "analytics", label: "效能分析", icon: "chart" },
    { key: "settings", label: "设置", icon: "settings" }
  ]
  const caseSelected = caseDraft || cases.find((x) => x.id === caseSelectedId) || null
  const activeNav = navItems.find((x) => x.key === page) || navItems[0]
  const pageMeta = {
    detection: "多模态内容识别与可解释分析",
    cases: "案件沉淀、状态推进与导出归档",
    history: "最近分析记录与快速回放",
    report: "结构化报告预览与导出",
    batch: "批量推理与结果落盘",
    analytics: "模型指标与误判样本诊断",
    settings: "阈值与功能开关统一管理"
  }
  const statusTone = (v) => {
    if (v === "已结案") return "tone-success"
    if (v === "审核中") return "tone-warning"
    return "tone-muted"
  }
  const decisionTone = (v) => {
    if (v === "真新闻") return "tone-success"
    if (v === "假新闻") return "tone-danger"
    if (v === "不确定") return "tone-warning"
    return "tone-muted"
  }
  const formatAttachmentSize = (bytes) => {
    const n = Number(bytes || 0)
    if (n <= 0) return "0 KB"
    const kb = n / 1024
    if (kb >= 1024) return `${(kb / 1024).toFixed(2)} MB`
    return `${kb.toFixed(2)} KB`
  }

  useEffect(() => {
    getJson("/api/settings").then((d) => {
      if (d?.settings) setSettings(normalizeSettings(d.settings))
    }).catch(() => {})
  }, [])

  useEffect(() => {
    if (page === "cases") {
      setLoadingPanel(true)
      getJson("/api/cases")
        .then((d) => {
          const items = Array.isArray(d?.items) ? d.items : []
          setCases(items)
          if (items.length > 0 && !caseSelectedId) setCaseSelectedId(items[0].id)
        })
        .finally(() => setLoadingPanel(false))
    } else if (page === "history") {
      setLoadingPanel(true)
      getJson("/api/history")
        .then((d) => {
          const items = Array.isArray(d?.items) ? d.items : []
          setHistoryItems(items)
          setHistorySelectedIdx(items.length > 0 ? items.length - 1 : 0)
        })
        .finally(() => setLoadingPanel(false))
    } else if (page === "report") {
      setLoadingPanel(true)
      getJson("/api/history")
        .then((d) => {
          const items = Array.isArray(d?.items) ? d.items : []
          setReportHistoryItems(items)
          if (items.length > 0 && reportSelectedIds.length === 0) {
            setReportSelectedIds([items[items.length - 1].id])
          } else if (items.length === 0) {
            setReportData(null)
            setReportPreviewMap({})
          }
        })
        .finally(() => setLoadingPanel(false))
    }
  }, [page])

  useEffect(() => {
    if (page !== "report") return
    if (reportHistoryItems.length === 0) {
      setReportPreviewMap({})
      return
    }
    setLoadingPanel(true)
    Promise.allSettled(
      reportHistoryItems.map(async (h) => {
        const d = await getJson(`/api/report/${h.id}`)
        return [h.id, d]
      })
    )
      .then((results) => {
        const nextMap = {}
        results.forEach((item) => {
          if (item.status === "fulfilled") {
            const [id, data] = item.value
            nextMap[id] = data
          }
        })
        setReportPreviewMap(nextMap)
      })
      .finally(() => setLoadingPanel(false))
  }, [reportHistoryItems, page])

  useEffect(() => {
    if (page === "report" && reportSelectedIds.length > 0) {
      setLoadingPanel(true)
      getJson(`/api/report/batch?ids=${reportSelectedIds.join(",")}`)
        .then((d) => setReportData(d))
        .catch(() => setReportData(null))
        .finally(() => setLoadingPanel(false))
    } else if (page === "report" && reportSelectedIds.length === 0) {
      setReportData(null)
    }
  }, [reportSelectedIds, page])

  useEffect(() => {
    if (!toastMsg) return
    const timer = window.setTimeout(() => setToastMsg(""), 1800)
    return () => window.clearTimeout(timer)
  }, [toastMsg])

  useEffect(() => {
    if (!showTranslationPane || !text.trim()) {
      setTranslationLoading(false)
      if (!showTranslationPane) setTranslatedText("")
      return
    }
    setTranslationLoading(true)
    const timer = window.setTimeout(() => {
      postJson("/api/translate-text", { text })
        .then((d) => {
          const used = String(d?.text_used || "").trim()
          const applied = Boolean(d?.text_meta?.translation_applied)
          if (applied && used && !hasCjk(used)) setTranslatedText(used)
          else setTranslatedText("")
        })
        .catch(() => setTranslatedText(""))
        .finally(() => setTranslationLoading(false))
    }, 350)
    return () => window.clearTimeout(timer)
  }, [showTranslationPane, text])

  useEffect(() => {
    if (page !== "cases" || !caseSelectedId) {
      setCaseDetail(null)
      return
    }
    getJson(`/api/cases/${caseSelectedId}/detail`).then((d) => setCaseDetail(d)).catch(() => setCaseDetail(null))
  }, [page, caseSelectedId, cases])

  const onSaveCaseFromCurrent = async () => {
    if (!predictResult || !text.trim() || caseSaved) return
    setErrorMsg("")
    try {
      const textUsed = showTranslationPane && translatedText.trim() ? translatedText.trim() : null
      await postJson("/api/cases", {
        text,
        text_used: textUsed,
        image_url: imageFile ? null : selectedImageUrl || null,
        image_base64: imageFile ? await toBase64(imageFile) : null,
        analysis_result: predictResult,
        threshold: Number(settings.threshold || 0.5),
        source: url ? "url" : (imageFile || selectedImageUrl ? "manual" : "manual"),
        source_url: url || null
      })
      setCaseSaved(true)
      setPage("cases")
    } catch (e) {
      setErrorMsg(String(e.message || e))
    }
  }

  const onSaveSettings = async () => {
    setErrorMsg("")
    try {
      const res = await putJson("/api/settings", settings)
      if (res?.settings) {
        setSettings(normalizeSettings(res.settings))
        setToastMsg("设置已生效")
      }
    } catch (e) {
      setErrorMsg(String(e.message || e))
    }
  }

  const onUpdateCase = async () => {
    if (!caseSelected || caseSaving) return
    setErrorMsg("")
    setCaseSaving(true)
    try {
      await putJson(`/api/cases/${caseSelected.id}`, {
        status: caseDraft?.status || caseSelected.status,
        decision: caseDraft?.decision ?? caseSelected.decision,
        tags: caseDraft?.tags ?? caseSelected.tags ?? [],
        notes: caseDraft?.notes || caseSelected.notes || ""
      })
      setCaseDraft(null)
      const d = await getJson("/api/cases")
      const items = Array.isArray(d?.items) ? d.items : []
      setCases(items)
    } catch (e) {
      setErrorMsg(String(e.message || e))
    } finally {
      setCaseSaving(false)
    }
  }

  
  const onDeleteCase = async () => {
    if (!caseSelected || !window.confirm("确定要删除该案例吗？")) return
    try {
      await fetch(`${apiBase}/api/cases/${caseSelected.id}`, { method: "DELETE" })
      setCases((prev) => prev.filter((c) => c.id !== caseSelected.id))
      setCaseSelectedId("")
    } catch (e) {
      setErrorMsg(String(e.message || e))
    }
  }

  const onUploadAttachment = async () => {
    if (!caseSelectedId || !caseAttachmentFile) return
    const fd = new FormData()
    fd.append("file", caseAttachmentFile)
    await postForm(`/api/cases/${caseSelectedId}/attachments`, fd)
    const d = await getJson(`/api/cases/${caseSelectedId}/detail`)
    setCaseDetail(d)
    setCaseAttachmentFile(null)
  }

  const onDeleteAttachment = async (name) => {
    if (!caseSelectedId || !name) return
    await deleteJson(`/api/cases/${caseSelectedId}/attachments/${encodeURIComponent(name)}`)
    const d = await getJson(`/api/cases/${caseSelectedId}/detail`)
    setCaseDetail(d)
  }

  const onClearHistory = async () => {
    await deleteJson("/api/history")
    const d = await getJson("/api/history")
    setHistoryItems(Array.isArray(d?.items) ? d.items : [])
  }

  const onLoadHistoryToDetection = (item) => {
    setPage("detection")
    setText(String(item?.text || ""))
    setTranslatedText("")
    setPredictResult(null)
    setExplainResult(null)
    setImageFile(null)
  }

  const onUploadBatchFile = async (f) => {
    if (!f) return
    setBatchCustomFile(f)
    const name = f.name.toLowerCase()
    try {
      let rows = []
      if (name.endsWith(".json")) {
        const text = await f.text()
        const data = JSON.parse(text)
        rows = Array.isArray(data) ? data : (data.items || data.rows || data.samples || [])
      } else {
        const txt = await f.text()
        const sep = name.endsWith(".tsv") ? "\t" : ","
        rows = parseDelimitedText(txt, sep)
      }
      setBatchRows(rows)
      setBatchMaxRows(rows.length)
      setBatchProcessCount(0)
      const cols = Object.keys(rows[0] || {})
      const textCol = cols.includes("content") ? "content" : (cols.includes("clean_title") ? "clean_title" : (cols[0] || ""))
      const imgCol = cols.includes("image_url") ? "image_url" : (cols[1] || "")
      setBatchTextCol(textCol)
      setBatchImageCol(imgCol)
    } catch (e) {
      setErrorMsg("文件解析失败：" + String(e.message || e))
    }
  }

  const onLoadBatchSample = async () => {
    const inputEl = document.getElementById("batch-file-input")
    if (inputEl) inputEl.value = ""
    setBatchCustomFile(null)
    const d = await getJson(`/api/batch/sample?n=${batchSampleCount}`)
    const items = Array.isArray(d?.items) ? d.items : []
    setBatchRows(items)
    setBatchMaxRows(items.length)
    setBatchProcessCount(0)
    const cols = Object.keys(items[0] || {})
    const textCol = cols.includes("clean_title") ? "clean_title" : (cols[0] || "")
    const imgCol = cols.includes("image_url") ? "image_url" : (cols[1] || "")
    setBatchTextCol(textCol)
    setBatchImageCol(imgCol)
  }

  const onRunBatch = async () => {
    if (!batchRows.length || !batchTextCol) return
    setLoadingPanel(true)
    try {
      const d = await postJson("/api/batch/run", {
        rows: batchRows,
        text_col: batchTextCol,
        image_col: batchImageCol || null,
        use_images: batchUseImages,
        max_rows: Math.min(5000, batchProcessCount > 0 ? batchProcessCount : batchRows.length),
        threshold: Number(settings.threshold || 0.5),
        translate_zh: Boolean(settings.enable_zh_translation)
      })
      setBatchResults(Array.isArray(d?.items) ? d.items : [])
    } finally {
      setLoadingPanel(false)
    }
  }

  const onSelectAnalyticsFile = async (f) => {
    setAnalyticsCustomFile(f)
    setAnalyticsTotalRows(0)
    setAnalyticsCols({ textCol: "", imageCol: "", labelCol: "" })
    if (!f) {
      setAnalyticsData(null)
      return
    }
    try {
      const text = await f.text()
      const name = String(f.name || "").toLowerCase()
      let items = []
      if (name.endsWith(".json")) {
        const data = JSON.parse(text)
        items = Array.isArray(data) ? data : (data.items || data.rows || data.samples || [])
      } else {
        const sep = name.endsWith(".tsv") ? "\t" : ","
        items = parseDelimitedText(text, sep)
      }
      setAnalyticsTotalRows(items.length)
      setAnalyticsRows(items)
      setAnalyticsCfg((p) => ({ ...p, n_samples: Math.min(50, Math.max(1, items.length || 0)) }))
      const cols = Object.keys(items[0] || {})
      const textCol = cols.includes("content") ? "content" : (cols.includes("clean_title") ? "clean_title" : (cols[0] || ""))
      const imgCol = cols.includes("image_url") ? "image_url" : (cols[1] || "")
      const labelCol = cols.includes("label") ? "label" : (cols.includes("2_way_label") ? "2_way_label" : (cols.includes("2_way_label_parsed") ? "2_way_label_parsed" : (cols.includes("ground_truth") ? "ground_truth" : "")))
      setAnalyticsCols({ textCol, imageCol: imgCol, labelCol })
    } catch (e) {
      setAnalyticsTotalRows(0)
      setErrorMsg("文件解析失败：" + String(e.message || e))
    }
  }

  const onRunAnalytics = async () => {
    setLoadingPanel(true)
    try {
      const requestedSamples = Number(analyticsCfg.n_samples || 50)
      const payload = {
        n_samples: requestedSamples,
        seed: Number(analyticsCfg.seed || 2026),
        include_images: Boolean(analyticsCfg.include_images),
        translate_zh: Boolean(settings.enable_zh_translation)
      }
      if (analyticsCustomFile && analyticsRows.length) {
        const seed = Number(analyticsCfg.seed || 2026)
        const rows = [...analyticsRows]
        for (let i = rows.length - 1; i > 0; i -= 1) {
          const j = Math.abs((seed * 9301 + i * 49297 + 233280) % (i + 1))
          ;[rows[i], rows[j]] = [rows[j], rows[i]]
        }
        const sampledRows = rows.slice(0, Math.min(rows.length, Math.max(1, requestedSamples)))
        payload.custom_data = JSON.stringify(sampledRows)
        payload.custom_data_type = "json"
        if (analyticsCols.textCol) payload.text_col = analyticsCols.textCol
        if (analyticsCols.imageCol) payload.image_col = analyticsCols.imageCol
        if (analyticsCols.labelCol) payload.label_col = analyticsCols.labelCol
      }
      const d = await postJson("/api/analytics/benchmark", payload)
      setAnalyticsData(d)
    } finally {
      setLoadingPanel(false)
    }
  }

  const filteredCases = cases.filter((c) => {
    if (caseStatusFilters.length > 0 && !caseStatusFilters.includes(String(c.status || ""))) return false
    if (caseDecisionFilters.length > 0 && !caseDecisionFilters.includes(String(c.decision || ""))) return false
    if (!caseQuery.trim()) return true
    const q = caseQuery.trim().toLowerCase()
    const t = String(c?.input?.text || "").toLowerCase()
    const tags = Array.isArray(c?.tags) ? c.tags.join(",").toLowerCase() : ""
    return t.includes(q) || tags.includes(q)
  })

  const paginatedCases = filteredCases.slice((casePage - 1) * CASE_PAGE_SIZE, casePage * CASE_PAGE_SIZE)
  const totalCasePages = Math.max(1, Math.ceil(filteredCases.length / CASE_PAGE_SIZE))

  const toggleCaseStatusFilter = (v) => {
    setCasePage(1)
    setCaseStatusFilters((prev) => prev.includes(v) ? prev.filter((x) => x !== v) : [...prev, v])
  }

  const toggleCaseDecisionFilter = (v) => {
    setCaseDecisionFilters((prev) => prev.includes(v) ? prev.filter((x) => x !== v) : [...prev, v])
    setCasePage(1)
  }

  const onThumbStripWheel = (e) => {
    if (Math.abs(e.deltaY) > Math.abs(e.deltaX)) {
      e.currentTarget.scrollLeft += e.deltaY
      e.preventDefault()
    }
  }

  const onSelectImageFile = (f) => {
    if (!f) return
    if (!f.type.startsWith("image/")) {
      setErrorMsg("仅支持图片文件上传")
      return
    }
    setErrorMsg("")
    setImageFile(f)
    updateSelectedImage("")
    setPredictResult(null)
    setExplainResult(null)
  }

  const onPreviewDrop = (e) => {
    e.preventDefault()
    setPreviewDragActive(false)
    const f = e.dataTransfer?.files?.[0] || null
    onSelectImageFile(f)
  }

  return (
    <div className={`shell ${sidebarCollapsed ? "sidebar-collapsed" : ""}`}>
      <aside className="sidebar">
        <div className="brand">
          <img className="brand-icon" src={iconPath("shield")} alt="logo" />
          <span className="brand-text">多模态假新闻智能检测系统</span>
        </div>
        <div className="nav">
          {navItems.map((item) => (
            <button
              key={item.key}
              className={`nav-btn ${page === item.key ? "active" : ""}`}
              onClick={() => setPage(item.key)}
            >
              <span className="nav-left">
                <img className="nav-icon" src={iconPath(item.icon)} alt={item.label} />
                <span className="nav-label">{item.label}</span>
              </span>
            </button>
          ))}
        </div>
      </aside>
      <button className="shell-sidebar-toggle" type="button" onClick={() => setSidebarCollapsed((v) => !v)}>
        {sidebarCollapsed ? (
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M9.5 5l7 7-7 7" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        ) : (
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M14.5 5l-7 7 7 7" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        )}
      </button>

      <main className={`page page-${page}`}>
        <div className="hero">
          <div>
            <h1 className="page-title">
              <img className="page-title-icon" src={iconPath(activeNav.icon)} alt={activeNav.label} />
              <span>{activeNav.label}</span>
            </h1>
            <div className="hero-sub">{pageMeta[page] || ""}</div>
          </div>
        </div>
        <div className="page-content" key={page}>
          {page === "detection" ? (
            <>
            <div className="layout">
              <section className="card">
                <h2>内容输入</h2>
                <div className="fetch-row">
                  <button onClick={onLoadRandomSample} disabled={loadingFetch}>随机样本</button>
                  <input value={url} onChange={(e) => setUrl(e.target.value)} placeholder="https://..." />
                  <select value={fillMode} onChange={(e) => setFillMode(e.target.value)}>
                    <option>标题+正文</option>
                    <option>仅标题</option>
                    <option>仅正文</option>
                    <option>仅摘要</option>
                  </select>
                  <button onClick={onFetch} disabled={loadingFetch}>{loadingFetch ? "抓取中..." : "网页抓取"}</button>
                </div>
                <div className="custom-sample-upload">
                  <div className="sub-title">自定义样本库</div>
                  <div className="custom-sample-upload-row">
                    <span style={{ display: "inline-flex", alignItems: "center", gap: "6px", padding: "0 8px 0 0", border: "1px solid var(--igp-border)", borderRadius: "6px", background: "#fff", height: "40px", boxSizing: "border-box" }}>
                    <input
                      id="detection-custom-file-input"
                      className="toolbar-file-input"
                      type="file"
                      accept=".csv,.tsv,.json"
                      onChange={(e) => onUploadCustomSampleFile(e.target.files?.[0] || null)}
                      style={{ height: "40px", boxSizing: "border-box", margin: 0 }}
                    />
                    {customSampleFile ? <span style={{ fontSize: "12px", color: "#64748b", lineHeight: "40px", paddingRight: "8px", whiteSpace: "nowrap" }}>{`已选: ${customSampleFile.name}${customSamples.length > 0 ? `（共 ${customSamples.length} 条样本）` : ""}`}</span> : null}
                  </span>
                    <span className="custom-sample-tip">支持 CSV、TSV、JSON 格式，需包含 content（或 clean_title / title）和 image_url 字段</span>
                    
                  </div>
                </div>
                {showTranslationPane ? (
                  <div className="dual-textarea">
                    <div>
                      <div className="sub-title">中文原文</div>
                      <textarea value={text} onChange={(e) => updateText(e.target.value)} placeholder="请输入中文文本..." />
                    </div>
                    <div>
                      <div className="sub-title">{translationLoading ? "英文翻译（生成中...）" : "英文翻译（可编辑）"}</div>
                      <textarea value={translatedText} onChange={(e) => setTranslatedText(e.target.value)} placeholder="自动翻译结果会显示在这里，可手动编辑..." />
                    </div>
                  </div>
                ) : (
                  <textarea value={text} onChange={(e) => updateText(e.target.value)} placeholder="请输入待验证的文本..." />
                )}
                {showTranslationPane && !translationLoading && !translatedText.trim() ? <div className="error">未获取到英文翻译，请在右侧手动输入英文后再分析</div> : null}
                <button className="primary" onClick={onPredict} disabled={loadingPredict || (!text.trim() && !selectedImageUrl && !imageFile)}>
                  {loadingPredict ? "分析中..." : "开始分析"}
                </button>
                <button className="ghost" onClick={onSaveCaseFromCurrent} disabled={!predictResult || caseSaved}>
                  保存为案例
                </button>
                {errorMsg ? <div className="error">{errorMsg}</div> : null}
              </section>

              <section className="card">
                <h2>候选图片</h2>
                <div className="thumb-strip" onWheel={onThumbStripWheel}>
                  {images.map((item, idx) => {
                    const u = String(item?.url || "")
                    if (!u) return null
                    const selected = selectedImageUrl === u && !imageFile
                    return (
                      <button
                        key={`${idx}-${u}`}
                        className={`thumb-card ${selected ? "selected" : ""}`}
                        onClick={() => {
                          updateSelectedImage(u)
                          setImageFile(null)
                          setPredictResult(null)
                          setExplainResult(null)
                        }}
                      >
                        <img src={u} alt={`candidate-${idx}`} />
                        <div>{`#${idx + 1} ${item?.source || "img"}`}</div>
                      </button>
                    )
                  })}
                </div>

                <h2>图像预览</h2>
                <div
                  className={`preview-dropzone ${previewDragActive ? "drag-active" : ""}`}
                  onDragOver={(e) => {
                    e.preventDefault()
                    setPreviewDragActive(true)
                  }}
                  onDragLeave={() => setPreviewDragActive(false)}
                  onDrop={onPreviewDrop}
                >
                  {selectedPreview ? <img className="preview" src={selectedPreview} alt="selected" /> : <div className="empty">拖拽图片到此区域，或点击下方按钮上传</div>}
                  <input
                    ref={previewFileInputRef}
                    className="image-picker-input"
                    type="file"
                    accept="image/png,image/jpeg,image/jpg"
                    onChange={(e) => onSelectImageFile(e.target.files?.[0] || null)}
                  />
                  <button className="ghost" type="button" onClick={() => previewFileInputRef.current?.click()}>本地上传图片</button>
                  <button className="ghost" type="button" onClick={onPasteFromClipboard}>从剪切板粘贴</button>
                </div>
              </section>
            </div>

            {predictResult ? (
              <section className="card">
                <h2>分析结果</h2>
                <div className={`result ${label === "真新闻" ? "real" : "fake"}`}>
                  <div>{label}</div>
                  <div>{`置信度 ${Math.round(confidence * 1000) / 10}%`}</div>
                </div>
                {showExplainPanel ? (
                  <div className="explain-wrap">
                    <h3>可解释性</h3>
                    {explainResult?.error ? <div className="error">{`解释生成失败：${explainResult.error}`}</div> : null}
                    {Boolean(settings.enable_shap) && shap ? (
                      <div>
                        <div className="sub-title">模态贡献（SHAP）</div>
                        <div className="shap">
                          <svg className="shap-chart" viewBox="0 0 320 230" aria-label="shap-pie-chart">
                            <defs>
                              <filter id="pieShadow" x="-20%" y="-20%" width="140%" height="140%">
                                <feDropShadow dx="0" dy="4" stdDeviation="6" floodColor="#0f172a" floodOpacity="0.15" />
                              </filter>
                            </defs>
                            <g filter="url(#pieShadow)">
                              <path d={`M ${shapCx} ${shapCy} L ${shapStartPt.x} ${shapStartPt.y} A ${shapRadius} ${shapRadius} 0 ${shapTextLargeArc} 1 ${shapSplitPt.x} ${shapSplitPt.y} Z`} fill="#1260A3"></path>
                              <path d={`M ${shapCx} ${shapCy} L ${shapSplitPt.x} ${shapSplitPt.y} A ${shapRadius} ${shapRadius} 0 ${shapImageLargeArc} 1 ${shapStartPt.x} ${shapStartPt.y} Z`} fill="#94a3b8"></path>
                            </g>
                            <polyline points={`${shapTextInnerPt.x},${shapTextInnerPt.y} ${shapTextMidPt.x},${shapTextMidPt.y} ${shapTextOuterPt.x},${shapTextOuterPt.y}`} className="shap-line-text"></polyline>
                            <polyline points={`${shapImageInnerPt.x},${shapImageInnerPt.y} ${shapImageMidPt.x},${shapImageMidPt.y} ${shapImageOuterPt.x},${shapImageOuterPt.y}`} className="shap-line-image"></polyline>
                            <text x={shapTextLabelX} y={shapTextOuterPt.y + 4} textAnchor={shapTextLabelAnchor} className="shap-pct-text">{`${(shapTextPct * 100).toFixed(1)}%`}</text>
                            <text x={shapImageLabelX} y={shapImageOuterPt.y + 4} textAnchor={shapImageLabelAnchor} className="shap-pct-image">{`${(shapImagePct * 100).toFixed(1)}%`}</text>
                          </svg>
                          <div className="shap-legend" aria-label="shap-legend">
                            <div className="shap-legend-item">
                              <span className="shap-legend-swatch shap-legend-swatch-text" aria-hidden="true"></span>
                              <span>文本</span>
                            </div>
                            <div className="shap-legend-item">
                              <span className="shap-legend-swatch shap-legend-swatch-image" aria-hidden="true"></span>
                              <span>视觉</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    ) : null}
                    {Boolean(settings.enable_text_heatmap) && tokenPairs.length > 0 ? (
                      <div>
                        <div className="sub-title">文本热力图</div>
                        <div className="token-heatmap-container">
                          <div className="token-heatmap-row">
                            <div className="token-heatmap">
                              {tokenPairs.map((item, idx) => {
                                const word = String(item?.[0] ?? "")
                                const score = Number(item?.[1] ?? 0)
                                const ratio = Math.min(1, Math.abs(score) / tokenMaxAbs)
                                const opacity = 0.08 + ratio * 0.62
                                const bg = `rgba(18,96,163,${opacity})`
                                return (
                                  <span
                                    key={`${idx}-${word}`}
                                    className="token-word"
                                    style={{ backgroundColor: bg }}
                                  >
                                    {word}
                                  </span>
                                )
                              })}
                            </div>
                            <div className="token-legend-panel">
                              <div className="token-legend-core">
                                <div className="token-legend-bar"></div>
                                <div className="token-legend-scale-vertical">
                                  <span>高</span>
                                  <span>中</span>
                                  <span>低</span>
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    ) : null}
                    {Boolean(settings.enable_image_heatmap) && heatmapSrc ? (
                      <div>
                        <div className="sub-title">视觉注意力热力图</div>
                        <div className="heatmap-row">
                          <img className="preview heatmap-main" src={heatmapSrc} alt="heatmap" />
                          <div className="heatmap-legend-panel">
                            <div className="heatmap-legend-core">
                              <div className="heatmap-legend-bar"></div>
                              <div className="heatmap-legend-scale-vertical">
                                <span>高</span>
                                <span>中</span>
                                <span>低</span>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    ) : null}
                    {Boolean(settings.enable_image_heatmap) && !heatmapSrc ? (
                      <div className="sub-title">{hasVisualInput ? "视觉热力图暂未生成" : "未提供图片，无法生成视觉热力图"}</div>
                    ) : null}
                  </div>
                ) : null}
              </section>
            ) : null}
            </>
          ) : page === "cases" ? (
            <section className="card">
            <h2>案例库</h2>
            <div className="filter-block">
              <div className="filter-title">状态筛选</div>
              <div className="filter-pills">
                {caseStatusOptions.map((v) => (
                  <button key={v} className={`pill-btn ${caseStatusFilters.includes(v) ? "active" : ""}`} onClick={() => toggleCaseStatusFilter(v)}>{v}</button>
                ))}
              </div>
              <div className="filter-title">结论筛选</div>
              <div className="filter-pills">
                {caseDecisionOptions.map((v) => (
                  <button key={v} className={`pill-btn ${caseDecisionFilters.includes(v) ? "active" : ""}`} onClick={() => toggleCaseDecisionFilter(v)}>{v}</button>
                ))}
              </div>
            </div>
            <div className="row-grid">
              <input value={caseQuery} onChange={(e) => { setCasePage(1); setCaseQuery(e.target.value) }} placeholder="搜索文本/标签" />
            </div>
            {loadingPanel ? <div className="sub-title">加载中...</div> : null}
            {!loadingPanel && filteredCases.length === 0 ? (
              <div className="empty">
                <div>
                  <div>没有匹配当前筛选条件的案例</div>
                  <div className="sub-title">可尝试放宽状态/结论筛选，或清空搜索词</div>
                  <button onClick={() => { setCaseStatusFilters([]); setCaseDecisionFilters([]); setCaseQuery(""); setCasePage(1) }}>重置筛选</button>
                </div>
              </div>
            ) : null}
            {filteredCases.length > 0 ? (
            <>
            <div className="table-wrap">
              <table style={{ tableLayout: "fixed" }}>
                <thead><tr><th style={{ width: "100px" }}>ID</th><th style={{ width: "80px" }}>状态</th><th style={{ width: "80px" }}>结论</th><th style={{ width: "80px" }}>置信度(真)</th><th style={{ width: "80px" }}>模型预测</th><th style={{ width: "140px" }}>更新时间</th><th className="action-cell" style={{ width: "128px" }}>操作</th></tr></thead>
                <tbody>
                  {paginatedCases.map((c) => (
                    <tr key={c.id} className={caseSelectedId === c.id ? "active-row" : ""} onClick={(e) => { e.stopPropagation(); if (caseSelectedId !== c.id) { setCaseSelectedId(c.id); setCaseDraft({ ...c, _fromList: true }) } }}>
                      <td>{String(c.id || "").slice(0, 10)}</td>
                      <td><span className={`pill ${statusTone(c.status)}`}>{c.status}</span></td>
                      <td><span className={`pill ${decisionTone(c.decision)}`}>{c.decision}</span></td>
                      <td>{Number(c?.prediction?.prob_real || 0).toFixed(3)}</td>
                      <td><span className={`pill ${decisionTone(c?.prediction?.label)}`}>{c?.prediction?.label || "—"}</span></td>
                      <td>{c.updated_at || c.created_at}</td>
                      <td className="action-cell">
                        <div className="table-row-actions">
                          <button className="ghost" onClick={(e) => { e.stopPropagation(); window.open(apiUrl(`/api/cases/${c.id}/export`), "_blank") }} disabled={c.status !== "已结案"}>ZIP</button>
                          <button className="ghost danger" onClick={(e) => { e.stopPropagation(); if (window.confirm("确定删除？")) { fetch(`${apiBase}/api/cases/${c.id}`, { method: "DELETE" }).then(() => setCases((prev) => prev.filter((x) => x.id !== c.id))).catch((e2) => setErrorMsg(String(e2.message || e2))) } }}>删除</button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "flex-end", gap: "8px", marginTop: "8px" }}>
              <button className="ghost" style={{ padding: "4px 12px", minWidth: "52px" }} onClick={() => setCasePage((p) => Math.max(1, p - 1))} disabled={casePage <= 1}>上页</button>
              <span style={{ fontSize: "13px", color: "#374151", whiteSpace: "nowrap" }}>第 {casePage} / {totalCasePages} 页，共 {filteredCases.length} 条</span>
              <button className="ghost" style={{ padding: "4px 12px", minWidth: "52px" }} onClick={() => setCasePage((p) => Math.min(totalCasePages, p + 1))} disabled={casePage >= totalCasePages}>下页</button>
            </div>
            {caseSelected ? (
              <div className="detail-grid">
                <select value={caseSelected.decision || ""} onChange={(e) => setCaseDraft((prev) => prev ? { ...prev, decision: e.target.value, status: e.target.value ? "已结案" : "待处理" } : null)}>
                  <option value="">未标记</option><option>真新闻</option><option>假新闻</option><option>不确定</option>
                </select>
                <input value={Array.isArray(caseSelected.tags) ? caseSelected.tags.join(",") : ""} onChange={(e) => setCaseDraft((prev) => prev ? { ...prev, tags: e.target.value.split(",").map((s) => s.trim()).filter(Boolean) } : null)} placeholder="标签，逗号分隔" />
                <textarea value={caseSelected.notes || ""} onChange={(e) => setCaseDraft((prev) => prev ? { ...prev, notes: e.target.value } : null)} placeholder="备注" />
                <div className="case-actions-row row-grid">
                  <div style={{ display: "flex", alignItems: "center", gap: "6px", height: "36px", boxSizing: "border-box" }}>
                    <input type="file" style={{ fontSize: "12px", height: "36px", lineHeight: "36px", padding: "0 8px", boxSizing: "border-box", margin: 0 }} onChange={(e) => setCaseAttachmentFile(e.target.files?.[0] || null)} />
                    <button onClick={onUploadAttachment} disabled={!caseAttachmentFile} style={{ whiteSpace: "nowrap", height: "36px", lineHeight: "36px", boxSizing: "border-box", margin: 0 }}>上传附件</button>
                  </div>
                  <button className="primary" onClick={onUpdateCase} disabled={!caseSelected || caseSaving} style={{ whiteSpace: "nowrap", height: "36px", lineHeight: "36px", boxSizing: "border-box", margin: 0 }}>{caseSaving ? "保存中..." : "保存更改"}</button>
                  {errorMsg ? <span style={{ fontSize: "12px", color: "#dc2626", marginLeft: "8px" }}>{errorMsg}</span> : null}
                </div>
                {Array.isArray(caseDetail?.attachments) && caseDetail.attachments.length > 0 ? (
                  <div className="table-wrap" style={{ marginTop: "8px" }}>
                    <table>
                      <thead><tr><th>文件名</th><th>大小</th><th>操作</th></tr></thead>
                      <tbody>
                        {caseDetail.attachments.map((a) => (
                          <tr key={a.name}>
                            <td>{a.name}</td>
                            <td>{formatAttachmentSize(a.size)}</td>
                            <td className="action-cell">
                              <button className="ghost" onClick={() => window.open(apiUrl(`/api/cases/${caseSelected.id}/attachments/${encodeURIComponent(a.name)}`), "_blank")}>下载</button>
                              <button className="ghost danger" onClick={() => { if (window.confirm("确定删除该附件？")) onDeleteAttachment(a.name) }}>删除</button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : null}
                <div className="sub-title">输入文本</div>
                <textarea value={String(caseSelected?.input?.text || "")} disabled />
                <div className="sub-title">输入图片</div>
                {caseDetail?.input_image_base64 ? (
                  <img className="preview" src={`data:image/png;base64,${caseDetail.input_image_base64}`} alt="case-image" />
                ) : caseDetail?.item?.input?.image_url ? (
                  <img className="preview" src={caseDetail.item.input.image_url} alt="case-image-url" />
                ) : null}
                {caseDetail?.heatmap_base64 ? (
                  <>
                    <div className="sub-title">注意力热力图</div>
                    <img className="preview" src={`data:image/png;base64,${caseDetail.heatmap_base64}`} alt="case-heatmap" />
                  </>
                ) : null}
              </div>
            ) : null}
            </>
            ) : null}
            </section>
          ) : page === "history" ? (
            <section className="card">
            <h2>历史记录</h2>
            <div className="row-grid">
              <button onClick={() => getJson("/api/history").then((d) => setHistoryItems(Array.isArray(d?.items) ? d.items : []))}>刷新</button>
              <button onClick={onClearHistory}>清空历史</button>
            </div>
            <div className="table-wrap">
              <table>
                <thead><tr><th>缩略图</th><th>时间</th><th>结论</th><th>P(Real)</th><th>文本</th><th>操作</th></tr></thead>
                <tbody>
                  {historyItems.map((h, idx) => (
                    <tr key={`${h.ts}-${idx}`} className={historySelectedIdx === idx ? "active-row" : ""} onClick={() => setHistorySelectedIdx(idx)}>
                      <td>
                        {h.image_url ? (
                          <img
                            className="history-thumb"
                            src={h.image_url}
                            alt="thumb"
                            style={{ cursor: "zoom-in" }}
                            onClick={() => setPreviewImageUrl(h.image_url)}
                          />
                        ) : (
                          <div className="history-thumb-placeholder">-</div>
                        )}
                      </td>
                      <td>{h.ts}</td>
                      <td><span className={`pill ${decisionTone(h.label)}`}>{h.label}</span></td>
                      <td>{Number(h.prob_real || 0).toFixed(3)}</td>
                      <td>{String(h.text || "").slice(0, 80)}</td>
                      <td><button onClick={(e) => { e.stopPropagation(); onLoadHistoryToDetection(h); }}>加载到检测页</button></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            </section>
          ) : page === "report" ? (
            <section className="card">
            <h2>批量分析报告</h2>
            <div className="report-topbar">
              <div className="report-topbar-actions row-grid" style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "8px" }}>
                <button className="ghost" onClick={() => { if (reportSelectMode) { setReportSelectedIds([]); } setReportSelectMode(!reportSelectMode) }}>{reportSelectMode ? "取消选择" : "选择"}</button>
                {reportSelectMode && <button className="ghost" onClick={() => setReportSelectedIds(reportHistoryItems.map((h) => h.id))}>全选</button>}
                {reportSelectMode && <button className="ghost danger" onClick={() => setReportSelectedIds([])}>清空</button>}
                {reportSelectMode && <button className="primary" disabled={reportSelectedIds.length === 0} onClick={() => downloadBlob(reportData.html, "batch_report.html", "text/html")}>下载合并报告 HTML</button>}
              </div>
              {reportSelectMode && <div className="report-topbar-meta">
                <div className="report-topbar-title">已选择 {reportSelectedIds.length} 条记录</div>
                <div className="report-topbar-sub">在下方每个报告标题旁直接勾选，合并导出一份 HTML 报告</div>
              </div>}
            </div>
            {reportHistoryItems.length > 0 ? (
              <div className="report-preview-list">
                {reportHistoryItems.slice().reverse().map((h) => (
                  <div key={`item-${h.id}`} className="report-preview-row">
                    <div className="report-preview-content">
                      <div key={`header-${h.id}`} className="report-preview-header">
                        <span className="report-preview-time">[{h.ts}]</span>
                        <span className={`pill ${decisionTone(h.label)}`}>{h.label}</span>
                        <span className="report-preview-text">{String(h.text || "").slice(0, 60)}...</span>
                      </div>
                      {reportPreviewMap[h.id]?.html ? (
                        <iframe key={`frame-${h.id}`} title={`report-preview-${h.id}`} className="report-frame" srcDoc={reportPreviewMap[h.id].html}></iframe>
                      ) : (
                        <div key={`empty-${h.id}`} className="empty">报告加载中或生成失败</div>
                      )}
                    </div>
                    {reportSelectMode && (
                      <div className="report-preview-check">
                        <input type="checkbox" checked={reportSelectedIds.includes(h.id)} onChange={(e) => {
                          if (e.target.checked) setReportSelectedIds([...reportSelectedIds, h.id])
                          else setReportSelectedIds(reportSelectedIds.filter((id) => id !== h.id))
                        }} />
                      </div>
                    )}
                  </div>
                ))}
              </div>
            ) : <div className="empty">暂无历史记录</div>}
            </section>
          ) : page === "batch" ? (
            <section className="card">
            <h2>批量处理</h2>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "10px", margin: "8px 0 10px", alignItems: "stretch", justifyContent: "flex-start" }}>
              <span style={{ display: "inline-flex", alignItems: "center", gap: "6px", padding: "0 8px 0 0", border: "1px solid var(--igp-border)", borderRadius: "6px", background: "#fff", height: "40px", boxSizing: "border-box" }}>
                <input id="batch-file-input" className="toolbar-file-input" type="file" accept=".csv,.tsv,.json" onChange={(e) => onUploadBatchFile(e.target.files?.[0] || null)} style={{ height: "40px", boxSizing: "border-box", margin: 0 }} />
                <span style={{ fontSize: "12px", color: "#64748b", lineHeight: "40px", paddingRight: "8px", whiteSpace: "nowrap" }}>{batchCustomFile ? `已选: ${batchCustomFile.name}` : "未选文件"}</span>
                {batchCustomFile && batchRows.length > 0 ? <span style={{ fontSize: "12px", color: "#64748b", whiteSpace: "nowrap", lineHeight: "40px", paddingRight: "8px" }}>{`共 ${batchRows.length} 行`}</span> : null}
              </span>
            </div>
            {batchCustomFile && batchRows.length > 0 ? (
              <div style={{ display: "flex", flexWrap: "wrap", gap: "10px", margin: "0 0 10px", alignItems: "stretch", justifyContent: "flex-start" }}>
                <select value={batchTextCol} onChange={(e) => setBatchTextCol(e.target.value)} style={{ height: "40px", padding: "0 8px", boxSizing: "border-box", fontSize: "12px" }}>
                  <option value="">文本列</option>
                  {Object.keys(batchRows[0] || {}).map((k) => <option key={k}>{k}</option>)}
                </select>
                <select value={batchImageCol} onChange={(e) => setBatchImageCol(e.target.value)} style={{ height: "40px", padding: "0 8px", boxSizing: "border-box", fontSize: "12px" }}>
                  <option value="">图像列</option>
                  {Object.keys(batchRows[0] || {}).map((k) => <option key={k}>{k}</option>)}
                </select>
                <span style={{ display: "flex", alignItems: "center", gap: "6px", flex: "0 0 auto", height: "40px", boxSizing: "border-box" }}>
                  <span style={{ fontSize: "12px", color: "#64748b", whiteSpace: "nowrap", lineHeight: "40px" }}>处理行数</span>
                  <input type="number" min="1" max={batchMaxRows || batchRows.length} value={batchProcessCount || (batchMaxRows || batchRows.length)} onChange={(e) => setBatchProcessCount(Math.max(1, Math.min(batchMaxRows || batchRows.length, Number(e.target.value || 0))))} style={{ width: "70px", margin: 0, height: "40px", padding: "0 8px", lineHeight: "40px", boxSizing: "border-box" }} />
                </span>
                <label className="switch-row" style={{ margin: 0, flex: "0 0 auto", display: "flex", alignItems: "center", justifyContent: "center", padding: "0 12px", height: "40px", boxSizing: "border-box" }}>
                  <input type="checkbox" checked={batchUseImages} onChange={(e) => setBatchUseImages(e.target.checked)} style={{ margin: 0 }} />
                  <span style={{ whiteSpace: "nowrap", marginLeft: "6px" }}>启用图像</span>
                </label>
                <button className="primary" onClick={onRunBatch} style={{ margin: 0, flex: "0 0 auto", width: "auto", whiteSpace: "nowrap", padding: "0 24px", height: "40px", boxSizing: "border-box", display: "flex", alignItems: "center", justifyContent: "center", lineHeight: 1 }}>开始批量推理</button>
              </div>
            ) : null}
            {loadingPanel ? <div className="sub-title">处理中...</div> : null}
            {batchResults.length > 0 ? (
              <>
                <div className="table-wrap">
                  <table>
                    <thead><tr><th>行号</th><th>P(Real)</th><th>预测</th><th>文本</th><th>缩略图</th></tr></thead>
                    <tbody>
                      {batchResults.map((r) => {
                        const srcRow = batchRows[r.row_index] || {}
                        const imgUrl = srcRow[batchImageCol] || null
                        return (
                          <tr key={r.row_index}>
                            <td>{r.row_index}</td>
                            <td>{Number(r.prob_real || 0).toFixed(3)}</td>
                            <td><span className={`pill ${decisionTone(r.prediction)}`}>{r.prediction}</span></td>
                            <td>{r.text}</td>
                            <td>
                              {imgUrl ? (
                                <img className="history-thumb" src={String(imgUrl)} alt="batch-thumb" />
                              ) : (
                                <div className="history-thumb-placeholder">-</div>
                              )}
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
                <button style={{ marginTop: "10px" }} onClick={() => downloadBlob(`row_index,prob_real,prediction,text\n${batchResults.map((r) => `${r.row_index},${r.prob_real},${r.prediction},"${String(r.text || "").replaceAll('"', '""')}"`).join("\n")}`, "batch_results.csv", "text/csv")}>下载 CSV</button>
              </>
            ) : null}
            </section>
          ) : page === "analytics" ? (
            <section className="card">
            <h2>模型效能分析</h2>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "10px", margin: "8px 0 10px", alignItems: "stretch", justifyContent: "flex-start" }}>
              <span style={{ display: "inline-flex", alignItems: "center", gap: "6px", padding: "0 8px 0 0", border: "1px solid var(--igp-border)", borderRadius: "6px", background: "#fff", height: "40px", boxSizing: "border-box" }}>
                <input
                  className="toolbar-file-input"
                  type="file"
                  accept=".csv,.tsv,.json"
                  style={{ height: "40px", boxSizing: "border-box", margin: 0 }}
                  id="analytics-file-input"
                  onChange={(e) => onSelectAnalyticsFile(e.target.files?.[0] || null)}
                />
                <span style={{ fontSize: "12px", color: "#64748b", lineHeight: "40px", paddingRight: "8px", whiteSpace: "nowrap" }}>{analyticsCustomFile ? `已选: ${analyticsCustomFile.name}` : "未选文件"}</span>
                {analyticsCustomFile ? <span style={{ fontSize: "12px", color: "#64748b", whiteSpace: "nowrap", lineHeight: "40px", paddingRight: "8px" }}>{`共 ${analyticsTotalRows} 行`}</span> : null}
              </span>
            </div>
            {analyticsCustomFile ? (
              <div style={{ display: "flex", flexWrap: "wrap", gap: "10px", margin: "0 0 10px", alignItems: "stretch", justifyContent: "flex-start" }}>
                <select value={analyticsCols.textCol} onChange={(e) => setAnalyticsCols((p) => ({ ...p, textCol: e.target.value }))} style={{ height: "40px", padding: "0 8px", boxSizing: "border-box", fontSize: "12px" }}>
                  <option value="">文本列</option>
                  {Object.keys(analyticsRows[0] || {}).map((k) => <option key={k}>{k}</option>)}
                </select>
                <select value={analyticsCols.imageCol} onChange={(e) => setAnalyticsCols((p) => ({ ...p, imageCol: e.target.value }))} style={{ height: "40px", padding: "0 8px", boxSizing: "border-box", fontSize: "12px" }}>
                  <option value="">图像列</option>
                  {Object.keys(analyticsRows[0] || {}).map((k) => <option key={k}>{k}</option>)}
                </select>
                <select value={analyticsCols.labelCol} onChange={(e) => setAnalyticsCols((p) => ({ ...p, labelCol: e.target.value }))} style={{ height: "40px", padding: "0 8px", boxSizing: "border-box", fontSize: "12px" }}>
                  <option value="">标签列</option>
                  {Object.keys(analyticsRows[0] || {}).map((k) => <option key={k}>{k}</option>)}
                </select>
                <span style={{ display: "flex", alignItems: "center", gap: "6px", flex: "0 0 auto", height: "40px", boxSizing: "border-box" }}>
                  <span style={{ fontSize: "12px", color: "#64748b", whiteSpace: "nowrap", lineHeight: "40px" }}>样本数</span>
                  <input type="number" value={analyticsCfg.n_samples} onChange={(e) => setAnalyticsCfg((p) => ({ ...p, n_samples: Number(e.target.value || 50) }))} style={{ margin: 0, width: "60px", height: "40px", padding: "0 8px", lineHeight: "40px", boxSizing: "border-box" }} />
                </span>
                <span style={{ display: "flex", alignItems: "center", gap: "6px", flex: "0 0 auto", height: "40px", boxSizing: "border-box" }}>
                  <span style={{ fontSize: "12px", color: "#64748b", whiteSpace: "nowrap", lineHeight: "40px" }}>随机种子</span>
                  <input type="number" value={analyticsCfg.seed} onChange={(e) => setAnalyticsCfg((p) => ({ ...p, seed: Number(e.target.value || 2026) }))} style={{ margin: 0, width: "70px", height: "40px", padding: "0 8px", lineHeight: "40px", boxSizing: "border-box" }} />
                </span>
                <label className="switch-row" style={{ margin: 0, flex: "0 0 auto", display: "flex", alignItems: "center", justifyContent: "center", padding: "0 12px", height: "40px", boxSizing: "border-box" }}>
                  <input type="checkbox" checked={analyticsCfg.include_images} onChange={(e) => setAnalyticsCfg((p) => ({ ...p, include_images: e.target.checked }))} style={{ margin: 0 }} />
                  <span style={{ whiteSpace: "nowrap", marginLeft: "6px" }}>启用图像</span>
                </label>
                <button className="primary" onClick={onRunAnalytics} style={{ margin: 0, flex: "0 0 auto", width: "auto", whiteSpace: "nowrap", padding: "0 24px", height: "40px", boxSizing: "border-box", display: "flex", alignItems: "center", justifyContent: "center", lineHeight: 1 }}>开始测试</button>
              </div>
            ) : null}

            {analyticsData?.metrics ? (
              <div className="metrics-grid">
                <div className="metric"><div>Accuracy</div><strong>{(analyticsData.metrics.accuracy * 100).toFixed(1)}%</strong></div>
                <div className="metric"><div>Precision</div><strong>{(analyticsData.metrics.precision * 100).toFixed(1)}%</strong></div>
                <div className="metric"><div>Recall</div><strong>{(analyticsData.metrics.recall * 100).toFixed(1)}%</strong></div>
                <div className="metric"><div>F1</div><strong>{analyticsData.metrics.f1.toFixed(3)}</strong></div>
              </div>
            ) : null}
            {analyticsData?.confusion ? (
              <div className="confusion-wrap">
                <div className="sub-title" style={{ marginBottom: "10px" }}>混淆矩阵</div>
                <div className="confusion-grid">
                  <div className="confusion-corner"></div>
                  <div className="confusion-header">预测假新闻</div>
                  <div className="confusion-header">预测真新闻</div>
                  <div className="confusion-header">实际假新闻</div>
                  <div className="confusion-cell confusion-tn">{analyticsData.confusion[0][0]}</div>
                  <div className="confusion-cell confusion-fp">{analyticsData.confusion[0][1]}</div>
                  <div className="confusion-header">实际真新闻</div>
                  <div className="confusion-cell confusion-fn">{analyticsData.confusion[1][0]}</div>
                  <div className="confusion-cell confusion-tp">{analyticsData.confusion[1][1]}</div>
                </div>
              </div>
            ) : null}
            {loadingPanel ? <div className="sub-title">模型评测中，请稍候...</div> : null}
            {analyticsData?.errors?.length ? (
              <div className="table-wrap">
                <table>
                  <thead><tr><th>索引</th><th>实际</th><th>预测</th><th>P(Real)</th><th>文本</th><th>缩略图</th></tr></thead>
                  <tbody>
                    {analyticsData.errors.map((e) => (
                      <tr key={e.index}>
                        <td>{e.index}</td>
                        <td><span className={`pill ${decisionTone(e.actual)}`}>{e.actual}</span></td>
                        <td><span className={`pill ${decisionTone(e.predicted)}`}>{e.predicted}</span></td>
                        <td>{Number(e.prob_real || 0).toFixed(3)}</td>
                        <td>{e.text}</td>
                        <td>
                          {e.image_url ? (
                            <img className="history-thumb" src={e.image_url} alt="err-thumb" />
                          ) : (
                            <div className="history-thumb-placeholder">-</div>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : null}
            </section>
          ) : page === "settings" ? (
            <section className="card">
            <h2>系统设置</h2>
            <div className="detail-grid">
              <label>判定阈值 <input type="number" step="0.01" min="0.05" max="0.95" value={settings.threshold} onChange={(e) => setSettings((p) => ({ ...p, threshold: Number(e.target.value || 0.5) }))} /></label>
              <label className="switch-row"><input type="checkbox" checked={settings.enable_shap} onChange={(e) => setSettings((p) => ({ ...p, enable_shap: e.target.checked }))} /><span>启用 SHAP 百分比</span></label>
              <label className="switch-row"><input type="checkbox" checked={settings.enable_image_heatmap} onChange={(e) => setSettings((p) => ({ ...p, enable_image_heatmap: e.target.checked }))} /><span>启用图像热力图</span></label>
              <label className="switch-row"><input type="checkbox" checked={settings.enable_text_heatmap} onChange={(e) => setSettings((p) => ({ ...p, enable_text_heatmap: e.target.checked }))} /><span>启用文本热力图</span></label>
              <label className="switch-row"><input type="checkbox" checked={settings.enable_zh_translation} onChange={(e) => setSettings((p) => ({ ...p, enable_zh_translation: e.target.checked }))} /><span>启用中英翻译</span></label>
              <button className="primary" onClick={onSaveSettings}>保存设置</button>
            </div>
            </section>
          ) : (
            <section className="card">
            <h2>模块建设中</h2>
            <div className="empty">{`${navItems.find((x) => x.key === page)?.label || "该模块"} 正在迁移到 React 端`}</div>
            </section>
          )}
        </div>
      </main>
      {toastMsg ? <div className="global-toast">{toastMsg}</div> : null}
      {previewImageUrl ? (
        <div className="img-modal" onClick={() => setPreviewImageUrl("")}>
          <img src={previewImageUrl} alt="preview-full" />
        </div>
      ) : null}
    </div>
  )
}
