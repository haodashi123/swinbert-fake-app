import json
import re
from html import unescape
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
import streamlit as st

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover
    BeautifulSoup = None


def _norm_space(s: str) -> str:
    s = unescape(s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_meta(html: str, key: str) -> Optional[str]:
    patterns = [
        rf'<meta[^>]+property=["\']{re.escape(key)}["\'][^>]+content=["\']([^"\']+)["\']',
        rf'<meta[^>]+name=["\']{re.escape(key)}["\'][^>]+content=["\']([^"\']+)["\']',
    ]
    for p in patterns:
        m = re.search(p, html, flags=re.IGNORECASE)
        if m:
            out = _norm_space(m.group(1))
            return out or None
    return None


def _extract_title(html: str) -> Optional[str]:
    t = _extract_meta(html, "og:title") or _extract_meta(html, "twitter:title")
    if t:
        return t
    m = re.search(r"(?is)<h1[^>]*>(.*?)</h1>", html)
    if m:
        h1 = _norm_space(re.sub(r"(?is)<[^>]+>", " ", m.group(1)))
        if h1:
            return h1
    m = re.search(r"(?is)<title[^>]*>(.*?)</title>", html)
    if m:
        title = _norm_space(m.group(1))
        return title or None
    return None


def _extract_description(html: str) -> Optional[str]:
    d = _extract_meta(html, "og:description") or _extract_meta(html, "description")
    return _norm_space(d) or None if d is not None else None


def _extract_site_name(html: str) -> Optional[str]:
    s = (
        _extract_meta(html, "og:site_name")
        or _extract_meta(html, "application-name")
        or _extract_meta(html, "apple-mobile-web-app-title")
        or _extract_meta(html, "twitter:site")
    )
    s = _norm_space(s or "")
    if s.startswith("@"):
        s = s[1:]
    return s or None


def _hostname_from_url(url: str) -> Optional[str]:
    try:
        host = (urlparse(url).netloc or "").strip().lower()
        if not host:
            return None
        if ":" in host:
            host = host.split(":", 1)[0]
        if host.startswith("www."):
            host = host[4:]
        return host or None
    except Exception:
        return None


def _tokenize_site_hint(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"^https?://", "", s)
    s = re.sub(r"^www\.", "", s)
    s = re.sub(r"[@#]", "", s)
    s = re.sub(r"[^\w\u4e00-\u9fff]+", "", s)
    return s


def _clean_title(title: Optional[str], site_name: Optional[str], hostname: Optional[str]) -> Optional[str]:
    t = _norm_space(title or "")
    if not t:
        return None

    site_tokens = set()
    if site_name:
        site_tokens.add(_tokenize_site_hint(site_name))
    if hostname:
        site_tokens.add(_tokenize_site_hint(hostname))
        base = hostname.split(".")[0]
        if base:
            site_tokens.add(_tokenize_site_hint(base))

    # Common patterns: "正文 - 站点名" / "正文 | 站点名" / "正文_站点名"
    delims = [" - ", " | ", " — ", " – ", " _ ", "-", "|", "—", "–", "_", "丨", "·"]
    for d in delims:
        if d not in t:
            continue
        idx = t.rfind(d)
        if idx <= 0:
            continue
        head = t[:idx].strip()
        tail = t[idx + len(d) :].strip()
        if not head or not tail:
            continue
        tail_tok = _tokenize_site_hint(tail)
        if tail_tok and tail_tok in site_tokens:
            if len(head) >= 6:
                t = head
            break

    # Trailing "(站点名)"
    m = re.match(r"^(.*?)[\s]*[\(\（]([^\)\）]{2,40})[\)\）]\s*$", t)
    if m:
        head, tail = m.group(1).strip(), m.group(2).strip()
        tail_tok = _tokenize_site_hint(tail)
        if head and tail_tok and tail_tok in site_tokens and len(head) >= 6:
            t = head

    # Prevent over-stripping.
    t = _norm_space(t)
    if len(t) < 4:
        return _norm_space(title or "") or None
    return t


def _strip_html_to_text(html: str) -> str:
    html = re.sub(r"(?is)<(script|style|noscript|svg)[^>]*>.*?</\1>", " ", html)
    html = re.sub(r"(?is)<!--.*?-->", " ", html)
    html = re.sub(r"(?is)</(p|div|section|article|h1|h2|h3|h4|li|blockquote)\s*>", "\n", html)
    html = re.sub(r"(?is)<br\s*/?>", "\n", html)
    html = re.sub(r"(?is)<[^>]+>", " ", html)
    text = unescape(html)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _find_jsonld(html: str) -> List[Any]:
    blocks = re.findall(
        r'(?is)<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html,
    )
    out: List[Any] = []
    for b in blocks:
        raw = b.strip()
        if not raw:
            continue
        raw = raw.replace("\u2028", " ").replace("\u2029", " ")
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        out.append(obj)
    return out


def _iter_jsonld_nodes(obj: Any) -> List[Dict[str, Any]]:
    nodes: List[Dict[str, Any]] = []
    if isinstance(obj, dict):
        if "@graph" in obj and isinstance(obj["@graph"], list):
            for x in obj["@graph"]:
                if isinstance(x, dict):
                    nodes.append(x)
        else:
            nodes.append(obj)
    elif isinstance(obj, list):
        for x in obj:
            if isinstance(x, dict):
                nodes.append(x)
    return nodes


def _pick_jsonld_article(nodes: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    best = None
    best_score = -1
    for n in nodes:
        t = n.get("@type")
        types = [t] if isinstance(t, str) else (t if isinstance(t, list) else [])
        types = [str(x).lower() for x in types if x]
        if not types:
            continue
        is_article = any(x in ("newsarticle", "article", "report", "blogposting") for x in types)
        if not is_article:
            continue
        body = n.get("articleBody") or n.get("text") or ""
        score = len(str(body))
        if score > best_score:
            best = n
            best_score = score
    return best


def _jsonld_image_to_url(image: Any) -> Optional[str]:
    if isinstance(image, str):
        return image.strip() or None
    if isinstance(image, list) and image:
        for x in image:
            u = _jsonld_image_to_url(x)
            if u:
                return u
    if isinstance(image, dict):
        for k in ("url", "@id", "contentUrl"):
            v = image.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return None


def _extract_main_html_candidates(html: str) -> List[Tuple[str, str]]:
    patterns = [
        ("article", r"(?is)<article\b[^>]*>(.*?)</article>"),
        ("main", r"(?is)<main\b[^>]*>(.*?)</main>"),
        ("content", r'(?is)<div\b[^>]*(?:id|class)=["\'][^"\']*(?:article|content|post|entry|main|detail|text)[^"\']*["\'][^>]*>(.*?)</div>'),
    ]
    cands: List[Tuple[str, str]] = []
    for label, p in patterns:
        for m in re.finditer(p, html):
            frag = m.group(1)
            if frag and len(frag) > 200:
                cands.append((label, frag))
    return cands


def _pick_best_body_html(html: str) -> Tuple[Optional[str], Optional[str]]:
    cands = _extract_main_html_candidates(html)
    best_html = None
    best_label = None
    best_score = -1
    for label, frag in cands:
        txt = _strip_html_to_text(frag)
        score = len(txt)
        if score > best_score:
            best_score = score
            best_html = frag
            best_label = label
    return best_html, best_label


_JUNK_IDCLASS_RE = re.compile(
    r"(cookie|consent|subscribe|newsletter|comment|reply|share|breadcrumb|related|recommend|promo|advert|ad[sx]?|sidebar|modal|popup|banner|toolbar|nav|footer|header)",
    re.IGNORECASE,
)
_CONTENT_HINT_RE = re.compile(r"(article|content|post|entry|main|detail|text|body)", re.IGNORECASE)
_BOILERPLATE_LINE_RE = re.compile(
    r"(cookie|隐私|隐私政策|订阅|newsletter|关注|扫码|下载|打开\s*app|客户端|注册|登录|评论|点赞|收藏|分享|转发|责任编辑|来源|免责声明|版权|copyright|相关阅读|相关推荐|更多精彩|更多内容|上一篇|下一篇|返回顶部|广告|推广)",
    re.IGNORECASE,
)


def _looks_like_boilerplate_line(line: str) -> bool:
    s = _norm_space(line)
    if not s:
        return True
    if len(s) <= 2:
        return True
    if _BOILERPLATE_LINE_RE.search(s):
        # Allow some longer in-article sentences; this is a heuristic.
        return True
    # Too "menu-like"
    if len(s) <= 18 and re.fullmatch(r"[\w\u4e00-\u9fff·\-\|/ ]{1,18}", s) and any(
        x in s for x in ("首页", "关于", "联系我们", "导航", "更多", "返回", "设置")
    ):
        return True
    return False


def _trim_boilerplate_blocks(blocks: List[str]) -> List[str]:
    if not blocks:
        return blocks
    start = 0
    end = len(blocks)
    while start < end and _looks_like_boilerplate_line(blocks[start]):
        start += 1
    while end > start and _looks_like_boilerplate_line(blocks[end - 1]):
        end -= 1
    blocks = blocks[start:end]

    # Drop obviously boilerplate lines inside (mostly short UI labels).
    out: List[str] = []
    for b in blocks:
        s = _norm_space(b)
        if not s:
            continue
        if _looks_like_boilerplate_line(s) and len(s) <= 60:
            continue
        out.append(s)

    # De-dup consecutive duplicates
    deduped: List[str] = []
    for s in out:
        if deduped and s == deduped[-1]:
            continue
        deduped.append(s)
    return deduped


def _extract_main_text_bs4(html: str) -> Tuple[Optional[str], Optional[str], List[str]]:
    if BeautifulSoup is None:
        return None, None, []

    soup = BeautifulSoup(html or "", "lxml")
    if not soup:
        return None, None, []

    def safe_id_class(tag: Any) -> str:
        attrs = getattr(tag, "attrs", None) or {}
        classes = attrs.get("class") or []
        if isinstance(classes, str):
            classes = [classes]
        tag_id = attrs.get("id") or ""
        return (" ".join([str(x) for x in classes if x]) + " " + str(tag_id)).strip()

    for tag in soup.find_all(["script", "style", "noscript", "svg", "canvas", "iframe", "form"]):
        tag.decompose()

    # Remove global chrome (keep header/footer inside article if any)
    for tag in soup.find_all(["header", "footer", "nav", "aside"]):
        if tag.find_parent("article") is None:
            tag.decompose()

    # Remove common junk containers
    for tag in soup.find_all(["div", "section", "aside"]):
        ident = safe_id_class(tag)
        if ident and _JUNK_IDCLASS_RE.search(ident):
            tag.decompose()

    def score_node(tag: Any) -> int:
        try:
            name = (tag.name or "").lower()
        except Exception:
            name = ""
        ident = safe_id_class(tag)
        text = tag.get_text(" ", strip=True) if tag else ""
        text = _norm_space(text)
        text_len = len(text)
        if text_len < 200:
            return -10_000
        p_count = len(tag.find_all("p")) if tag else 0
        li_count = len(tag.find_all("li")) if tag else 0
        h_count = len(tag.find_all(["h2", "h3", "h4"])) if tag else 0
        a_count = len(tag.find_all("a")) if tag else 0

        sc = text_len + p_count * 180 + li_count * 20 + h_count * 60 - a_count * 25
        if name == "article":
            sc += 700
        elif name == "main":
            sc += 400
        if ident and _CONTENT_HINT_RE.search(ident):
            sc += 250
        if ident and _JUNK_IDCLASS_RE.search(ident):
            sc -= 1200
        return sc

    candidates: List[Tuple[int, str, Any]] = []
    for tag in soup.find_all(["article", "main"]):
        candidates.append((score_node(tag), tag.name.lower(), tag))

    # Content-like div/section candidates (bounded)
    divs = soup.find_all(["div", "section"], limit=1800)
    for tag in divs:
        ident = safe_id_class(tag)
        if not ident or not _CONTENT_HINT_RE.search(ident):
            continue
        candidates.append((score_node(tag), tag.name.lower(), tag))

    if not candidates:
        return None, None, []

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_kind, best = candidates[0]
    if best_score <= -5000 or best is None:
        return None, None, []

    blocks: List[str] = []
    for el in best.find_all(["p", "h2", "h3", "h4", "li", "blockquote"]):
        t = _norm_space(el.get_text(" ", strip=True))
        if not t:
            continue
        # Avoid very short UI fragments
        if len(t) <= 3:
            continue
        if _looks_like_boilerplate_line(t) and len(t) <= 80:
            continue
        blocks.append(t)

    blocks = _trim_boilerplate_blocks(blocks)
    text = "\n\n".join(blocks).strip()
    if len(text) < 200:
        # Fallback to raw text if we trimmed too aggressively.
        text = _norm_space(best.get_text("\n", strip=True))
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # Extract images from chosen container
    imgs: List[str] = []
    for img in best.find_all("img"):
        attrs = img.attrs or {}
        cand = (
            attrs.get("data-src")
            or attrs.get("data-original")
            or attrs.get("data-url")
            or attrs.get("data-lazy-src")
            or attrs.get("src")
        )
        if not cand:
            srcset = attrs.get("srcset") or attrs.get("data-srcset")
            if srcset:
                parts = [p.strip().split(" ")[0] for p in str(srcset).split(",") if p.strip()]
                if parts:
                    cand = parts[-1]
        cand = (cand or "").strip()
        if not cand or cand.startswith("data:"):
            continue
        imgs.append(cand)

    return text or None, f"bs4:{best_kind}", imgs


def _extract_image_urls_from_html(html: str) -> List[str]:
    urls: List[str] = []
    meta = _extract_meta(html, "og:image") or _extract_meta(html, "twitter:image")
    if meta:
        urls.append(meta)
    for m in re.finditer(r"(?is)<img\b[^>]*>", html):
        tag = m.group(0)
        attrs = {}
        for am in re.finditer(r'([a-zA-Z_:][-a-zA-Z0-9_:.]*)\s*=\s*["\']([^"\']+)["\']', tag):
            attrs[am.group(1).lower()] = am.group(2)
        cand = (
            attrs.get("data-src")
            or attrs.get("data-original")
            or attrs.get("data-url")
            or attrs.get("src")
        )
        if not cand:
            srcset = attrs.get("srcset") or attrs.get("data-srcset")
            if srcset:
                parts = [p.strip().split(" ")[0] for p in srcset.split(",") if p.strip()]
                if parts:
                    cand = parts[-1]
        if cand:
            urls.append(cand)
    out = []
    seen = set()
    for u in urls:
        u = (u or "").strip()
        if not u or u.startswith("data:"):
            continue
        key = u.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(u)
    return out


def _rank_images(candidates: List[Tuple[str, str]], base_url: str) -> List[Dict[str, Any]]:
    def score_url(u: str, source: str) -> int:
        s = (u or "").lower()
        sc = 0
        if source == "jsonld":
            sc += 120
        elif source == "og":
            sc += 110
        elif source == "twitter":
            sc += 95
        elif source.startswith("content"):
            sc += 70
        else:
            sc += 40

        if s.startswith("http"):
            sc += 10
        if any(s.endswith(x) for x in (".jpg", ".jpeg", ".png", ".webp")):
            sc += 18
        if s.endswith(".gif"):
            sc -= 6
        if s.endswith(".svg"):
            sc -= 25
        if any(x in s for x in ("logo", "icon", "sprite", "avatar", "emoji", "favicon")):
            sc -= 40
        if any(x in s for x in ("thumb", "thumbnail", "small", "40x40", "60x60", "80x80", "120x120")):
            sc -= 25
        if any(x in s for x in ("large", "original", "1200", "1920", "1080")):
            sc += 8
        return sc

    seen = set()
    ranked: List[Dict[str, Any]] = []
    for raw_url, source in candidates:
        u = (raw_url or "").strip()
        if not u or u.startswith("data:"):
            continue
        abs_u = urljoin(base_url, u)
        key = abs_u.split("#", 1)[0].lower()
        if key in seen:
            continue
        seen.add(key)
        ranked.append({"url": abs_u, "source": source, "score": score_url(abs_u, source)})

    ranked.sort(key=lambda x: (int(x.get("score", 0)), x.get("source", "")), reverse=True)
    return ranked


def _download_image(url: str, referer: str) -> Tuple[Optional[bytes], Optional[str], Optional[str]]:
    if not url:
        return None, None, "empty_image_url"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Referer": referer,
    }
    try:
        try:
            r = requests.get(url, headers=headers, timeout=12, allow_redirects=True, stream=True)
        except requests.exceptions.SSLError:
            r = requests.get(url, headers=headers, timeout=12, allow_redirects=True, stream=True, verify=False)
        r.raise_for_status()
        ctype = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype and not ctype.startswith("image/"):
            return None, None, f"not_image:{ctype}"
        max_bytes = 5 * 1024 * 1024
        chunks = []
        total = 0
        for chunk in r.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                return None, None, "image_too_large"
        data = b"".join(chunks)
        if not data:
            return None, None, "empty_image_bytes"
        return data, (ctype or "image/jpeg"), None
    except Exception as e:
        return None, None, str(e)


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_url_payload(url: str) -> Dict[str, Any]:
    url = (url or "").strip()
    if not url:
        return {"ok": False, "error": "empty_url", "url": url}
    if not re.match(r"^https?://", url, flags=re.IGNORECASE):
        url = "https://" + url

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }
    try:
        def do_get(target: str, hdrs: Dict[str, str], verify: bool):
            return requests.get(target, headers=hdrs, timeout=15, allow_redirects=True, verify=verify)

        tls_verify = True
        try:
            resp = do_get(url, headers, True)
        except requests.exceptions.SSLError:
            tls_verify = False
            resp = do_get(url, headers, False)

        if resp.status_code in (403, 429, 451, 503):
            alt_headers = dict(headers)
            alt_headers["User-Agent"] = "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"
            alt_headers["Accept-Language"] = "en-US,en;q=0.9"
            try:
                resp2 = do_get(url, alt_headers, tls_verify)
                if resp2.ok:
                    resp = resp2
            except Exception:
                pass

        if not resp.ok:
            code = int(resp.status_code)
            if code == 403:
                msg = "站点拒绝抓取(403)，可能需要登录或存在反爬限制"
            elif code == 404:
                msg = "网页不存在(404)"
            elif code == 429:
                msg = "站点限流(429)，请稍后重试"
            elif code == 451:
                msg = "站点拒绝提供内容(451)"
            elif code == 503:
                msg = "站点服务不可用(503)"
            else:
                msg = f"抓取失败(HTTP {code})"
            return {"ok": False, "error": msg, "url": url, "status_code": code, "final_url": str(resp.url), "tls_verify": tls_verify}

        final_url = str(resp.url)
        # requests automatically decodes resp.text using resp.encoding at first access.
        # If the server sends no charset in headers, requests defaults to ISO-8859-1,
        # but the page is often UTF-8/GBK, causing mojibake. Since resp.encoding is
        # already consumed at this point, we must re-decode from the raw bytes using
        # apparent_encoding (the real encoding detected by charset-normalizer).
        detected_encoding = resp.apparent_encoding or "utf-8"
        safe_encoding = detected_encoding if detected_encoding.lower() != "iso-8859-1" else "utf-8"
        html = resp.content.decode(safe_encoding, errors="replace")
    except Exception as e:
        err = str(e)
        if "WinError 10061" in err:
            err = "连接被拒绝(WinError 10061)，目标站点可能屏蔽访问或当前网络不可达"
        elif "NameResolutionError" in err or "getaddrinfo failed" in err:
            err = "域名解析失败，可能是网络或 DNS 问题"
        elif "ConnectTimeoutError" in err or "ReadTimeout" in err:
            err = "连接超时，目标站点响应过慢或网络受限"
        return {"ok": False, "error": err, "url": url}

    site_name = _extract_site_name(html)
    hostname = _hostname_from_url(final_url) or _hostname_from_url(url)

    title_raw = _extract_title(html)
    title = _clean_title(title_raw, site_name=site_name, hostname=hostname)
    desc = _extract_description(html)

    jsonld_blocks = _find_jsonld(html)
    jsonld_nodes: List[Dict[str, Any]] = []
    for b in jsonld_blocks:
        jsonld_nodes.extend(_iter_jsonld_nodes(b))
    art = _pick_jsonld_article(jsonld_nodes)

    body_text = None
    body_source = None
    image_from_jsonld = None
    if art:
        headline = art.get("headline")
        if not title and isinstance(headline, str):
            title = _clean_title(headline, site_name=site_name, hostname=hostname) or title
        body = art.get("articleBody") or art.get("text")
        if isinstance(body, str) and body.strip():
            body_text = _norm_space(body)
            body_source = "jsonld"
        image_from_jsonld = _jsonld_image_to_url(art.get("image"))

    bs4_text, bs4_source, bs4_imgs = _extract_main_text_bs4(html)
    main_html, main_source = _pick_best_body_html(html)
    if not main_source and bs4_source:
        main_source = bs4_source

    if not body_text:
        if bs4_text:
            body_text = bs4_text
            body_source = bs4_source or "bs4"
        elif main_html:
            body_text = _strip_html_to_text(main_html)
            body_source = main_source or "html"
        else:
            body_text = _strip_html_to_text(html)
            body_source = "full_html"

    body_text = body_text or ""
    body_snippet = body_text[:2500]

    image_candidates: List[Tuple[str, str]] = []
    if image_from_jsonld:
        image_candidates.append((image_from_jsonld, "jsonld"))

    og_img = _extract_meta(html, "og:image")
    if og_img:
        image_candidates.append((og_img, "og"))
    tw_img = _extract_meta(html, "twitter:image")
    if tw_img:
        image_candidates.append((tw_img, "twitter"))

    for u in bs4_imgs[:60]:
        image_candidates.append((u, "content"))
    if main_html:
        for u in _extract_image_urls_from_html(main_html)[:80]:
            image_candidates.append((u, "content_html"))
    for u in _extract_image_urls_from_html(html)[:120]:
        image_candidates.append((u, "full_html"))

    ranked_images = _rank_images(image_candidates, base_url=final_url)
    img_url = ranked_images[0]["url"] if ranked_images else None

    image_bytes = None
    image_mime = None
    image_error = None
    if img_url:
        image_bytes, image_mime, image_error = _download_image(img_url, referer=final_url)

    text_default = ""
    if title and body_text:
        text_default = f"{title}\n\n{body_text}"
    elif title:
        text_default = title
    elif desc:
        text_default = desc
    else:
        text_default = body_snippet[:600]

    return {
        "ok": True,
        "url": url,
        "final_url": final_url,
        "tls_verify": tls_verify,
        "site_name": site_name,
        "hostname": hostname,
        "title_raw": title_raw,
        "title": title,
        "description": desc,
        "body_text": body_text,
        "body_snippet": body_snippet,
        "text": text_default,
        "image_url": img_url,
        "images": ranked_images[:30],
        "image_urls": [x["url"] for x in ranked_images[:30]],
        "image_bytes": image_bytes,
        "image_mime": image_mime,
        "image_error": image_error,
        "body_source": body_source,
    }
