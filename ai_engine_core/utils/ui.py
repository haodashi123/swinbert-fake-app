from __future__ import annotations

import streamlit as st

from utils.icons import inline_svg


def inject_global_styles() -> None:
    st.markdown(
        """
        <style>
          :root {
            --igp-bg: #f5f7fb;
            --igp-surface: #ffffff;
            --igp-surface-2: #f8fafc;
            --igp-border: #e5e7eb;
            --igp-text: #0f172a;
            --igp-muted: #64748b;
            --igp-primary: #1260A3;
            --igp-primary-rgb: 18, 96, 163;
            --igp-primary-hover: #0f4f86;
            --igp-primary-active: #0b3a63;
            --igp-primary-weak: rgba(var(--igp-primary-rgb), 0.12);
            --igp-success: #2e7d32;
            --igp-danger: #c62828;
            --igp-radius: 12px;
          }

          /* Hide Streamlit chrome to feel like an app */
          #MainMenu { visibility: hidden; }
          footer { visibility: hidden; }
          header[data-testid="stHeader"] { display: none; }
          div[data-testid="stToolbar"] { visibility: hidden; height: 0; }
          div[data-testid="stDecoration"] { display: none; }

          html, body, .stApp {
            background: var(--igp-bg);
            color: var(--igp-text);
            font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial,
              "Apple Color Emoji", "Segoe UI Emoji";
          }

          .igp-svg svg {
            width: 1em;
            height: 1em;
            display: block;
          }

          section.main > div.block-container {
            padding-top: 1.25rem;
            padding-bottom: 1.25rem;
            max-width: 1440px;
          }

          h1, h2, h3, h4 {
            color: var(--igp-text);
            letter-spacing: -0.02em;
          }

          /* Sidebar: neutral "desktop app" panel */
          section[data-testid="stSidebar"] {
            background: var(--igp-surface);
            border-right: 1px solid var(--igp-border);
          }
          section[data-testid="stSidebar"] > div {
            padding-top: 1rem;
          }
          [data-testid="stSidebarNav"] {
            display: none !important;
          }
          section[data-testid="stSidebar"] nav[aria-label="Streamlit navigation"] {
            display: none !important;
          }
          section[data-testid="stSidebar"] nav[aria-label="Pages"] {
            display: none !important;
          }

          /* Sidebar brand block */
          [data-testid="stSidebar"] .igp-brand {
            display: flex;
            gap: 0.75rem;
            align-items: center;
            padding: 0.25rem 0.25rem 0.75rem 0.25rem;
          }
          [data-testid="stSidebar"] .igp-brand__icon {
            width: 36px;
            height: 36px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: var(--igp-primary-weak);
            border: 1px solid rgba(var(--igp-primary-rgb), 0.18);
            font-size: 18px;
          }
          [data-testid="stSidebar"] .igp-brand__name {
            font-weight: 800;
            font-size: 1.05rem;
            line-height: 1.1;
          }
          [data-testid="stSidebar"] .igp-brand__sub {
            color: var(--igp-muted);
            font-size: 0.82rem;
            margin-top: 0.15rem;
            line-height: 1.25;
          }
          [data-testid="stSidebar"] .igp-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.1rem 0.5rem;
            border-radius: 999px;
            background: var(--igp-surface-2);
            border: 1px solid var(--igp-border);
            color: var(--igp-muted);
            font-size: 0.75rem;
            margin-left: 0.35rem;
          }

          /* Sidebar nav buttons: align like a menu */
          [data-testid="stSidebar"] div.stButton button {
            width: 100%;
            justify-content: flex-start;
            gap: 0.5rem;
            border-radius: 12px;
            padding: 0.55rem 0.75rem;
            font-weight: 650;
          }

          [data-testid="stSidebar"] .igp-nav-icon {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 2.25rem;
          }

          /* Buttons/inputs: softer radius for "app" feel */
          .stApp div.stButton button,
          .stApp div.stDownloadButton button,
          .stApp div.stFormSubmitButton button,
          .stApp button[kind="secondary"],
          .stApp button[kind="primary"] {
            border-radius: 12px;
            font-weight: 650;
            transition: background-color 120ms ease, border-color 120ms ease, color 120ms ease, transform 80ms ease;
          }

          /* Secondary buttons */
          .stApp div.stButton button,
          .stApp div.stDownloadButton button,
          .stApp div.stFormSubmitButton button,
          .stApp button[kind="secondary"] {
            border: 1px solid var(--igp-border) !important;
            background: var(--igp-surface) !important;
            color: var(--igp-text) !important;
          }
          .stApp div.stButton button:hover,
          .stApp div.stDownloadButton button:hover,
          .stApp div.stFormSubmitButton button:hover,
          .stApp button[kind="secondary"]:hover {
            border-color: rgba(var(--igp-primary-rgb), 0.45) !important;
            background: rgba(var(--igp-primary-rgb), 0.06) !important;
          }
          .stApp div.stButton button:active,
          .stApp div.stDownloadButton button:active,
          .stApp div.stFormSubmitButton button:active,
          .stApp button[kind="secondary"]:active {
            transform: translateY(0.5px);
          }

          /* Primary buttons */
          .stApp div.stButton button[kind="primary"],
          .stApp div.stDownloadButton button[kind="primary"],
          .stApp div.stFormSubmitButton button[kind="primary"],
          .stApp button[kind="primary"],
          .stApp button[data-testid="baseButton-primary"] {
            background: var(--igp-primary) !important;
            border: 1px solid var(--igp-primary) !important;
            color: #ffffff !important;
          }
          .stApp div.stButton button[kind="primary"]:hover,
          .stApp div.stDownloadButton button[kind="primary"]:hover,
          .stApp div.stFormSubmitButton button[kind="primary"]:hover,
          .stApp button[kind="primary"]:hover,
          .stApp button[data-testid="baseButton-primary"]:hover {
            background: var(--igp-primary-hover) !important;
            border-color: var(--igp-primary-hover) !important;
          }
          .stApp div.stButton button[kind="primary"]:active,
          .stApp div.stDownloadButton button[kind="primary"]:active,
          .stApp div.stFormSubmitButton button[kind="primary"]:active,
          .stApp button[kind="primary"]:active,
          .stApp button[data-testid="baseButton-primary"]:active {
            background: var(--igp-primary-active) !important;
            border-color: var(--igp-primary-active) !important;
          }

          /* Focus + disabled */
          .stApp div.stButton button:focus-visible,
          .stApp div.stDownloadButton button:focus-visible,
          .stApp div.stFormSubmitButton button:focus-visible,
          .stApp button[kind="secondary"]:focus-visible,
          .stApp button[kind="primary"]:focus-visible {
            outline: 3px solid rgba(var(--igp-primary-rgb), 0.25);
            outline-offset: 2px;
          }
          .stApp div.stButton button:disabled,
          .stApp div.stDownloadButton button:disabled,
          .stApp div.stFormSubmitButton button:disabled,
          .stApp button[kind="secondary"]:disabled,
          .stApp button[kind="primary"]:disabled {
            opacity: 0.55;
            cursor: not-allowed;
            transform: none;
          }
          textarea {
            border-radius: 12px !important;
          }
          div[data-baseweb="input"] > div,
          div[data-baseweb="input"] input {
            border-radius: 12px;
          }
          .stApp div[data-baseweb="input"] input:focus,
          .stApp div[data-baseweb="input"] input:focus-visible,
          .stApp div[data-baseweb="input"] > div:focus-within {
            border-color: rgba(var(--igp-primary-rgb), 0.75) !important;
            box-shadow: 0 0 0 3px rgba(var(--igp-primary-rgb), 0.18) !important;
          }
          .stApp textarea:focus,
          .stApp textarea:focus-visible {
            border-color: rgba(var(--igp-primary-rgb), 0.75) !important;
            box-shadow: 0 0 0 3px rgba(var(--igp-primary-rgb), 0.18) !important;
          }

          /* Slider: align thumb + active rail to theme primary */
          .stApp div[data-testid="stSlider"] [data-baseweb="slider"] > div {
            /* Keep large hit-area, remove the "big grey bar" background */
            background: transparent !important;
            background-color: transparent !important;
            padding: 10px 8px !important;
          }
          .stApp div[data-testid="stSlider"] [data-baseweb="slider"] > div > div {
            /* Inner track */
            height: 4px !important;
            border-radius: 999px !important;
          }
          .stApp div[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
            background: var(--igp-primary) !important;
            border-color: var(--igp-primary) !important;
            box-shadow: none !important;
          }
          .stApp div[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"]:focus,
          .stApp div[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"]:focus-visible {
            box-shadow: 0 0 0 4px rgba(var(--igp-primary-rgb), 0.20) !important;
          }
          /* Let Streamlit theme drive the rail gradient (overrides here tend to break layout). */

          /* Progress bars */
          .stApp div[data-testid="stProgress"] div[role="progressbar"] {
            background: var(--igp-border) !important;
          }
          .stApp div[data-testid="stProgress"] div[role="progressbar"] > div {
            background: var(--igp-primary) !important;
          }

          /* Spinners */
          .stApp div[data-testid="stSpinner"] svg {
            color: var(--igp-primary) !important;
            fill: var(--igp-primary) !important;
          }

          /* Toggle (st.toggle uses stCheckbox + BaseWeb toggle track) */
          .stApp div[data-testid="stCheckbox"] div[data-baseweb="checkbox"]:has(> input) > div:first-child:has(> div) {
            background: rgba(var(--igp-primary-rgb), 0.22) !important;
          }
          .stApp div[data-testid="stCheckbox"] div[data-baseweb="checkbox"]:has(> input:checked) > div:first-child:has(> div) {
            background: var(--igp-primary) !important;
          }
          .stApp div[data-testid="stCheckbox"] div[data-baseweb="checkbox"] > div:first-child:has(> div) > div {
            background: var(--igp-surface) !important;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.18) !important;
          }
          .stApp div[data-testid="stCheckbox"] div[data-baseweb="checkbox"] > div:first-child:has(> div) > div:focus,
          .stApp div[data-testid="stCheckbox"] div[data-baseweb="checkbox"] > div:first-child:has(> div) > div:focus-visible,
          .stApp div[data-testid="stCheckbox"] div[data-baseweb="checkbox"]:has(> input:focus-visible) > div:first-child:has(> div) > div {
            box-shadow: 0 0 0 3px rgba(var(--igp-primary-rgb), 0.20) !important;
          }

          /* Metrics: avoid Streamlit brand red (#FF4B4B) for deltas */
          .stApp div[data-testid="stMetricDelta"] svg {
            color: var(--igp-primary) !important;
            fill: var(--igp-primary) !important;
          }
          .stApp div[data-testid="stMetricDelta"] {
            color: var(--igp-muted) !important;
          }

          /* Alerts (st.info/st.warning/st.error/st.success) -> brand styling */
          div[data-testid="stAlert"] {
            border-radius: 12px;
            border: 1px solid rgba(var(--igp-primary-rgb), 0.28) !important;
            background: rgba(var(--igp-primary-rgb), 0.07) !important;
          }
          div[data-testid="stAlert"] * {
            color: var(--igp-text) !important;
          }
          div[data-testid="stAlert"] svg {
            color: var(--igp-primary) !important;
            fill: var(--igp-primary) !important;
          }

          /* Page header helper */
          .igp-page-header {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 1rem;
            margin: 0 0 1.25rem 0;
          }
          .igp-page-title {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            font-size: 1.45rem;
            font-weight: 800;
            line-height: 1.15;
          }
          .igp-page-icon {
            width: 34px;
            height: 34px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: var(--igp-surface);
            border: 1px solid var(--igp-border);
          }
          .igp-page-subtitle {
            color: var(--igp-muted);
            margin-top: 0.25rem;
            font-size: 0.95rem;
            line-height: 1.35;
          }

          /* Global image limits and force centering */
          div[data-testid="stImage"] {
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            width: 100% !important;
            text-align: center !important;
          }
          div[data-testid="stImage"] > div {
            display: flex !important;
            justify-content: center !important;
            width: 100% !important;
          }
          div[data-testid="stImage"] img {
            max-width: 100% !important;
            max-height: 400px !important;
            width: auto !important;
            height: auto !important;
            object-fit: contain !important;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(title: str, subtitle: str | None = None, icon: str | None = None) -> None:
    icon_html = (
        f"<span class='igp-page-icon'>{inline_svg(icon, size_px=18, color='var(--igp-primary)')}</span>"
        if icon
        else ""
    )
    subtitle_html = f"<div class='igp-page-subtitle'>{subtitle}</div>" if subtitle else ""
    st.markdown(
        f"""
        <div class="igp-page-header">
          <div>
            <div class="igp-page-title">{icon_html}<span>{title}</span></div>
            {subtitle_html}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
