import re
import time
import unicodedata
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Customer Intelligence", page_icon="📊", layout="wide")

COLORS = ["#2563EB", "#10B981", "#8B5CF6", "#F59E0B", "#EC4899", "#06B6D4", "#64748B", "#EF4444"]
MONEY = ["VND", "USD", "EUR", "GBP", "JPY", "CNY", "KRW", "AUD", "CAD", "SGD", "THB", "MYR", "INR", "BRL"]
SYMBOLS = {"₫": "VND", "€": "EUR", "£": "GBP", "₩": "KRW", "₹": "INR", "R$": "BRL"}
DATES = ["Tự nhận diện", "DD/MM/YYYY", "MM/DD/YYYY", "YYYY-MM-DD", "Theo cột quốc gia"]
COUNTRY_DATE = {
    "vietnam": "DD/MM/YYYY", "viet nam": "DD/MM/YYYY", "united kingdom": "DD/MM/YYYY",
    "uk": "DD/MM/YYYY", "france": "DD/MM/YYYY", "germany": "DD/MM/YYYY",
    "australia": "DD/MM/YYYY", "india": "DD/MM/YYYY", "thailand": "DD/MM/YYYY",
    "singapore": "DD/MM/YYYY", "united states": "MM/DD/YYYY", "usa": "MM/DD/YYYY",
    "us": "MM/DD/YYYY", "japan": "YYYY-MM-DD", "china": "YYYY-MM-DD",
    "south korea": "YYYY-MM-DD", "korea": "YYYY-MM-DD", "canada": "YYYY-MM-DD",
}

# ===========================================================
# 1. GIAO DIỆN SÁNG - UPLOAD LỚN - CHỮ DỄ ĐỌC
# ===========================================================
st.markdown("""
<style>
:root {color-scheme:light;}
.stApp,[data-testid="stAppViewContainer"] {background:#F6F8FC!important;color:#17243B!important;}
.block-container {max-width:1250px;padding-top:1.4rem;padding-bottom:3rem;}
[data-testid="stHeader"] {background:transparent!important;}
.stApp [data-testid="stMarkdownContainer"] h1,
.stApp [data-testid="stMarkdownContainer"] h2,
.stApp [data-testid="stMarkdownContainer"] h3,
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stExpander"] summary {color:#17243B!important;}
.stApp [data-testid="stCaptionContainer"] p {color:#586980!important;}
.stApp [data-baseweb="select"]>div,
.stApp [data-baseweb="input"]>div,
.stApp input {background:#FFF!important;color:#17243B!important;}
.stApp [data-baseweb="select"] span,
.stApp [data-baseweb="select"] input {color:#17243B!important;}
.stApp [data-testid="stExpander"] {background:#FFF;border:1px solid #E2E8F0;border-radius:12px;}
.brand {background:#FFF;border:1px solid #DFE7F2;border-radius:14px;
 padding:16px 22px;color:#1D4ED8;font-weight:800;margin-bottom:18px;}
.hero-tag {text-align:center;color:#2563EB;font-size:12px;font-weight:800;letter-spacing:1.1px;}
.hero-title {text-align:center;color:#15253C;font-weight:800;
 font-size:clamp(26px,3.8vw,40px);margin:10px 0;}
.hero-sub {text-align:center;color:#586980;font-size:14px;margin-bottom:18px;}
[data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stFileUploader"]) {
 background:#FFF!important;border:1px solid #DFE7F2!important;
 border-radius:20px!important;padding:20px 22px 25px;
 box-shadow:0 9px 28px rgba(30,70,110,.05);}
.stApp [data-testid="stFileUploaderDropzone"] {
 min-height:245px!important;padding:25px!important;background:#F7FAFF!important;
 border:2px dashed #91B6ED!important;border-radius:15px!important;
 display:flex!important;flex-direction:column!important;
 align-items:center!important;justify-content:center!important;gap:12px;}
.stApp [data-testid="stFileUploaderDropzone"]::before {
 content:"⬆  UPLOAD FILE";display:block;color:#1D4ED8;font-size:24px;
 font-weight:800;letter-spacing:1px;text-align:center;}
.stApp [data-testid="stFileUploaderDropzone"] section,
.stApp [data-testid="stFileUploaderDropzone"] span,
.stApp [data-testid="stFileUploaderDropzone"] small {color:#334155!important;text-align:center;}
.stApp [data-testid="stFileUploaderDropzone"] button {
 background:#2563EB!important;color:#FFF!important;border:0!important;border-radius:10px;}
.stApp [data-testid="stFileUploaderDropzone"] button * {color:#FFF!important;}
.kpi {background:#FFF;border:1px solid #E2E8F0;border-radius:14px;padding:18px;min-height:104px;}
.kpi-label {color:#586980;font-size:12px;}
.kpi-value {color:#1D4ED8;font-size:25px;font-weight:800;margin-top:8px;overflow-wrap:anywhere;}
.matrix-scroll {overflow-x:auto;background:#FFF;border:1px solid #DFE7F2;border-radius:13px;padding:8px;}
.evo-matrix {width:100%;min-width:590px;border-spacing:5px;text-align:center;font-size:12px;}
.evo-matrix th {background:#EFF5FF;border-radius:7px;padding:11px 6px;color:#234365;}
.evo-matrix td {border-radius:8px;padding:12px 6px;color:#17243B;}
.evo-matrix strong {display:block;font-size:19px;color:#17243B;}
.evo-matrix small {display:block;margin-top:4px;color:#475569;}
@media(max-width:640px){.stApp [data-testid="stFileUploaderDropzone"]{min-height:205px!important;padding:12px!important;}}
</style>
""", unsafe_allow_html=True)


def heading(value):
    st.markdown(f"### {value}")


def metric_card(label, value):
    st.markdown(
        f'<div class="kpi"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div></div>', unsafe_allow_html=True
