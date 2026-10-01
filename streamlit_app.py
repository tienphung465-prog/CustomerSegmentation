
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

st.set_page_config(
    page_title="Customer Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

COLORS = [
    "#2563eb", "#10b981", "#8b5cf6", "#f59e0b",
    "#ec4899", "#06b6d4", "#6366f1", "#ef4444"
]

FORMATS = [
    "Tự nhận diện",
    "DD/MM/YYYY",
    "MM/DD/YYYY",
    "YYYY-MM-DD",
    "Theo cột quốc gia",
]

COUNTRY_FORMAT = {
    "vietnam": "DD/MM/YYYY",
    "viet nam": "DD/MM/YYYY",
    "united kingdom": "DD/MM/YYYY",
    "uk": "DD/MM/YYYY",
    "france": "DD/MM/YYYY",
    "germany": "DD/MM/YYYY",
    "australia": "DD/MM/YYYY",
    "india": "DD/MM/YYYY",
    "thailand": "DD/MM/YYYY",
    "united states": "MM/DD/YYYY",
    "usa": "MM/DD/YYYY",
    "us": "MM/DD/YYYY",
    "canada": "YYYY-MM-DD",
    "japan": "YYYY-MM-DD",
    "china": "YYYY-MM-DD",
    "south korea": "YYYY-MM-DD",
    "korea": "YYYY-MM-DD",
    "singapore": "DD/MM/YYYY",
}

CURRENCY_CODES = [
    "VND", "USD", "EUR", "GBP", "JPY",
    "CNY", "KRW", "AUD", "CAD", "SGD",
    "THB", "MYR", "INR",
]

SYMBOL_TO_CODE = {
    "₫": "VND",
    "€": "EUR",
    "£": "GBP",
    "₩": "KRW",
    "₹": "INR",
}


# =====================================================
# 1. CSS - GIAO DIỆN SÁNG + UPLOAD LỚN
# =====================================================

st.markdown("""
<style>
:root {
    color-scheme: light;
}

.stApp,
[data-testid="stAppViewContainer"] {
    background: #f6f8fc !important;
    color: #17243b !important;
}

.block-container {
    max-width: 1280px;
    padding-top: 1.35rem;
    padding-bottom: 3rem;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

/* Đảm bảo chữ luôn dễ đọc trên nền trắng */

.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stMarkdownContainer"] li,
.stApp [data-testid="stMarkdownContainer"] h1,
.stApp [data-testid="stMarkdownContainer"] h2,
.stApp [data-testid="stMarkdownContainer"] h3,
.stApp [data-testid="stMarkdownContainer"] h4,
.stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stCaptionContainer"] p,
.stApp [data-testid="stExpander"] summary,
.stApp [data-testid="stMetricLabel"],
.stApp [data-testid="stMetricValue"] {
    color: #17243b !important;
}

.stApp [data-testid="stCaptionContainer"] p {
    color: #64748b !important;
}

/* Chữ trong các ô chọn và ô nhập */

.stApp [data-baseweb="select"] > div,
.stApp [data-baseweb="input"] > div,
.stApp [data-baseweb="input"] input,
.stApp [data-testid="stTextInput"] input {
    background-color: #ffffff !important;
    color: #17243b !important;
}

.stApp [data-baseweb="select"] span,
.stApp [data-baseweb="select"] input,
.stApp [data-baseweb="select"] div[role="button"] {
    color: #17243b !important;
}

.stApp [data-testid="stExpander"] {
    background: #ffffff;
    border: 1px solid #dce5f1;
    border-radius: 12px;
}

/* Thanh thương hiệu */

.brand {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 16px 22px;
    color: #1d4ed8 !important;
    font-weight: 800;
    letter-spacing: .3px;
    margin-bottom: 16px;
}

/* Thẻ trắng chứa phần tải dữ liệu */

[data-testid="stVerticalBlockBorderWrapper"]:has(
    [data-testid="stFileUploader"]
) {
    background: #ffffff !important;
    border: 1px solid #dbe5f1 !important;
    border-radius: 22px !important;
    box-shadow: 0 9px 28px rgba(36,72,130,.055);
    padding: 22px 24px 26px;
}

.hero-title {
    text-align: center;
    font-size: clamp(26px,4vw,39px);
    font-weight: 800;
    line-height: 1.25;
    color: #15253c;
    margin: 8px 0 12px;
}

.hero-eyebrow {
    text-align: center;
    font-size: 11px;
    font-weight: 800;
    color: #2563eb;
    letter-spacing: 1.3px;
}

.hero-sub {
    text-align: center;
    font-size: 14px;
    color: #596a83;
    margin-bottom: 24px;
}

/* KHU VỰC UPLOAD - CHỮ NẰM TRONG KHUNG */

.stApp [data-testid="stFileUploaderDropzone"] {
    min-height: 250px !important;
    padding: 24px 30px 30px !important;
    background: #f8fbff !important;
    border: 2px dashed #8fb5ee !important;
    border-radius: 17px !important;

    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 14px;

    transition: background-color .2s, border-color .2s;
}

.stApp [data-testid="stFileUploaderDropzone"]:hover {
    background: #eff6ff !important;
    border-color: #2563eb !important;
}

.stApp [data-testid="stFileUploaderDropzone"]::before {
    content: "⬆  UPLOAD FILE";
    display: block;
    color: #1d4ed8 !important;
    font-size: clamp(18px,3vw,25px);
    font-weight: 800;
    letter-spacing: 1.5px;
    text-align: center;
}

.stApp [data-testid="stFileUploaderDropzone"] > div {
    margin: 0 auto;
    text-align: center;
}

.stApp [data-testid="stFileUploaderDropzone"] section,
.stApp [data-testid="stFileUploaderDropzone"] span,
.stApp [data-testid="stFileUploaderDropzone"] small,
.stApp [data-testid="stFileUploaderDropzone"]
[data-testid="stFileUploaderDropzoneInstructions"] {
    color: #334155 !important;
    text-align: center;
}

.stApp [data-testid="stFileUploaderDropzone"] button {
    border: 1px solid #2563eb !important;
    border-radius: 10px !important;
    background: #2563eb !important;
    color: #ffffff !important;
    font-weight: 700;
}

.stApp [data-testid="stFileUploaderDropzone"] button * {
    color: #ffffff !important;
}

.upload-help {
    font-size: 12px;
    color: #64748b;
    text-align: center;
    margin: 4px 0 8px;
}

/* Thẻ thống kê */

.kpi {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 18px;
    min-height: 105px;
}

.kpi small {
    color: #64748b !important;
    font-size: 12px;
}

.kpi strong {
    display: block;
    font-size: 25px;
    color: #1d4ed8 !important;
    margin-top: 8px;
    overflow-wrap: anywhere;
}

/* Điện thoại */

@media (max-width: 640px) {
    .stApp [data-testid="stFileUploaderDropzone"] {
        min-height: 215px !important;
        padding: 16px !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"]:has(
        [data-testid="stFileUploader"]
    ) {
        padding: 12px !important;
    }
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# 2. HIỂN THỊ
# =====================================================

def heading(text):
    st.markdown(f"### {text}")


def kpi(label, value):
    st.markdown(
        f'<div class="kpi">'
        f'<small>{label}</small>'
        f'<strong>{value}</strong>'
        '</div>',
        unsafe_allow_html=True,
    )


def graph(fig, height=370):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=25, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# =====================================================
# 3. MA TRẬN CHUYỂN NHÓM
# =====================================================

def show_transition_matrix(matrix, pairs, k):

    names_before = [
        f"Nhóm {i + 1} tháng trước"
        for i in range(k)
    ]

    names_after = [
        f"Nhóm {i + 1} tháng sau"
        for i in range(k)
    ]

    counts = matrix.reindex(
        index=range(k),
        columns=range(k),
        fill_value=0,
    ).astype(int)

    total = int(counts.to_numpy().sum())
    stayed = int(np.trace(counts.to_numpy()))
    moved = total - stayed

    if total == 0:
        st.info(
            "Chưa có khách hàng giao dịch "
            "ở hai tháng liên tiếp để so sánh nhóm."
        )
        return

    a, b, c = st.columns(3)

    a.metric("Tổng lượt đối chiếu", f"{total:,}")

    b.metric(
        "🟢 Giữ nguyên nhóm",
        f"{stayed:,}",
        f"{stayed / total:.1%} lượt",
    )

    c.metric(
        "🔵 Chuyển sang nhóm khác",
        f"{moved:,}",
        f"{moved / total:.1%} lượt",
    )

    st.caption(
        "**Nhóm tháng trước:** các hàng (↓). "
        "**Nhóm tháng sau:** các cột (→). "
        "Xanh lá: giữ nhóm · Xanh dương: chuyển nhóm. "
        "Tỷ lệ dưới mỗi ô tính theo hàng."
    )

    st.markdown("""
    <style>
    .matrix-scroll {
        overflow-x: auto;
        max-width: 100%;
        border: 1px solid #dce5f1;
        border-radius: 14px;
        background: #ffffff;
        padding: 8px;
    }

    table.evo-matrix {
        width: 100%;
        min-width: 590px;
        border-collapse: separate;
        border-spacing: 5px;
        font-size: 13px;
        text-align: center;
    }

    .evo-matrix th {
        padding: 10px 6px;
        background: #eff5ff;
        border-radius: 8px;
        color: #274267;
        white-space: nowrap;
        font-weight: 700;
    }

    .evo-matrix .axis {
        background: #e1eafb;
        color: #1e40af;
    }

    .evo-matrix td {
        border-radius: 9px;
        padding: 10px 5px;
        min-width: 72px;
    }

    .evo-matrix .num {
        display: block;
        font-size: 20px;
        font-weight: 800;
        color: #17243b;
    }

    .evo-matrix .pct {
        display: block;
        font-size: 11px;
        margin-top: 5px;
        color: #475569;
    }

    .evo-matrix .sum {
        background: #f1f5f9;
        font-weight: 800;
        color: #334155;
    }
    </style>
    """, unsafe_allow_html=True)

    max_cell = max(
        int(counts.to_numpy().max()),
        1,
    )

    parts = [
        '<div class="matrix-scroll">',
        '<table class="evo-matrix">',
        '<thead><tr>',
        '<th class="axis" rowspan="2">',
        'Nhóm tháng trước ↓',
        '</th>',
        f'<th class="axis" colspan="{k}">',
        'Nhóm tháng sau →',
        '</th>',
        '<th class="axis" rowspan="2">',
        'Tổng lượt theo nhóm trước',
        '</th>',
        '</tr><tr>',
    ]

    for name in names_after:
        parts.append(f"<th>{name}</th>")

    parts.append("</tr></thead><tbody>")

    for i in range(k):
        row_total = int(counts.loc[i].sum())

        parts.append(
            f"<tr><th>{names_before[i]}</th>"
        )

        for j in range(k):
            n = int(counts.loc[i, j])

            pct = (
                100 * n / row_total
                if row_total
                else 0
            )

            alpha = (
                0.07 + 0.48 * np.sqrt(n / max_cell)
                if n
                else 0.035
            )

            color = (
                f"rgba(16,185,129,{alpha:.3f})"
                if i == j
                else f"rgba(37,99,235,{alpha:.3f})"
            )

            parts.append(
                f'<td style="background:{color}">'
                f'<span class="num">{n:,}</span>'
                f'<span class="pct">{pct:.1f}%</span>'
                '</td>'
            )

        parts.append(
            f'<td class="sum">{row_total:,}</td>'
            '</tr>'
        )

    parts.append(
        '<tr><th class="axis">'
        'Tổng lượt theo nhóm sau</th>'
    )

    for j in range(k):
        parts.append(
            f'<td class="sum">'
            f'{int(counts[j].sum()):,}</td>'
        )

    parts.append(
        f'<td class="sum">{total:,}</td>'
        '</tr></tbody></table></div>'
    )

    st.markdown(
        "".join(parts),
        unsafe_allow_html=True,
    )

    movements = []

    for i in range(k):
        outgoing = int(counts.loc[i].sum())

        for j in range(k):
            n = int(counts.loc[i, j])

            if i != j and n > 0:
                movements.append({
                    "Nhóm tháng trước": names_before[i],
                    "Nhóm tháng sau": names_after[j],
                    "Số lượt": n,
                    "Tỷ lệ trong nhóm trước": (
                        f"{n / outgoing:.1%}"
                        if outgoing
                        else "0%"
                    ),
                })

    if movements:
        st.markdown(
            "**Chi tiết các hướng chuyển nhóm:**"
        )

        details = (
            pd.DataFrame(movements)
            .sort_values(
                "Số lượt",
                ascending=False,
            )
        )

        st.dataframe(
            details,
            hide_index=True,
            use_container_width=True,
        )

        top = details.iloc[0]

        st.info(
            f"Có **{top['Số lượt']:,} lượt** chuyển "
            f"từ **{top['Nhóm tháng trước']}** "
            f"sang **{top['Nhóm tháng sau']}**."
        )

    else:
        st.success(
            "Không ghi nhận lượt chuyển "
            "sang nhóm khác."
        )

    st.caption(
        "Một khách hàng có thể đóng góp "
        "nhiều lượt qua nhiều cặp tháng. "
        "Không bao gồm những tháng "
        "không có giao dịch."
    )


# =====================================================
# 4. CHUẨN HÓA TÊN CỘT
# =====================================================

def normalize(value):
    value = unicodedata.normalize(
        "NFD",
        str(value).strip().lower(),
    ).replace("đ", "d")

    text = "".join(
        c for c in value
        if unicodedata.category(c) != "Mn"
    )

    return re.sub(r"[^a-z0-9]", "", text)


ALIASES = {
    "CustomerID": [
        "customer id", "client id", "customer",
        "mã khách hàng", "khách hàng"
    ],
    "InvoiceNo": [
        "invoice", "invoice no", "order id",
        "order number", "transaction id",
        "mã hóa đơn", "mã đơn hàng"
    ],
    "InvoiceDate": [
        "invoice date", "order date",
        "purchase date", "transaction date",
        "ngày mua", "ngày giao dịch", "date"
    ],
    "TotalAmount": [
        "total amount", "amount", "sales",
        "revenue", "thành tiền", "tổng tiền",
        "doanh thu"
    ],
    "Quantity": [
        "quantity", "qty", "số lượng"
    ],
    "UnitPrice": [
        "price", "unit price", "đơn giá",
        "giá bán"
    ],
    "Currency": [
        "currency", "currency code",
        "loại tiền", "đơn vị tiền tệ"
    ],
    "Country": [
        "country", "quốc gia", "nước"
    ],
    "Status": [
        "status", "order status",
        "trạng thái", "trạng thái đơn hàng"
    ],
}

ALIASES = {
    name: [name] + variants
    for name, variants in ALIASES.items()
}

LOOKUP = {
    normalize(alias): name
    for name, aliases in ALIASES.items()
    for alias in aliases
}


# =====================================================
# 5. ĐỌC FILE
# =====================================================

@st.cache_data(show_spinner=False)
def list_sheets(blob, filename):
    if filename.lower().endswith(".csv"):
        return ["CSV"]

    return pd.ExcelFile(
        BytesIO(blob)
    ).sheet_names


@st.cache_data(show_spinner=False)
def read_data(blob, filename, selected):

    if filename.lower().endswith(".csv"):

        for encoding in (
            "utf-8-sig",
            "cp1258",
            "latin1",
        ):
            try:
                frame = pd.read_csv(
                    BytesIO(blob),
                    sep=None,
                    engine="python",
                    encoding=encoding,
                )
                break
            except UnicodeDecodeError:
                continue

    else:
        frame = pd.concat(
            [
                pd.read_excel(
                    BytesIO(blob),
                    sheet_name=sheet,
                )
                for sheet in selected
            ],
            ignore_index=True,
        )

    frame.columns = [
        str(col).strip()
        for col in frame.columns
    ]

    if frame.columns.duplicated().any():
        raise ValueError(
            "File có tên cột trùng nhau."
        )

    return frame


# =====================================================
# 6. ĐƠN VỊ TIỀN TỆ
# =====================================================

def currency_of(value):
    if pd.isna(value):
        return None

    value = str(value).strip().upper()

    if value in ("VNĐ", "Đ", "DONG"):
        return "VND"

    if value in CURRENCY_CODES:
        return value

    if value in SYMBOL_TO_CODE:
        return SYMBOL_TO_CODE[value]

    return None


def currency_in_text(value):
    if pd.isna(value):
        return None

    text = str(value)

    found = {
        code
        for symbol, code in SYMBOL_TO_CODE.items()
        if symbol in text
    }

    found |= {
        code
        for code in CURRENCY_CODES
        if re.search(rf"\b{code}\b", text, re.I)
    }

    return (
        next(iter(found))
        if len(found) == 1
        else None
    )


def parse_number(series, decimal_style):

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(
            series,
            errors="coerce",
        )

    s = (
        series.astype("string")
        .str.replace(
            r"[^0-9.,\-+]",
            "",
            regex=True,
        )
    )

    if decimal_style == "1.234,56":
        s = (
            s.str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
        )

    elif decimal_style == "1,234.56":
        s = s.str.replace(
            ",",
            "",
            regex=False,
        )

    else:
        ambiguous = s.str.fullmatch(
            r"[+-]?\d{1,3}[,.]\d{3}",
            na=False,
        )

        if ambiguous.any():
            raise ValueError(
                "Có số tiền dạng 1.234 hoặc 1,234 "
                "chưa rõ dấu thập phân. "
                "Hãy chọn kiểu ghi số tiền."
            )

        comma_decimal = (
            s.str.contains(
                ",",
                regex=False,
                na=False,
            )
            &
            ~s.str.contains(
                ".",
                regex=False,
                na=False,
            )
        )

        s.loc[comma_decimal] = (
            s.loc[comma_decimal]
            .str.replace(
                ",",
                ".",
                regex=False,
            )
        )

        both = (
            s.str.contains(
                ",",
                regex=False,
                na=False,
            )
            &
            s.str.contains(
                ".",
                regex=False,
                na=False,
            )
        )

        for idx in s.index[both]:
            value = s.at[idx]

            if value.rfind(",") > value.rfind("."):
                s.at[idx] = (
                    value.replace(".", "")
                    .replace(",", ".")
                )
            else:
                s.at[idx] = value.replace(",", "")

    return pd.to_numeric(
        s,
        errors="coerce",
    )


# =====================================================
# 7. NGÀY THÁNG
# =====================================================

def date_format_hint(series):

    if pd.api.types.is_datetime64_any_dtype(
        series
    ):
        return "Ngày Excel đã có định dạng ngày"

    sample = (
        series.dropna()
        .astype(str)
        .str.strip()
        .head(1000)
    )

    parts = sample.str.extract(
        r"^(\d{1,4})[-/](\d{1,2})[-/](\d{1,4})"
    )

    if parts.dropna().empty:
        return "Không đủ mẫu để tự xác định"

    first = pd.to_numeric(
        parts[0],
        errors="coerce",
    )

    second = pd.to_numeric(
        parts[1],
        errors="coerce",
    )

    if (first > 31).all():
        return "YYYY-MM-DD"

    if (
        (first > 12).any()
        and (second > 12).any()
    ):
        return "Định dạng ngày không thống nhất"

    if (first > 12).any():
        return "DD/MM/YYYY"

    if (second > 12).any():
        return "MM/DD/YYYY"

    return (
        "Ngày/tháng chưa rõ. "
        "Hãy chọn định dạng thủ công."
    )


def parse_dates(series, mode, countries=None):

    if pd.api.types.is_datetime64_any_dtype(
        series
    ):
        return pd.to_datetime(
            series,
            errors="coerce",
        )

    if mode == "Tự nhận diện":

        hint = date_format_hint(series)

        if hint not in (
            "DD/MM/YYYY",
            "MM/DD/YYYY",
            "YYYY-MM-DD",
        ):
            raise ValueError(
                "Ngày tháng chưa được nhận diện "
                "chắc chắn. Hãy chọn định dạng."
            )

        mode = hint

    if mode == "Theo cột quốc gia":

        if countries is None:
            raise ValueError(
                "Cần chọn cột Quốc gia."
            )

        output = pd.Series(
            pd.NaT,
            index=series.index,
            dtype="datetime64[ns]",
        )

        countries = (
            countries.astype("string")
            .str.strip()
            .str.lower()
        )

        if countries.isna().any():
            raise ValueError(
                "Có dòng thiếu quốc gia."
            )

        for country in countries.dropna().unique():

            local_format = COUNTRY_FORMAT.get(
                str(country)
            )

            if local_format is None:
                raise ValueError(
                    f"Chưa có định dạng ngày cho "
                    f"quốc gia {country}. "
                    "Hãy chọn định dạng thủ công."
                )

            mask = (
                countries == country
            ).fillna(False)

            output.loc[mask] = parse_dates(
                series.loc[mask],
                local_format,
            )

        return output

    return pd.to_datetime(
        series,
        format="mixed",
        dayfirst=(mode == "DD/MM/YYYY"),
        errors="coerce",
    )


# =====================================================
# 8. TIỀN XỬ LÝ
# =====================================================

@st.cache_data(show_spinner=False)
def prepare(
    raw,
    mapping,
    currency_choice,
    date_mode,
    number_style,
    cancel_mode,
    cancel_words,
    one_row,
):

    used = [
        col for col in mapping.values()
        if col
    ]

    if len(used) != len(set(used)):
        raise ValueError(
            "Một cột không thể đại diện "
            "cho nhiều trường."
        )

    df = raw.rename(
        columns={
            original: canonical
            for canonical, original in mapping.items()
            if original
        }
    ).copy()

    for required in (
        "CustomerID",
        "InvoiceDate",
    ):
        if required not in df.columns:
            raise ValueError(
                f"Thiếu thông tin {required}."
            )

    if (
        "InvoiceNo" not in df
        and not one_row
    ):
        raise ValueError(
            "Cần mã hóa đơn hoặc chọn "
            "mỗi dòng là một giao dịch."
        )

    if (
        "TotalAmount" not in df
        and not {
            "Quantity",
            "UnitPrice",
        }.issubset(df.columns)
    ):
        raise ValueError(
            "Cần Thành tiền hoặc "
            "Số lượng × Đơn giá."
        )

    original = len(df)

    # Đơn vị tiền tệ

    if "Currency" in df:
        row_currency = df[
            "Currency"
        ].map(currency_of)

    else:
        amount_col = (
            "TotalAmount"
            if "TotalAmount" in df
            else "UnitPrice"
        )

        row_currency = df[
            amount_col
        ].map(currency_in_text)

    observed = set(
        row_currency.dropna().unique()
    )

    if len(observed) > 1:

        if currency_choice == "Chưa xác định":
            raise ValueError(
                "File có nhiều đơn vị tiền tệ. "
                "Hãy chọn một đơn vị."
            )

        if row_currency.isna().any():
            raise ValueError(
                "Một số dòng chưa xác định tiền tệ. "
                "Hãy bổ sung cột Currency."
            )

        df = df.loc[
            row_currency == currency_choice
        ].copy()

    elif len(observed) == 1:

        detected = next(iter(observed))

        if currency_choice not in (
            "Chưa xác định",
            detected,
        ):
            raise ValueError(
                "Tiền tệ đã chọn không khớp "
                "với dữ liệu."
            )

    if df.empty:
        raise ValueError(
            "Không có giao dịch với "
            "loại tiền đã chọn."
        )

    # Đọc và sắp xếp ngày

    country_series = (
        df["Country"]
        if "Country" in df
        else None
    )

    df["InvoiceDate"] = parse_dates(
        df["InvoiceDate"],
        date_mode,
        country_series,
    )

    # Mã khách hàng

    df["CustomerID"] = (
        df["CustomerID"]
        .astype("string")
        .str.strip()
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
        .replace(
            ["", "nan", "None", "<NA>"],
            pd.NA,
        )
    )

    # Mã hóa đơn

    if "InvoiceNo" not in df:

        df["InvoiceNo"] = pd.Series(
            df.index.astype(str),
            index=df.index,
        )

    else:

        df["InvoiceNo"] = (
            df["InvoiceNo"]
            .astype("string")
            .str.strip()
            .replace("", pd.NA)
        )

    # Thành tiền

    if "TotalAmount" in df:

        df["TotalAmount"] = parse_number(
            df["TotalAmount"],
            number_style,
        )

    else:

        df["Quantity"] = parse_number(
            df["Quantity"],
            number_style,
        )

        df["UnitPrice"] = parse_number(
            df["UnitPrice"],
            number_style,
        )

        df["TotalAmount"] = (
            df["Quantity"]
            *
            df["UnitPrice"]
        )

    critical = [
        "CustomerID",
        "InvoiceNo",
        "InvoiceDate",
        "TotalAmount",
    ]

    missing = int(
        df[critical]
        .isna()
        .any(axis=1)
        .sum()
    )

    df = df.dropna(
        subset=critical
    ).copy()

    # Hóa đơn hủy

    cancelled = 0

    if cancel_mode == "Theo trạng thái":

        if "Status" not in df:
            raise ValueError(
                "Chưa có cột Trạng thái."
            )

        values = {
            normalize(value)
            for value in cancel_words.split(",")
            if value.strip()
        }

        if not values:
            raise ValueError(
                "Chưa nhập trạng thái hủy."
            )

        mask = (
            df["Status"]
            .fillna("")
            .map(normalize)
            .isin(values)
        )

        cancelled = int(mask.sum())

        df = df.loc[
            ~mask
        ].copy()

    elif cancel_mode == "Mã bắt đầu bằng C":

        if not mapping.get("InvoiceNo"):
            raise ValueError(
                "Không có mã hóa đơn gốc "
                "để kiểm tra chữ C."
            )

        mask = (
            df["InvoiceNo"]
            .astype(str)
            .str.upper()
            .str.startswith("C")
        )

        cancelled = int(mask.sum())

        df = df.loc[
            ~mask
        ].copy()

    # Giá trị không hợp lệ

    invalid = (
        ~np.isfinite(df["TotalAmount"])
    ) | (
        df["TotalAmount"] <= 0
    )

    if mapping["TotalAmount"] is None:
        invalid |= (
            (df["Quantity"] <= 0)
            |
            (df["UnitPrice"] <= 0)
        )

    invalid_count = int(invalid.sum())

    df = df.loc[
        ~invalid
    ].copy()

    duplicates = int(
        df.duplicated().sum()
    )

    df = (
        df.drop_duplicates()
        .sort_values("InvoiceDate")
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError(
            "Không còn giao dịch hợp lệ."
        )

    stats = {
        "Ban đầu": original,
        "Thiếu thông tin": missing,
        "Đơn hủy": cancelled,
        "Giá trị không hợp lệ": invalid_count,
        "Trùng lặp": duplicates,
        "Hợp lệ": len(df),
    }

    return df, stats


# =====================================================
# 9. RFM
# =====================================================

@st.cache_data(show_spinner=False)
def rfm_table(df):

    reference = (
        df["InvoiceDate"]
        .max()
        .normalize()
        +
        pd.Timedelta(days=1)
    )

    rfm = (
        df.groupby("CustomerID")
        .agg(
            Last=("InvoiceDate", "max"),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalAmount", "sum"),
        )
        .reset_index()
    )

    rfm["Recency"] = (
        reference
        -
        rfm["Last"].dt.normalize()
    ).dt.days

    return rfm[
        [
            "CustomerID",
            "Recency",
            "Frequency",
            "Monetary",
        ]
    ]


# =====================================================
# 10. K-MEANS
# =====================================================

@st.cache_data(show_spinner=False)
def clustering(rfm, k):

    features = [
        "Recency",
        "Frequency",
        "Monetary",
    ]

    X = np.log1p(
        rfm[features].astype(float)
    )

    scaler = StandardScaler()

    z = scaler.fit_transform(X)

    if (
        len(z) <= k
        or len(np.unique(z, axis=0)) < k
    ):
        raise ValueError(
            "Không đủ khách hàng có RFM "
            "khác nhau. Hãy giảm K."
        )

    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42,
    )

    result = rfm.copy()

    result["Cluster"] = model.fit_predict(z)

    return result, z, scaler, model


# =====================================================
# 11. CUSTOMER EVOLUTION
# =====================================================

@st.cache_data(show_spinner=False)
def monthly_rfm(df):

    m = df[
        [
            "CustomerID",
            "InvoiceDate",
            "InvoiceNo",
            "TotalAmount",
        ]
    ].copy()

    m["Month"] = (
        m["InvoiceDate"].dt.to_period("M")
    )

    m = (
        m.groupby(
            ["CustomerID", "Month"]
        )
        .agg(
            Last=("InvoiceDate", "max"),
            MonthF=("InvoiceNo", "nunique"),
            MonthM=("TotalAmount", "sum"),
        )
        .reset_index()
        .sort_values(
            ["CustomerID", "Month"]
        )
    )

    m["Frequency"] = (
        m.groupby("CustomerID")["MonthF"]
        .cumsum()
    )

    m["Monetary"] = (
        m.groupby("CustomerID")["MonthM"]
        .cumsum()
    )

    month_end = (
        m["Month"]
        .dt.to_timestamp(how="end")
        .dt.normalize()
    )

    m["Recency"] = (
        month_end
        -
        m["Last"].dt.normalize()
    ).dt.days.clip(lower=0)

    m["Tháng"] = m["Month"].astype(str)

    return m


def evolution_model(monthly, scaler, model):

    evo = monthly.copy()

    features = [
        "Recency",
        "Frequency",
        "Monetary",
    ]

    X = np.log1p(
        evo[features].astype(float)
    )

    z = scaler.transform(X)

    evo["Cluster"] = model.predict(z)

    evo["Nhóm khách hàng"] = (
        "Nhóm khách hàng "
        +
        (evo["Cluster"] + 1).astype(str)
    )

    return evo


def transition_matrix(evo, k):

    history = evo.sort_values(
        ["CustomerID", "Month"]
    ).copy()

    history["BeforeGroup"] = (
        history.groupby("CustomerID")["Cluster"]
        .shift()
    )

    history["BeforeMonth"] = (
        history.groupby("CustomerID")["Month"]
        .shift()
    )

    now = (
        history["Month"].dt.year * 12
        +
        history["Month"].dt.month
    )

    before = (
        history["BeforeMonth"].dt.year * 12
        +
        history["BeforeMonth"].dt.month
    )

    pairs = history.loc[
        history["BeforeMonth"].notna()
        &
        ((now - before) == 1)
    ].copy()

    matrix = pd.crosstab(
        pairs["BeforeGroup"],
        pairs["Cluster"],
    )

    matrix = matrix.reindex(
        index=range(k),
        columns=range(k),
        fill_value=0,
    )

    return matrix, pairs


# =====================================================
# 12. ELBOW VÀ ĐÁNH GIÁ
# =====================================================

@st.cache_data(show_spinner=False)
def diagnostics(z, k):

    rng = np.random.default_rng(42)

    indices = rng.choice(
        len(z),
        min(1000, len(z)),
        replace=False,
    )

    sample = z[indices]

    unique = len(
        np.unique(sample, axis=0)
    )

    elbow = []

    max_k = min(
        8,
        len(sample) - 1,
        unique,
    )

    for count in range(2, max_k + 1):

        km = KMeans(
            n_clusters=count,
            n_init=5,
            random_state=42,
        )

        km.fit(sample)

        elbow.append({
            "K": count,
            "Inertia": km.inertia_,
        })

    scores = []

    if len(sample) > k and unique >= k:

        methods = [
            (
                "K-Means",
                KMeans(
                    n_clusters=k,
                    n_init=10,
                    random_state=42,
                ),
            ),
            (
                "Agglomerative",
                AgglomerativeClustering(
                    n_clusters=k,
                ),
            ),
        ]

        for name, method in methods:

            labels = method.fit_predict(sample)

            if (
                1 < len(np.unique(labels))
                < len(sample)
            ):
                scores.append({
                    "Thuật toán": name,
                    "Silhouette": silhouette_score(
                        sample,
                        labels,
                    ),
                    "Davies-Bouldin": (
                        davies_bouldin_score(
                            sample,
                            labels,
                        )
                    ),
                })

    return (
        pd.DataFrame(elbow),
        pd.DataFrame(scores),
    )


# =====================================================
# 13. BỘ ĐẾM SAU XUẤT CSV
# =====================================================

def mark_export():
    st.session_state["export_time"] = time.time()


def format_elapsed(seconds):

    seconds = int(
        max(0, seconds)
    )

    return (
        f"{seconds // 3600:02d}:"
        f"{seconds % 3600 // 60:02d}:"
        f"{seconds % 60:02d}"
    )


@st.fragment(run_every="1s")
def export_timer():

    started = st.session_state.get(
        "export_time"
    )

    if started is not None:

        elapsed = (
            time.time() - started
        )

        st.info(
            "⏱️ Thời gian kể từ khi nhấn xuất CSV: "
            +
            format_elapsed(elapsed)
        )

        if st.button(
            "Dừng / đặt lại bộ đếm",
            key="stop_timer",
        ):
            st.session_state["export_time"] = None

            st.rerun(scope="fragment")


# =====================================================
# 14. GIAO DIỆN TẢI FILE
# =====================================================

st.markdown(
    '<div class="brand">'
    '◈ CUSTOMER INTELLIGENCE'
    '</div>',
    unsafe_allow_html=True,
)

with st.container(border=True):

    st.markdown(
        '<div class="hero-eyebrow">'
        'DATA ANALYTICS PLATFORM'
        '</div>'
        '<div class="hero-title">'
        'Phân nhóm khách hàng &amp; Customer Evolution'
        '</div>'
        '<div class="hero-sub">'
        'RFM · K-Means · '
        'Theo dõi sự thay đổi nhóm khách hàng '
        'theo thời gian'
        '</div>',
        unsafe_allow_html=True,
    )

    upload_left, upload_center, upload_right = (
        st.columns([0.5, 5, 0.5])
    )

    with upload_center:

        uploaded = st.file_uploader(
            "Tải lên tệp giao dịch",
            type=["csv", "xlsx", "xls"],
            label_visibility="collapsed",
        )

        st.markdown(
            '<div class="upload-help">'
            'Kéo thả tệp hoặc nhấn Browse files'
            ' · Hỗ trợ CSV, XLSX, XLS'
            '</div>',
            unsafe_allow_html=True,
        )


# Tách thiết lập khỏi thẻ Upload

settings_left, settings_center, settings_right = (
    st.columns([1, 2.5, 1])
)

if uploaded is None:
    st.stop()


# =====================================================
# 15. ĐỌC FILE ĐÃ TẢI
# =====================================================

started_processing = time.perf_counter()

blob = uploaded.getvalue()

try:

    sheets = list_sheets(
        blob,
        uploaded.name,
    )

    with settings_center:

        sheet = st.selectbox(
            "Chọn sheet",
            sheets
            +
            (
                ["Tất cả các sheet"]
                if len(sheets) > 1
                else []
            ),
        )

    selected = (
        tuple(sheets)
        if sheet == "Tất cả các sheet"
        else (sheet,)
    )

    with st.spinner("Đang đọc dữ liệu..."):

        raw = read_data(
            blob,
            uploaded.name,
            selected,
        )

except Exception as exc:

    st.error(
        f"Không đọc được dữ liệu: {exc}"
    )

    st.stop()


# =====================================================
# 16. THIẾT LẬP CỘT
# =====================================================

FIELD_NAMES = {
    "CustomerID": "Mã khách hàng *",
    "InvoiceDate": "Ngày giao dịch *",
    "InvoiceNo": "Mã hóa đơn",
    "TotalAmount": "Thành tiền",
    "Quantity": "Số lượng",
    "UnitPrice": "Đơn giá",
    "Currency": "Đơn vị tiền tệ",
    "Country": "Quốc gia",
    "Status": "Trạng thái đơn",
}

with settings_center:

    with st.expander(
        "Thiết lập cột "
        "(mở khi nhận diện chưa đúng)"
    ):

        mapping = {}

        options = [
            "— Không có —"
        ] + list(raw.columns)

        for column, label in FIELD_NAMES.items():

            guess = next(
                (
                    col for col in raw.columns
                    if LOOKUP.get(
                        normalize(col)
                    ) == column
                ),
                None,
            )

            picked = st.selectbox(
                label,
                options,
                index=(
                    options.index(guess)
                    if guess in options
                    else 0
                ),
                key="map_" + column,
            )

            mapping[column] = (
                None
                if picked == "— Không có —"
                else picked
            )

    one_row = st.checkbox(
        "Mỗi dòng là một giao dịch "
        "(khi không có mã hóa đơn)",
        value=False,
        disabled=(
            mapping["InvoiceNo"] is not None
        ),
    )

    date_mode = st.selectbox(
        "Định dạng ngày trong file",
        FORMATS,
        index=0,
    )

    if mapping["InvoiceDate"]:

        st.caption(
            "Gợi ý ngày: "
            +
            date_format_hint(
                raw[
                    mapping["InvoiceDate"]
                ]
            )
        )

    else:
        st.caption(
            "Chưa chọn cột ngày."
        )

    if date_mode == "Theo cột quốc gia":

        st.caption(
            "Chỉ chọn khi ngày trong file "
            "thật sự theo quy ước "
            "của quốc gia ở từng dòng."
        )

    # Tiền tệ

    currency_col = mapping["Currency"]

    amount_col = (
        mapping["TotalAmount"]
        or mapping["UnitPrice"]
    )

    if currency_col:

        detected_values = (
            raw[currency_col]
            .map(currency_of)
        )

    elif amount_col:

        detected_values = (
            raw[amount_col]
            .head(2000)
            .map(currency_in_text)
        )

    else:

        detected_values = pd.Series(
            dtype="object"
        )

    known_currencies = sorted(
        set(
            detected_values.dropna().unique()
        )
    )

    currency_default = (
        known_currencies[0]
        if len(known_currencies) == 1
        else "Chưa xác định"
    )

    currency_options = (
        ["Chưa xác định"]
        +
        CURRENCY_CODES
    )

    currency_choice = st.selectbox(
        "Đơn vị tiền tệ",
        currency_options,
        index=currency_options.index(
            currency_default
        ),
    )

    if known_currencies:

        st.caption(
            "Đã phát hiện: "
            +
            ", ".join(
                known_currencies
            )
        )

    else:

        st.caption(
            "Không phát hiện rõ loại tiền. "
            "Có thể chọn thủ công. "
            "Ứng dụng không tự quy đổi."
        )

    number_style = st.selectbox(
        "Cách ghi số tiền",
        [
            "Tự động",
            "1,234.56",
            "1.234,56",
        ],
    )

    # Hóa đơn hủy

    cancel = st.checkbox(
        "Nếu có hóa đơn bị hủy, "
        "loại khỏi phân tích",
        value=False,
    )

    cancel_mode = "Không lọc"

    cancel_words = (
        "cancelled, canceled, "
        "void, hủy, đã hủy"
    )

    if cancel:

        choices = []

        if mapping["Status"]:
            choices.append("Theo trạng thái")

        if mapping["InvoiceNo"]:
            choices.append("Mã bắt đầu bằng C")

        if not choices:

            st.warning(
                "Cần chọn cột Trạng thái "
                "hoặc Mã hóa đơn."
            )

            st.stop()

        cancel_mode = st.selectbox(
            "Cách xác định đơn hủy",
            choices,
        )

        if cancel_mode == "Theo trạng thái":

            cancel_words = st.text_input(
                "Trạng thái hủy "
                "(ngăn cách bằng dấu phẩy)",
                cancel_words,
            )

        else:

            st.caption(
                "Chỉ chọn khi chữ C ở đầu mã "
                "thực sự có nghĩa là đơn hủy."
            )


# =====================================================
# 17. TIỀN XỬ LÝ VÀ RFM
# =====================================================

try:

    with st.spinner(
        "Đang làm sạch dữ liệu và tính RFM..."
    ):

        clean, stats = prepare(
            raw,
            mapping,
            currency_choice,
            date_mode,
            number_style,
            cancel_mode,
            cancel_words,
            one_row,
        )

        rfm = rfm_table(clean)

except Exception as exc:

    st.error(
        f"Không thể chuẩn bị dữ liệu: {exc}"
    )

    st.stop()


if len(rfm) < 3:

    st.error(
        "Cần ít nhất 3 khách hàng "
        "có giao dịch hợp lệ."
    )

    st.stop()


with settings_center:

    k = st.slider(
        "Số nhóm khách hàng (K)",
        2,
        min(8, len(rfm) - 1),
        min(4, len(rfm) - 1),
    )


# =====================================================
# 18. PHÂN NHÓM VÀ CUSTOMER EVOLUTION
# =====================================================

try:

    with st.spinner(
        "Đang chạy K-Means "
        "và Customer Evolution..."
    ):

        groups, z, scaler, model = (
            clustering(rfm, k)
        )

        profiles = (
            groups.groupby("Cluster")
            .agg(
                Khach_hang=(
                    "CustomerID",
                    "count",
                ),
                R=("Recency", "mean"),
                F=("Frequency", "mean"),
                M=("Monetary", "mean"),
            )
            .reset_index()
        )

        profiles["Nhóm khách hàng"] = (
            "Nhóm khách hàng "
            +
            (profiles["Cluster"] + 1).astype(str)
        )

        evolution = evolution_model(
            monthly_rfm(clean),
            scaler,
            model,
        )

        matrix, pairs = transition_matrix(
            evolution,
            k,
        )

except Exception as exc:

    st.error(
        f"Phân tích thất bại: {exc}"
    )

    st.stop()


elapsed_processing = (
    time.perf_counter()
    -
    started_processing
)

st.success(
    "✅ Phân tích hoàn tất! "
    "Cuộn xuống để xem Dashboard."
)

st.caption(
    f"Thời gian xử lý lượt này: "
    f"{elapsed_processing:.2f} giây "
    "(có thể sử dụng bộ nhớ đệm)."
)

st.divider()


# =====================================================
# 19. DASHBOARD
# =====================================================

heading("📊 Tổng quan kết quả")

metrics = [
    (
        "Khách hàng",
        len(rfm),
    ),
    (
        "Hóa đơn",
        clean["InvoiceNo"].nunique(),
    ),
    (
        f"Giá trị mua ({currency_choice})",
        clean["TotalAmount"].sum(),
    ),
    (
        "Nhóm khách hàng",
        k,
    ),
]

for col, (label, value) in zip(
    st.columns(4),
    metrics,
):

    with col:
        kpi(
            label,
            f"{value:,.0f}",
        )


# Phân bố nhóm

heading("👥 Phân bố nhóm khách hàng")

col1, col2 = st.columns(2)

with col1:

    fig = px.bar(
        profiles,
        x="Nhóm khách hàng",
        y="Khach_hang",
        color="Nhóm khách hàng",
        text="Khach_hang",
        color_discrete_sequence=COLORS,
        labels={
            "Khach_hang": "Số khách hàng",
        },
    )

    fig.update_layout(
        showlegend=False
    )

    graph(fig)


with col2:

    fig = px.pie(
        profiles,
        names="Nhóm khách hàng",
        values="Khach_hang",
        hole=0.55,
        color_discrete_sequence=COLORS,
    )

    graph(fig)


# Đặc điểm từng nhóm

heading("📋 Đặc điểm từng nhóm khách hàng")

profile_view = profiles[
    [
        "Nhóm khách hàng",
        "Khach_hang",
        "R",
        "F",
        "M",
    ]
].rename(
    columns={
        "Khach_hang": "Số khách hàng",
        "R": "R: Số ngày",
        "F": "F: Số hóa đơn",
        "M": f"M: Tiền mua ({currency_choice})",
    }
)

st.dataframe(
    profile_view.round(2),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "R: số ngày từ lần mua gần nhất. "
    "F: số hóa đơn. "
    "M: tổng giá trị mua."
)


# =====================================================
# 20. CUSTOMER EVOLUTION TỔNG QUAN
# =====================================================

heading("🔄 Customer Evolution")

st.info(
    "Biểu đồ thống kê số khách hàng "
    "có giao dịch mỗi tháng. "
    "RFM tích lũy được áp dụng "
    "cùng một mô hình K-Means."
)

evo_counts = (
    evolution.groupby(
        [
            "Tháng",
            "Nhóm khách hàng",
        ]
    )
    .size()
    .reset_index(
        name="Số khách hàng"
    )
)

fig = px.line(
    evo_counts,
    x="Tháng",
    y="Số khách hàng",
    color="Nhóm khách hàng",
    markers=True,
    color_discrete_sequence=COLORS,
)

graph(fig, 410)

if not pairs.empty:

    moved = int(
        (
            pairs["BeforeGroup"]
            !=
            pairs["Cluster"]
        ).sum()
    )

    st.caption(
        f"Có {moved:,} lượt chuyển nhóm "
        f"trên {len(pairs):,} lượt "
        "đối chiếu hai tháng liên tiếp."
    )


# =====================================================
# 21. CHI TIẾT PHÂN TÍCH
# =====================================================

heading("🔎 Chi tiết phân tích")


with st.expander("1. Dữ liệu"):

    st.caption(
        f"File: {uploaded.name} | "
        f"Sheet: {', '.join(selected)} | "
        f"Định dạng ngày: {date_mode}"
    )

    st.dataframe(
        raw.head(30),
        hide_index=True,
        use_container_width=True,
    )


with st.expander("2. Tiền xử lý"):

    stats_table = pd.DataFrame(
        list(stats.items()),
        columns=[
            "Công đoạn",
            "Số dòng",
        ],
    )

    st.dataframe(
        stats_table,
        hide_index=True,
        use_container_width=True,
    )

    st.dataframe(
        clean.head(30),
        hide_index=True,
        use_container_width=True,
    )


with st.expander("3. RFM"):

    st.write(
        "**R (Recency):** Số ngày "
        "từ lần mua gần nhất. "
        "**F (Frequency):** Số hóa đơn. "
        "**M (Monetary):** Tổng giá trị mua."
    )

    st.dataframe(
        rfm.head(100),
        hide_index=True,
        use_container_width=True,
    )


with st.expander("4. Chuẩn hóa"):

    st.write(
        "Sử dụng log1p và StandardScaler "
        "trước khi chạy K-Means."
    )

    standardized = pd.DataFrame(
        z[:30],
        columns=[
            "R chuẩn hóa",
            "F chuẩn hóa",
            "M chuẩn hóa",
        ],
    )

    st.dataframe(
        standardized.round(3),
        hide_index=True,
        use_container_width=True,
    )


with st.expander("5. K-Means"):

    st.write(
        f"Đã chia thành "
        f"**{k} nhóm khách hàng**."
    )

    if st.button("Xem Elbow"):

        elbow, _ = diagnostics(
            z,
            k,
        )

        if not elbow.empty:

            fig = px.line(
                elbow,
                x="K",
                y="Inertia",
                markers=True,
            )

            graph(fig)


with st.expander(
    "6. Phân tích nhóm khách hàng"
):

    st.dataframe(
        profile_view.round(2),
        hide_index=True,
        use_container_width=True,
    )


with st.expander(
    "7. Đánh giá và so sánh"
):

    st.caption(
        "So sánh K-Means và Agglomerative "
        "trên cùng mẫu tối đa "
        "1.000 khách hàng."
    )

    if st.button("Chạy so sánh"):

        _, scores = diagnostics(z, k)

        st.dataframe(
            scores.round(4),
            hide_index=True,
            use_container_width=True,
        )


# =====================================================
# 22. CUSTOMER EVOLUTION CÓ VÍ DỤ
# =====================================================

with st.expander(
    "8. Customer Evolution – "
    "Ví dụ và hành trình khách hàng"
):

    st.markdown(
        "#### Ví dụ minh họa"
    )

    st.caption(
        "Dữ liệu giả định, "
        "không phải kết quả phân tích thật."
    )

    example = pd.DataFrame({

        "Khoảng tháng": [
            "01/2025 → 02/2025",
            "02/2025 → 03/2025",
            "03/2025 → 04/2025",
        ],

        "Nhóm tháng trước": [
            "Nhóm 1 tháng trước",
            "Nhóm 1 tháng trước",
            "Nhóm 2 tháng trước",
        ],

        "Nhóm tháng sau": [
            "Nhóm 1 tháng sau",
            "Nhóm 2 tháng sau",
            "Nhóm 3 tháng sau",
        ],

        "Kết quả": [
            "Giữ nguyên nhóm",
            "Chuyển nhóm",
            "Chuyển nhóm",
        ],
    })

    st.dataframe(
        example,
        hide_index=True,
        use_container_width=True,
    )

    st.write(
        "**Cách đọc ví dụ:** "
        "Nhóm 1 tháng trước → "
        "Nhóm 1 tháng sau là giữ nguyên. "
        "Nhóm 1 tháng trước → "
        "Nhóm 2 tháng sau là chuyển nhóm. "
        "Các số 1, 2, 3... là mã nhóm, "
        "không phải thứ hạng tốt hoặc xấu."
    )

    st.markdown(
        "#### 📘 Chú thích RFM"
    )

    st.write(
        "**R:** Số ngày từ lần mua gần nhất "
        "đến cuối tháng (thấp là mua gần hơn). "
        "**F:** Số hóa đơn tích lũy. "
        "**M:** Tổng tiền mua tích lũy "
        "theo đơn vị tiền của file."
    )

    st.markdown(
        "#### 🔄 Ma trận nhóm tháng trước "
        "và nhóm tháng sau"
    )

    show_transition_matrix(
        matrix,
        pairs,
        k,
    )

    st.markdown(
        "#### 👤 Hành trình khách hàng"
    )

    customer = st.selectbox(
        "Mã khách hàng",
        sorted(
            evolution["CustomerID"]
            .unique()
            .tolist()
        ),
    )

    history = (
        evolution.loc[
            evolution["CustomerID"] == customer
        ]
        .sort_values("Month")
    )

    hist_view = history[
        [
            "Tháng",
            "Nhóm khách hàng",
            "Recency",
            "Frequency",
            "Monetary",
        ]
    ].rename(
        columns={
            "Recency": "R – số ngày",
            "Frequency": "F – hóa đơn tích lũy",
            "Monetary": (
                f"M – tiền mua ({currency_choice})"
            ),
        }
    )

    st.dataframe(
        hist_view.round(2),
        hide_index=True,
        use_container_width=True,
    )

    if len(history) > 1:

        plot_history = history.assign(
            NhomHienThi=history["Cluster"] + 1
        )

        fig = px.line(
            plot_history,
            x="Tháng",
            y="NhomHienThi",
            markers=True,
            labels={
                "NhomHienThi": "Nhóm khách hàng",
            },
        )

        fig.update_yaxes(
            tickvals=list(
                range(1, k + 1)
            ),
            ticktext=[
                f"Nhóm {i}"
                for i in range(1, k + 1)
            ],
        )

        graph(fig, 310)

    else:

        st.info(
            "Khách hàng này chỉ có dữ liệu "
            "ở một tháng giao dịch."
        )

    st.caption(
        "Chỉ theo dõi các tháng có giao dịch. "
        "Không dự báo sự thay đổi trong tương lai."
    )


with st.expander(
    "9. Streamlit và kết quả"
):

    st.write(
        f"Đã phân tích **{len(rfm):,} "
        f"khách hàng** thành "
        f"**{k} nhóm khách hàng**."
    )

    st.write(
        "Dữ liệu → Tiền xử lý → RFM → "
        "Chuẩn hóa → K-Means → "
        "Phân tích khách hàng → "
        "Đánh giá & so sánh → "
        "Customer Evolution → Kết quả."
    )


# =====================================================
# 23. XUẤT FILE VÀ BỘ ĐẾM
# =====================================================

heading("📥 Xuất kết quả phân tích")

export_groups = groups.copy()

export_groups["Nhóm khách hàng"] = (
    "Nhóm khách hàng "
    +
    (export_groups["Cluster"] + 1).astype(str)
)

export_groups["Cluster"] = (
    export_groups["Cluster"] + 1
)

export_groups = export_groups.rename(
    columns={
        "Cluster": "Mã nhóm",
    }
)

export_evo = evolution[
    [
        "CustomerID",
        "Tháng",
        "Nhóm khách hàng",
        "Recency",
        "Frequency",
        "Monetary",
    ]
]

col1, col2 = st.columns(2)

with col1:

    st.download_button(
        "⬇️ Xuất danh sách nhóm khách hàng",
        data=export_groups.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="nhom_khach_hang.csv",
        mime="text/csv",
        on_click=mark_export,
    )


with col2:

    st.download_button(
        "⬇️ Xuất Customer Evolution",
        data=export_evo.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="customer_evolution.csv",
        mime="text/csv",
        on_click=mark_export,
    )


export_timer()

st.caption(
    "Bộ đếm tính từ lúc nhấn xuất CSV; "
    "Streamlit không xác nhận được thời điểm "
    "trình duyệt lưu file xong."
)

st.divider()

st.caption(
    "CUSTOMER INTELLIGENCE · "
    "RFM · K-Means · Customer Evolution"
)
