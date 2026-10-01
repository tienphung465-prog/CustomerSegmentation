
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
)


COLORS = [
    "#2563EB", "#10B981", "#8B5CF6", "#F59E0B",
    "#EC4899", "#06B6D4", "#64748B", "#EF4444",
]

MONEY = [
    "VND", "USD", "EUR", "GBP", "JPY", "CNY",
    "KRW", "AUD", "CAD", "SGD", "THB", "MYR", "INR",
]

SYMBOLS = {
    "₫": "VND",
    "€": "EUR",
    "£": "GBP",
    "₩": "KRW",
    "₹": "INR",
}

DATES = [
    "Tự nhận diện",
    "DD/MM/YYYY",
    "MM/DD/YYYY",
    "YYYY-MM-DD",
    "Theo cột quốc gia",
]

COUNTRY_DATE = {
    "vietnam": "DD/MM/YYYY",
    "viet nam": "DD/MM/YYYY",
    "united kingdom": "DD/MM/YYYY",
    "uk": "DD/MM/YYYY",
    "france": "DD/MM/YYYY",
    "germany": "DD/MM/YYYY",
    "australia": "DD/MM/YYYY",
    "india": "DD/MM/YYYY",
    "thailand": "DD/MM/YYYY",
    "singapore": "DD/MM/YYYY",
    "united states": "MM/DD/YYYY",
    "usa": "MM/DD/YYYY",
    "us": "MM/DD/YYYY",
    "japan": "YYYY-MM-DD",
    "china": "YYYY-MM-DD",
    "south korea": "YYYY-MM-DD",
    "korea": "YYYY-MM-DD",
    "canada": "YYYY-MM-DD",
}


# =====================================================
# 1. GIAO DIỆN
# =====================================================

st.markdown("""
<style>
:root {
    color-scheme: light;
}

.stApp,
[data-testid="stAppViewContainer"] {
    background: #F6F8FC !important;
    color: #17243B !important;
}

.block-container {
    max-width: 1250px;
    padding-top: 1.4rem;
    padding-bottom: 3rem;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

.stApp [data-testid="stMarkdownContainer"] h1,
.stApp [data-testid="stMarkdownContainer"] h2,
.stApp [data-testid="stMarkdownContainer"] h3,
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stExpander"] summary {
    color: #17243B !important;
}

.stApp [data-testid="stCaptionContainer"] p {
    color: #586980 !important;
}

.stApp [data-baseweb="select"] > div,
.stApp [data-baseweb="input"] > div,
.stApp input {
    background: #FFFFFF !important;
    color: #17243B !important;
}

.stApp [data-baseweb="select"] span,
.stApp [data-baseweb="select"] input {
    color: #17243B !important;
}

.stApp [data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
}

.brand {
    background: #FFFFFF;
    border: 1px solid #DFE7F2;
    border-radius: 14px;
    padding: 16px 22px;
    color: #1D4ED8;
    font-weight: 800;
    margin-bottom: 18px;
}

.hero-tag {
    text-align: center;
    color: #2563EB;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1.1px;
}

.hero-title {
    text-align: center;
    color: #15253C;
    font-weight: 800;
    font-size: clamp(26px, 3.8vw, 40px);
    margin: 10px 0;
}

.hero-sub {
    text-align: center;
    color: #586980;
    font-size: 14px;
    margin-bottom: 18px;
}

[data-testid="stVerticalBlockBorderWrapper"]:has(
    [data-testid="stFileUploader"]
) {
    background: #FFFFFF !important;
    border: 1px solid #DFE7F2 !important;
    border-radius: 20px !important;
    padding: 20px 22px 25px;
    box-shadow: 0 9px 28px rgba(30,70,110,.05);
}

.stApp [data-testid="stFileUploaderDropzone"] {
    min-height: 245px !important;
    padding: 25px !important;
    background: #F7FAFF !important;
    border: 2px dashed #91B6ED !important;
    border-radius: 15px !important;

    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 12px;
}

.stApp [data-testid="stFileUploaderDropzone"]::before {
    content: "⬆  UPLOAD FILE";
    display: block;
    color: #1D4ED8;
    font-size: 24px;
    font-weight: 800;
    letter-spacing: 1px;
    text-align: center;
}

.stApp [data-testid="stFileUploaderDropzone"] section,
.stApp [data-testid="stFileUploaderDropzone"] span,
.stApp [data-testid="stFileUploaderDropzone"] small {
    color: #334155 !important;
    text-align: center;
}

.stApp [data-testid="stFileUploaderDropzone"] button {
    background: #2563EB !important;
    color: #FFFFFF !important;
    border: 0 !important;
    border-radius: 10px;
}

.stApp [data-testid="stFileUploaderDropzone"] button * {
    color: #FFFFFF !important;
}

.kpi {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 18px;
    min-height: 104px;
}

.kpi-label {
    color: #586980;
    font-size: 12px;
}

.kpi-value {
    color: #1D4ED8;
    font-size: 25px;
    font-weight: 800;
    margin-top: 8px;
    overflow-wrap: anywhere;
}

.matrix-scroll {
    overflow-x: auto;
    background: #FFFFFF;
    border: 1px solid #DFE7F2;
    border-radius: 13px;
    padding: 8px;
}

.evo-matrix {
    width: 100%;
    min-width: 590px;
    border-spacing: 5px;
    text-align: center;
    font-size: 12px;
}

.evo-matrix th {
    background: #EFF5FF;
    border-radius: 7px;
    padding: 11px 6px;
    color: #234365;
}

.evo-matrix td {
    border-radius: 8px;
    padding: 12px 6px;
    color: #17243B;
}

.evo-matrix strong {
    display: block;
    font-size: 19px;
    color: #17243B;
}

.evo-matrix small {
    display: block;
    margin-top: 4px;
    color: #475569;
}

@media (max-width: 640px) {
    .stApp [data-testid="stFileUploaderDropzone"] {
        min-height: 205px !important;
        padding: 12px !important;
    }
}
</style>
""", unsafe_allow_html=True)


def heading(value):
    st.markdown(f"### {value}")


def metric_card(label, value):
    st.markdown(
        f'<div class="kpi">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# =====================================================
# 2. HÀM VẼ BIỂU ĐỒ - ĐÃ SỬA MÀU CHỮ
# =====================================================

def draw(fig, height=385):
    """
    Cố định màu chữ trong Plotly.
    Không để theme Streamlit ghi đè.
    """

    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",

        font=dict(
            family="Arial, sans-serif",
            size=13,
            color="#243247",
        ),

        margin=dict(
            l=24,
            r=24,
            t=32,
            b=76,
        ),

        legend=dict(
            font=dict(
                color="#17243B",
                size=13,
            ),
            title_font=dict(
                color="#17243B",
                size=13,
            ),
            bgcolor="#FFFFFF",
            orientation="v",
        ),

        hoverlabel=dict(
            bgcolor="#FFFFFF",
            font=dict(
                color="#17243B",
            ),
        ),
    )

    fig.update_xaxes(
        tickfont=dict(
            color="#334155",
            size=12,
        ),
        title_font=dict(
            color="#334155",
            size=13,
        ),
        linecolor="#CBD5E1",
        gridcolor="#E7EDF5",
        automargin=True,
    )

    fig.update_yaxes(
        tickfont=dict(
            color="#334155",
            size=12,
        ),
        title_font=dict(
            color="#334155",
            size=13,
        ),
        linecolor="#CBD5E1",
        gridcolor="#E7EDF5",
        automargin=True,
    )

    if any(
        trace.type == "pie"
        for trace in fig.data
    ):
        fig.update_traces(
            textposition="inside",
            textfont=dict(
                color="#FFFFFF",
                size=13,
            ),
            selector=dict(type="pie"),
        )

    # Không để Streamlit tự đổi màu chữ của Plotly.
    st.plotly_chart(
        fig,
        use_container_width=True,
        theme=None,
    )


# =====================================================
# 3. NHẬN DIỆN TÊN CỘT
# =====================================================

def slug(x):
    x = unicodedata.normalize(
        "NFD",
        str(x).strip().lower(),
    ).replace("đ", "d")

    return re.sub(
        r"[^a-z0-9]",
        "",
        "".join(
            c for c in x
            if unicodedata.category(c) != "Mn"
        ),
    )


ALIASES = {
    "CustomerID": [
        "customer",
        "customer id",
        "client id",
        "mã khách hàng",
        "khách hàng",
    ],

    "InvoiceNo": [
        "invoice",
        "invoice no",
        "order id",
        "transaction id",
        "mã hóa đơn",
        "mã đơn hàng",
    ],

    "InvoiceDate": [
        "date",
        "invoice date",
        "order date",
        "purchase date",
        "ngày mua",
        "ngày giao dịch",
    ],

    "TotalAmount": [
        "total amount",
        "amount",
        "sales",
        "revenue",
        "thành tiền",
        "tổng tiền",
        "doanh thu",
    ],

    "Quantity": [
        "quantity",
        "qty",
        "số lượng",
    ],

    "UnitPrice": [
        "unit price",
        "price",
        "đơn giá",
        "giá bán",
    ],

    "Currency": [
        "currency",
        "currency code",
        "loại tiền",
        "đơn vị tiền tệ",
    ],

    "Country": [
        "country",
        "quốc gia",
        "nước",
    ],

    "Status": [
        "status",
        "order status",
        "trạng thái",
        "trạng thái đơn hàng",
    ],
}

LOOKUP = {
    slug(alias): field
    for field, choices in ALIASES.items()
    for alias in [field, *choices]
}


# =====================================================
# 4. ĐỌC FILE
# =====================================================

@st.cache_data(show_spinner=False)
def sheets_of(blob, filename):

    if filename.lower().endswith(".csv"):
        return ["CSV"]

    return pd.ExcelFile(
        BytesIO(blob)
    ).sheet_names


@st.cache_data(show_spinner=False)
def read_data(blob, filename, sheets):

    if filename.lower().endswith(".csv"):

        for enc in (
            "utf-8-sig",
            "cp1258",
            "latin1",
        ):
            try:
                df = pd.read_csv(
                    BytesIO(blob),
                    sep=None,
                    engine="python",
                    encoding=enc,
                )
                break

            except UnicodeDecodeError:
                continue

    else:
        df = pd.concat(
            [
                pd.read_excel(
                    BytesIO(blob),
                    sheet_name=s,
                )
                for s in sheets
            ],
            ignore_index=True,
        )

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    if df.columns.duplicated().any():
        raise ValueError(
            "File có tên cột bị trùng."
        )

    return df


# =====================================================
# 5. NHẬN DIỆN TIỀN TỆ
# =====================================================

def currency_code(v):

    if pd.isna(v):
        return None

    text = str(v).strip().upper()

    if text in (
        "VNĐ",
        "Đ",
        "DONG",
    ):
        return "VND"

    if text in MONEY:
        return text

    return SYMBOLS.get(text)


def currency_in_amount(v):

    if pd.isna(v):
        return None

    s = str(v)

    found = {
        code
        for symbol, code in SYMBOLS.items()
        if symbol in s
    }

    found |= {
        code
        for code in MONEY
        if re.search(
            rf"\b{code}\b",
            s,
            re.I,
        )
    }

    if len(found) == 1:
        return next(iter(found))

    return None


def parse_amount(series, style):

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(
            series,
            errors="coerce",
        )

    clean = (
        series.astype("string")
        .str.replace(
            r"[^0-9.,+\-]",
            "",
            regex=True,
        )
    )

    if style == "1,234.56":

        clean = clean.str.replace(
            ",",
            "",
            regex=False,
        )

    elif style == "1.234,56":

        clean = (
            clean.str.replace(
                ".",
                "",
                regex=False,
            )
            .str.replace(
                ",",
                ".",
                regex=False,
            )
        )

    else:

        ambiguous = clean.str.fullmatch(
            r"[+-]?\d{1,3}[.,]\d{3}",
            na=False,
        )

        if ambiguous.any():
            raise ValueError(
                "Số tiền có dấu phân cách mơ hồ. "
                "Hãy chọn kiểu ghi số tiền."
            )

        comma_only = (
            clean.str.contains(
                ",",
                regex=False,
                na=False,
            )
            &
            ~clean.str.contains(
                ".",
                regex=False,
                na=False,
            )
        )

        clean.loc[comma_only] = (
            clean.loc[comma_only]
            .str.replace(
                ",",
                ".",
                regex=False,
            )
        )

        mixed = (
            clean.str.contains(
                ",",
                regex=False,
                na=False,
            )
            &
            clean.str.contains(
                ".",
                regex=False,
                na=False,
            )
        )

        for ix in clean.index[mixed]:

            s = clean.at[ix]

            if s.rfind(",") > s.rfind("."):
                clean.at[ix] = (
                    s.replace(".", "")
                    .replace(",", ".")
                )

            else:
                clean.at[ix] = s.replace(",", "")

    return pd.to_numeric(
        clean,
        errors="coerce",
    )


# =====================================================
# 6. ĐỊNH DẠNG NGÀY
# =====================================================

def guess_date(series):

    if pd.api.types.is_datetime64_any_dtype(
        series
    ):
        return "Ngày Excel đã lưu đúng kiểu"

    samples = (
        series.dropna()
        .astype(str)
        .head(1000)
    )

    match = (
        samples.str.extract(
            r"^(\d{1,4})[-/](\d{1,2})[-/](\d{1,4})"
        )
        .dropna()
    )

    if match.empty:
        return "Không đủ thông tin"

    a = pd.to_numeric(match[0])
    b = pd.to_numeric(match[1])

    if (a > 31).all():
        return "YYYY-MM-DD"

    if (
        (a > 12).any()
        and (b > 12).any()
    ):
        return "Ngày tháng lẫn định dạng"

    if (a > 12).any():
        return "DD/MM/YYYY"

    if (b > 12).any():
        return "MM/DD/YYYY"

    return (
        "Chưa phân biệt được ngày và tháng"
    )


def parse_date(series, style, country=None):

    if pd.api.types.is_datetime64_any_dtype(
        series
    ):
        return pd.to_datetime(
            series,
            errors="coerce",
        )

    if style == "Tự nhận diện":

        style = guess_date(series)

        if style not in DATES[1:4]:
            raise ValueError(
                "Ngày/tháng chưa thể nhận diện "
                "chắc chắn. Hãy chọn định dạng thủ công."
            )

    if style == "Theo cột quốc gia":

        if country is None:
            raise ValueError(
                "Hãy chọn cột quốc gia hoặc "
                "chọn định dạng ngày thủ công."
            )

        if country.isna().any():
            raise ValueError(
                "Có giao dịch không rõ quốc gia. "
                "Hãy chọn định dạng ngày thủ công."
            )

        result = pd.Series(
            pd.NaT,
            index=series.index,
            dtype="datetime64[ns]",
        )

        country_names = (
            country.astype("string")
            .str.strip()
            .str.lower()
            .dropna()
            .unique()
        )

        for name in country_names:

            fmt = COUNTRY_DATE.get(
                str(name)
            )

            if fmt is None:
                raise ValueError(
                    f"Chưa có quy tắc ngày cho "
                    f"'{name}'. Chọn định dạng thủ công."
                )

            idx = (
                country.astype("string")
                .str.strip()
                .str.lower()
                .eq(name)
                .fillna(False)
            )

            result.loc[idx] = parse_date(
                series.loc[idx],
                fmt,
            )

        return result

    return pd.to_datetime(
        series,
        format="mixed",
        dayfirst=(
            style == "DD/MM/YYYY"
        ),
        errors="coerce",
    )


# =====================================================
# 7. TIỀN XỬ LÝ
# =====================================================

@st.cache_data(show_spinner=False)
def prepare(
    raw,
    mapping,
    currency,
    date_style,
    number_style,
    cancel_style,
    cancel_words,
    each_line,
):

    chosen = [
        v for v in mapping.values()
        if v
    ]

    if len(chosen) != len(set(chosen)):
        raise ValueError(
            "Bạn đang chọn cùng một cột "
            "cho nhiều trường."
        )

    df = raw.rename(
        columns={
            original: canonical
            for canonical, original in mapping.items()
            if original
        }
    ).copy()

    for req in (
        "CustomerID",
        "InvoiceDate",
    ):
        if req not in df.columns:
            raise ValueError(
                f"Thiếu {req}; kiểm tra "
                "phần Thiết lập cột."
            )

    if (
        "InvoiceNo" not in df
        and not each_line
    ):
        raise ValueError(
            "Cần mã hóa đơn hoặc tùy chọn "
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
            "Cần Thành tiền hoặc cả "
            "Số lượng và Đơn giá."
        )

    original = len(df)

    # Nhận diện tiền tệ.

    money_col = (
        "TotalAmount"
        if "TotalAmount" in df
        else "UnitPrice"
    )

    if "Currency" in df:
        line_money = (
            df["Currency"].map(currency_code)
        )

    else:
        line_money = (
            df[money_col].map(currency_in_amount)
        )

    observed = set(
        line_money.dropna().unique()
    )

    if len(observed) > 1:

        if currency == "Chưa xác định":
            raise ValueError(
                "File có nhiều tiền tệ; "
                "hãy chọn một loại để phân tích."
            )

        if line_money.isna().any():
            raise ValueError(
                "Dữ liệu có nhiều loại tiền nhưng "
                "có dòng không xác định tiền tệ."
            )

        df = df.loc[
            line_money == currency
        ].copy()

    elif len(observed) == 1:

        detected = next(iter(observed))

        if currency not in (
            "Chưa xác định",
            detected,
        ):
            raise ValueError(
                "Tiền tệ được chọn "
                "không khớp dữ liệu."
            )

    if df.empty:
        raise ValueError(
            "Không có giao dịch "
            "thuộc loại tiền này."
        )

    # Ngày tháng.

    df["InvoiceDate"] = parse_date(
        df["InvoiceDate"],
        date_style,
        df.get("Country"),
    )

    # Mã khách hàng.

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
            [
                "",
                "nan",
                "None",
                "<NA>",
            ],
            pd.NA,
        )
    )

    # Mã hóa đơn.

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
            .replace(
                "",
                pd.NA,
            )
        )

    # Tổng giá trị mua.

    if "TotalAmount" in df:

        df["TotalAmount"] = parse_amount(
            df["TotalAmount"],
            number_style,
        )

    else:

        df["Quantity"] = parse_amount(
            df["Quantity"],
            number_style,
        )

        df["UnitPrice"] = parse_amount(
            df["UnitPrice"],
            number_style,
        )

        df["TotalAmount"] = (
            df["Quantity"] * df["UnitPrice"]
        )

    required = [
        "CustomerID",
        "InvoiceDate",
        "InvoiceNo",
        "TotalAmount",
    ]

    missing = int(
        df[required].isna().any(axis=1).sum()
    )

    df = df.dropna(
        subset=required
    ).copy()

    # Loại hóa đơn hủy nếu được chọn.

    cancelled = 0

    if cancel_style != "Không lọc":

        if cancel_style == "Theo trạng thái":

            if "Status" not in df:
                raise ValueError(
                    "Bạn chưa chọn cột trạng thái."
                )

            words = {
                slug(w)
                for w in cancel_words.split(",")
                if w.strip()
            }

            if not words:
                raise ValueError(
                    "Cần khai báo trạng thái hủy."
                )

            mask = (
                df["Status"]
                .fillna("")
                .map(slug)
                .isin(words)
            )

        else:

            if not mapping["InvoiceNo"]:
                raise ValueError(
                    "Không có mã hóa đơn gốc "
                    "để lọc tiền tố C."
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

    # Giá trị không hợp lệ.

    invalid = (
        ~np.isfinite(df["TotalAmount"])
    ) | (
        df["TotalAmount"] <= 0
    )

    if not mapping["TotalAmount"]:

        invalid |= (
            (df["Quantity"] <= 0)
            |
            (df["UnitPrice"] <= 0)
        )

    invalid_n = int(invalid.sum())

    df = df.loc[
        ~invalid
    ].copy()

    duplicate = int(
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

    summary = {
        "Dòng ban đầu": original,
        "Thiếu thông tin": missing,
        "Hóa đơn hủy": cancelled,
        "Giá trị không hợp lệ": invalid_n,
        "Dòng trùng": duplicate,
        "Dòng hợp lệ": len(df),
    }

    return df, summary


# =====================================================
# 8. RFM VÀ K-MEANS
# =====================================================

@st.cache_data(show_spinner=False)
def rfm_data(df):

    ref = (
        df["InvoiceDate"].max().normalize()
        +
        pd.Timedelta(days=1)
    )

    grouped = (
        df.groupby("CustomerID")
        .agg(
            Last=("InvoiceDate", "max"),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalAmount", "sum"),
        )
        .reset_index()
    )

    grouped["Recency"] = (
        ref
        -
        grouped["Last"].dt.normalize()
    ).dt.days

    return grouped[
        [
            "CustomerID",
            "Recency",
            "Frequency",
            "Monetary",
        ]
    ]


@st.cache_data(show_spinner=False)
def train(rfm, k):

    x = np.log1p(
        rfm[
            [
                "Recency",
                "Frequency",
                "Monetary",
            ]
        ].astype(float)
    )

    scaler = StandardScaler()

    scaled = scaler.fit_transform(x)

    if (
        len(scaled) <= k
        or len(np.unique(scaled, axis=0)) < k
    ):
        raise ValueError(
            "Số mẫu RFM khác nhau chưa đủ. "
            "Hãy giảm K."
        )

    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42,
    )

    groups = rfm.copy()

    groups["Cluster"] = (
        model.fit_predict(scaled)
    )

    return (
        groups,
        scaled,
        scaler,
        model,
    )


# =====================================================
# 9. CUSTOMER EVOLUTION
# =====================================================

@st.cache_data(show_spinner=False)
def monthly_rfm(df):

    tmp = df[
        [
            "CustomerID",
            "InvoiceNo",
            "InvoiceDate",
            "TotalAmount",
        ]
    ].copy()

    tmp["Month"] = (
        tmp["InvoiceDate"].dt.to_period("M")
    )

    monthly = (
        tmp.groupby(
            [
                "CustomerID",
                "Month",
            ]
        )
        .agg(
            Last=("InvoiceDate", "max"),
            MonthF=("InvoiceNo", "nunique"),
            MonthM=("TotalAmount", "sum"),
        )
        .reset_index()
        .sort_values(
            [
                "CustomerID",
                "Month",
            ]
        )
    )

    monthly["Frequency"] = (
        monthly.groupby("CustomerID")[
            "MonthF"
        ].cumsum()
    )

    monthly["Monetary"] = (
        monthly.groupby("CustomerID")[
            "MonthM"
        ].cumsum()
    )

    end_month = (
        monthly["Month"]
        .dt.to_timestamp(how="end")
        .dt.normalize()
    )

    monthly["Recency"] = (
        end_month
        -
        monthly["Last"].dt.normalize()
    ).dt.days.clip(lower=0)

    monthly["Tháng"] = (
        monthly["Month"].astype(str)
    )

    return monthly


def assign_evolution(monthly, scaler, model):

    # Không cache đối tượng sklearn.

    evo = monthly.copy()

    x = np.log1p(
        evo[
            [
                "Recency",
                "Frequency",
                "Monetary",
            ]
        ].astype(float)
    )

    evo["Cluster"] = (
        model.predict(
            scaler.transform(x)
        )
    )

    evo["Nhóm khách hàng"] = (
        "Nhóm khách hàng "
        +
        (evo["Cluster"] + 1).astype(str)
    )

    return evo


def transitions(evo, k):

    result = evo.sort_values(
        [
            "CustomerID",
            "Month",
        ]
    ).copy()

    result["PreviousGroup"] = (
        result.groupby("CustomerID")[
            "Cluster"
        ].shift()
    )

    result["PreviousMonth"] = (
        result.groupby("CustomerID")[
            "Month"
        ].shift()
    )

    current = (
        result["Month"].dt.year * 12
        +
        result["Month"].dt.month
    )

    previous = (
        result["PreviousMonth"].dt.year * 12
        +
        result["PreviousMonth"].dt.month
    )

    pairs = result.loc[
        result["PreviousMonth"].notna()
        &
        ((current - previous) == 1)
    ].copy()

    matrix = (
        pd.crosstab(
            pairs["PreviousGroup"],
            pairs["Cluster"],
        )
        .reindex(
            index=range(k),
            columns=range(k),
            fill_value=0,
        )
        .astype(int)
    )

    return matrix, pairs


# =====================================================
# 10. MA TRẬN CHUYỂN NHÓM DỄ ĐỌC
# =====================================================

def show_matrix(matrix, k):

    data = (
        matrix.reindex(
            index=range(k),
            columns=range(k),
            fill_value=0,
        )
        .astype(int)
    )

    total = int(
        data.to_numpy().sum()
    )

    if total == 0:
        st.info(
            "Chưa có khách hàng mua ở "
            "hai tháng liên tiếp để lập ma trận."
        )
        return

    same = int(
        np.trace(data.to_numpy())
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Tổng lượt đối chiếu",
        f"{total:,}",
    )

    c2.metric(
        "🟢 Giữ nhóm",
        f"{same:,}",
        f"{same / total:.1%}",
    )

    c3.metric(
        "🔵 Chuyển nhóm",
        f"{total - same:,}",
        f"{(total - same) / total:.1%}",
    )

    st.caption(
        "Hàng = Nhóm tháng trước ↓; "
        "cột = Nhóm tháng sau →. "
        "Ô xanh lá giữ nhóm; "
        "ô xanh dương chuyển nhóm. "
        "Phần trăm tính theo hàng."
    )

    biggest = max(
        1,
        int(data.to_numpy().max()),
    )

    html = [
        '<div class="matrix-scroll">',
        '<table class="evo-matrix">',
        '<thead><tr>',
        '<th rowspan="2">Nhóm tháng trước ↓</th>',
        f'<th colspan="{k}">Nhóm tháng sau →</th>',
        '<th rowspan="2">Tổng</th>',
        '</tr><tr>',
    ]

    html += [
        f'<th>Nhóm {j + 1} tháng sau</th>'
        for j in range(k)
    ]

    html += [
        '</tr></thead><tbody>'
    ]

    for i in range(k):

        row_total = int(
            data.loc[i].sum()
        )

        html.append(
            f'<tr><th>Nhóm {i + 1} tháng trước</th>'
        )

        for j in range(k):

            amount = int(
                data.loc[i, j]
            )

            percent = (
                100 * amount / row_total
                if row_total
                else 0
            )

            opacity = (
                0.05
                +
                0.5 * np.sqrt(amount / biggest)
                if amount
                else 0.04
            )

            if i == j:
                bg = (
                    f'rgba(16,185,129,{opacity:.3f})'
                )
            else:
                bg = (
                    f'rgba(37,99,235,{opacity:.3f})'
                )

            html.append(
                f'<td style="background:{bg}">'
                f'<strong>{amount:,}</strong>'
                f'<small>{percent:.1f}%</small>'
                '</td>'
            )

        html.append(
            '<td style="background:#F1F5F9">'
            f'{row_total:,}</td></tr>'
        )

    html.append(
        '<tr><th>Tổng nhóm tháng sau</th>'
    )

    html += [
        '<td style="background:#F1F5F9">'
        f'{int(data[j].sum()):,}</td>'
        for j in range(k)
    ]

    html.append(
        '<td style="background:#E2E8F0">'
        f'{total:,}</td>'
        '</tr></tbody></table></div>'
    )

    st.markdown(
        "".join(html),
        unsafe_allow_html=True,
    )

    moves = [
        {
            "Nhóm tháng trước":
                f"Nhóm {i + 1} tháng trước",
            "Nhóm tháng sau":
                f"Nhóm {j + 1} tháng sau",
            "Số lượt":
                int(data.loc[i, j]),
        }

        for i in range(k)
        for j in range(k)

        if i != j and data.loc[i, j] > 0
    ]

    if moves:

        table = (
            pd.DataFrame(moves)
            .sort_values(
                "Số lượt",
                ascending=False,
            )
        )

        st.dataframe(
            table,
            hide_index=True,
            use_container_width=True,
        )

    st.caption(
        "Một khách hàng có thể được đối chiếu "
        "nhiều lần ở các cặp tháng khác nhau. "
        "Không tính tháng không có giao dịch."
    )


# =====================================================
# 11. ELBOW VÀ ĐÁNH GIÁ
# =====================================================

@st.cache_data(show_spinner=False)
def model_scores(scaled, k):

    rng = np.random.default_rng(42)

    sample = scaled[
        rng.choice(
            len(scaled),
            min(len(scaled), 1000),
            replace=False,
        )
    ]

    unique = len(
        np.unique(sample, axis=0)
    )

    elbow = []

    for n in range(
        2,
        min(
            8,
            len(sample) - 1,
            unique,
        ) + 1,
    ):

        km = KMeans(
            n_clusters=n,
            n_init=5,
            random_state=42,
        ).fit(sample)

        elbow.append({
            "K": n,
            "Inertia": km.inertia_,
        })

    comparison = []

    if (
        unique >= k
        and len(sample) > k
    ):

        models = [
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

        for name, model in models:

            labels = (
                model.fit_predict(sample)
            )

            if (
                1 < len(np.unique(labels))
                < len(sample)
            ):

                comparison.append({
                    "Thuật toán": name,
                    "Silhouette":
                        silhouette_score(
                            sample,
                            labels,
                        ),
                    "Davies-Bouldin":
                        davies_bouldin_score(
                            sample,
                            labels,
                        ),
                })

    return (
        pd.DataFrame(elbow),
        pd.DataFrame(comparison),
    )


# =====================================================
# 12. BỘ ĐẾM XUẤT CSV
# =====================================================

def begin_download():
    st.session_state[
        "download_started"
    ] = time.time()


@st.fragment(run_every="1s")
def export_timer():

    start = st.session_state.get(
        "download_started"
    )

    if start is None:
        return

    seconds = max(
        0,
        int(time.time() - start),
    )

    st.info(
        "⏱️ Thời gian từ khi nhấn xuất file: "
        f"{seconds // 3600:02d}:"
        f"{seconds % 3600 // 60:02d}:"
        f"{seconds % 60:02d}"
    )

    if st.button(
        "Dừng bộ đếm",
        key="reset_export",
    ):

        st.session_state[
            "download_started"
        ] = None

        st.rerun(
            scope="fragment"
        )


# =====================================================
# 13. GIAO DIỆN TẢI FILE
# =====================================================

st.markdown(
    '<div class="brand">'
    '◈ CUSTOMER INTELLIGENCE'
    '</div>',
    unsafe_allow_html=True,
)

with st.container(border=True):

    st.markdown(
        '<div class="hero-tag">'
        'DATA ANALYTICS PLATFORM'
        '</div>'
        '<div class="hero-title">'
        'Phân nhóm khách hàng '
        '&amp; Customer Evolution'
        '</div>'
        '<div class="hero-sub">'
        'RFM · K-Means · '
        'Phân tích hành vi khách hàng '
        'theo thời gian'
        '</div>',
        unsafe_allow_html=True,
    )

    u1, umid, u3 = st.columns(
        [0.35, 5, 0.35]
    )

    with umid:

        uploaded = st.file_uploader(
            "Tải dữ liệu CSV / Excel",
            type=[
                "csv",
                "xlsx",
                "xls",
            ],
            label_visibility="collapsed",
        )

        st.caption(
            "Kéo thả hoặc nhấn Browse files "
            "để chọn dữ liệu · CSV, XLSX, XLS"
        )


# Chưa có file thì không hiển thị Dashboard.

if uploaded is None:
    st.stop()


# =====================================================
# 14. ĐỌC DỮ LIỆU
# =====================================================

started = time.perf_counter()

contents = uploaded.getvalue()

try:

    sheet_list = sheets_of(
        contents,
        uploaded.name,
    )

    _, setting_col, _ = st.columns(
        [1, 3, 1]
    )

    with setting_col:

        sheet = st.selectbox(
            "Chọn sheet",
            sheet_list
            +
            (
                ["Tất cả các sheet"]
                if len(sheet_list) > 1
                else []
            ),
        )

    selected = (
        tuple(sheet_list)
        if sheet == "Tất cả các sheet"
        else (sheet,)
    )

    with st.spinner(
        "Đang đọc dữ liệu..."
    ):

        raw = read_data(
            contents,
            uploaded.name,
            selected,
        )

except Exception as error:

    st.error(
        f"Không đọc được file: {error}"
    )

    st.stop()


# =====================================================
# 15. THIẾT LẬP CỘT
# =====================================================

LABELS = {
    "CustomerID": "Mã khách hàng *",
    "InvoiceDate": "Ngày giao dịch *",
    "InvoiceNo": "Mã hóa đơn",
    "TotalAmount": "Thành tiền",
    "Quantity": "Số lượng",
    "UnitPrice": "Đơn giá",
    "Currency": "Đơn vị tiền tệ",
    "Country": "Quốc gia",
    "Status": "Trạng thái hóa đơn",
}


with setting_col:

    with st.expander(
        "Thiết lập cột "
        "(mở nếu nhận diện chưa đúng)"
    ):

        mapping = {}

        choices = [
            "— Không có —"
        ] + list(raw.columns)

        for field, label in LABELS.items():

            detected = next(
                (
                    c for c in raw.columns
                    if LOOKUP.get(
                        slug(c)
                    ) == field
                ),
                None,
            )

            picked = st.selectbox(
                label,
                choices,
                index=(
                    choices.index(detected)
                    if detected in choices
                    else 0
                ),
                key="column_" + field,
            )

            mapping[field] = (
                None
                if picked == "— Không có —"
                else picked
            )

    each_line = st.checkbox(
        "Không có mã hóa đơn: "
        "mỗi dòng là một giao dịch",
        value=False,
        disabled=(
            mapping["InvoiceNo"] is not None
        ),
    )

    date_style = st.selectbox(
        "Định dạng ngày",
        DATES,
    )

    if mapping["InvoiceDate"]:

        st.caption(
            "Nhận diện ngày: "
            +
            guess_date(
                raw[
                    mapping["InvoiceDate"]
                ]
            )
        )

    if date_style == "Theo cột quốc gia":

        st.caption(
            "Chỉ dùng khi ngày của mỗi dòng "
            "thật sự tuân theo cách ghi "
            "của quốc gia đó."
        )

    # Nhận diện tiền tệ.

    money_source = mapping["Currency"]

    money_amount = (
        mapping["TotalAmount"]
        or mapping["UnitPrice"]
    )

    if money_source:

        found = (
            raw[money_source]
            .map(currency_code)
        )

    elif money_amount:

        found = (
            raw[money_amount]
            .head(2000)
            .map(currency_in_amount)
        )

    else:

        found = pd.Series(
            dtype="object"
        )

    detected_money = sorted(
        set(
            found.dropna().unique()
        )
    )

    suggested = (
        detected_money[0]
        if len(detected_money) == 1
        else "Chưa xác định"
    )

    currency = st.selectbox(
        "Đơn vị tiền tệ",
        [
            "Chưa xác định",
            *MONEY,
        ],
        index=(
            ["Chưa xác định", *MONEY]
            .index(suggested)
        ),
    )

    if detected_money:

        st.caption(
            "Phát hiện: "
            +
            ", ".join(detected_money)
        )

    else:

        st.caption(
            "Không nhận diện chắc chắn tiền tệ. "
            "Có thể chọn thủ công; "
            "không tự đổi tỷ giá."
        )

    number_style = st.selectbox(
        "Dấu phân cách số tiền",
        [
            "Tự động",
            "1,234.56",
            "1.234,56",
        ],
    )

    # Hóa đơn hủy.

    has_cancel = st.checkbox(
        "Nếu có hóa đơn bị hủy, "
        "loại khỏi phân tích"
    )

    cancel_style = "Không lọc"

    cancel_words = (
        "cancelled, canceled, "
        "void, hủy, đã hủy"
    )

    if has_cancel:

        cancel_types = (
            (
                ["Theo trạng thái"]
                if mapping["Status"]
                else []
            )
            +
            (
                ["Mã bắt đầu bằng C"]
                if mapping["InvoiceNo"]
                else []
            )
        )

        if not cancel_types:

            st.warning(
                "Cần cột trạng thái "
                "hoặc mã hóa đơn "
                "để lọc đơn hủy."
            )

            st.stop()

        cancel_style = st.selectbox(
            "Nhận biết hóa đơn hủy",
            cancel_types,
        )

        if cancel_style == "Theo trạng thái":

            cancel_words = st.text_input(
                "Các trạng thái hủy "
                "(ngăn cách dấu phẩy)",
                cancel_words,
            )

        else:

            st.caption(
                "Chỉ dùng nếu tiền tố C "
                "thực sự nghĩa là đơn hủy "
                "trong bộ dữ liệu này."
            )


# =====================================================
# 16. TIỀN XỬ LÝ VÀ PHÂN NHÓM
# =====================================================

try:

    with st.spinner(
        "Đang tiền xử lý, tính RFM..."
    ):

        clean, summary = prepare(
            raw,
            mapping,
            currency,
            date_style,
            number_style,
            cancel_style,
            cancel_words,
            each_line,
        )

        rfm = rfm_data(clean)

    if len(rfm) < 3:
        raise ValueError(
            "Cần ít nhất 3 khách hàng "
            "hợp lệ để phân nhóm."
        )

    with setting_col:

        k = st.slider(
            "Số nhóm khách hàng (K)",
            2,
            min(8, len(rfm) - 1),
            min(4, len(rfm) - 1),
        )

    with st.spinner(
        "Đang phân nhóm và tính "
        "Customer Evolution..."
    ):

        groups, scaled, scaler, model = train(
            rfm,
            k,
        )

        profiles = (
            groups.groupby("Cluster")
            .agg(
                Customers=(
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
            (
                profiles["Cluster"] + 1
            ).astype(str)
        )

        evolution = assign_evolution(
            monthly_rfm(clean),
            scaler,
            model,
        )

        matrix, pairs = transitions(
            evolution,
            k,
        )

except Exception as error:

    st.error(
        f"Phân tích chưa hoàn tất: {error}"
    )

    st.stop()


st.success(
    "✅ Phân tích hoàn tất! "
    "Cuộn xuống để xem Dashboard."
)

st.caption(
    f"Thời gian xử lý lượt này: "
    f"{time.perf_counter() - started:.2f} giây "
    "(có thể dùng bộ nhớ đệm)."
)

st.divider()


# =====================================================
# 17. DASHBOARD
# =====================================================

heading("📊 Tổng quan kết quả")

metrics = [
    ("Khách hàng", len(rfm)),
    (
        "Hóa đơn",
        clean["InvoiceNo"].nunique(),
    ),
    (
        f"Giá trị mua ({currency})",
        clean["TotalAmount"].sum(),
    ),
    (
        "Nhóm khách hàng",
        k,
    ),
]

for col, (name, value) in zip(
    st.columns(4),
    metrics,
):

    with col:

        metric_card(
            name,
            f"{value:,.0f}",
        )


# =====================================================
# 18. PHÂN BỐ NHÓM - ĐÃ SỬA MÀU BIỂU ĐỒ
# =====================================================

heading("👥 Phân bố nhóm khách hàng")

bar_col, pie_col = st.columns(2)

order = [
    f"Nhóm khách hàng {i}"
    for i in range(1, k + 1)
]

color_map = {
    name: COLORS[i]
    for i, name in enumerate(order)
}


with bar_col:

    fig = px.bar(
        profiles,
        x="Nhóm khách hàng",
        y="Customers",
        color="Nhóm khách hàng",
        color_discrete_map=color_map,
        category_orders={
            "Nhóm khách hàng": order,
        },
        text="Customers",
        labels={
            "Customers": "Số khách hàng",
        },
    )

    # Chữ trắng nằm bên trong cột màu.
    fig.update_traces(
        textposition="inside",
        textfont=dict(
            color="#FFFFFF",
            size=13,
        ),
    )

    fig.update_layout(
        showlegend=False
    )

    fig.update_xaxes(
        tickangle=-12
    )

    draw(fig)


with pie_col:

    fig = px.pie(
        profiles,
        names="Nhóm khách hàng",
        values="Customers",
        hole=0.55,
        color="Nhóm khách hàng",
        color_discrete_map=color_map,
        category_orders={
            "Nhóm khách hàng": order,
        },
    )

    fig.update_layout(
        legend=dict(
            x=1.02,
            xanchor="left",
            y=0.5,
            yanchor="middle",
        ),
    )

    draw(fig)


# =====================================================
# 19. ĐẶC ĐIỂM TỪNG NHÓM
# =====================================================

heading("📋 Đặc điểm từng nhóm khách hàng")

profile_table = profiles[
    [
        "Nhóm khách hàng",
        "Customers",
        "R",
        "F",
        "M",
    ]
].rename(
    columns={
        "Customers": "Số khách hàng",
        "R": "R (ngày)",
        "F": "F (hóa đơn)",
        "M": f"M (giá trị mua - {currency})",
    }
)

st.dataframe(
    profile_table.round(2),
    hide_index=True,
    use_container_width=True,
)

st.caption(
    "R: số ngày từ lần mua gần nhất · "
    "F: số hóa đơn · "
    "M: tổng giá trị mua."
)


# =====================================================
# 20. CUSTOMER EVOLUTION TỔNG QUAN
# =====================================================

heading("🔄 Customer Evolution")

st.info(
    "Theo dõi nhóm khách hàng "
    "trong từng tháng có giao dịch. "
    "RFM tích lũy được phân loại "
    "bằng cùng một mô hình K-Means."
)

counts = (
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

line = px.line(
    counts,
    x="Tháng",
    y="Số khách hàng",
    color="Nhóm khách hàng",
    markers=True,
    color_discrete_map=color_map,
    category_orders={
        "Nhóm khách hàng": order,
    },
)

draw(line, 420)


if not pairs.empty:

    n_moved = int(
        (
            pairs["PreviousGroup"]
            !=
            pairs["Cluster"]
        ).sum()
    )

    st.caption(
        f"Có {n_moved:,} lượt chuyển nhóm "
        f"/ {len(pairs):,} lượt đối chiếu "
        "hai tháng liền kề."
    )


# =====================================================
# 21. CHI TIẾT PHÂN TÍCH
# =====================================================

heading("🔎 Chi tiết phân tích")


with st.expander("1. Dữ liệu"):

    st.caption(
        f"File: {uploaded.name} · "
        f"Sheet: {', '.join(selected)}"
    )

    st.dataframe(
        raw.head(40),
        hide_index=True,
        use_container_width=True,
    )


with st.expander("2. Tiền xử lý"):

    st.dataframe(
        pd.DataFrame(
            summary.items(),
            columns=[
                "Tiêu chí",
                "Số dòng",
            ],
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.dataframe(
        clean.head(40),
        hide_index=True,
        use_container_width=True,
    )


with st.expander("3. RFM"):

    st.write(
        "**R:** Ngày từ lần mua gần nhất. "
        "**F:** Số hóa đơn. "
        "**M:** Tổng giá trị mua."
    )

    st.dataframe(
        rfm.head(100),
        hide_index=True,
        use_container_width=True,
    )


with st.expander("4. Chuẩn hóa"):

    st.write(
        "Log1p, sau đó StandardScaler "
        "để đưa R, F, M về thang đo "
        "tương đương."
    )

    st.dataframe(
        pd.DataFrame(
            scaled[:40],
            columns=[
                "R chuẩn hóa",
                "F chuẩn hóa",
                "M chuẩn hóa",
            ],
        ),
        hide_index=True,
        use_container_width=True,
    )


with st.expander("5. K-Means"):

    st.write(
        f"Đã chia thành **{k} nhóm khách hàng**, "
        f"từ Nhóm 1 đến Nhóm {k}."
    )

    if st.button("Tính Elbow"):

        elbow, _ = model_scores(
            scaled,
            k,
        )

        if not elbow.empty:

            draw(
                px.line(
                    elbow,
                    x="K",
                    y="Inertia",
                    markers=True,
                )
            )


with st.expander(
    "6. Phân tích nhóm khách hàng"
):

    st.dataframe(
        profile_table.round(2),
        hide_index=True,
        use_container_width=True,
    )


with st.expander(
    "7. Đánh giá và so sánh"
):

    st.caption(
        "K-Means và Agglomerative "
        "được so sánh trên cùng mẫu "
        "tối đa 1.000 khách hàng."
    )

    if st.button(
        "Chạy so sánh thuật toán"
    ):

        _, scores = model_scores(
            scaled,
            k,
        )

        st.dataframe(
            scores.round(4),
            hide_index=True,
            use_container_width=True,
        )


# =====================================================
# 22. CUSTOMER EVOLUTION - VÍ DỤ VÀ MA TRẬN
# =====================================================

with st.expander(
    "8. Customer Evolution – "
    "Ví dụ và hành trình khách hàng"
):

    st.markdown(
        "#### Ví dụ minh họa "
        "(dữ liệu giả định)"
    )

    demo = pd.DataFrame({

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
        demo,
        hide_index=True,
        use_container_width=True,
    )

    st.write(
        "Ví dụ: **Nhóm 1 tháng trước → "
        "Nhóm 2 tháng sau** nghĩa là chuyển nhóm. "
        "Số 1, 2, 3 là mã nhóm, "
        "không phải thứ hạng tốt/xấu."
    )

    # Chú thích RFM.

    st.markdown(
        "#### Chú thích RFM"
    )

    guide = pd.DataFrame(
        [
            [
                "R – Recency",
                "Ngày từ lần mua gần nhất "
                "đến cuối tháng",
                "R nhỏ: mua gần hơn",
            ],
            [
                "F – Frequency",
                "Số hóa đơn tích lũy",
                "F lớn: mua nhiều lần hơn",
            ],
            [
                "M – Monetary",
                "Tổng giá trị mua tích lũy",
                "M lớn: tổng mua nhiều hơn",
            ],
        ],
        columns=[
            "Chỉ số",
            "Ý nghĩa",
            "Cách đọc",
        ],
    )

    st.dataframe(
        guide,
        hide_index=True,
        use_container_width=True,
    )

    # Ma trận chuyển nhóm.

    st.markdown(
        "#### Ma trận chuyển nhóm "
        "từ dữ liệu thật"
    )

    show_matrix(
        matrix,
        k,
    )

    # Hành trình khách hàng.

    st.markdown(
        "#### Hành trình khách hàng"
    )

    customer = st.selectbox(
        "Chọn mã khách hàng",
        sorted(
            evolution["CustomerID"]
            .unique()
            .tolist()
        ),
    )

    history = (
        evolution.loc[
            evolution["CustomerID"]
            ==
            customer
        ]
        .sort_values("Month")
    )

    view = history[
        [
            "Tháng",
            "Nhóm khách hàng",
            "Recency",
            "Frequency",
            "Monetary",
        ]
    ].rename(
        columns={
            "Recency": "R – ngày",
            "Frequency": "F – hóa đơn tích lũy",
            "Monetary": (
                f"M – tiền mua ({currency})"
            ),
        }
    )

    st.dataframe(
        view.round(2),
        hide_index=True,
        use_container_width=True,
    )

    if len(history) > 1:

        path = history.assign(
            Nhom=history["Cluster"] + 1
        )

        fig = px.line(
            path,
            x="Tháng",
            y="Nhom",
            markers=True,
            labels={
                "Nhom": "Nhóm khách hàng",
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

        draw(fig, 320)

    st.caption(
        "Lịch sử hiện chỉ gồm tháng "
        "phát sinh giao dịch. "
        "Đây không phải dự báo tương lai."
    )


with st.expander(
    "9. Streamlit và kết quả"
):

    st.write(
        f"Đã phân tích "
        f"**{len(rfm):,} khách hàng** "
        f"thành **{k} nhóm khách hàng**."
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

export_groups["Mã nhóm"] = (
    export_groups["Cluster"] + 1
)

export_groups["Nhóm khách hàng"] = (
    "Nhóm khách hàng "
    +
    export_groups["Mã nhóm"].astype(str)
)

export_groups = export_groups.drop(
    columns="Cluster"
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

c1, c2 = st.columns(2)

with c1:

    st.download_button(
        "⬇️ Xuất danh sách nhóm khách hàng",
        export_groups.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="nhom_khach_hang.csv",
        mime="text/csv",
        on_click=begin_download,
    )


with c2:

    st.download_button(
        "⬇️ Xuất Customer Evolution",
        export_evo.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="customer_evolution.csv",
        mime="text/csv",
        on_click=begin_download,
    )


export_timer()

st.caption(
    "Đồng hồ bắt đầu khi nhấn nút xuất CSV, "
    "không xác nhận thời điểm trình duyệt "
    "lưu file xong."
)

st.divider()

st.caption(
    "CUSTOMER INTELLIGENCE · "
    "RFM · K-Means · Customer Evolution"
)
