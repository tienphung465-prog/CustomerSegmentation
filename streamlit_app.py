
import re
import time
import unicodedata
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.cluster import (
    AgglomerativeClustering,
    KMeans,
)
from sklearn.metrics import (
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


# =====================================================
# 1. CẤU HÌNH
# =====================================================

st.set_page_config(
    page_title="Customer Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

COLORS = [
    "#2563eb",
    "#10b981",
    "#8b5cf6",
    "#f59e0b",
    "#ec4899",
    "#06b6d4",
    "#6366f1",
    "#ef4444",
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
    "VND",
    "USD",
    "EUR",
    "GBP",
    "JPY",
    "CNY",
    "KRW",
    "AUD",
    "CAD",
    "SGD",
    "THB",
    "MYR",
    "INR",
]

SYMBOL_TO_CODE = {
    "₫": "VND",
    "€": "EUR",
    "£": "GBP",
    "₩": "KRW",
    "₹": "INR",
}


# =====================================================
# 2. GIAO DIỆN NỀN TRẮNG
# =====================================================

st.markdown("""
<style>

.stApp {
    background: #f6f8fc;
    color: #16243b;
}

.block-container {
    max-width: 1280px;
    padding-top: 1.4rem;
}

[data-testid="stHeader"] {
    background: transparent;
}

.brand {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 16px 22px;
    color: #1d4ed8;
    font-weight: 800;
}

.hero {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    text-align: center;
    padding: 26px 14px 16px;
    margin: 17px 0 18px;
}

.hero h1 {
    font-size: clamp(27px, 4vw, 40px);
    color: #15253c;
    line-height: 1.2;
    margin: 12px 0;
}

.hero p {
    font-size: 14px;
    color: #64748b;
}

[data-testid="stFileUploaderDropzone"] {
    background: #f8fbff;
    border: 2px dashed #b9cff1;
    border-radius: 12px;
}

.kpi {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 13px;
    padding: 16px;
    min-height: 105px;
}

.kpi small {
    color: #64748b;
    font-size: 12px;
}

.kpi strong {
    display: block;
    font-size: 24px;
    color: #2563eb;
    margin-top: 8px;
    overflow-wrap: anywhere;
}

[data-testid="stExpander"] {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


def heading(text):
    st.markdown(f"### {text}")


def kpi(label, value):
    st.markdown(
        f'<div class="kpi">'
        f'<small>{label}</small>'
        f'<strong>{value}</strong>'
        f'</div>',
        unsafe_allow_html=True,
    )


def graph(fig, height=370):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(
            l=10,
            r=10,
            t=25,
            b=30,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# =====================================================
# 3. NHẬN DIỆN TÊN CỘT
# =====================================================

def normalize(s):
    s = unicodedata.normalize(
        "NFD",
        str(s).strip().lower(),
    ).replace("đ", "d")

    s = "".join(
        x for x in s
        if unicodedata.category(x) != "Mn"
    )

    return re.sub(
        r"[^a-z0-9]",
        "",
        s,
    )


ALIASES = {
    "CustomerID": [
        "customer id",
        "client id",
        "customer",
        "mã khách hàng",
        "khách hàng",
    ],

    "InvoiceNo": [
        "invoice",
        "invoice no",
        "order id",
        "order number",
        "transaction id",
        "mã hóa đơn",
        "mã đơn hàng",
    ],

    "InvoiceDate": [
        "invoice date",
        "order date",
        "purchase date",
        "transaction date",
        "ngày mua",
        "ngày giao dịch",
        "date",
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
        "price",
        "unit price",
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

ALIASES = {
    k: [k] + v
    for k, v in ALIASES.items()
}

LOOKUP = {
    normalize(alias): name
    for name, aliases in ALIASES.items()
    for alias in aliases
}


# =====================================================
# 4. ĐỌC FILE
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
                    sheet_name=s,
                )
                for s in selected
            ],
            ignore_index=True,
        )

    frame.columns = [
        str(x).strip()
        for x in frame.columns
    ]

    if frame.columns.duplicated().any():
        raise ValueError(
            "File có các tên cột trùng nhau."
        )

    return frame


# =====================================================
# 5. NHẬN DIỆN ĐƠN VỊ TIỀN TỆ
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

    # Không tự đoán $ hay ¥ vì không duy nhất.
    return None


def currency_in_text(value):

    if pd.isna(value):
        return None

    s = str(value)

    found = {
        code
        for symbol, code in SYMBOL_TO_CODE.items()
        if symbol in s
    }

    found |= {
        code
        for code in CURRENCY_CODES
        if re.search(
            rf"\b{code}\b",
            s,
            re.I,
        )
    }

    if len(found) == 1:
        return next(iter(found))

    return None


# =====================================================
# 6. CHUYỂN ĐỔI GIÁ TRỊ TIỀN
# =====================================================

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
            s.str.replace(
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
                "Hãy chọn cách ghi số tiền."
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

            v = s.at[idx]

            if v.rfind(",") > v.rfind("."):
                s.at[idx] = (
                    v.replace(".", "")
                    .replace(",", ".")
                )

            else:
                s.at[idx] = v.replace(",", "")

    return pd.to_numeric(
        s,
        errors="coerce",
    )


# =====================================================
# 7. NHẬN DIỆN ĐỊNH DẠNG NGÀY
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

    a = sample.str.extract(
        r"^(\d{1,4})[-/](\d{1,2})[-/](\d{1,4})"
    )

    if a.dropna().empty:
        return "Không đủ mẫu để tự xác định"

    first = pd.to_numeric(
        a[0],
        errors="coerce",
    )

    second = pd.to_numeric(
        a[1],
        errors="coerce",
    )

    if (first > 31).all():
        return "YYYY-MM-DD"

    if (
        (first > 12).any()
        and (second > 12).any()
    ):
        return (
            "Dữ liệu có định dạng lẫn lộn; "
            "hãy kiểm tra"
        )

    if (first > 12).any():
        return "DD/MM/YYYY"

    if (second > 12).any():
        return "MM/DD/YYYY"

    return (
        "Ngày/tháng mơ hồ; "
        "chọn định dạng thủ công"
    )


def parse_dates(
    series,
    mode,
    countries=None,
):

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
                "Ngày tháng chưa thể tự nhận diện "
                "an toàn. Hãy chọn định dạng thủ công."
            )

        mode = hint

    if mode == "Theo cột quốc gia":

        if countries is None:
            raise ValueError(
                "Cần chọn cột Quốc gia."
            )

        out = pd.Series(
            pd.NaT,
            index=series.index,
            dtype="datetime64[ns]",
        )

        country_values = (
            countries.astype("string")
            .str.strip()
            .str.lower()
        )

        if country_values.isna().any():
            raise ValueError(
                "Có dòng thiếu quốc gia. "
                "Hãy chọn định dạng thủ công."
            )

        for country in (
            country_values.dropna().unique()
        ):

            local_format = COUNTRY_FORMAT.get(
                str(country)
            )

            if local_format is None:
                raise ValueError(
                    f"Chưa có quy tắc ngày cho "
                    f"quốc gia '{country}'. "
                    "Hãy chọn định dạng thủ công."
                )

            mask = (
                country_values == country
            ).fillna(False)

            out.loc[mask] = parse_dates(
                series.loc[mask],
                local_format,
            )

        return out

    return pd.to_datetime(
        series,
        format="mixed",
        dayfirst=(mode == "DD/MM/YYYY"),
        errors="coerce",
    )


# =====================================================
# 8. TIỀN XỬ LÝ GIAO DỊCH
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
        v for v in mapping.values()
        if v
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

    if any(
        c not in df
        for c in (
            "CustomerID",
            "InvoiceDate",
        )
    ):
        raise ValueError(
            "Thiếu Mã khách hàng "
            "hoặc Ngày giao dịch."
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
            "Số lượng và Đơn giá."
        )

    original = len(df)

    # -------------------------------------------------
    # ĐƠN VỊ TIỀN TỆ
    # -------------------------------------------------

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
                "Phát hiện nhiều tiền tệ. "
                "Hãy chọn một loại để phân tích."
            )

        if row_currency.isna().any():
            raise ValueError(
                "Có dòng chưa rõ tiền tệ "
                "trong file nhiều loại tiền. "
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
                "Tiền tệ bạn chọn không khớp "
                "với tiền tệ phát hiện trong file."
            )

    if df.empty:
        raise ValueError(
            "Không còn giao dịch "
            "trong đơn vị tiền đã chọn."
        )

    # -------------------------------------------------
    # NGÀY THÁNG
    # -------------------------------------------------

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

    # -------------------------------------------------
    # MÃ KHÁCH HÀNG
    # -------------------------------------------------

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

    # -------------------------------------------------
    # GIÁ TRỊ GIAO DỊCH
    # -------------------------------------------------

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

    required = [
        "CustomerID",
        "InvoiceNo",
        "InvoiceDate",
        "TotalAmount",
    ]

    missing = int(
        df[required]
        .isna()
        .any(axis=1)
        .sum()
    )

    df = df.dropna(
        subset=required
    ).copy()

    # -------------------------------------------------
    # HÓA ĐƠN HỦY
    # -------------------------------------------------

    cancelled = 0

    if cancel_mode == "Theo trạng thái":

        if "Status" not in df:
            raise ValueError(
                "Bạn chưa chọn cột Trạng thái."
            )

        values = {
            normalize(s)
            for s in cancel_words.split(",")
            if s.strip()
        }

        if not values:
            raise ValueError(
                "Cần nhập trạng thái "
                "tương ứng với đơn hủy."
            )

        mask = (
            df["Status"]
            .fillna("")
            .map(normalize)
            .isin(values)
        )

        cancelled = int(
            mask.sum()
        )

        df = df.loc[
            ~mask
        ].copy()

    elif cancel_mode == "Mã bắt đầu bằng C":

        if not mapping.get("InvoiceNo"):
            raise ValueError(
                "Không có mã hóa đơn gốc "
                "để xác định chữ C."
            )

        mask = (
            df["InvoiceNo"]
            .astype(str)
            .str.upper()
            .str.startswith("C")
        )

        cancelled = int(
            mask.sum()
        )

        df = df.loc[
            ~mask
        ].copy()

    # -------------------------------------------------
    # DỮ LIỆU KHÔNG HỢP LỆ
    # -------------------------------------------------

    invalid = (
        ~np.isfinite(df["TotalAmount"])
    ) | (
        df["TotalAmount"] <= 0
    )

    if (
        "TotalAmount" not in mapping
        or mapping["TotalAmount"] is None
    ):
        invalid |= (
            (df["Quantity"] <= 0)
            |
            (df["UnitPrice"] <= 0)
        )

    invalid_count = int(
        invalid.sum()
    )

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
            "Dữ liệu không có giao dịch "
            "hợp lệ sau khi xử lý."
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
# 9. TÍNH RFM
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
            Last=(
                "InvoiceDate",
                "max",
            ),
            Frequency=(
                "InvoiceNo",
                "nunique",
            ),
            Monetary=(
                "TotalAmount",
                "sum",
            ),
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

    X = np.log1p(
        rfm[
            [
                "Recency",
                "Frequency",
                "Monetary",
            ]
        ].astype(float)
    )

    scaler = StandardScaler()

    z = scaler.fit_transform(X)

    if (
        len(z) <= k
        or len(np.unique(z, axis=0)) < k
    ):
        raise ValueError(
            "Không đủ khách hàng có "
            "RFM khác biệt để chia K nhóm."
        )

    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42,
    )

    result = rfm.copy()

    result["Cluster"] = (
        model.fit_predict(z)
    )

    return (
        result,
        z,
        scaler,
        model,
    )


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
        m["InvoiceDate"]
        .dt.to_period("M")
    )

    m = (
        m.groupby(
            [
                "CustomerID",
                "Month",
            ]
        )
        .agg(
            Last=(
                "InvoiceDate",
                "max",
            ),
            MonthF=(
                "InvoiceNo",
                "nunique",
            ),
            MonthM=(
                "TotalAmount",
                "sum",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "CustomerID",
                "Month",
            ]
        )
    )

    m["Frequency"] = (
        m.groupby("CustomerID")[
            "MonthF"
        ].cumsum()
    )

    m["Monetary"] = (
        m.groupby("CustomerID")[
            "MonthM"
        ].cumsum()
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
    ).dt.days.clip(
        lower=0
    )

    m["Tháng"] = (
        m["Month"].astype(str)
    )

    return m


def evolution_model(
    monthly,
    scaler,
    model,
):

    # Không dùng cache cho scaler/model.

    evo = monthly.copy()

    z = scaler.transform(
        np.log1p(
            evo[
                [
                    "Recency",
                    "Frequency",
                    "Monetary",
                ]
            ].astype(float)
        )
    )

    evo["Cluster"] = (
        model.predict(z)
    )

    evo["Nhóm khách hàng"] = (
        "Nhóm khách hàng "
        +
        evo["Cluster"].astype(str)
    )

    return evo


def transition_matrix(evo, k):

    a = evo.sort_values(
        [
            "CustomerID",
            "Month",
        ]
    ).copy()

    a["BeforeGroup"] = (
        a.groupby("CustomerID")[
            "Cluster"
        ].shift()
    )

    a["BeforeMonth"] = (
        a.groupby("CustomerID")[
            "Month"
        ].shift()
    )

    month_now = (
        a["Month"].dt.year * 12
        +
        a["Month"].dt.month
    )

    month_before = (
        a["BeforeMonth"].dt.year * 12
        +
        a["BeforeMonth"].dt.month
    )

    changes = a.loc[
        a["BeforeMonth"].notna()
        &
        (
            (month_now - month_before) == 1
        )
    ]

    matrix = pd.crosstab(
        changes["BeforeGroup"],
        changes["Cluster"],
    )

    return (
        matrix.reindex(
            index=range(k),
            columns=range(k),
            fill_value=0,
        ),
        changes,
    )


# =====================================================
# 12. ELBOW VÀ SO SÁNH THUẬT TOÁN
# =====================================================

@st.cache_data(show_spinner=False)
def diagnostics(z, k):

    rng = np.random.default_rng(42)

    sample = z[
        rng.choice(
            len(z),
            min(
                1000,
                len(z),
            ),
            replace=False,
        )
    ]

    unique = len(
        np.unique(
            sample,
            axis=0,
        )
    )

    elbow = []

    for count in range(
        2,
        min(
            8,
            len(sample) - 1,
            unique,
        ) + 1,
    ):

        km = KMeans(
            n_clusters=count,
            n_init=5,
            random_state=42,
        ).fit(sample)

        elbow.append({
            "K": count,
            "Inertia": km.inertia_,
        })

    scores = []

    if (
        len(sample) > k
        and unique >= k
    ):

        methods = (
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
        )

        for name, method in methods:

            labels = method.fit_predict(
                sample
            )

            if (
                1
                <
                len(np.unique(labels))
                <
                len(sample)
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
# 13. BỘ ĐẾM SAU KHI XUẤT CSV
# =====================================================

def mark_export():
    st.session_state[
        "export_time"
    ] = time.time()


def format_elapsed(seconds):

    seconds = int(
        max(
            0,
            seconds,
        )
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
            "⏱️ Thời gian kể từ "
            "khi nhấn xuất CSV: "
            +
            format_elapsed(elapsed)
        )

        if st.button(
            "Dừng / đặt lại bộ đếm",
            key="stop_timer",
        ):

            st.session_state[
                "export_time"
            ] = None

            st.rerun(
                scope="fragment"
            )


# =====================================================
# 14. GIAO DIỆN ĐẦU TRANG
# =====================================================

st.markdown(
    '<div class="brand">'
    '◈ CUSTOMER INTELLIGENCE'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero">'
    '<h1>Phân nhóm khách hàng '
    '&amp; Customer Evolution</h1>'
    '<p>RFM · K-Means · '
    'Theo dõi sự thay đổi nhóm khách hàng '
    'theo thời gian</p>'
    '</div>',
    unsafe_allow_html=True,
)

left, center, right = st.columns(
    [1, 2.5, 1]
)

with center:

    uploaded = st.file_uploader(
        "Tải file Excel hoặc CSV",
        type=[
            "csv",
            "xlsx",
            "xls",
        ],
    )


# Chưa tải file thì không có Dashboard.

if uploaded is None:
    st.stop()


# =====================================================
# 15. ĐỌC VÀ CHỌN DỮ LIỆU
# =====================================================

started_processing = (
    time.perf_counter()
)

blob = uploaded.getvalue()

try:

    sheets = list_sheets(
        blob,
        uploaded.name,
    )

    with center:

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

    with st.spinner(
        "Đang đọc dữ liệu..."
    ):

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
# 16. THIẾT LẬP CỘT VÀ TÙY CHỌN
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

with center:

    with st.expander(
        "Thiết lập cột "
        "(mở khi nhận diện chưa đúng)"
    ):

        mapping = {}

        options = [
            "— Không có —"
        ] + list(raw.columns)

        for column, label in (
            FIELD_NAMES.items()
        ):

            guess = next(
                (
                    c for c in raw.columns
                    if LOOKUP.get(
                        normalize(c)
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
            mapping["InvoiceNo"]
            is not None
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
            "Chưa chọn cột ngày"
        )

    if date_mode == "Theo cột quốc gia":

        st.caption(
            "Chỉ dùng khi ngày trong từng dòng "
            "thực sự được ghi theo quy ước "
            "của quốc gia ở dòng đó."
        )

    currency_col = mapping[
        "Currency"
    ]

    amount_col = (
        mapping["TotalAmount"]
        or mapping["UnitPrice"]
    )

    if currency_col:

        detected_values = raw[
            currency_col
        ].map(currency_of)

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
            detected_values
            .dropna()
            .unique()
        )
    )

    currency_default = (
        known_currencies[0]
        if len(known_currencies) == 1
        else "Chưa xác định"
    )

    currency_options = [
        "Chưa xác định"
    ] + CURRENCY_CODES

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
            "Không tìm thấy ký hiệu tiền rõ ràng; "
            "hãy chọn tiền tệ nếu biết. "
            "Hệ thống không tự quy đổi."
        )

    number_style = st.selectbox(
        "Cách ghi số tiền",
        [
            "Tự động",
            "1,234.56",
            "1.234,56",
        ],
    )

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

            choices.append(
                "Theo trạng thái"
            )

        if mapping["InvoiceNo"]:

            choices.append(
                "Mã bắt đầu bằng C"
            )

        if not choices:

            st.warning(
                "Cần chọn cột Trạng thái "
                "hoặc Mã hóa đơn để lọc đơn hủy."
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
                "Chỉ chọn khi mã C thực sự "
                "là ký hiệu hóa đơn hủy."
            )


# =====================================================
# 17. CHUẨN BỊ RFM
# =====================================================

try:

    with st.spinner(
        "Đang làm sạch dữ liệu "
        "và tính RFM..."
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
        "Không thể chuẩn bị dữ liệu: "
        f"{exc}"
    )

    st.stop()


if len(rfm) < 3:

    st.error(
        "Cần ít nhất 3 khách hàng "
        "có giao dịch hợp lệ."
    )

    st.stop()


with center:

    k = st.slider(
        "Số nhóm khách hàng (K)",
        2,
        min(
            8,
            len(rfm) - 1,
        ),
        min(
            4,
            len(rfm) - 1,
        ),
    )


# =====================================================
# 18. CHẠY TOÀN BỘ PHÂN TÍCH
# =====================================================

try:

    with st.spinner(
        "Đang chạy K-Means "
        "và Customer Evolution..."
    ):

        (
            groups,
            z,
            scaler,
            model,
        ) = clustering(
            rfm,
            k,
        )

        profiles = (
            groups.groupby("Cluster")
            .agg(
                Khach_hang=(
                    "CustomerID",
                    "count",
                ),
                R=(
                    "Recency",
                    "mean",
                ),
                F=(
                    "Frequency",
                    "mean",
                ),
                M=(
                    "Monetary",
                    "mean",
                ),
            )
            .reset_index()
        )

        profiles["Nhóm khách hàng"] = (
            "Nhóm khách hàng "
            +
            profiles["Cluster"].astype(str)
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


# Chỉ hiển thị Dashboard khi đã thành công.

elapsed_processing = (
    time.perf_counter()
    -
    started_processing
)

st.success(
    "✅ Phân tích hoàn tất! "
    "Cuộn xuống để xem kết quả."
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

heading(
    "📊 Tổng quan kết quả"
)

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


# =====================================================
# 20. PHÂN BỐ NHÓM
# =====================================================

heading(
    "👥 Phân bố nhóm khách hàng"
)

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

    graph(
        px.pie(
            profiles,
            names="Nhóm khách hàng",
            values="Khach_hang",
            hole=0.55,
            color_discrete_sequence=COLORS,
        )
    )


# =====================================================
# 21. ĐẶC ĐIỂM CÁC NHÓM
# =====================================================

heading(
    "📋 Đặc điểm từng nhóm khách hàng"
)

profile_view = (
    profiles[
        [
            "Nhóm khách hàng",
            "Khach_hang",
            "R",
            "F",
            "M",
        ]
    ]
    .rename(
        columns={
            "Khach_hang": "Số khách hàng",
            "R": "R: Số ngày",
            "F": "F: Số hóa đơn",
            "M": (
                f"M: Tiền mua "
                f"({currency_choice})"
            ),
        }
    )
)

st.dataframe(
    profile_view.round(2),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "R: số ngày từ lần mua gần nhất "
    "(thấp là gần hơn); "
    "F: số hóa đơn; "
    "M: tổng giá trị mua."
)


# =====================================================
# 22. DASHBOARD CUSTOMER EVOLUTION
# =====================================================

heading(
    "🔄 Customer Evolution"
)

st.info(
    "Biểu đồ đếm những khách hàng có mua "
    "ở mỗi tháng. RFM tích lũy được gán nhóm "
    "bởi cùng một mô hình K-Means."
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

graph(
    px.line(
        evo_counts,
        x="Tháng",
        y="Số khách hàng",
        color="Nhóm khách hàng",
        markers=True,
        color_discrete_sequence=COLORS,
    ),
    410,
)

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
        "so sánh tháng liền kề."
    )


# =====================================================
# 23. CHI TIẾT PHÂN TÍCH
# =====================================================

heading(
    "🔎 Chi tiết phân tích"
)


with st.expander("1. Dữ liệu"):

    st.caption(
        f"File: {uploaded.name} | "
        f"Sheet: {', '.join(selected)} | "
        f"Định dạng ngày: {date_mode}"
    )

    st.dataframe(
        raw.head(30),
        use_container_width=True,
        hide_index=True,
    )


with st.expander("2. Tiền xử lý"):

    st.dataframe(
        pd.DataFrame(
            list(stats.items()),
            columns=[
                "Công đoạn",
                "Số dòng",
            ],
        ),
        hide_index=True,
    )

    st.dataframe(
        clean.head(30),
        use_container_width=True,
        hide_index=True,
    )


with st.expander("3. RFM"):

    st.write(
        "**R (Recency):** Số ngày từ lần mua gần nhất. "
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
        "Biến đổi log1p rồi dùng StandardScaler "
        "trước khi chạy K-Means."
    )

    st.dataframe(
        pd.DataFrame(
            z[:30],
            columns=[
                "R chuẩn hóa",
                "F chuẩn hóa",
                "M chuẩn hóa",
            ],
        ).round(3),
        hide_index=True,
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

            graph(
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
        profile_view.round(2),
        hide_index=True,
        use_container_width=True,
    )


with st.expander(
    "7. Đánh giá và so sánh"
):

    st.caption(
        "So sánh K-Means và Agglomerative "
        "trên cùng mẫu tối đa 1.000 khách hàng."
    )

    if st.button(
        "Chạy so sánh"
    ):

        _, scores = diagnostics(
            z,
            k,
        )

        st.dataframe(
            scores.round(4),
            hide_index=True,
            use_container_width=True,
        )


# =====================================================
# 24. CUSTOMER EVOLUTION CÓ VÍ DỤ
# =====================================================

with st.expander(
    "8. Customer Evolution – "
    "Ví dụ và hành trình khách hàng"
):

    st.markdown(
        "#### Ví dụ minh họa"
    )

    st.caption(
        "Ví dụ giả định, "
        "không phải kết quả thật."
    )

    example = pd.DataFrame({
        "Tháng": [
            "01/2025",
            "02/2025",
            "03/2025",
        ],

        "Nhóm khách hàng": [
            "Nhóm 2",
            "Nhóm 2",
            "Nhóm 1",
        ],

        "R – số ngày": [
            12,
            6,
            3,
        ],

        "F – hóa đơn tích lũy": [
            1,
            2,
            4,
        ],

        "M – tổng tiền tích lũy": [
            120000,
            350000,
            850000,
        ],
    })

    st.dataframe(
        example,
        hide_index=True,
        use_container_width=True,
    )

    st.write(
        "Tháng 1 → 2 giữ nhóm 2, "
        "tháng 2 → 3 chuyển từ nhóm 2 "
        "sang nhóm 1. Mã nhóm "
        "không có thứ tự tốt/xấu."
    )

    st.markdown(
        "#### Chú thích RFM"
    )

    st.write(
        "**R:** Số ngày từ lần mua gần nhất "
        "đến cuối tháng (thấp là gần hơn). "
        "**F:** Số hóa đơn tích lũy. "
        "**M:** Tổng tiền mua tích lũy "
        "theo đơn vị tiền của file."
    )

    st.markdown(
        "#### Ma trận chuyển nhóm "
        "(dữ liệu thật)"
    )

    st.caption(
        "Hàng: nhóm tháng trước. "
        "Cột: nhóm tháng sau. "
        "Đường chéo: giữ nhóm. "
        "Ô khác: chuyển nhóm."
    )

    mat = matrix.copy()

    mat.index = [
        f"Nhóm {x}"
        for x in mat.index
    ]

    mat.columns = [
        f"Nhóm {x}"
        for x in mat.columns
    ]

    graph(
        px.imshow(
            mat,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Blues",
        ),
        420,
    )

    st.markdown(
        "#### Hành trình khách hàng"
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

    hist_view = (
        history[
            [
                "Tháng",
                "Nhóm khách hàng",
                "Recency",
                "Frequency",
                "Monetary",
            ]
        ]
        .rename(
            columns={
                "Recency": "R – số ngày",
                "Frequency": "F – hóa đơn tích lũy",
                "Monetary": (
                    f"M – tiền mua "
                    f"({currency_choice})"
                ),
            }
        )
    )

    st.dataframe(
        hist_view.round(2),
        hide_index=True,
        use_container_width=True,
    )

    if len(history) > 1:

        fig = px.line(
            history,
            x="Tháng",
            y="Cluster",
            markers=True,
        )

        fig.update_yaxes(
            tickvals=list(range(k)),
            ticktext=[
                f"Nhóm {i}"
                for i in range(k)
            ],
        )

        graph(
            fig,
            310,
        )

    st.caption(
        "Chỉ xét các tháng có giao dịch. "
        "Không dự báo sự thay đổi "
        "trong tương lai."
    )


with st.expander(
    "9. Streamlit và kết quả"
):

    st.write(
        f"Đã phân tích "
        f"{len(rfm):,} khách hàng "
        f"thành {k} nhóm."
    )

    st.write(
        "Dữ liệu → Tiền xử lý → RFM → "
        "Chuẩn hóa → K-Means → Phân tích → "
        "Đánh giá → Customer Evolution → Kết quả."
    )


# =====================================================
# 25. XUẤT FILE VÀ BỘ ĐẾM
# =====================================================

heading(
    "📥 Xuất kết quả phân tích"
)

export_groups = groups.copy()

export_groups["Nhóm khách hàng"] = (
    "Nhóm khách hàng "
    +
    export_groups["Cluster"].astype(str)
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
        "⬇️ Xuất danh sách nhóm khách hàng (CSV)",
        data=export_groups.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="nhom_khach_hang.csv",
        mime="text/csv",
        on_click=mark_export,
    )


with col2:

    st.download_button(
        "⬇️ Xuất Customer Evolution (CSV)",
        data=export_evo.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="customer_evolution.csv",
        mime="text/csv",
        on_click=mark_export,
    )


export_timer()

st.caption(
    "Bộ đếm tính từ lúc bạn bấm xuất CSV. "
    "Streamlit không xác nhận thời điểm "
    "trình duyệt lưu xong file."
)

st.divider()

st.caption(
    "CUSTOMER INTELLIGENCE · "
    "RFM · K-Means · Customer Evolution"
)
