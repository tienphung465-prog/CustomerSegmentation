
import re
import time
import unicodedata
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score


# ==================================================
# 1. CẤU HÌNH
# ==================================================

st.set_page_config(
    page_title="Customer Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

COLORS = [
    "#2563eb", "#10b981", "#8b5cf6", "#f59e0b",
    "#ec4899", "#06b6d4", "#6366f1", "#ef4444",
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


# ==================================================
# 2. GIAO DIỆN
# ==================================================

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

.matrix-scroll {
    overflow-x: auto;
    max-width: 100%;
    border: 1px solid #dce5f1;
    border-radius: 14px;
    background: white;
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
    line-height: 1.2;
    font-weight: 800;
    color: #17243b;
}

.evo-matrix .pct {
    display: block;
    margin-top: 5px;
    font-size: 11px;
    color: #475569;
}

.evo-matrix .sum {
    background: #f1f5f9;
    font-weight: 800;
    color: #334155;
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
    st.plotly_chart(fig, use_container_width=True)


# ==================================================
# 3. MA TRẬN CHUYỂN NHÓM
# ==================================================

def show_transition_matrix(matrix, pairs, k):
    """
    K-Means dùng mã 0..K-1 bên trong.
    Người dùng nhìn thấy nhóm 1..K.
    """

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

    a.metric(
        "Tổng lượt đối chiếu",
        f"{total:,}",
    )

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
        "Xanh lá: giữ nguyên nhóm. "
        "Xanh dương: chuyển nhóm. "
        "Phần trăm trong ô tính theo hàng."
    )

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

            if i == j:
                color = (
                    f"rgba(16,185,129,{alpha:.3f})"
                )
            else:
                color = (
                    f"rgba(37,99,235,{alpha:.3f})"
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
        'Tổng lượt theo nhóm sau'
        '</th>'
    )

    for j in range(k):
        parts.append(
            '<td class="sum">'
            f'{int(counts[j].sum()):,}'
            '</td>'
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
            f"sang **{top['Nhóm tháng sau']}** "
            "theo hướng chuyển được ghi nhận "
            "nhiều nhất."
        )

    else:
        st.success(
            "Không ghi nhận lượt chuyển "
            "sang nhóm khác."
        )

    st.caption(
        "Một khách hàng có thể đóng góp nhiều "
        "lượt qua các cặp tháng khác nhau. "
        "Không tính tháng không có giao dịch."
    )


# ==================================================
# 4. NHẬN DIỆN CỘT
# ==================================================

def normalize(value):
    value = unicodedata.normalize(
        "NFD",
        str(value).strip().lower(),
    ).replace("đ", "d")

    return re.sub(
        r"[^a-z0-9]",
        "",
        "".join(
            c for c in value
            if unicodedata.category(c) != "Mn"
        ),
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

LOOKUP = {
    normalize(alias): name
    for name, aliases in ALIASES.items()
    for alias in [name, *aliases]
}


# ==================================================
# 5. ĐỌC FILE
# ==================================================

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
        str(c).strip()
        for c in frame.columns
    ]

    if frame.columns.duplicated().any():
        raise ValueError(
            "File có tên cột trùng nhau."
        )

    return frame


# ==================================================
# 6. TIỀN TỆ
# ==================================================

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
        if re.search(
            rf"\b{code}\b",
            text,
            re.I,
        )
    }

    if len(found) == 1:
        return next(iter(found))

    return None


def parse_number(series, style):

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

    if style == "1.234,56":

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

    elif style == "1,234.56":

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
                "Dấu phân cách số tiền chưa rõ. "
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

            v = s.at[idx]

            if v.rfind(",") > v.rfind("."):
                s.at[idx] = (
                    v.replace(".", "")
                    .replace(",", ".")
                )
            else:
                s.at[idx] = (
                    v.replace(",", "")
                )

    return pd.to_numeric(
        s,
        errors="coerce",
    )


# ==================================================
# 7. XỬ LÝ NGÀY THÁNG
# ==================================================

def date_format_hint(series):

    if pd.api.types.is_datetime64_any_dtype(
        series
    ):
        return "Ngày Excel đã được định dạng"

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
        return "Không đủ mẫu để nhận diện"

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

    return "Chưa rõ ngày/tháng, hãy chọn thủ công"


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

            local = COUNTRY_FORMAT.get(
                str(country)
            )

            if local is None:
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
                local,
            )

        return output

    return pd.to_datetime(
        series,
        format="mixed",
        dayfirst=(mode == "DD/MM/YYYY"),
        errors="coerce",
    )


# ==================================================
# 8. TIỀN XỬ LÝ
# ==================================================

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
        value
        for value in mapping.values()
        if value
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

    for col in ("CustomerID", "InvoiceDate"):
        if col not in df.columns:
            raise ValueError(
                f"Thiếu thông tin {col}."
            )

    if (
        "InvoiceNo" not in df
        and not one_row
    ):
        raise ValueError(
            "Cần mã hóa đơn hoặc bật tùy chọn "
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

    # Xác định đơn vị tiền tệ.

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
                "Một số dòng không rõ tiền tệ. "
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

    # Định dạng ngày.

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

    # Chuẩn hóa mã khách hàng.

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

    # Thành tiền.

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

    # Loại hóa đơn hủy theo tùy chọn.

    cancelled = 0

    if cancel_mode == "Theo trạng thái":

        if "Status" not in df:
            raise ValueError(
                "Chưa có cột Trạng thái."
            )

        values = {
            normalize(s)
            for s in cancel_words.split(",")
            if s.strip()
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
        df = df.loc[~mask].copy()

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
        df = df.loc[~mask].copy()

    # Loại giá trị không hợp lệ.

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


# ==================================================
# 9. RFM VÀ K-MEANS
# ==================================================

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

    groups = rfm.copy()

    # K-Means vẫn dùng 0..K-1 để xử lý.
    groups["Cluster"] = (
        model.fit_predict(z)
    )

    return groups, z, scaler, model


# ==================================================
# 10. CUSTOMER EVOLUTION
# ==================================================

@st.cache_data(show_spinner=False)
def monthly_rfm(df):

    data = df[
        [
            "CustomerID",
            "InvoiceDate",
            "InvoiceNo",
            "TotalAmount",
        ]
    ].copy()

    data["Month"] = (
        data["InvoiceDate"]
        .dt.to_period("M")
    )

    monthly = (
        data.groupby(
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

    month_end = (
        monthly["Month"]
        .dt.to_timestamp(how="end")
        .dt.normalize()
    )

    monthly["Recency"] = (
        month_end
        -
        monthly["Last"].dt.normalize()
    ).dt.days.clip(lower=0)

    monthly["Tháng"] = (
        monthly["Month"].astype(str)
    )

    return monthly


def evolution_model(monthly, scaler, model):

    result = monthly.copy()

    features = [
        "Recency",
        "Frequency",
        "Monetary",
    ]

    values = np.log1p(
        result[features].astype(float)
    )

    z = scaler.transform(values)

    result["Cluster"] = (
        model.predict(z)
    )

    # HIỂN THỊ BẮT ĐẦU TỪ 1.
    result["Nhóm khách hàng"] = (
        "Nhóm khách hàng "
        +
        (result["Cluster"] + 1).astype(str)
    )

    return result


def transition_matrix(evolution, k):

    history = evolution.sort_values(
        [
            "CustomerID",
            "Month",
        ]
    ).copy()

    history["BeforeGroup"] = (
        history.groupby("CustomerID")[
            "Cluster"
        ].shift()
    )

    history["BeforeMonth"] = (
        history.groupby("CustomerID")[
            "Month"
        ].shift()
    )

    current = (
        history["Month"].dt.year * 12
        +
        history["Month"].dt.month
    )

    previous = (
        history["BeforeMonth"].dt.year * 12
        +
        history["BeforeMonth"].dt.month
    )

    pairs = history.loc[
        history["BeforeMonth"].notna()
        &
        ((current - previous) == 1)
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


# ==================================================
# 11. ĐÁNH GIÁ
# ==================================================

@st.cache_data(show_spinner=False)
def diagnostics(z, k):

    rng = np.random.default_rng(42)

    sample = z[
        rng.choice(
            len(z),
            min(1000, len(z)),
            replace=False,
        )
    ]

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

    if (
        len(sample) > k
        and unique >= k
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

        for name, method in models:

            labels = method.fit_predict(
                sample
            )

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


# ==================================================
# 12. BỘ ĐẾM SAU KHI XUẤT CSV
# ==================================================

def mark_export():
    st.session_state["export_time"] = (
        time.time()
    )


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
            "⏱️ Thời gian từ lúc nhấn xuất CSV: "
            +
            format_elapsed(elapsed)
        )

        if st.button(
            "Dừng / đặt lại bộ đếm",
            key="stop_timer",
        ):

            st.session_state["export_time"] = None

            st.rerun(
                scope="fragment"
            )


# ==================================================
# 13. TRANG CHÍNH
# ==================================================

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
        type=["csv", "xlsx", "xls"],
    )

if uploaded is None:
    st.stop()


# ==================================================
# 14. ĐỌC VÀ THIẾT LẬP DỮ LIỆU
# ==================================================

start_processing = time.perf_counter()

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


FIELDS = {
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

        for canonical, label in FIELDS.items():

            guess = next(
                (
                    col
                    for col in raw.columns
                    if LOOKUP.get(
                        normalize(col)
                    ) == canonical
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
                key="map_" + canonical,
            )

            mapping[canonical] = (
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
    )

    if mapping["InvoiceDate"]:

        st.caption(
            "Gợi ý ngày: "
            +
            date_format_hint(
                raw[mapping["InvoiceDate"]]
            )
        )

    if date_mode == "Theo cột quốc gia":

        st.caption(
            "Chỉ sử dụng nếu ngày thực sự "
            "được ghi theo định dạng của "
            "quốc gia trong từng dòng."
        )

    # Tiền tệ.

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

    known = sorted(
        set(
            detected_values
            .dropna()
            .unique()
        )
    )

    currency_default = (
        known[0]
        if len(known) == 1
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

    if known:
        st.caption(
            "Đã nhận diện: "
            +
            ", ".join(known)
        )

    else:
        st.caption(
            "Không nhận diện được đơn vị tiền "
            "từ nội dung file. Có thể chọn thủ công."
        )

    number_style = st.selectbox(
        "Cách ghi số tiền",
        [
            "Tự động",
            "1,234.56",
            "1.234,56",
        ],
    )

    # Hóa đơn hủy.

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
                "hoặc Mã hóa đơn."
            )

            st.stop()

        cancel_mode = st.selectbox(
            "Cách xác định đơn hủy",
            choices,
        )

        if cancel_mode == "Theo trạng thái":

            cancel_words = st.text_input(
                "Các trạng thái hủy",
                cancel_words,
            )

        else:

            st.caption(
                "Chỉ sử dụng khi chữ C "
                "thật sự có nghĩa là đơn hủy."
            )


# ==================================================
# 15. CHẠY PHÂN TÍCH
# ==================================================

try:

    with st.spinner(
        "Đang tiền xử lý và tính RFM..."
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
        f"Không thể xử lý dữ liệu: {exc}"
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
        min(8, len(rfm) - 1),
        min(4, len(rfm) - 1),
    )


try:

    with st.spinner(
        "Đang chạy K-Means "
        "và Customer Evolution..."
    ):

        groups, z, scaler, model = clustering(
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
                R=("Recency", "mean"),
                F=("Frequency", "mean"),
                M=("Monetary", "mean"),
            )
            .reset_index()
        )

        # NHÓM HIỂN THỊ BẮT ĐẦU TỪ 1.
        profiles["Nhóm khách hàng"] = (
            "Nhóm khách hàng "
            +
            (profiles["Cluster"] + 1).astype(str)
        )

        monthly = monthly_rfm(clean)

        evolution = evolution_model(
            monthly,
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


elapsed = (
    time.perf_counter()
    -
    start_processing
)

st.success(
    "✅ Phân tích hoàn tất! "
    "Cuộn xuống để xem Dashboard."
)

st.caption(
    f"Thời gian xử lý: {elapsed:.2f} giây "
    "(có thể sử dụng bộ nhớ đệm)."
)

st.divider()


# ==================================================
# 16. DASHBOARD
# ==================================================

heading("📊 Tổng quan kết quả")

metrics = [
    ("Khách hàng", len(rfm)),
    (
        "Hóa đơn",
        clean["InvoiceNo"].nunique(),
    ),
    (
        f"Giá trị mua ({currency_choice})",
        clean["TotalAmount"].sum(),
    ),
    ("Số nhóm khách hàng", k),
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


heading("👥 Phân bố nhóm khách hàng")

a, b = st.columns(2)

with a:

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


with b:

    fig = px.pie(
        profiles,
        names="Nhóm khách hàng",
        values="Khach_hang",
        hole=0.55,
        color_discrete_sequence=COLORS,
    )

    graph(fig)


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


# ==================================================
# 17. CUSTOMER EVOLUTION TỔNG QUAN
# ==================================================

heading("🔄 Customer Evolution")

st.info(
    "Biểu đồ theo dõi số khách hàng có "
    "giao dịch mỗi tháng. RFM tích lũy "
    "được phân nhóm bằng một mô hình "
    "K-Means chung."
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
        f"trên {len(pairs):,} lượt đối chiếu "
        "hai tháng giao dịch liên tiếp."
    )


# ==================================================
# 18. CHI TIẾT PHÂN TÍCH
# ==================================================

heading("🔎 Chi tiết phân tích")


with st.expander("1. Dữ liệu"):

    st.caption(
        f"File: {uploaded.name} | "
        f"Sheet: {', '.join(selected)}"
    )

    st.dataframe(
        raw.head(30),
        hide_index=True,
        use_container_width=True,
    )


with st.expander("2. Tiền xử lý"):

    stats_table = pd.DataFrame(
        list(stats.items()),
        columns=["Công đoạn", "Số dòng"],
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
        "**R:** Số ngày từ lần mua gần nhất. "
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
        "Biến đổi log1p và sử dụng "
        "StandardScaler trước K-Means."
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
        f"Đã phân thành "
        f"**{k} nhóm khách hàng** "
        f"(Nhóm 1 đến Nhóm {k})."
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
        "trên cùng mẫu tối đa 1.000 khách hàng."
    )

    if st.button("Chạy so sánh"):

        _, scores = diagnostics(
            z,
            k,
        )

        st.dataframe(
            scores.round(4),
            hide_index=True,
            use_container_width=True,
        )


# ==================================================
# 19. MỤC 8 - CUSTOMER EVOLUTION
# ==================================================

with st.expander(
    "8. Customer Evolution – "
    "Ví dụ và hành trình khách hàng"
):

    st.markdown(
        "#### Ví dụ minh họa"
    )

    st.caption(
        "Đây là số liệu giả định, "
        "không phải kết quả thật."
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

    st.info(
        "**Cách đọc:**\n\n"
        "- Nhóm 1 tháng trước → "
        "Nhóm 1 tháng sau: giữ nguyên.\n"
        "- Nhóm 1 tháng trước → "
        "Nhóm 2 tháng sau: chuyển nhóm.\n"
        "- Nhóm 2 tháng trước → "
        "Nhóm 3 tháng sau: chuyển nhóm."
    )

    st.caption(
        "Các mã nhóm 1, 2, 3... không phải "
        "thứ hạng tốt hoặc xấu."
    )

    # Chú thích RFM.

    st.markdown(
        "#### 📘 Chú thích các chỉ số RFM"
    )

    guide = pd.DataFrame({
        "Chỉ số": [
            "R – Recency",
            "F – Frequency",
            "M – Monetary",
        ],

        "Ý nghĩa": [
            "Số ngày từ lần mua gần nhất",
            "Số hóa đơn tích lũy",
            "Tổng giá trị mua tích lũy",
        ],

        "Cách hiểu": [
            "R thấp: mua gần hơn",
            "F cao: mua nhiều lần hơn",
            "M cao: tổng giá trị mua lớn hơn",
        ],
    })

    st.dataframe(
        guide,
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "Đơn vị M lấy từ dữ liệu gốc. "
        "R trong lịch sử được tính "
        "đến cuối tháng tương ứng."
    )

    # Ma trận chuyển nhóm thật.

    st.markdown(
        "#### 🔄 Ma trận chuyển nhóm "
        "(dữ liệu thật)"
    )

    show_transition_matrix(
        matrix,
        pairs,
        k,
    )

    # Hành trình khách hàng.

    st.markdown(
        "#### 👤 Hành trình từng khách hàng"
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

        # Trục biểu đồ cũng bắt đầu từ Nhóm 1.
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
        "Chỉ theo dõi những tháng có "
        "giao dịch. Đây là phân tích "
        "lịch sử, không phải dự báo."
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


# ==================================================
# 20. XUẤT CSV VÀ BỘ ĐẾM
# ==================================================

heading("📥 Xuất kết quả phân tích")

export_groups = groups.copy()

# Đổi toàn bộ mã nhóm xuất ra bắt đầu từ 1.
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

export_evolution = evolution[
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
        data=export_evolution.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="customer_evolution.csv",
        mime="text/csv",
        on_click=mark_export,
    )


export_timer()

st.caption(
    "Bộ đếm tính từ thời điểm nhấn xuất CSV, "
    "không phải thời điểm trình duyệt "
    "hoàn tất tải xuống."
)

st.divider()

st.caption(
    "CUSTOMER INTELLIGENCE · "
    "RFM · K-Means · Customer Evolution"
)
