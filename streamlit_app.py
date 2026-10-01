
import re
import unicodedata
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


# =====================================================
# 1. CẤU HÌNH GIAO DIỆN
# =====================================================

st.set_page_config(
    page_title="Customer Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

COLORS = [
    "#2563EB",
    "#10B981",
    "#8B5CF6",
    "#F59E0B",
    "#EC4899",
    "#06B6D4",
    "#6366F1",
    "#EF4444",
]

st.markdown("""
<style>
.stApp,
[data-testid="stAppViewContainer"] {
    background: #F6F8FC;
    color: #17233B;
}

.block-container {
    max-width: 1320px;
    padding-top: 1.5rem;
    padding-bottom: 4rem;
}

[data-testid="stHeader"] {
    background: transparent;
}

.brand {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    padding: 17px 22px;
    border-radius: 15px;
    font-size: 17px;
    font-weight: 800;
    color: #1D4ED8;
    margin-bottom: 18px;
}

.hero-label {
    text-align: center;
    color: #2563EB;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1px;
    margin-top: 20px;
}

.hero-title {
    text-align: center;
    color: #17243A;
    font-size: clamp(27px, 4vw, 40px);
    font-weight: 850;
    margin: 16px 0 12px;
}

.hero-desc {
    text-align: center;
    color: #64748B;
    font-size: 14px;
    margin-bottom: 24px;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF;
    border-color: #E2E8F0;
    border-radius: 17px;
}

[data-testid="stFileUploaderDropzone"] {
    background: #F8FAFF;
    border: 2px dashed #BBD0F3;
    border-radius: 12px;
}

.section-title {
    font-size: 22px;
    font-weight: 800;
    color: #17243A;
    margin-top: 26px;
    margin-bottom: 15px;
}

.metric-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 15px;
    padding: 20px;
    min-height: 112px;
}

.metric-label {
    color: #64748B;
    font-size: 12px;
    font-weight: 600;
}

.metric-value {
    color: #2563EB;
    font-size: 27px;
    font-weight: 800;
    margin-top: 9px;
    overflow-wrap: anywhere;
}

.metric-note {
    color: #94A3B8;
    font-size: 11px;
    margin-top: 4px;
}

[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    margin-bottom: 9px;
}

.stDownloadButton button {
    background: white;
    border: 1px solid #BFDBFE;
    color: #1D4ED8;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


def section(title):
    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True,
    )


def metric(label, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def fmt(value):
    return f"{value:,.0f}".replace(",", ".")


def style_chart(fig, height=380):
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#475569"),
        margin=dict(l=12, r=12, t=30, b=26),
        legend_title_text="Nhóm khách hàng",
    )

    fig.update_yaxes(
        gridcolor="#EAF0F7",
        zeroline=False,
    )

    return fig


# =====================================================
# 2. NHẬN DIỆN TÊN CỘT
# =====================================================

def normalize_name(value):
    text = unicodedata.normalize(
        "NFD",
        str(value).strip().lower(),
    )

    text = text.replace("đ", "d")

    text = "".join(
        c for c in text
        if unicodedata.category(c) != "Mn"
    )

    return re.sub(
        r"[^a-z0-9]",
        "",
        text,
    )


COLUMN_ALIASES = {
    "CustomerID": [
        "customer id",
        "customer",
        "customer_id",
        "client id",
        "user id",
        "buyer id",
        "mã khách hàng",
        "khách hàng",
    ],

    "InvoiceDate": [
        "invoice date",
        "order date",
        "purchase date",
        "transaction date",
        "date",
        "ngày mua",
        "ngày đặt hàng",
        "ngày hóa đơn",
        "ngày giao dịch",
    ],

    "InvoiceNo": [
        "invoice",
        "invoice no",
        "invoice number",
        "order id",
        "order no",
        "transaction id",
        "bill no",
        "mã hóa đơn",
        "mã đơn hàng",
        "mã giao dịch",
    ],

    "TotalAmount": [
        "total amount",
        "total",
        "total price",
        "amount",
        "sales",
        "revenue",
        "order value",
        "thành tiền",
        "tổng tiền",
        "doanh thu",
    ],

    "Quantity": [
        "quantity",
        "qty",
        "số lượng",
        "sl",
    ],

    "UnitPrice": [
        "unit price",
        "price",
        "đơn giá",
        "giá bán",
    ],

    "Status": [
        "status",
        "order status",
        "transaction status",
        "invoice status",
        "payment status",
        "trạng thái",
        "trạng thái đơn hàng",
        "tình trạng",
    ],
}


ALIAS_LOOKUP = {}

for canonical, aliases in COLUMN_ALIASES.items():
    for alias in [canonical] + aliases:
        ALIAS_LOOKUP[
            normalize_name(alias)
        ] = canonical


def suggest_column(columns, canonical):
    for column in columns:
        if (
            ALIAS_LOOKUP.get(
                normalize_name(column)
            ) == canonical
        ):
            return column

    return None


# =====================================================
# 3. ĐỌC FILE
# =====================================================

@st.cache_data(show_spinner=False)
def get_sheets(content, filename):
    if filename.lower().endswith(".csv"):
        return ["CSV"]

    excel = pd.ExcelFile(
        BytesIO(content)
    )

    return excel.sheet_names


@st.cache_data(show_spinner=False)
def read_data(content, filename, sheets):

    if filename.lower().endswith(".csv"):

        try:
            df = pd.read_csv(
                BytesIO(content),
                sep=None,
                engine="python",
            )

        except UnicodeDecodeError:
            df = pd.read_csv(
                BytesIO(content),
                sep=None,
                engine="python",
                encoding="latin1",
            )

    else:
        frames = []

        for sheet in sheets:
            frames.append(
                pd.read_excel(
                    BytesIO(content),
                    sheet_name=sheet,
                )
            )

        df = pd.concat(
            frames,
            ignore_index=True,
        )

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    if df.columns.duplicated().any():
        raise ValueError(
            "File có tên cột trùng nhau."
        )

    return df


def apply_mapping(df, mapping):
    selected = [
        col for col in mapping.values()
        if col is not None
    ]

    if len(selected) != len(set(selected)):
        raise ValueError(
            "Một cột đang được chọn cho "
            "nhiều thông tin khác nhau."
        )

    rename_map = {
        original: canonical
        for canonical, original in mapping.items()
        if original is not None
    }

    result = df.rename(
        columns=rename_map
    ).copy()

    return result


# =====================================================
# 4. TIỀN XỬ LÝ VÀ HÓA ĐƠN HỦY
# =====================================================

def normalize_status(value):
    return normalize_name(value)


@st.cache_data(show_spinner=False)
def preprocess(
    raw,
    mapping,
    dayfirst,
    cancel_enabled,
    cancel_mode,
    cancel_words,
    each_row_order,
):

    df = apply_mapping(raw, mapping)

    required = [
        "CustomerID",
        "InvoiceDate",
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if (
        "TotalAmount" not in df.columns
        and not {
            "Quantity",
            "UnitPrice",
        }.issubset(df.columns)
    ):
        missing.append(
            "Thành tiền hoặc Số lượng + Đơn giá"
        )

    if (
        "InvoiceNo" not in df.columns
        and not each_row_order
    ):
        missing.append(
            "Mã hóa đơn hoặc tùy chọn "
            "mỗi dòng là một giao dịch"
        )

    if missing:
        raise ValueError(
            "Thiếu thông tin: "
            + ", ".join(missing)
        )

    original_rows = len(df)

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"],
        errors="coerce",
        dayfirst=dayfirst,
    )

    df["CustomerID"] = (
        df["CustomerID"]
        .astype("string")
        .str.strip()
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
    )

    df["CustomerID"] = (
        df["CustomerID"].replace({
            "": pd.NA,
            "nan": pd.NA,
            "None": pd.NA,
            "<NA>": pd.NA,
        })
    )

    if (
        each_row_order
        and "InvoiceNo" not in df.columns
    ):
        df["InvoiceNo"] = (
            pd.Series(
                np.arange(len(df)),
                index=df.index,
            ).astype(str)
        )

    else:
        df["InvoiceNo"] = (
            df["InvoiceNo"]
            .astype("string")
            .str.strip()
            .replace({
                "": pd.NA,
                "nan": pd.NA,
                "None": pd.NA,
            })
        )

    if "TotalAmount" in df.columns:

        df["TotalAmount"] = pd.to_numeric(
            df["TotalAmount"],
            errors="coerce",
        )

    else:
        df["Quantity"] = pd.to_numeric(
            df["Quantity"],
            errors="coerce",
        )

        df["UnitPrice"] = pd.to_numeric(
            df["UnitPrice"],
            errors="coerce",
        )

        df["TotalAmount"] = (
            df["Quantity"] * df["UnitPrice"]
        )

    important = [
        "CustomerID",
        "InvoiceNo",
        "InvoiceDate",
        "TotalAmount",
    ]

    missing_count = int(
        df[important].isna().any(axis=1).sum()
    )

    df = df.dropna(
        subset=important
    ).copy()

    # Xử lý hóa đơn hủy nếu người dùng bật.
    cancelled_count = 0

    if cancel_enabled:

        if cancel_mode == "Theo cột trạng thái":

            if "Status" not in df.columns:
                raise ValueError(
                    "Chưa chọn cột trạng thái."
                )

            cancelled_values = {
                normalize_status(x.strip())
                for x in cancel_words.split(",")
                if x.strip()
            }

            if not cancelled_values:
                raise ValueError(
                    "Cần nhập trạng thái hủy."
                )

            cancelled_flag = (
                df["Status"]
                .fillna("")
                .astype(str)
                .map(normalize_status)
                .isin(cancelled_values)
            )

        elif cancel_mode == "Mã hóa đơn bắt đầu bằng C":

            if (
                each_row_order
                and mapping.get("InvoiceNo") is None
            ):
                raise ValueError(
                    "Không có mã hóa đơn gốc "
                    "để kiểm tra chữ C."
                )

            cancelled_flag = (
                df["InvoiceNo"]
                .astype(str)
                .str.upper()
                .str.startswith("C")
            )

        else:
            raise ValueError(
                "Chưa chọn cách nhận biết "
                "hóa đơn hủy."
            )

        cancelled_count = int(
            cancelled_flag.sum()
        )

        df = df.loc[
            ~cancelled_flag
        ].copy()

    # Loại giá trị tiền không hợp lệ.
    invalid = (
        ~np.isfinite(df["TotalAmount"])
    ) | (
        df["TotalAmount"] <= 0
    )

    if (
        mapping.get("TotalAmount") is None
        and "Quantity" in df.columns
    ):
        invalid |= (
            pd.to_numeric(
                df["Quantity"],
                errors="coerce",
            ) <= 0
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

    df = df.drop_duplicates().copy()

    if df.empty:
        raise ValueError(
            "Không còn giao dịch hợp lệ."
        )

    stats = {
        "Ban đầu": original_rows,
        "Thiếu thông tin": missing_count,
        "Hóa đơn hủy": cancelled_count,
        "Giá trị không hợp lệ": invalid_count,
        "Trùng lặp": duplicates,
        "Còn lại": len(df),
    }

    return df, stats


# =====================================================
# 5. TÍNH RFM
# =====================================================

@st.cache_data(show_spinner=False)
def compute_rfm(df):

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
            LastDate=(
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
        rfm["LastDate"].dt.normalize()
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
# 6. CHUẨN HÓA VÀ K-MEANS
# =====================================================

@st.cache_data(show_spinner=False)
def run_kmeans(rfm, k):

    values = np.log1p(
        rfm[
            [
                "Recency",
                "Frequency",
                "Monetary",
            ]
        ].astype(float)
    )

    scaler = StandardScaler()

    X = scaler.fit_transform(
        values
    )

    if (
        len(rfm) <= k
        or len(np.unique(X, axis=0)) < k
    ):
        raise ValueError(
            "Không đủ khách hàng có RFM "
            "khác nhau. Hãy giảm K."
        )

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
    )

    grouped = rfm.copy()

    grouped["Cluster"] = (
        model.fit_predict(X)
    )

    return (
        grouped,
        X,
        scaler,
        model,
    )


def analyze_groups(grouped):

    profiles = (
        grouped.groupby("Cluster")
        .agg(
            Customers=(
                "CustomerID",
                "count",
            ),
            Recency=(
                "Recency",
                "mean",
            ),
            Frequency=(
                "Frequency",
                "mean",
            ),
            Monetary=(
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

    return profiles


# =====================================================
# 7. CUSTOMER EVOLUTION
# =====================================================

@st.cache_data(show_spinner=False)
def calculate_monthly_rfm(df):

    data = df[
        [
            "CustomerID",
            "InvoiceNo",
            "InvoiceDate",
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
            LastPurchase=(
                "InvoiceDate",
                "max",
            ),
            MonthlyFrequency=(
                "InvoiceNo",
                "nunique",
            ),
            MonthlyMonetary=(
                "TotalAmount",
                "sum",
            ),
        )
        .reset_index()
        .sort_values([
            "CustomerID",
            "Month",
        ])
    )

    monthly["Frequency"] = (
        monthly.groupby("CustomerID")[
            "MonthlyFrequency"
        ].cumsum()
    )

    monthly["Monetary"] = (
        monthly.groupby("CustomerID")[
            "MonthlyMonetary"
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
        monthly["LastPurchase"]
        .dt.normalize()
    ).dt.days.clip(lower=0)

    monthly["Tháng"] = (
        monthly["Month"].astype(str)
    )

    return monthly


def customer_evolution(monthly, scaler, model):
    # Không dùng cache_data cho đối tượng sklearn.

    result = monthly.copy()

    X = np.log1p(
        result[
            [
                "Recency",
                "Frequency",
                "Monetary",
            ]
        ].astype(float)
    )

    X_scaled = scaler.transform(X)

    result["Cluster"] = model.predict(
        X_scaled
    )

    result["Nhóm khách hàng"] = (
        "Nhóm khách hàng "
        +
        result["Cluster"].astype(str)
    )

    return result


def calculate_transitions(evolution):

    history = evolution.sort_values(
        [
            "CustomerID",
            "Month",
        ]
    ).copy()

    history["PreviousCluster"] = (
        history.groupby("CustomerID")[
            "Cluster"
        ].shift()
    )

    history["PreviousMonth"] = (
        history.groupby("CustomerID")[
            "Month"
        ].shift()
    )

    now = (
        history["Month"].dt.year * 12
        +
        history["Month"].dt.month
    )

    before = (
        history["PreviousMonth"].dt.year * 12
        +
        history["PreviousMonth"].dt.month
    )

    valid = (
        history["PreviousMonth"].notna()
        &
        ((now - before) == 1)
    )

    changes = history.loc[
        valid
    ].copy()

    if changes.empty:
        return pd.DataFrame(), changes

    matrix = pd.crosstab(
        changes["PreviousCluster"].astype(int),
        changes["Cluster"],
    )

    return matrix, changes


# =====================================================
# 8. ELBOW
# =====================================================

@st.cache_data(show_spinner=False)
def calculate_elbow(X):

    if len(X) > 2500:
        rng = np.random.default_rng(42)

        index = rng.choice(
            len(X),
            size=2500,
            replace=False,
        )

        X = X[index]

    unique = len(
        np.unique(X, axis=0)
    )

    max_k = min(
        10,
        len(X) - 1,
        unique,
    )

    results = []

    for k in range(2, max_k + 1):

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=5,
        )

        model.fit(X)

        results.append({
            "K": k,
            "Inertia": model.inertia_,
        })

    return pd.DataFrame(results)


# =====================================================
# 9. SO SÁNH THUẬT TOÁN
# =====================================================

@st.cache_data(show_spinner=False)
def compare_models(X, k):

    if len(X) > 1200:
        rng = np.random.default_rng(42)

        index = rng.choice(
            len(X),
            size=1200,
            replace=False,
        )

        X = X[index]

    if (
        len(X) <= k
        or len(np.unique(X, axis=0)) < k
    ):
        return pd.DataFrame()

    methods = {
        "K-Means": KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
        ),
        "Agglomerative": AgglomerativeClustering(
            n_clusters=k,
        ),
    }

    results = []

    for name, algorithm in methods.items():

        labels = algorithm.fit_predict(X)

        if not (
            2 <= len(np.unique(labels)) < len(X)
        ):
            continue

        results.append({
            "Thuật toán": name,

            "Silhouette": silhouette_score(
                X,
                labels,
                sample_size=min(
                    800,
                    len(X),
                ),
                random_state=42,
            ),

            "Davies-Bouldin": davies_bouldin_score(
                X,
                labels,
            ),

            "Calinski-Harabasz": (
                calinski_harabasz_score(
                    X,
                    labels,
                )
            ),
        })

    return pd.DataFrame(results)


# =====================================================
# 10. ĐẦU TRANG VÀ TẢI FILE
# =====================================================

st.markdown(
    '<div class="brand">'
    '◈ CUSTOMER INTELLIGENCE'
    '</div>',
    unsafe_allow_html=True,
)

with st.container(border=True):

    st.markdown(
        '<div class="hero-label">'
        'DATA ANALYTICS PLATFORM'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="hero-title">'
        'Phân nhóm khách hàng '
        '&amp; Customer Evolution'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="hero-desc">'
        'RFM · K-Means · '
        'Phân tích hành vi khách hàng'
        '</div>',
        unsafe_allow_html=True,
    )

    left, middle, right = st.columns(
        [1, 3, 1]
    )

    with middle:
        uploaded = st.file_uploader(
            "Tải file Excel hoặc CSV",
            type=[
                "csv",
                "xlsx",
                "xls",
            ],
            label_visibility="collapsed",
        )


# Chưa tải file thì dừng.
# Dashboard chưa xuất hiện.

if uploaded is None:
    st.stop()


# =====================================================
# 11. CHỌN SHEET
# =====================================================

try:
    sheets = get_sheets(
        uploaded.getvalue(),
        uploaded.name,
    )

except Exception as error:
    st.error(
        f"Không đọc được file: {error}"
    )
    st.stop()


with middle:

    if len(sheets) > 1:

        sheet_choice = st.selectbox(
            "Chọn sheet dữ liệu",
            sheets + [
                "Tất cả các sheet"
            ],
        )

        selected_sheets = (
            tuple(sheets)
            if sheet_choice == "Tất cả các sheet"
            else (sheet_choice,)
        )

    else:
        selected_sheets = tuple(sheets)


# =====================================================
# 12. ĐỌC DỮ LIỆU
# =====================================================

try:

    with st.spinner("Đang đọc dữ liệu..."):

        raw = read_data(
            uploaded.getvalue(),
            uploaded.name,
            selected_sheets,
        )

except Exception as error:

    st.error(
        f"Không mở được dữ liệu: {error}"
    )
    st.stop()


# =====================================================
# 13. CHỌN CỘT DỮ LIỆU
# =====================================================

with middle:

    with st.expander(
        "Thiết lập cột dữ liệu",
        expanded=False,
    ):

        st.caption(
            "Hệ thống tự gợi ý tên cột. "
            "Bạn có thể thay đổi nếu cần."
        )

        labels = {
            "CustomerID": "Mã khách hàng *",
            "InvoiceDate": "Ngày giao dịch *",
            "InvoiceNo": "Mã hóa đơn / đơn hàng",
            "TotalAmount": "Thành tiền",
            "Quantity": "Số lượng",
            "UnitPrice": "Đơn giá",
            "Status": "Trạng thái đơn hàng (nếu có)",
        }

        mapping = {}

        options = [
            "— Không có —"
        ] + list(raw.columns)

        for canonical, label in labels.items():

            suggested = suggest_column(
                raw.columns,
                canonical,
            )

            selected_index = (
                options.index(suggested)
                if suggested in options
                else 0
            )

            selected = st.selectbox(
                label,
                options,
                index=selected_index,
                key="column_" + canonical,
            )

            mapping[canonical] = (
                None
                if selected == "— Không có —"
                else selected
            )


# =====================================================
# 14. CÁC TÙY CHỌN DỮ LIỆU
# =====================================================

with middle:

    if mapping["InvoiceNo"] is None:

        each_row_order = st.checkbox(
            "Không có mã hóa đơn: "
            "coi mỗi dòng là một giao dịch",
            value=False,
        )

        if each_row_order:
            st.caption(
                "Chỉ chọn khi một dòng "
                "thực sự tương ứng một giao dịch."
            )

    else:
        each_row_order = False

    dayfirst = st.checkbox(
        "Ngày trong file theo dạng ngày/tháng/năm",
        value=False,
    )

    cancel_enabled = st.checkbox(
        "Nếu có hóa đơn bị hủy, loại khỏi phân tích",
        value=False,
    )

    cancel_mode = "Không lọc"

    cancel_words = (
        "cancelled, canceled, cancel, "
        "void, voided, hủy, huỷ, đã hủy, đã huỷ"
    )

    if cancel_enabled:

        modes = []

        if mapping["Status"] is not None:
            modes.append(
                "Theo cột trạng thái"
            )

        if mapping["InvoiceNo"] is not None:
            modes.append(
                "Mã hóa đơn bắt đầu bằng C"
            )

        if not modes:

            st.warning(
                "Hãy chọn cột trạng thái "
                "hoặc mã hóa đơn ở phần "
                "'Thiết lập cột dữ liệu'."
            )

        else:

            cancel_mode = st.selectbox(
                "Cách nhận biết hóa đơn bị hủy",
                modes,
            )

            if cancel_mode == "Theo cột trạng thái":

                cancel_words = st.text_input(
                    "Các trạng thái hủy "
                    "(ngăn cách bằng dấu phẩy)",
                    value=cancel_words,
                )

            else:
                st.caption(
                    "Chỉ dùng cách này nếu "
                    "mã bắt đầu bằng C thực sự "
                    "có nghĩa là hóa đơn hủy."
                )


if (
    cancel_enabled
    and cancel_mode == "Không lọc"
):
    st.stop()


# =====================================================
# 15. CHUẨN BỊ DỮ LIỆU
# =====================================================

try:

    with st.spinner(
        "Đang làm sạch dữ liệu và tính RFM..."
    ):

        cleaned, stats = preprocess(
            raw,
            mapping,
            dayfirst,
            cancel_enabled,
            cancel_mode,
            cancel_words,
            each_row_order,
        )

        rfm = compute_rfm(cleaned)

except Exception as error:

    st.error(
        f"Lỗi dữ liệu: {error}"
    )
    st.stop()


if len(rfm) < 3:

    st.error(
        "Cần ít nhất 3 khách hàng "
        "hợp lệ để phân nhóm."
    )
    st.stop()


# =====================================================
# 16. CHỌN SỐ NHÓM KHÁCH HÀNG
# =====================================================

with middle:

    k = st.slider(
        "Số nhóm khách hàng (K)",
        min_value=2,
        max_value=min(
            8,
            len(rfm) - 1,
        ),
        value=min(
            4,
            len(rfm) - 1,
        ),
    )


# =====================================================
# 17. PHÂN TÍCH
# =====================================================

try:

    with st.spinner(
        "Đang chạy RFM, K-Means "
        "và Customer Evolution..."
    ):

        (
            grouped,
            X_scaled,
            scaler,
            model,
        ) = run_kmeans(
            rfm,
            k,
        )

        profiles = analyze_groups(
            grouped
        )

        monthly = calculate_monthly_rfm(
            cleaned
        )

        evolution = customer_evolution(
            monthly,
            scaler,
            model,
        )

        matrix, transitions = (
            calculate_transitions(
                evolution
            )
        )

except Exception as error:

    st.error(
        f"Phân tích chưa hoàn tất: {error}"
    )
    st.stop()


# Dashboard chỉ bắt đầu tại đây.

st.success(
    "✅ Phân tích hoàn tất! "
    "Cuộn xuống để xem kết quả."
)

st.divider()


# =====================================================
# 18. DASHBOARD TỔNG QUAN
# =====================================================

section(
    "📊 Tổng quan kết quả"
)

a, b, c, d = st.columns(4)

with a:
    metric(
        "👥 Khách hàng",
        fmt(len(rfm)),
        "Khách hàng hợp lệ",
    )

with b:
    metric(
        "🧾 Hóa đơn",
        fmt(
            cleaned["InvoiceNo"].nunique()
        ),
        "Giao dịch riêng biệt",
    )

with c:
    metric(
        "💰 Giá trị mua",
        fmt(
            cleaned["TotalAmount"].sum()
        ),
        "Đơn vị tiền theo file",
    )

with d:
    metric(
        "🎯 Nhóm khách hàng",
        str(k),
        "RFM + K-Means",
    )


# =====================================================
# 19. PHÂN BỐ NHÓM KHÁCH HÀNG
# =====================================================

section(
    "👥 Phân bố nhóm khách hàng"
)

col1, col2 = st.columns(2)

with col1:

    fig = px.bar(
        profiles,
        x="Nhóm khách hàng",
        y="Customers",
        color="Nhóm khách hàng",
        text="Customers",
        color_discrete_sequence=COLORS,
    )

    fig.update_layout(
        showlegend=False
    )

    fig.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        style_chart(fig),
        use_container_width=True,
    )


with col2:

    fig = px.pie(
        profiles,
        names="Nhóm khách hàng",
        values="Customers",
        hole=0.55,
        color_discrete_sequence=COLORS,
    )

    st.plotly_chart(
        style_chart(fig),
        use_container_width=True,
    )


# =====================================================
# 20. ĐẶC ĐIỂM NHÓM KHÁCH HÀNG
# =====================================================

section(
    "📋 Đặc điểm từng nhóm khách hàng"
)

profile_table = profiles[
    [
        "Nhóm khách hàng",
        "Customers",
        "Recency",
        "Frequency",
        "Monetary",
    ]
].copy()

profile_table.columns = [
    "Nhóm khách hàng",
    "Số khách hàng",
    "R trung bình",
    "F trung bình",
    "M trung bình",
]

st.dataframe(
    profile_table.round(2),
    hide_index=True,
    use_container_width=True,
)


# =====================================================
# 21. CUSTOMER EVOLUTION
# =====================================================

section(
    "🔄 Customer Evolution"
)

st.info(
    "Biểu đồ thể hiện số khách hàng "
    "có giao dịch trong từng tháng. "
    "RFM được tích lũy và sử dụng "
    "chung một mô hình K-Means."
)

evolution_count = (
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
    evolution_count,
    x="Tháng",
    y="Số khách hàng",
    color="Nhóm khách hàng",
    markers=True,
    color_discrete_sequence=COLORS,
)

st.plotly_chart(
    style_chart(fig, 430),
    use_container_width=True,
)


if not transitions.empty:

    moved = int(
        (
            transitions["PreviousCluster"]
            !=
            transitions["Cluster"]
        ).sum()
    )

    t1, t2 = st.columns(2)

    with t1:
        metric(
            "Lượt chuyển nhóm khách hàng",
            fmt(moved),
        )

    with t2:
        metric(
            "Lượt so sánh tháng liên tiếp",
            fmt(len(transitions)),
        )


# =====================================================
# 22. CHI TIẾT PHÂN TÍCH
# =====================================================

section(
    "🔎 Chi tiết phân tích"
)


# ---------------- BƯỚC 1 ----------------

with st.expander(
    "1. Dữ liệu"
):

    st.write(
        "**Tên file:**",
        uploaded.name,
    )

    st.write(
        "**Sheet:**",
        ", ".join(selected_sheets),
    )

    st.write(
        "**Các cột được sử dụng:**"
    )

    st.json({
        key: value
        for key, value in mapping.items()
        if value is not None
    })

    st.dataframe(
        raw.head(40),
        hide_index=True,
        use_container_width=True,
    )


# ---------------- BƯỚC 2 ----------------

with st.expander(
    "2. Tiền xử lý"
):

    stats_table = pd.DataFrame(
        stats.items(),
        columns=[
            "Tiêu chí",
            "Số dòng",
        ],
    )

    st.dataframe(
        stats_table,
        hide_index=True,
        use_container_width=True,
    )

    st.dataframe(
        cleaned.head(40),
        hide_index=True,
        use_container_width=True,
    )


# ---------------- BƯỚC 3 ----------------

with st.expander(
    "3. RFM"
):

    st.markdown("""
    **Recency:** Số ngày từ lần mua gần nhất.

    **Frequency:** Số giao dịch.

    **Monetary:** Tổng giá trị mua hàng.
    """)

    st.dataframe(
        rfm.head(100),
        hide_index=True,
        use_container_width=True,
    )

    st.download_button(
        "⬇️ Tải dữ liệu RFM",
        rfm.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="rfm.csv",
        mime="text/csv",
    )


# ---------------- BƯỚC 4 ----------------

with st.expander(
    "4. Chuẩn hóa"
):

    st.write(
        "Dữ liệu được biến đổi bằng log1p "
        "và chuẩn hóa với StandardScaler."
    )

    standard_table = pd.DataFrame(
        X_scaled[:40],
        columns=[
            "R chuẩn hóa",
            "F chuẩn hóa",
            "M chuẩn hóa",
        ],
    )

    st.dataframe(
        standard_table.round(3),
        hide_index=True,
        use_container_width=True,
    )


# ---------------- BƯỚC 5 ----------------

with st.expander(
    "5. K-Means"
):

    st.write(
        f"**Số nhóm khách hàng:** {k}"
    )

    if st.button(
        "Tính biểu đồ Elbow"
    ):

        elbow_data = calculate_elbow(
            X_scaled
        )

        if elbow_data.empty:

            st.info(
                "Chưa đủ dữ liệu cho Elbow."
            )

        else:

            fig = px.line(
                elbow_data,
                x="K",
                y="Inertia",
                markers=True,
            )

            st.plotly_chart(
                style_chart(fig),
                use_container_width=True,
            )

    plot_data = grouped.sample(
        min(1200, len(grouped)),
        random_state=42,
    ).copy()

    plot_data["Nhóm khách hàng"] = (
        "Nhóm khách hàng "
        +
        plot_data["Cluster"].astype(str)
    )

    fig_3d = px.scatter_3d(
        plot_data,
        x="Recency",
        y="Frequency",
        z="Monetary",
        color="Nhóm khách hàng",
        color_discrete_sequence=COLORS,
    )

    st.plotly_chart(
        style_chart(fig_3d, 450),
        use_container_width=True,
    )


# ---------------- BƯỚC 6 ----------------

with st.expander(
    "6. Phân tích nhóm khách hàng"
):

    st.dataframe(
        profile_table.round(2),
        hide_index=True,
        use_container_width=True,
    )

    export_groups = grouped.copy()

    export_groups["Nhóm khách hàng"] = (
        "Nhóm khách hàng "
        +
        export_groups["Cluster"].astype(str)
    )

    st.download_button(
        "⬇️ Tải bảng phân nhóm",
        export_groups.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="customer_groups.csv",
        mime="text/csv",
    )


# ---------------- BƯỚC 7 ----------------

with st.expander(
    "7. Đánh giá và so sánh"
):

    st.write(
        "So sánh K-Means và Agglomerative "
        "trên cùng một mẫu dữ liệu."
    )

    if st.button(
        "Chạy so sánh thuật toán"
    ):

        with st.spinner(
            "Đang đánh giá thuật toán..."
        ):

            scores = compare_models(
                X_scaled,
                k,
            )

        if scores.empty:

            st.warning(
                "Chưa đủ dữ liệu để so sánh."
            )

        else:

            st.dataframe(
                scores.round(4),
                hide_index=True,
                use_container_width=True,
            )


# ---------------- BƯỚC 8 ----------------

with st.expander(
    "8. Customer Evolution"
):

    st.subheader(
        "Ma trận chuyển nhóm khách hàng"
    )

    if matrix.empty:

        st.info(
            "Chưa có giao dịch ở "
            "hai tháng liên tiếp."
        )

    else:

        matrix_display = matrix.reindex(
            index=range(k),
            columns=range(k),
            fill_value=0,
        )

        matrix_display.index = [
            f"Nhóm khách hàng {i}"
            for i in matrix_display.index
        ]

        matrix_display.columns = [
            f"Nhóm khách hàng {i}"
            for i in matrix_display.columns
        ]

        heatmap = px.imshow(
            matrix_display,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Blues",
        )

        st.plotly_chart(
            style_chart(heatmap, 430),
            use_container_width=True,
        )

    st.subheader(
        "Hành trình từng khách hàng"
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
        evolution[
            evolution["CustomerID"]
            ==
            customer
        ]
        .sort_values("Month")
    )

    st.dataframe(
        history[
            [
                "Tháng",
                "Nhóm khách hàng",
                "Recency",
                "Frequency",
                "Monetary",
            ]
        ],
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
                f"Nhóm khách hàng {i}"
                for i in range(k)
            ],
        )

        st.plotly_chart(
            style_chart(fig),
            use_container_width=True,
        )

    export_cols = [
        "CustomerID",
        "Tháng",
        "Nhóm khách hàng",
        "Recency",
        "Frequency",
        "Monetary",
    ]

    st.download_button(
        "⬇️ Tải Customer Evolution",
        evolution[
            export_cols
        ].to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="customer_evolution.csv",
        mime="text/csv",
    )


# ---------------- BƯỚC 9 ----------------

with st.expander(
    "9. Streamlit và kết quả"
):

    st.write(
        f"Đã phân tích **{fmt(len(rfm))} "
        f"khách hàng** thành "
        f"**{k} nhóm khách hàng**."
    )

    st.caption(
        "Dữ liệu → Tiền xử lý → RFM → "
        "Chuẩn hóa → K-Means → "
        "Phân tích khách hàng → "
        "Đánh giá & so sánh → "
        "Customer Evolution → Kết quả"
    )


# =====================================================
# 23. CUỐI TRANG
# =====================================================

st.divider()

st.caption(
    "CUSTOMER INTELLIGENCE · "
    "RFM · K-Means · Customer Evolution"
)
