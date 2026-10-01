
import re
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
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


# ==================================================
# 1. CẤU HÌNH VÀ GIAO DIỆN
# ==================================================

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
    "#EF4444",
    "#64748B",
]

st.markdown("""
<style>
.stApp,
[data-testid="stAppViewContainer"] {
    background: #F6F8FC;
    color: #17233B;
}

.block-container {
    max-width: 1300px;
    padding-top: 1.6rem;
    padding-bottom: 4rem;
}

[data-testid="stHeader"] {
    background: transparent;
}

.brand {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 15px;
    padding: 17px 23px;
    font-size: 16px;
    font-weight: 800;
    color: #1D4ED8;
}

.hero-title {
    font-size: clamp(26px, 4vw, 39px);
    font-weight: 800;
    color: #14243C;
    text-align: center;
    margin: 13px 0 8px;
}

.hero-sub {
    font-size: 14px;
    color: #64748B;
    text-align: center;
    margin-bottom: 22px;
}

.hero-label {
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1px;
    color: #2563EB;
    text-align: center;
    margin-top: 24px;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF;
    border-color: #E4EAF3;
    border-radius: 16px;
}

[data-testid="stFileUploaderDropzone"] {
    background: #F8FAFF;
    border: 2px dashed #BBD0F3;
    border-radius: 12px;
}

.section {
    font-size: 22px;
    font-weight: 800;
    color: #14243C;
    margin: 24px 0 12px;
}

.metric-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 15px;
    padding: 18px;
    min-height: 108px;
    margin-bottom: 8px;
}

.metric-label {
    font-size: 12px;
    color: #64748B;
    font-weight: 600;
}

.metric-value {
    font-size: 25px;
    font-weight: 850;
    color: #2563EB;
    margin-top: 7px;
    overflow-wrap: anywhere;
}

.metric-note {
    font-size: 11px;
    color: #94A3B8;
    margin-top: 5px;
}

[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 11px;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)


def section(text):
    st.markdown(
        f'<div class="section">{text}</div>',
        unsafe_allow_html=True,
    )


def metric(label, value, note=""):
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-note">{note}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def fmt(value):
    return f"{value:,.0f}".replace(",", ".")


def style_fig(fig, height=380):
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#475569"),
        margin=dict(
            l=10,
            r=10,
            t=28,
            b=30,
        ),
        legend_title_text="Nhóm khách hàng",
    )

    fig.update_yaxes(
        gridcolor="#EAF0F7",
        zeroline=False,
    )

    return fig


# ==================================================
# 2. ĐỌC FILE VÀ CHUẨN HÓA TÊN CỘT
# ==================================================

def key(text):
    text = unicodedata.normalize(
        "NFD",
        str(text).strip().lower(),
    ).replace("đ", "d")

    text = "".join(
        c for c in text
        if unicodedata.category(c) != "Mn"
    )

    return re.sub(
        r"[^a-z0-9]",
        "",
        text,
    )


ALIASES = {
    "InvoiceNo": [
        "invoice",
        "invoice no",
        "order id",
        "order no",
        "transaction id",
        "mã hóa đơn",
        "mã đơn hàng",
    ],

    "InvoiceDate": [
        "invoice date",
        "order date",
        "transaction date",
        "purchase date",
        "ngày mua",
        "ngày giao dịch",
    ],

    "CustomerID": [
        "customer id",
        "customer",
        "client id",
        "user id",
        "mã khách hàng",
        "khách hàng",
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

    "TotalAmount": [
        "total amount",
        "amount",
        "sales",
        "revenue",
        "total price",
        "thành tiền",
        "tổng tiền",
        "doanh thu",
    ],
}


COLUMN_MAP = {
    key(alias): canonical
    for canonical, names in ALIASES.items()
    for alias in [canonical, *names]
}


def normalize_columns(df):
    result = df.copy()

    result.columns = [
        COLUMN_MAP.get(
            key(c),
            str(c).strip(),
        )
        for c in result.columns
    ]

    if result.columns.duplicated().any():
        raise ValueError(
            "Có tên cột bị trùng sau chuẩn hóa. "
            "Hãy đổi tên cột trùng trong file."
        )

    return result


@st.cache_data(show_spinner=False)
def get_sheets(content, filename):
    if filename.lower().endswith(".csv"):
        return ["CSV"]

    return pd.ExcelFile(
        BytesIO(content)
    ).sheet_names


@st.cache_data(show_spinner=False)
def load_data(content, filename, selected):
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
        df = pd.concat(
            [
                pd.read_excel(
                    BytesIO(content),
                    sheet_name=s,
                )
                for s in selected
            ],
            ignore_index=True,
        )

    return normalize_columns(df)


# ==================================================
# 3. TIỀN XỬ LÝ
# ==================================================

@st.cache_data(show_spinner=False)
def preprocess(raw, remove_c, dayfirst):

    required = [
        "InvoiceNo",
        "InvoiceDate",
        "CustomerID",
    ]

    missing = [
        c for c in required
        if c not in raw.columns
    ]

    if (
        "TotalAmount" not in raw
        and not {
            "Quantity",
            "UnitPrice",
        }.issubset(raw.columns)
    ):
        missing.append(
            "TotalAmount hoặc Quantity + UnitPrice"
        )

    if missing:
        raise ValueError(
            "Thiếu cột: " + ", ".join(missing)
        )

    df = raw.copy()

    before = len(df)

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

    if "TotalAmount" in df:
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

    missing_rows = int(
        df[
            required + ["TotalAmount"]
        ].isna().any(axis=1).sum()
    )

    df = df.dropna(
        subset=required + ["TotalAmount"]
    ).copy()

    invalid = (
        ~np.isfinite(df["TotalAmount"])
    ) | (
        df["TotalAmount"] <= 0
    )

    if {
        "Quantity",
        "UnitPrice",
    }.issubset(df.columns):

        invalid = invalid | (
            pd.to_numeric(
                df["Quantity"],
                errors="coerce",
            ) <= 0
        )

        invalid = invalid | (
            pd.to_numeric(
                df["UnitPrice"],
                errors="coerce",
            ) <= 0
        )

    invalid_rows = int(invalid.sum())

    df = df.loc[
        ~invalid
    ].copy()

    cancelled_rows = 0

    if remove_c:
        is_cancelled = (
            df["InvoiceNo"]
            .astype(str)
            .str.upper()
            .str.startswith("C")
        )

        cancelled_rows = int(
            is_cancelled.sum()
        )

        df = df.loc[
            ~is_cancelled
        ].copy()

    duplicates = int(
        df.duplicated().sum()
    )

    df = df.drop_duplicates().copy()

    if df.empty:
        raise ValueError(
            "Không còn giao dịch hợp lệ "
            "sau tiền xử lý."
        )

    stats = {
        "Dòng ban đầu": before,
        "Thiếu dữ liệu": missing_rows,
        "Giá trị không hợp lệ": invalid_rows,
        "Hóa đơn hủy": cancelled_rows,
        "Dòng trùng": duplicates,
        "Dòng hợp lệ": len(df),
    }

    return df, stats


# ==================================================
# 4. RFM VÀ K-MEANS
# ==================================================

@st.cache_data(show_spinner=False)
def build_rfm(df):

    reference = (
        df["InvoiceDate"].max().normalize()
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


@st.cache_data(show_spinner=False)
def train(rfm, k):

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

    X = scaler.fit_transform(values)

    if (
        len(X) <= k
        or len(np.unique(X, axis=0)) < k
    ):
        raise ValueError(
            "Không đủ giá trị RFM khác nhau "
            "cho số nhóm đã chọn. Hãy giảm K."
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


def profiles_of(grouped):

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


# ==================================================
# 5. CUSTOMER EVOLUTION
# ==================================================

@st.cache_data(show_spinner=False)
def monthly_rfm(df):

    work = df[
        [
            "CustomerID",
            "InvoiceNo",
            "InvoiceDate",
            "TotalAmount",
        ]
    ].copy()

    work["Month"] = (
        work["InvoiceDate"].dt.to_period("M")
    )

    monthly = (
        work.groupby(
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


def predict_monthly(monthly, scaler, model):

    # Không dùng @st.cache_data ở đây
    # để tránh lỗi hash đối tượng sklearn.

    evolution = monthly.copy()

    values = np.log1p(
        evolution[
            [
                "Recency",
                "Frequency",
                "Monetary",
            ]
        ].astype(float)
    )

    evolution["Cluster"] = model.predict(
        scaler.transform(values)
    )

    evolution["Nhóm khách hàng"] = (
        "Nhóm khách hàng "
        +
        evolution["Cluster"].astype(str)
    )

    return evolution


def transition_report(evolution):

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

    current_num = (
        history["Month"].dt.year * 12
        +
        history["Month"].dt.month
    )

    previous_num = (
        history["PreviousMonth"].dt.year * 12
        +
        history["PreviousMonth"].dt.month
    )

    pairs = history.loc[
        history["PreviousMonth"].notna()
        &
        ((current_num - previous_num) == 1)
    ].copy()

    if len(pairs):
        matrix = pd.crosstab(
            pairs["PreviousCluster"].astype(int),
            pairs["Cluster"],
        )

    else:
        matrix = pd.DataFrame()

    return matrix, pairs


# ==================================================
# 6. ELBOW VÀ ĐÁNH GIÁ
# ==================================================

@st.cache_data(show_spinner=False)
def elbow_scores(X):

    if len(X) > 2500:

        X = X[
            np.random.default_rng(42).choice(
                len(X),
                2500,
                replace=False,
            )
        ]

    unique = len(np.unique(X, axis=0))

    rows = []

    for k in range(
        2,
        min(10, len(X) - 1, unique) + 1,
    ):

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=5,
        ).fit(X)

        rows.append({
            "K": k,
            "Inertia": model.inertia_,
        })

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def compare(X, k):

    if len(X) > 1200:

        X = X[
            np.random.default_rng(42).choice(
                len(X),
                1200,
                replace=False,
            )
        ]

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

        "Agglomerative":
            AgglomerativeClustering(
                n_clusters=k,
            ),
    }

    rows = []

    for name, method in methods.items():

        labels = method.fit_predict(X)

        if not (
            2 <= len(np.unique(labels)) < len(X)
        ):
            continue

        rows.append({
            "Thuật toán": name,

            "Silhouette": silhouette_score(
                X,
                labels,
                sample_size=min(800, len(X)),
                random_state=42,
            ),

            "Davies-Bouldin":
                davies_bouldin_score(
                    X,
                    labels,
                ),

            "Calinski-Harabasz":
                calinski_harabasz_score(
                    X,
                    labels,
                ),
        })

    return pd.DataFrame(rows)


# ==================================================
# 7. ĐẦU TRANG VÀ Ô TẢI FILE
# ==================================================

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
        '<div class="hero-sub">'
        'RFM · K-Means · '
        'Phân tích hành vi khách hàng'
        '</div>',
        unsafe_allow_html=True,
    )

    left, middle, right = st.columns(
        [1, 3, 1]
    )

    with middle:
        upload = st.file_uploader(
            "Tải file Excel hoặc CSV",
            type=[
                "xlsx",
                "xls",
                "csv",
            ],
            label_visibility="collapsed",
        )


# Chưa tải file thì dừng ngay.
# Không hiển thị bất kỳ Dashboard nào.

if upload is None:
    st.stop()


# ==================================================
# 8. LỰA CHỌN DỮ LIỆU
# ==================================================

content = upload.getvalue()

try:
    sheets = get_sheets(
        content,
        upload.name,
    )

except Exception as exc:
    st.error(
        f"Không mở được file: {exc}"
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

        if sheet_choice == "Tất cả các sheet":
            selected_sheets = tuple(sheets)

        else:
            selected_sheets = (
                sheet_choice,
            )

    else:
        selected_sheets = (
            sheets[0],
        )

    remove_c = st.checkbox(
        "Bỏ hóa đơn có mã bắt đầu bằng C "
        "(nếu đó là ký hiệu hủy)",
        value=False,
    )

    dayfirst = st.checkbox(
        "Ngày trong file theo dạng "
        "ngày/tháng/năm",
        value=False,
    )


# ==================================================
# 9. CHUẨN BỊ DỮ LIỆU
# ==================================================

try:

    with st.spinner(
        "Đang đọc và làm sạch dữ liệu..."
    ):

        raw = load_data(
            content,
            upload.name,
            selected_sheets,
        )

        clean, stats = preprocess(
            raw,
            remove_c,
            dayfirst,
        )

        rfm = build_rfm(
            clean
        )

except Exception as exc:

    st.error(
        f"Không thể xử lý dữ liệu: {exc}"
    )

    st.caption(
        "Cần mã khách hàng, mã hóa đơn, "
        "ngày mua và thành tiền "
        "(hoặc số lượng + đơn giá)."
    )

    st.stop()


if len(rfm) < 3:

    st.error(
        "Cần ít nhất 3 khách hàng hợp lệ "
        "để phân nhóm."
    )

    st.stop()


with middle:

    max_k = min(
        8,
        len(rfm) - 1,
    )

    k = st.slider(
        "Số nhóm khách hàng (K)",
        2,
        max_k,
        min(4, max_k),
    )


# ==================================================
# 10. PHÂN TÍCH TRƯỚC KHI HIỆN DASHBOARD
# ==================================================

try:

    with st.spinner(
        "Đang phân tích RFM, K-Means "
        "và Customer Evolution..."
    ):

        (
            grouped,
            X_scaled,
            scaler,
            model,
        ) = train(
            rfm,
            k,
        )

        profiles = profiles_of(
            grouped
        )

        monthly = monthly_rfm(
            clean
        )

        evolution = predict_monthly(
            monthly,
            scaler,
            model,
        )

        matrix, pairs = transition_report(
            evolution
        )

except Exception as exc:

    st.error(
        f"Phân tích chưa hoàn tất: {exc}"
    )

    st.stop()


# Chỉ khi xử lý thành công
# mới chạy các đoạn Dashboard bên dưới.

st.success(
    "Phân tích hoàn tất! "
    "Cuộn xuống để xem Dashboard."
)

st.divider()


# ==================================================
# 11. DASHBOARD TỔNG QUAN
# ==================================================

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
            clean["InvoiceNo"].nunique()
        ),
        "Mã hóa đơn riêng biệt",
    )

with c:
    metric(
        "💰 Giá trị mua",
        fmt(
            clean["TotalAmount"].sum()
        ),
        "Đơn vị tiền theo file",
    )

with d:
    metric(
        "🎯 Nhóm khách hàng",
        str(k),
        "K-Means",
    )


# ==================================================
# 12. BIỂU ĐỒ PHÂN NHÓM
# ==================================================

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
        labels={
            "Customers": "Số khách hàng",
        },
    )

    fig.update_traces(
        textposition="outside"
    )

    fig.update_layout(
        showlegend=False
    )

    st.plotly_chart(
        style_fig(fig),
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
        style_fig(fig),
        use_container_width=True,
    )


# ==================================================
# 13. BẢNG THỐNG KÊ NHÓM
# ==================================================

section(
    "📋 Đặc điểm nhóm khách hàng"
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


# ==================================================
# 14. CUSTOMER EVOLUTION
# ==================================================

section(
    "🔄 Customer Evolution"
)

st.info(
    "Biểu đồ thống kê khách hàng có giao dịch "
    "từng tháng. RFM được tích lũy theo thời gian "
    "và áp dụng một mô hình K-Means chung."
)

monthly_counts = (
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
    monthly_counts,
    x="Tháng",
    y="Số khách hàng",
    color="Nhóm khách hàng",
    markers=True,
    color_discrete_sequence=COLORS,
)

st.plotly_chart(
    style_fig(fig, 420),
    use_container_width=True,
)


if len(pairs):

    changed = int(
        (
            pairs["PreviousCluster"]
            !=
            pairs["Cluster"]
        ).sum()
    )

    u, v = st.columns(2)

    with u:
        metric(
            "Lượt chuyển nhóm",
            fmt(changed),
            "Hai tháng giao dịch liền kề",
        )

    with v:
        metric(
            "Lượt so sánh",
            fmt(len(pairs)),
            "Không phải số khách hàng riêng biệt",
        )


# ==================================================
# 15. CHI TIẾT PHÂN TÍCH
# ==================================================

section(
    "🔎 Chi tiết phân tích"
)

st.caption(
    "Nhấn mở mục bạn muốn xem. "
    "Mọi mục đều được thu gọn mặc định."
)


# ---------- BƯỚC 1 ----------

with st.expander("1. Dữ liệu"):

    st.write(
        "**Tên file:**",
        upload.name,
    )

    st.write(
        "**Sheet:**",
        ", ".join(selected_sheets),
    )

    st.dataframe(
        raw.head(50),
        hide_index=True,
        use_container_width=True,
    )


# ---------- BƯỚC 2 ----------

with st.expander("2. Tiền xử lý"):

    st.dataframe(
        pd.DataFrame(
            stats.items(),
            columns=[
                "Tiêu chí",
                "Số dòng",
            ],
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.dataframe(
        clean.head(50),
        hide_index=True,
        use_container_width=True,
    )


# ---------- BƯỚC 3 ----------

with st.expander("3. RFM"):

    st.write(
        "**Recency:** Số ngày từ lần mua gần nhất. "
        "**Frequency:** Số hóa đơn. "
        "**Monetary:** Tổng giá trị mua."
    )

    st.dataframe(
        rfm.head(100),
        hide_index=True,
        use_container_width=True,
    )

    st.download_button(
        "Tải RFM (CSV)",
        rfm.to_csv(
            index=False
        ).encode("utf-8-sig"),
        "rfm.csv",
        "text/csv",
    )


# ---------- BƯỚC 4 ----------

with st.expander("4. Chuẩn hóa"):

    st.write(
        "Dùng log1p và StandardScaler "
        "để cân bằng thang đo R, F, M."
    )

    standardized = pd.DataFrame(
        X_scaled[:50],
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


# ---------- BƯỚC 5 ----------

with st.expander("5. K-Means"):

    st.write(
        f"**Số nhóm khách hàng:** {k}"
    )

    if st.button(
        "Tính biểu đồ Elbow"
    ):

        elbow = elbow_scores(
            X_scaled
        )

        if len(elbow):

            fig = px.line(
                elbow,
                x="K",
                y="Inertia",
                markers=True,
            )

            st.plotly_chart(
                style_fig(fig),
                use_container_width=True,
            )

        else:
            st.warning(
                "Chưa đủ dữ liệu cho Elbow."
            )

    sample = grouped.sample(
        min(
            1200,
            len(grouped),
        ),
        random_state=42,
    ).copy()

    sample["Nhóm khách hàng"] = (
        "Nhóm khách hàng "
        +
        sample["Cluster"].astype(str)
    )

    fig = px.scatter_3d(
        sample,
        x="Recency",
        y="Frequency",
        z="Monetary",
        color="Nhóm khách hàng",
        color_discrete_sequence=COLORS,
    )

    st.plotly_chart(
        style_fig(fig, 450),
        use_container_width=True,
    )


# ---------- BƯỚC 6 ----------

with st.expander(
    "6. Phân tích nhóm khách hàng"
):

    st.dataframe(
        profile_table.round(2),
        hide_index=True,
        use_container_width=True,
    )

    exported = grouped.copy()

    exported["Nhóm khách hàng"] = (
        "Nhóm khách hàng "
        +
        exported["Cluster"].astype(str)
    )

    st.download_button(
        "Tải kết quả phân nhóm",
        exported.to_csv(
            index=False
        ).encode("utf-8-sig"),
        "customer_groups.csv",
        "text/csv",
    )


# ---------- BƯỚC 7 ----------

with st.expander(
    "7. Đánh giá và so sánh"
):

    st.write(
        "So sánh K-Means và Agglomerative "
        "bằng Silhouette, Davies-Bouldin, "
        "Calinski-Harabasz trên cùng mẫu "
        "tối đa 1.200 khách hàng."
    )

    if st.button(
        "Chạy đánh giá"
    ):

        with st.spinner(
            "Đang đánh giá..."
        ):
            scores = compare(
                X_scaled,
                k,
            )

        if scores.empty:
            st.warning(
                "Mẫu chưa đủ điều kiện đánh giá."
            )

        else:
            st.dataframe(
                scores.round(4),
                hide_index=True,
                use_container_width=True,
            )


# ---------- BƯỚC 8 ----------

with st.expander(
    "8. Customer Evolution"
):

    st.subheader(
        "Ma trận chuyển nhóm khách hàng"
    )

    if matrix.empty:

        st.info(
            "Chưa có giao dịch trong hai tháng "
            "liền nhau để so sánh."
        )

    else:

        mat = matrix.reindex(
            index=range(k),
            columns=range(k),
            fill_value=0,
        )

        mat.index = [
            f"Nhóm khách hàng {i}"
            for i in mat.index
        ]

        mat.columns = [
            f"Nhóm khách hàng {i}"
            for i in mat.columns
        ]

        heatmap = px.imshow(
            mat,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Blues",
        )

        st.plotly_chart(
            style_fig(heatmap, 420),
            use_container_width=True,
        )

    st.subheader(
        "Hành trình khách hàng"
    )

    customer = st.selectbox(
        "Chọn mã khách hàng",
        sorted(
            evolution["CustomerID"]
            .unique()
            .tolist()
        ),
    )

    history = evolution.loc[
        evolution["CustomerID"] == customer
    ].sort_values("Month")

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
            style_fig(fig),
            use_container_width=True,
        )

    cols = [
        "CustomerID",
        "Tháng",
        "Nhóm khách hàng",
        "Recency",
        "Frequency",
        "Monetary",
    ]

    st.download_button(
        "Tải Customer Evolution",
        evolution[
            cols
        ].to_csv(
            index=False
        ).encode("utf-8-sig"),
        "customer_evolution.csv",
        "text/csv",
    )


# ---------- BƯỚC 9 ----------

with st.expander(
    "9. Streamlit và kết quả"
):

    st.write(
        f"Đã phân tích **{fmt(len(rfm))} "
        f"khách hàng**, thành "
        f"**{k} nhóm khách hàng**."
    )

    st.write(
        "Quy trình: Dữ liệu → Tiền xử lý → "
        "RFM → Chuẩn hóa → K-Means → "
        "Phân tích khách hàng → "
        "Đánh giá & so sánh → "
        "Customer Evolution → Kết quả."
    )


# ==================================================
# 16. CUỐI TRANG
# ==================================================

st.divider()

st.caption(
    "CUSTOMER INTELLIGENCE · "
    "RFM · K-Means · Customer Evolution · "
    "Phân tích lịch sử giao dịch"
)
