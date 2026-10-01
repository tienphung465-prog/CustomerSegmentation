
import re
import unicodedata
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


# =====================================================
# 1. THIẾT LẬP VÀ GIAO DIỆN
# =====================================================

st.set_page_config(
    page_title="Phân nhóm khách hàng",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PALETTE = [
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
    color: #17243A;
}

.block-container {
    max-width: 1390px;
    padding-top: 1.5rem;
    padding-bottom: 4rem;
}

h1, h2, h3 {
    color: #17243A;
}

.header {
    background: white;
    border: 1px solid #E3EAF3;
    border-radius: 15px;
    padding: 17px 23px;
    font-size: 17px;
    font-weight: 800;
    color: #1D4ED8;
    margin-bottom: 18px;
}

.hero {
    text-align: center;
    padding: 22px 12px 25px;
}

.eyebrow {
    display: inline-block;
    border-radius: 24px;
    background: #E9F2FF;
    color: #2563EB;
    padding: 8px 16px;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: .8px;
}

.hero-title {
    font-size: clamp(28px, 4vw, 44px);
    line-height: 1.2;
    color: #14233B;
    font-weight: 850;
    margin-top: 17px;
}

.hero-desc {
    color: #64748B;
    font-size: 14px;
    margin-top: 11px;
    line-height: 1.65;
}

.upload-title {
    text-align: center;
    font-size: 20px;
    color: #17243A;
    font-weight: 800;
    margin: 6px 0 13px;
}

[data-testid="stFileUploader"] {
    background: white;
    border: 1px solid #DDE6F3;
    border-radius: 15px;
    padding: 12px;
}

[data-testid="stFileUploaderDropzone"] {
    background: #F8FAFF;
    border: 2px dashed #B6CBEB;
    border-radius: 11px;
}

.section {
    font-size: 21px;
    font-weight: 800;
    color: #17243A;
    margin: 29px 0 14px;
}

.metric {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 15px;
    padding: 19px;
    min-height: 110px;
    box-shadow: 0 3px 15px #18335A08;
    margin-bottom: 8px;
}

.metric-label {
    font-size: 12px;
    font-weight: 600;
    color: #64748B;
}

.metric-value {
    font-size: 27px;
    font-weight: 800;
    color: #1D4ED8;
    margin-top: 9px;
    overflow-wrap: anywhere;
}

.metric-note {
    font-size: 11px;
    color: #94A3B8;
    margin-top: 4px;
}

.info {
    border: 1px solid #CFE0FD;
    background: #EEF5FF;
    color: #1E40AF;
    padding: 16px 19px;
    border-radius: 12px;
    line-height: 1.7;
}

[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    margin-bottom: 9px;
}

.stDownloadButton button {
    border: 1px solid #BFDBFE;
    color: #1D4ED8;
    background: white;
    border-radius: 9px;
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
        f"""
        <div class="metric">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def number(value):
    return f"{value:,.0f}".replace(",", ".")


def plot_style(fig, height=380):
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=12, t=20, b=20),
        font=dict(color="#475569"),
        legend_title_text="Nhóm khách hàng",
    )

    fig.update_yaxes(
        gridcolor="#EAF0F7",
        zeroline=False,
    )

    return fig


# =====================================================
# 2. ĐỌC VÀ CHUẨN HÓA TÊN CỘT
# =====================================================

def normalized_key(name):
    text = unicodedata.normalize(
        "NFD",
        str(name).strip().lower(),
    )

    text = "".join(
        c for c in text
        if unicodedata.category(c) != "Mn"
    )

    text = text.replace("đ", "d")

    return re.sub(
        r"[^a-z0-9]",
        "",
        text,
    )


ALIASES = {
    "InvoiceNo": [
        "invoice",
        "invoice no",
        "invoice id",
        "invoice number",
        "order id",
        "order no",
        "transaction id",
        "mã hóa đơn",
        "mã đơn hàng",
        "ma giao dich",
    ],
    "InvoiceDate": [
        "invoice date",
        "order date",
        "transaction date",
        "purchase date",
        "date",
        "ngày mua",
        "ngày giao dịch",
        "ngày hóa đơn",
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
        "unitprice",
        "đơn giá",
        "giá bán",
    ],
    "TotalAmount": [
        "total amount",
        "totalamount",
        "amount",
        "sales",
        "revenue",
        "total price",
        "thành tiền",
        "tổng tiền",
        "doanh thu",
    ],
}

MAP = {
    normalized_key(alias): name
    for name, aliases in ALIASES.items()
    for alias in [name] + aliases
}


def standardize(df):
    df = df.copy()

    df.columns = [
        MAP.get(
            normalized_key(c),
            str(c).strip(),
        )
        for c in df.columns
    ]

    if df.columns.duplicated().any():
        raise ValueError(
            "Có cột bị trùng sau khi chuẩn hóa tên; "
            "hãy đổi tên cột trùng trong file."
        )

    return df


@st.cache_data(show_spinner=False)
def sheet_list(content, filename):
    if filename.lower().endswith(".csv"):
        return ["Dữ liệu CSV"]

    return pd.ExcelFile(
        BytesIO(content)
    ).sheet_names


@st.cache_data(show_spinner=False)
def load_data(
    content,
    filename,
    chosen_sheets,
):
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
                for s in chosen_sheets
            ],
            ignore_index=True,
        )

    return standardize(df)


# =====================================================
# 3. TIỀN XỬ LÝ
# =====================================================

@st.cache_data(show_spinner=False)
def clean_data(
    raw,
    cancellation_prefix,
):
    required = [
        "InvoiceNo",
        "InvoiceDate",
        "CustomerID",
    ]

    missing = [
        x for x in required
        if x not in raw.columns
    ]

    if (
        "TotalAmount" not in raw.columns
        and not {
            "Quantity",
            "UnitPrice",
        }.issubset(raw.columns)
    ):
        missing.append(
            "TotalAmount hoặc cặp Quantity + UnitPrice"
        )

    if missing:
        raise ValueError(
            "Thiếu dữ liệu bắt buộc: "
            + ", ".join(missing)
            + "."
        )

    df = raw.copy()
    start = len(df)

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"],
        errors="coerce",
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

    required += ["TotalAmount"]

    missing_count = int(
        df[required]
        .isna()
        .any(axis=1)
        .sum()
    )

    df = df.dropna(
        subset=required
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
        invalid |= (
            pd.to_numeric(
                df["Quantity"],
                errors="coerce",
            ) <= 0
        ) | (
            pd.to_numeric(
                df["UnitPrice"],
                errors="coerce",
            ) <= 0
        )

    invalid_count = int(invalid.sum())

    df = df.loc[
        ~invalid
    ].copy()

    cancelled_count = 0

    if cancellation_prefix:
        cancelled = (
            df["InvoiceNo"]
            .astype(str)
            .str.upper()
            .str.startswith("C")
        )

        cancelled_count = int(
            cancelled.sum()
        )

        df = df.loc[
            ~cancelled
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
        "Dòng ban đầu": start,
        "Thiếu dữ liệu": missing_count,
        "Giá trị không hợp lệ": invalid_count,
        "Hóa đơn hủy": cancelled_count,
        "Trùng lặp": duplicates,
        "Dòng hợp lệ": len(df),
    }

    return df, stats


# =====================================================
# 4. RFM + CHUẨN HÓA + K-MEANS
# =====================================================

@st.cache_data(show_spinner=False)
def build_rfm(df):
    reference = (
        df["InvoiceDate"]
        .max()
        .normalize()
        +
        pd.Timedelta(days=1)
    )

    result = (
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

    result["Recency"] = (
        reference
        -
        result["LastDate"]
        .dt.normalize()
    ).dt.days

    return result[
        [
            "CustomerID",
            "Recency",
            "Frequency",
            "Monetary",
        ]
    ]


@st.cache_data(show_spinner=False)
def run_clustering(rfm, k):
    if len(rfm) <= k:
        raise ValueError(
            "Số khách hàng phải lớn hơn "
            "số nhóm K."
        )

    X_log = np.log1p(
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
        X_log
    )

    if len(np.unique(X, axis=0)) < k:
        raise ValueError(
            "Không đủ mẫu RFM khác nhau "
            "để phân thành K nhóm. "
            "Hãy giảm K."
        )

    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42,
    )

    groups = rfm.copy()

    groups["Cluster"] = (
        model.fit_predict(X)
    )

    return (
        groups,
        X,
        scaler,
        model,
    )


def group_profiles(groups):
    report = (
        groups.groupby("Cluster")
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

    report["Nhóm khách hàng"] = (
        "Nhóm khách hàng "
        +
        report["Cluster"].astype(str)
    )

    return report


# =====================================================
# 5. SO SÁNH THUẬT TOÁN VÀ ELBOW
# =====================================================

@st.cache_data(show_spinner=False)
def calculate_elbow(X):
    if len(X) > 2500:
        X = X[
            np.random.default_rng(42).choice(
                len(X),
                2500,
                replace=False,
            )
        ]

    rows = []

    for k in range(
        2,
        min(10, len(X) - 1) + 1,
    ):
        if len(np.unique(X, axis=0)) < k:
            break

        model = KMeans(
            n_clusters=k,
            n_init=5,
            random_state=42,
        ).fit(X)

        rows.append({
            "K": k,
            "Inertia": model.inertia_,
        })

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def evaluate(X, k):
    if len(X) > 1200:
        X = X[
            np.random.default_rng(42).choice(
                len(X),
                1200,
                replace=False,
            )
        ]

    if len(np.unique(X, axis=0)) < k:
        return pd.DataFrame()

    algorithms = {
        "K-Means": KMeans(
            n_clusters=k,
            n_init=10,
            random_state=42,
        ),
        "Agglomerative": AgglomerativeClustering(
            n_clusters=k
        ),
    }

    rows = []

    for name, algorithm in algorithms.items():
        labels = algorithm.fit_predict(X)

        if not (
            2 <= len(np.unique(labels)) < len(X)
        ):
            continue

        rows.append({
            "Thuật toán": name,
            "Silhouette": silhouette_score(
                X,
                labels,
                sample_size=min(
                    len(X),
                    800,
                ),
                random_state=42,
            ),
            "Davies-Bouldin": (
                davies_bouldin_score(
                    X,
                    labels,
                )
            ),
            "Calinski-Harabasz": (
                calinski_harabasz_score(
                    X,
                    labels,
                )
            ),
        })

    return pd.DataFrame(rows)


# =====================================================
# 6. CUSTOMER EVOLUTION
# =====================================================

# Không dùng cache_data vì hàm nhận scaler, model.

def build_evolution(
    df,
    scaler,
    model,
):
    """
    Theo dõi khách hàng ở các tháng có giao dịch.
    RFM được tích lũy đến tháng đang xét.
    """

    work = df[
        [
            "CustomerID",
            "InvoiceNo",
            "InvoiceDate",
            "TotalAmount",
        ]
    ].copy()

    work["Month"] = (
        work["InvoiceDate"]
        .dt.to_period("M")
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

    X = np.log1p(
        monthly[
            [
                "Recency",
                "Frequency",
                "Monetary",
            ]
        ].astype(float)
    )

    monthly["Cluster"] = (
        model.predict(
            scaler.transform(X)
        )
    )

    monthly["Tháng"] = (
        monthly["Month"].astype(str)
    )

    monthly["Nhóm khách hàng"] = (
        "Nhóm khách hàng "
        +
        monthly["Cluster"].astype(str)
    )

    return monthly


def transitions_table(evo):
    data = evo.sort_values(
        [
            "CustomerID",
            "Month",
        ]
    ).copy()

    data["PreviousCluster"] = (
        data.groupby("CustomerID")[
            "Cluster"
        ].shift()
    )

    data["PreviousMonth"] = (
        data.groupby("CustomerID")[
            "Month"
        ].shift()
    )

    current_num = (
        data["Month"].dt.year * 12
        +
        data["Month"].dt.month
    )

    previous_num = (
        data["PreviousMonth"].dt.year * 12
        +
        data["PreviousMonth"].dt.month
    )

    adjacent = (
        data["PreviousMonth"].notna()
        &
        (
            (current_num - previous_num) == 1
        )
    )

    pairs = data.loc[
        adjacent
    ].copy()

    if len(pairs):
        matrix = pd.crosstab(
            pairs["PreviousCluster"].astype(int),
            pairs["Cluster"],
        )
    else:
        matrix = pd.DataFrame()

    return matrix, pairs


# =====================================================
# 7. TRANG ĐẦU - CHỈ HIỆN VÙNG TẢI FILE
# =====================================================

st.markdown(
    '<div class="header">'
    '◈ CUSTOMER INTELLIGENCE'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">
            DATA ANALYTICS PLATFORM
        </div>

        <div class="hero-title">
            Phân nhóm khách hàng
            <br>
            &amp; Customer Evolution
        </div>

        <div class="hero-desc">
            Phân tích hành vi mua sắm
            bằng RFM và K-Means.
            <br>
            Theo dõi sự thay đổi nhóm
            khách hàng theo thời gian.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="upload-title">'
    '📂 Tải dữ liệu khách hàng'
    '</div>',
    unsafe_allow_html=True,
)

left, center, right = st.columns(
    [1, 2.2, 1]
)

with center:
    uploaded = st.file_uploader(
        "Chọn file Excel hoặc CSV",
        type=[
            "xlsx",
            "xls",
            "csv",
        ],
        label_visibility="collapsed",
    )


# Chưa có file thì dừng tại đây.
# Dashboard hoàn toàn chưa được tạo.

if uploaded is None:
    st.info(
        "Tải dữ liệu ở giữa trang. "
        "Sau khi phân tích xong, "
        "kết quả mới xuất hiện bên dưới."
    )

    st.stop()


# =====================================================
# 8. CHỌN DỮ LIỆU
# =====================================================

content = uploaded.getvalue()

try:
    sheets = sheet_list(
        content,
        uploaded.name,
    )

except Exception as error:
    st.error(
        f"Không đọc được file: {error}"
    )

    st.stop()


with center:
    if len(sheets) > 1:
        choice = st.selectbox(
            "Chọn phạm vi dữ liệu",
            sheets + [
                "Tất cả các sheet"
            ],
        )

        chosen = (
            tuple(sheets)
            if choice == "Tất cả các sheet"
            else (choice,)
        )

    else:
        chosen = (
            sheets[0],
        )

    remove_c = st.checkbox(
        "Bỏ hóa đơn có mã bắt đầu bằng C "
        "(nếu C là ký hiệu hủy)",
        value=False,
    )


# =====================================================
# 9. ĐỌC VÀ CHUẨN BỊ DỮ LIỆU
# =====================================================

try:
    with st.spinner(
        "Đang đọc và làm sạch dữ liệu..."
    ):
        raw = load_data(
            content,
            uploaded.name,
            chosen,
        )

        clean, stats = clean_data(
            raw,
            remove_c,
        )

        rfm = build_rfm(
            clean
        )

except Exception as error:
    st.error(
        f"Lỗi dữ liệu: {error}"
    )

    st.caption(
        "File cần mã khách hàng, mã giao dịch, "
        "ngày giao dịch và thành tiền "
        "(hoặc số lượng + đơn giá)."
    )

    st.stop()


if len(rfm) < 3:
    st.error(
        "Cần tối thiểu 3 khách hàng "
        "có giao dịch hợp lệ để phân nhóm."
    )

    st.stop()


with center:
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


# =====================================================
# 10. CHẠY PHÂN TÍCH TRƯỚC KHI HIỆN DASHBOARD
# =====================================================

try:
    with st.spinner(
        "Đang phân tích RFM, K-Means "
        "và Customer Evolution. "
        "Vui lòng chờ..."
    ):
        (
            groups,
            X_scaled,
            scaler,
            model,
        ) = run_clustering(
            rfm,
            k,
        )

        profiles = group_profiles(
            groups
        )

        evolution = build_evolution(
            clean,
            scaler,
            model,
        )

        (
            transition_matrix,
            transitions,
        ) = transitions_table(
            evolution
        )

except Exception as error:
    st.error(
        f"Lỗi phân tích: {error}"
    )

    st.stop()


# Chỉ đến đoạn này nếu phân tích đã thành công.

st.success(
    "✅ Phân tích hoàn tất! "
    "Cuộn xuống để xem Dashboard."
)

st.divider()


# =====================================================
# 11. DASHBOARD TỔNG QUAN
# =====================================================

section(
    "📊 Tổng quan kết quả"
)

cols = st.columns(4)

with cols[0]:
    metric(
        "👥 Tổng khách hàng",
        number(len(rfm)),
        "Khách hàng hợp lệ",
    )

with cols[1]:
    metric(
        "🧾 Tổng hóa đơn",
        number(
            clean["InvoiceNo"].nunique()
        ),
        "Hóa đơn hợp lệ",
    )

with cols[2]:
    metric(
        "💰 Tổng giá trị mua",
        number(
            clean["TotalAmount"].sum()
        ),
        "Đơn vị tiền theo file",
    )

with cols[3]:
    metric(
        "🎯 Số nhóm khách hàng",
        str(k),
        "Mô hình K-Means",
    )


# =====================================================
# 12. PHÂN BỐ NHÓM KHÁCH HÀNG
# =====================================================

section(
    "👥 Phân bố nhóm khách hàng"
)

chart_col, pie_col = st.columns(2)

with chart_col:
    fig = px.bar(
        profiles,
        x="Nhóm khách hàng",
        y="Customers",
        color="Nhóm khách hàng",
        text="Customers",
        color_discrete_sequence=PALETTE,
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
        plot_style(fig),
        use_container_width=True,
    )


with pie_col:
    fig = px.pie(
        profiles,
        names="Nhóm khách hàng",
        values="Customers",
        hole=0.55,
        color_discrete_sequence=PALETTE,
    )

    st.plotly_chart(
        plot_style(fig),
        use_container_width=True,
    )


# =====================================================
# 13. ĐẶC ĐIỂM TỪNG NHÓM
# =====================================================

section(
    "📋 Đặc điểm từng nhóm khách hàng"
)

display = profiles[
    [
        "Nhóm khách hàng",
        "Customers",
        "Recency",
        "Frequency",
        "Monetary",
    ]
].copy()

display.columns = [
    "Nhóm khách hàng",
    "Số khách hàng",
    "R trung bình",
    "F trung bình",
    "M trung bình",
]

st.dataframe(
    display.round(2),
    use_container_width=True,
    hide_index=True,
)


# =====================================================
# 14. MỘT SỐ PHÁT HIỆN
# =====================================================

section(
    "💡 Một số phát hiện"
)

largest = profiles.loc[
    profiles["Customers"].idxmax()
]

recent = profiles.loc[
    profiles["Recency"].idxmin()
]

spender = profiles.loc[
    profiles["Monetary"].idxmax()
]

a, b, c = st.columns(3)

with a:
    metric(
        "Nhóm có nhiều khách hàng",
        largest["Nhóm khách hàng"],
    )

with b:
    metric(
        "Nhóm có R trung bình thấp nhất",
        recent["Nhóm khách hàng"],
    )

with c:
    metric(
        "Nhóm có M trung bình cao nhất",
        spender["Nhóm khách hàng"],
    )


# =====================================================
# 15. CUSTOMER EVOLUTION
# =====================================================

section(
    "🔄 Customer Evolution"
)

st.markdown(
    """
    <div class="info">
        <b>
        Theo dõi sự thay đổi nhóm khách hàng
        theo thời gian.
        </b>
        <br>
        Áp dụng một mô hình K-Means cố định
        lên RFM tích lũy của những khách hàng
        có giao dịch theo tháng.
        Biểu đồ đếm khách hàng phát sinh
        giao dịch từng tháng, không đại diện
        cho toàn bộ khách hàng ở từng thời điểm.
    </div>
    """,
    unsafe_allow_html=True,
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

fig = px.line(
    counts,
    x="Tháng",
    y="Số khách hàng",
    color="Nhóm khách hàng",
    markers=True,
    color_discrete_sequence=PALETTE,
)

st.plotly_chart(
    plot_style(fig, 440),
    use_container_width=True,
)


if len(transitions):
    changed = int(
        (
            transitions["PreviousCluster"]
            !=
            transitions["Cluster"]
        ).sum()
    )

    a, b = st.columns(2)

    with a:
        metric(
            "Lượt chuyển nhóm khách hàng",
            number(changed),
            "Giữa hai tháng giao dịch liền nhau",
        )

    with b:
        metric(
            "Tổng lượt được so sánh",
            number(len(transitions)),
            "Lượt, không phải khách hàng riêng biệt",
        )

else:
    st.info(
        "Chưa có cặp tháng giao dịch liền nhau "
        "để tính số lượt chuyển nhóm."
    )


# =====================================================
# 16. CHI TIẾT PHÂN TÍCH
# =====================================================

section(
    "🔎 Chi tiết phân tích"
)

st.caption(
    "Nhấn vào mỗi mục bên dưới để xem "
    "quy trình và dữ liệu kỹ thuật."
)


# -----------------------------------------------------
# BƯỚC 1 - DỮ LIỆU ĐẦU VÀO
# -----------------------------------------------------

with st.expander(
    "1. Dữ liệu đầu vào"
):
    st.write(
        "**Tệp:**",
        uploaded.name,
    )

    st.write(
        "**Sheet:**",
        ", ".join(chosen),
    )

    st.write(
        f"**Số dòng đầu vào:** {number(len(raw))}"
    )

    st.dataframe(
        raw.head(40),
        use_container_width=True,
        hide_index=True,
    )


# -----------------------------------------------------
# BƯỚC 2 - TIỀN XỬ LÝ
# -----------------------------------------------------

with st.expander(
    "2. Tiền xử lý"
):
    st.dataframe(
        pd.DataFrame(
            stats.items(),
            columns=[
                "Công đoạn",
                "Số dòng",
            ],
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.dataframe(
        clean.head(40),
        use_container_width=True,
        hide_index=True,
    )


# -----------------------------------------------------
# BƯỚC 3 - RFM
# -----------------------------------------------------

with st.expander(
    "3. RFM"
):
    st.markdown(
        """
        **Recency:** Số ngày từ lần mua gần nhất
        đến ngày sau giao dịch cuối.

        **Frequency:** Số hóa đơn riêng biệt.

        **Monetary:** Tổng giá trị mua.
        """
    )

    st.dataframe(
        rfm.head(100),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "⬇️ Tải bảng RFM",
        rfm.to_csv(
            index=False
        ).encode("utf-8-sig"),
        "rfm.csv",
        "text/csv",
    )


# -----------------------------------------------------
# BƯỚC 4 - CHUẨN HÓA
# -----------------------------------------------------

with st.expander(
    "4. Chuẩn hóa"
):
    st.write(
        "Dùng log1p rồi StandardScaler "
        "để các đặc trưng R, F, M "
        "có thang đo phù hợp."
    )

    st.dataframe(
        pd.DataFrame(
            X_scaled[:40],
            columns=[
                "R chuẩn hóa",
                "F chuẩn hóa",
                "M chuẩn hóa",
            ],
        ).round(3),
        use_container_width=True,
        hide_index=True,
    )


# -----------------------------------------------------
# BƯỚC 5 - K-MEANS
# -----------------------------------------------------

with st.expander(
    "5. K-Means"
):
    st.write(
        f"**Số nhóm khách hàng:** {k}"
    )

    st.caption(
        "Cluster là tên cột kỹ thuật. "
        "Giao diện sử dụng 'Nhóm khách hàng'."
    )

    if st.button(
        "Xem biểu đồ Elbow"
    ):
        elbow = calculate_elbow(
            X_scaled
        )

        if len(elbow):
            fig = px.line(
                elbow,
                x="K",
                y="Inertia",
                markers=True,
                labels={
                    "K": "Số nhóm khách hàng",
                    "Inertia": "Inertia",
                },
            )

            st.plotly_chart(
                plot_style(fig),
                use_container_width=True,
            )

        else:
            st.info(
                "Không đủ mẫu khác biệt "
                "để tính Elbow."
            )

    sample = groups.sample(
        min(
            1300,
            len(groups),
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
        color_discrete_sequence=PALETTE,
    )

    st.plotly_chart(
        plot_style(fig, 440),
        use_container_width=True,
    )


# -----------------------------------------------------
# BƯỚC 6 - PHÂN TÍCH NHÓM KHÁCH HÀNG
# -----------------------------------------------------

with st.expander(
    "6. Phân tích nhóm khách hàng"
):
    st.dataframe(
        display.round(2),
        hide_index=True,
        use_container_width=True,
    )

    exported = groups.copy()

    exported["Nhóm khách hàng"] = (
        "Nhóm khách hàng "
        +
        exported["Cluster"].astype(str)
    )

    st.download_button(
        "⬇️ Tải kết quả phân nhóm",
        exported.to_csv(
            index=False
        ).encode("utf-8-sig"),
        "customer_groups.csv",
        "text/csv",
    )


# -----------------------------------------------------
# BƯỚC 7 - ĐÁNH GIÁ VÀ SO SÁNH
# -----------------------------------------------------

with st.expander(
    "7. Đánh giá và so sánh thuật toán"
):
    st.write(
        "So sánh K-Means với Agglomerative "
        "trên cùng mẫu tối đa 1.200 khách hàng."
    )

    st.caption(
        "Silhouette và Calinski-Harabasz: "
        "cao thường tốt hơn. "
        "Davies-Bouldin: thấp thường tốt hơn."
    )

    if st.button(
        "Chạy so sánh"
    ):
        with st.spinner(
            "Đang tính các chỉ số..."
        ):
            result = evaluate(
                X_scaled,
                k,
            )

        if result.empty:
            st.warning(
                "Chưa đủ dữ liệu phù hợp "
                "để so sánh."
            )

        else:
            st.dataframe(
                result.round(4),
                use_container_width=True,
                hide_index=True,
            )


# -----------------------------------------------------
# BƯỚC 8 - CUSTOMER EVOLUTION
# -----------------------------------------------------

with st.expander(
    "8. Customer Evolution"
):
    st.subheader(
        "Ma trận chuyển nhóm khách hàng"
    )

    if transition_matrix.empty:
        st.info(
            "Chưa có dữ liệu chuyển nhóm "
            "giữa hai tháng liền nhau."
        )

    else:
        mat = transition_matrix.reindex(
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

        fig = px.imshow(
            mat,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Blues",
            labels={
                "x": "Nhóm khách hàng tháng sau",
                "y": "Nhóm khách hàng tháng trước",
                "color": "Lượt",
            },
        )

        st.plotly_chart(
            plot_style(fig, 450),
            use_container_width=True,
        )

    st.subheader(
        "Lịch sử từng khách hàng"
    )

    selected = st.selectbox(
        "Chọn mã khách hàng",
        sorted(
            evolution["CustomerID"]
            .unique()
            .tolist()
        ),
    )

    history = (
        evolution[
            evolution["CustomerID"] == selected
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
        use_container_width=True,
        hide_index=True,
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
            plot_style(fig),
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
        "⬇️ Tải lịch sử chuyển nhóm",
        evolution[
            export_cols
        ].to_csv(
            index=False
        ).encode("utf-8-sig"),
        "customer_evolution.csv",
        "text/csv",
    )


# -----------------------------------------------------
# BƯỚC 9 - KẾT QUẢ
# -----------------------------------------------------

with st.expander(
    "9. Streamlit và kết quả"
):
    st.write(
        f"**Số khách hàng:** {number(len(rfm))}"
    )

    st.write(
        f"**Số nhóm khách hàng:** {k}"
    )

    st.write(
        "**Thời gian dữ liệu:**",
        clean["InvoiceDate"]
        .min()
        .strftime("%d/%m/%Y"),
        "đến",
        clean["InvoiceDate"]
        .max()
        .strftime("%d/%m/%Y"),
    )

    st.write(
        "Quy trình: Dữ liệu → Tiền xử lý → "
        "RFM → Chuẩn hóa → K-Means → "
        "Phân tích khách hàng → "
        "Đánh giá & so sánh → "
        "Customer Evolution → "
        "Streamlit/Kết quả."
    )


# =====================================================
# 17. KẾT LUẬN
# =====================================================

section(
    "✅ Hoàn thành"
)

st.success(
    f"Đã xử lý {number(len(rfm))} khách hàng "
    f"và phân thành {k} nhóm khách hàng. "
    f"Đây là phân tích dữ liệu lịch sử, "
    f"không phải dự báo tương lai."
)

st.caption(
    "CUSTOMER INTELLIGENCE · "
    "RFM · K-Means · Customer Evolution"
)
