
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


# ============================================================
# 1. CẤU HÌNH VÀ GIAO DIỆN
# ============================================================

st.set_page_config(
    page_title="Customer Intelligence",
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
    max-width: 1350px;
    padding-top: 1.6rem;
    padding-bottom: 4rem;
}

header[data-testid="stHeader"] {
    background: transparent;
}

h1, h2, h3 {
    color: #14243C;
}

.brand {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 15px;
    padding: 17px 22px;
    color: #1D4ED8;
    font-size: 17px;
    font-weight: 850;
    box-shadow: 0 4px 20px #20305008;
}

.hero {
    text-align: center;
    padding: 34px 12px 20px;
}

.eyebrow {
    display: inline-block;
    background: #EAF2FF;
    color: #2563EB;
    padding: 8px 15px;
    border-radius: 25px;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1px;
}

.hero-title {
    font-size: clamp(29px, 4vw, 44px);
    font-weight: 850;
    line-height: 1.22;
    color: #14243C;
    margin: 18px 0 12px;
}

.hero-caption {
    color: #64748B;
    font-size: 14px;
    line-height: 1.75;
}

.upload-label {
    font-size: 20px;
    text-align: center;
    color: #17243B;
    font-weight: 800;
    margin: 15px 0 8px;
}

.upload-hint {
    text-align: center;
    color: #64748B;
    font-size: 13px;
    margin-bottom: 15px;
}

[data-testid="stFileUploader"] {
    background: white;
    border-radius: 16px;
    border: 1px solid #DEE7F2;
    padding: 12px;
    box-shadow: 0 5px 24px #20305008;
}

[data-testid="stFileUploaderDropzone"] {
    background: #F9FBFF;
    border: 2px dashed #B5C8E6;
    border-radius: 11px;
}

.section {
    color: #17233B;
    font-size: 22px;
    font-weight: 800;
    margin: 26px 0 13px;
}

.kpi {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    min-height: 112px;
    padding: 20px;
    margin-bottom: 8px;
    box-shadow: 0 3px 16px #20305007;
}

.kpi-label {
    color: #64748B;
    font-size: 12px;
    font-weight: 650;
}

.kpi-value {
    font-size: clamp(19px, 2.3vw, 27px);
    font-weight: 850;
    color: #2563EB;
    margin-top: 9px;
    overflow-wrap: anywhere;
}

.kpi-note {
    font-size: 11px;
    color: #94A3B8;
    margin-top: 5px;
}

.note {
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    color: #1E40AF;
    border-radius: 14px;
    padding: 17px 20px;
    line-height: 1.7;
}

[data-testid="stExpander"] {
    background: white;
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


def title(text):
    st.markdown(
        f'<div class="section">{text}</div>',
        unsafe_allow_html=True,
    )


def kpi(label, value, note=""):
    st.markdown(
        f'<div class="kpi"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def nfmt(value):
    return f"{value:,.0f}".replace(",", ".")


def chart_theme(fig, height=390):
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#475569"),
        margin=dict(l=12, r=12, t=32, b=20),
        legend_title_text="Nhóm khách hàng",
    )

    fig.update_yaxes(
        gridcolor="#EAF0F7",
        zeroline=False,
    )

    return fig


# ============================================================
# 2. ĐỌC DỮ LIỆU NHIỀU ĐỊNH DẠNG
# ============================================================

def key_name(value):
    text = unicodedata.normalize(
        "NFD",
        str(value).strip().lower(),
    )

    text = "".join(
        c for c in text
        if unicodedata.category(c) != "Mn"
    )

    return re.sub(
        r"[^a-z0-9]",
        "",
        text.replace("đ", "d"),
    )


ALIASES = {
    "InvoiceNo": [
        "invoice",
        "invoice no",
        "invoice id",
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

COLMAP = {
    key_name(alias): canonical
    for canonical, names in ALIASES.items()
    for alias in [canonical, *names]
}


def standardize(df):
    result = df.copy()

    result.columns = [
        COLMAP.get(
            key_name(col),
            str(col).strip(),
        )
        for col in df.columns
    ]

    if result.columns.duplicated().any():
        raise ValueError(
            "Có cột bị trùng sau khi chuẩn hóa tên cột."
        )

    return result


@st.cache_data(show_spinner=False)
def sheet_list(content, filename):
    if filename.lower().endswith(".csv"):
        return ["Dữ liệu CSV"]

    return pd.ExcelFile(
        BytesIO(content)
    ).sheet_names


@st.cache_data(show_spinner=False)
def read_file(content, filename, selected_sheets):
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
                for s in selected_sheets
            ],
            ignore_index=True,
        )

    return standardize(df)


# ============================================================
# 3. TIỀN XỬ LÝ
# ============================================================

@st.cache_data(show_spinner=False)
def prepare_transactions(raw, remove_c, dayfirst):
    required = [
        "InvoiceNo",
        "InvoiceDate",
        "CustomerID",
    ]

    missing = [
        name
        for name in required
        if name not in raw.columns
    ]

    if (
        "TotalAmount" not in raw.columns
        and not {"Quantity", "UnitPrice"}.issubset(
            raw.columns
        )
    ):
        missing.append(
            "TotalAmount hoặc Quantity + UnitPrice"
        )

    if missing:
        raise ValueError(
            "Thiếu cột: " + ", ".join(missing)
        )

    df = raw.copy()
    initial = len(df)

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"],
        dayfirst=dayfirst,
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
        .replace({
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

    missing_count = int(
        df[
            [*required, "TotalAmount"]
        ].isna().any(axis=1).sum()
    )

    df = df.dropna(
        subset=[*required, "TotalAmount"]
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

    invalid_count = int(
        invalid.sum()
    )

    df = df.loc[
        ~invalid
    ].copy()

    cancelled = 0

    if remove_c:
        cancel_flag = (
            df["InvoiceNo"]
            .astype(str)
            .str.upper()
            .str.startswith("C")
        )

        cancelled = int(
            cancel_flag.sum()
        )

        df = df.loc[
            ~cancel_flag
        ].copy()

    duplicate_count = int(
        df.duplicated().sum()
    )

    df = df.drop_duplicates().copy()

    if df.empty:
        raise ValueError(
            "Sau khi làm sạch không còn giao dịch hợp lệ."
        )

    stats = {
        "Ban đầu": initial,
        "Thiếu dữ liệu": missing_count,
        "Giá trị không hợp lệ": invalid_count,
        "Hóa đơn hủy": cancelled,
        "Trùng lặp": duplicate_count,
        "Còn lại": len(df),
    }

    return df, stats


# ============================================================
# 4. RFM, CHUẨN HÓA VÀ K-MEANS
# ============================================================

@st.cache_data(show_spinner=False)
def create_rfm(df):
    cutoff = (
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
        cutoff
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
def train_groups(rfm, k):
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

    if (
        len(rfm) <= k
        or len(np.unique(X, axis=0)) < k
    ):
        raise ValueError(
            "Không đủ khách hàng có RFM khác nhau "
            "cho K này. Hãy giảm K."
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

    return grouped, X, scaler, model


def summarize_groups(grouped):
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


# ============================================================
# 5. CUSTOMER EVOLUTION
# ============================================================

@st.cache_data(show_spinner=False)
def monthly_features(df):
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

    monthly["Tháng"] = (
        monthly["Month"].astype(str)
    )

    return monthly


def predict_evolution(monthly, scaler, model):
    # Không dùng st.cache_data với đối tượng sklearn.

    evo = monthly.copy()

    features = np.log1p(
        evo[
            [
                "Recency",
                "Frequency",
                "Monetary",
            ]
        ].astype(float)
    )

    evo["Cluster"] = model.predict(
        scaler.transform(features)
    )

    evo["Nhóm khách hàng"] = (
        "Nhóm khách hàng "
        +
        evo["Cluster"].astype(str)
    )

    return evo


def get_transitions(evo):
    result = evo.sort_values(
        [
            "CustomerID",
            "Month",
        ]
    ).copy()

    result["PreviousCluster"] = (
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

    if len(pairs):
        matrix = pd.crosstab(
            pairs["PreviousCluster"].astype(int),
            pairs["Cluster"],
        )

    else:
        matrix = pd.DataFrame()

    return matrix, pairs


# ============================================================
# 6. ĐÁNH GIÁ THUẬT TOÁN VÀ ELBOW
# ============================================================

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
            random_state=42,
            n_init=5,
        ).fit(X)

        rows.append({
            "K": k,
            "Inertia": model.inertia_,
        })

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def compare_methods(X, k):
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
        "Agglomerative": AgglomerativeClustering(
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


# ============================================================
# 7. MÀN HÌNH CHỜ TẢI FILE
# ============================================================

st.markdown(
    '<div class="brand">'
    '◈ CUSTOMER INTELLIGENCE'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown("""
<div class="hero">

    <div class="eyebrow">
        DATA ANALYTICS PLATFORM
    </div>

    <div class="hero-title">
        Phân nhóm khách hàng
        <br>
        &amp; Customer Evolution
    </div>

    <div class="hero-caption">
        Phân tích hành vi mua sắm
        bằng RFM và K-Means.
        <br>
        Theo dõi sự thay đổi nhóm khách hàng
        theo thời gian.
    </div>

</div>

<div class="upload-label">
    📂 Tải dữ liệu khách hàng
</div>

<div class="upload-hint">
    Tải Excel hoặc CSV;
    kết quả chỉ xuất hiện sau khi xử lý xong.
</div>
""", unsafe_allow_html=True)


space1, mid, space2 = st.columns(
    [1, 2.2, 1]
)

with mid:
    upload = st.file_uploader(
        "Chọn file Excel hoặc CSV",
        type=[
            "xlsx",
            "xls",
            "csv",
        ],
        label_visibility="collapsed",
    )


# Quan trọng:
# Chưa tải file thì dừng ngay.
# Không tạo Dashboard bên dưới.

if upload is None:
    st.stop()


# ============================================================
# 8. CHỌN DỮ LIỆU
# ============================================================

content = upload.getvalue()

try:
    sheets = sheet_list(
        content,
        upload.name,
    )

except Exception as exc:
    st.error(
        f"Không mở được tệp: {exc}"
    )
    st.stop()


with mid:
    if len(sheets) > 1:
        sheet_choice = st.selectbox(
            "Chọn sheet",
            sheets + ["Tất cả các sheet"],
        )

        chosen_sheets = (
            tuple(sheets)
            if sheet_choice == "Tất cả các sheet"
            else (sheet_choice,)
        )

    else:
        chosen_sheets = tuple(sheets)

    remove_c = st.checkbox(
        "Bỏ hóa đơn bắt đầu bằng C "
        "(nếu C biểu thị hủy)",
        value=False,
    )

    dayfirst = st.checkbox(
        "Ngày ở dạng ngày/tháng/năm "
        "(nếu file dùng định dạng này)",
        value=False,
    )


# ============================================================
# 9. ĐỌC VÀ CHUẨN BỊ DỮ LIỆU
# ============================================================

try:
    with st.spinner(
        "Đang đọc và làm sạch dữ liệu..."
    ):
        raw = read_file(
            content,
            upload.name,
            chosen_sheets,
        )

        cleaned, cleaning_stats = (
            prepare_transactions(
                raw,
                remove_c,
                dayfirst,
            )
        )

        rfm = create_rfm(cleaned)

except Exception as exc:
    st.error(
        f"Lỗi dữ liệu: {exc}"
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


with mid:
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


# ============================================================
# 10. CHẠY PHÂN TÍCH TRƯỚC KHI HIỆN DASHBOARD
# ============================================================

try:
    with st.spinner(
        "Đang phân tích RFM, K-Means "
        "và Customer Evolution. "
        "Vui lòng chờ..."
    ):
        (
            grouped,
            X_scaled,
            scaler,
            model,
        ) = train_groups(rfm, k)

        profiles = summarize_groups(
            grouped
        )

        monthly = monthly_features(
            cleaned
        )

        evolution = predict_evolution(
            monthly,
            scaler,
            model,
        )

        (
            transition_matrix,
            pairs,
        ) = get_transitions(evolution)

except Exception as exc:
    st.error(
        f"Không thể hoàn thành phân tích: {exc}"
    )
    st.stop()


# Chỉ bắt đầu hiển thị Dashboard
# sau khi toàn bộ bước trên thành công.

st.success(
    "✅ Phân tích hoàn tất! "
    "Cuộn xuống để xem toàn bộ kết quả."
)

st.divider()


# ============================================================
# 11. DASHBOARD TỔNG QUAN
# ============================================================

title("📊 Tổng quan kết quả")

c1, c2, c3, c4 = st.columns(4)

with c1:
    kpi(
        "👥 Khách hàng",
        nfmt(len(rfm)),
        "Khách hàng có giao dịch hợp lệ",
    )

with c2:
    kpi(
        "🧾 Hóa đơn",
        nfmt(
            cleaned["InvoiceNo"].nunique()
        ),
        "Mã hóa đơn riêng biệt",
    )

with c3:
    kpi(
        "💰 Giá trị mua",
        nfmt(
            cleaned["TotalAmount"].sum()
        ),
        "Theo đơn vị tiền của dữ liệu",
    )

with c4:
    kpi(
        "🎯 Số nhóm khách hàng",
        str(k),
        "RFM + K-Means",
    )


# ============================================================
# 12. PHÂN BỐ NHÓM KHÁCH HÀNG
# ============================================================

title("👥 Phân bố nhóm khách hàng")

chart_col, pie_col = st.columns(2)

with chart_col:
    bar = px.bar(
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

    bar.update_traces(
        textposition="outside"
    )

    bar.update_layout(
        showlegend=False
    )

    st.plotly_chart(
        chart_theme(bar),
        use_container_width=True,
    )


with pie_col:
    pie = px.pie(
        profiles,
        names="Nhóm khách hàng",
        values="Customers",
        hole=0.54,
        color_discrete_sequence=PALETTE,
    )

    st.plotly_chart(
        chart_theme(pie),
        use_container_width=True,
    )


# ============================================================
# 13. ĐẶC ĐIỂM TỪNG NHÓM KHÁCH HÀNG
# ============================================================

title("📋 Đặc điểm từng nhóm khách hàng")

profiles_view = profiles[
    [
        "Nhóm khách hàng",
        "Customers",
        "Recency",
        "Frequency",
        "Monetary",
    ]
].copy()

profiles_view.columns = [
    "Nhóm khách hàng",
    "Số khách hàng",
    "R trung bình",
    "F trung bình",
    "M trung bình",
]

st.dataframe(
    profiles_view.round(2),
    hide_index=True,
    use_container_width=True,
)


# ============================================================
# 14. PHÁT HIỆN TỪ DỮ LIỆU
# ============================================================

title("💡 Phát hiện từ dữ liệu")

most_customers = profiles.loc[
    profiles["Customers"].idxmax()
]

most_recent = profiles.loc[
    profiles["Recency"].idxmin()
]

highest_spending = profiles.loc[
    profiles["Monetary"].idxmax()
]

a, b, c = st.columns(3)

with a:
    kpi(
        "Nhóm đông khách nhất",
        most_customers["Nhóm khách hàng"],
    )

with b:
    kpi(
        "Nhóm có R trung bình thấp nhất",
        most_recent["Nhóm khách hàng"],
    )

with c:
    kpi(
        "Nhóm có M trung bình cao nhất",
        highest_spending["Nhóm khách hàng"],
    )


# ============================================================
# 15. CUSTOMER EVOLUTION
# ============================================================

title("🔄 Customer Evolution")

st.markdown("""
<div class="note">

    <b>
        Theo dõi sự thay đổi nhóm khách hàng
        theo thời gian
    </b>

    <br>

    Mỗi điểm tháng dựa trên những khách hàng
    có giao dịch trong tháng đó.

    RFM tính tích lũy đến tháng đang xét,
    và các tháng dùng cùng một mô hình K-Means
    để đối chiếu nhóm.

    Biểu đồ không bao gồm khách hàng
    không giao dịch trong tháng.

</div>
""", unsafe_allow_html=True)


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

line = px.line(
    monthly_counts,
    x="Tháng",
    y="Số khách hàng",
    color="Nhóm khách hàng",
    markers=True,
    color_discrete_sequence=PALETTE,
)

st.plotly_chart(
    chart_theme(line, height=445),
    use_container_width=True,
)


if len(pairs):
    moved = int(
        (
            pairs["PreviousCluster"]
            !=
            pairs["Cluster"]
        ).sum()
    )

    t1, t2 = st.columns(2)

    with t1:
        kpi(
            "Lượt chuyển nhóm khách hàng",
            nfmt(moved),
            "Trong các tháng giao dịch liền kề",
        )

    with t2:
        kpi(
            "Lượt được đối chiếu",
            nfmt(len(pairs)),
            "Lượt, không phải số khách hàng riêng biệt",
        )

else:
    st.info(
        "Chưa có đủ giao dịch ở các tháng "
        "liền kề để thống kê chuyển nhóm."
    )


# ============================================================
# 16. CHI TIẾT QUY TRÌNH
# ============================================================

title("🔍 Chi tiết phân tích")

st.caption(
    "Chỉ mở mục cần xem; "
    "phần kỹ thuật không làm rối Dashboard."
)


# ------------------------------------------------------------
# BƯỚC 1 - DỮ LIỆU
# ------------------------------------------------------------

with st.expander("1. Dữ liệu"):
    st.write(
        "**Tên file:**",
        upload.name,
    )

    st.write(
        "**Sheet:**",
        ", ".join(chosen_sheets),
    )

    st.write(
        "**Số dòng ban đầu:**",
        nfmt(len(raw)),
    )

    st.dataframe(
        raw.head(50),
        hide_index=True,
        use_container_width=True,
    )


# ------------------------------------------------------------
# BƯỚC 2 - TIỀN XỬ LÝ
# ------------------------------------------------------------

with st.expander("2. Tiền xử lý"):
    st.dataframe(
        pd.DataFrame(
            cleaning_stats.items(),
            columns=[
                "Tiêu chí",
                "Số dòng",
            ],
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.dataframe(
        cleaned.head(50),
        hide_index=True,
        use_container_width=True,
    )


# ------------------------------------------------------------
# BƯỚC 3 - RFM
# ------------------------------------------------------------

with st.expander("3. RFM"):
    st.markdown(
        """
        **R:** Số ngày từ lần mua gần nhất.

        **F:** Số hóa đơn.

        **M:** Tổng giá trị mua.
        """
    )

    st.dataframe(
        rfm.head(100),
        hide_index=True,
        use_container_width=True,
    )

    st.download_button(
        "⬇️ Tải bảng RFM",
        rfm.to_csv(
            index=False
        ).encode("utf-8-sig"),
        "rfm.csv",
        "text/csv",
    )


# ------------------------------------------------------------
# BƯỚC 4 - CHUẨN HÓA
# ------------------------------------------------------------

with st.expander("4. Chuẩn hóa"):
    st.write(
        "Sử dụng log1p và StandardScaler "
        "trước khi phân nhóm."
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
        hide_index=True,
        use_container_width=True,
    )


# ------------------------------------------------------------
# BƯỚC 5 - K-MEANS
# ------------------------------------------------------------

with st.expander("5. K-Means"):
    st.write(
        f"**Số nhóm khách hàng (K): {k}**"
    )

    if st.button("Tính biểu đồ Elbow"):
        elbow = calculate_elbow(
            X_scaled
        )

        if not elbow.empty:
            elbow_fig = px.line(
                elbow,
                x="K",
                y="Inertia",
                markers=True,
            )

            st.plotly_chart(
                chart_theme(elbow_fig),
                use_container_width=True,
            )

        else:
            st.info(
                "Chưa đủ mẫu khác nhau "
                "để tính Elbow."
            )

    sample = grouped.sample(
        min(1200, len(grouped)),
        random_state=42,
    ).copy()

    sample["Nhóm khách hàng"] = (
        "Nhóm khách hàng "
        +
        sample["Cluster"].astype(str)
    )

    scatter = px.scatter_3d(
        sample,
        x="Recency",
        y="Frequency",
        z="Monetary",
        color="Nhóm khách hàng",
        color_discrete_sequence=PALETTE,
    )

    st.plotly_chart(
        chart_theme(scatter, 480),
        use_container_width=True,
    )


# ------------------------------------------------------------
# BƯỚC 6 - PHÂN TÍCH NHÓM KHÁCH HÀNG
# ------------------------------------------------------------

with st.expander(
    "6. Phân tích nhóm khách hàng"
):
    st.dataframe(
        profiles_view.round(2),
        hide_index=True,
        use_container_width=True,
    )

    output = grouped.copy()

    output["Nhóm khách hàng"] = (
        "Nhóm khách hàng "
        +
        output["Cluster"].astype(str)
    )

    st.download_button(
        "⬇️ Tải bảng phân nhóm",
        output.to_csv(
            index=False
        ).encode("utf-8-sig"),
        "customer_groups.csv",
        "text/csv",
    )


# ------------------------------------------------------------
# BƯỚC 7 - ĐÁNH GIÁ VÀ SO SÁNH
# ------------------------------------------------------------

with st.expander(
    "7. Đánh giá và so sánh thuật toán"
):
    st.write(
        "So sánh K-Means với Agglomerative "
        "bằng Silhouette, Davies-Bouldin "
        "và Calinski-Harabasz."
    )

    if st.button(
        "Chạy so sánh thuật toán"
    ):
        with st.spinner(
            "Đang đánh giá trên cùng một mẫu..."
        ):
            scores = compare_methods(
                X_scaled,
                k,
            )

        if scores.empty:
            st.warning(
                "Mẫu chưa phù hợp để tính "
                "các chỉ số."
            )

        else:
            st.dataframe(
                scores.round(4),
                hide_index=True,
                use_container_width=True,
            )

            st.caption(
                "Dùng tối đa 1.200 khách hàng "
                "để giới hạn bộ nhớ khi so sánh."
            )


# ------------------------------------------------------------
# BƯỚC 8 - CUSTOMER EVOLUTION
# ------------------------------------------------------------

with st.expander(
    "8. Customer Evolution"
):
    st.subheader(
        "Ma trận chuyển nhóm khách hàng"
    )

    if transition_matrix.empty:
        st.info(
            "Chưa có cặp tháng "
            "giao dịch liền kề."
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

        heatmap = px.imshow(
            mat,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Blues",
            labels={
                "x": "Nhóm tháng sau",
                "y": "Nhóm tháng trước",
                "color": "Lượt",
            },
        )

        st.plotly_chart(
            chart_theme(heatmap, 440),
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

    history = evolution.loc[
        evolution["CustomerID"] == customer
    ].sort_values("Month")

    history_view = history[
        [
            "Tháng",
            "Nhóm khách hàng",
            "Recency",
            "Frequency",
            "Monetary",
        ]
    ]

    st.dataframe(
        history_view.round(2),
        hide_index=True,
        use_container_width=True,
    )

    if len(history) > 1:
        path_fig = px.line(
            history,
            x="Tháng",
            y="Cluster",
            markers=True,
        )

        path_fig.update_yaxes(
            tickvals=list(range(k)),
            ticktext=[
                f"Nhóm khách hàng {i}"
                for i in range(k)
            ],
        )

        st.plotly_chart(
            chart_theme(path_fig),
            use_container_width=True,
        )

    st.download_button(
        "⬇️ Tải Customer Evolution",
        evolution[
            [
                "CustomerID",
                "Tháng",
                "Nhóm khách hàng",
                "Recency",
                "Frequency",
                "Monetary",
            ]
        ].to_csv(
            index=False
        ).encode("utf-8-sig"),
        "customer_evolution.csv",
        "text/csv",
    )


# ------------------------------------------------------------
# BƯỚC 9 - STREAMLIT VÀ KẾT QUẢ
# ------------------------------------------------------------

with st.expander(
    "9. Streamlit và kết quả"
):
    st.write(
        "**Khách hàng hợp lệ:**",
        nfmt(len(rfm)),
    )

    st.write(
        "**Số nhóm khách hàng:**",
        k,
    )

    st.write(
        "**Thời gian dữ liệu:**",
        cleaned["InvoiceDate"]
        .min()
        .strftime("%d/%m/%Y"),
        "đến",
        cleaned["InvoiceDate"]
        .max()
        .strftime("%d/%m/%Y"),
    )

    st.write(
        "Quy trình: Dữ liệu → Tiền xử lý → "
        "RFM → Chuẩn hóa → K-Means → "
        "Phân tích khách hàng → "
        "Đánh giá & so sánh → "
        "Customer Evolution → Kết quả."
    )


# ============================================================
# 17. CUỐI TRANG
# ============================================================

st.divider()

st.caption(
    "CUSTOMER INTELLIGENCE • "
    "RFM • K-Means • Customer Evolution • "
    "Phân tích dữ liệu lịch sử"
)
