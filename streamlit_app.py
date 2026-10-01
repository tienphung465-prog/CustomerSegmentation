
import re
import unicodedata
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
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

st.markdown("""
<style>
.stApp {
    background:#f5f7fb;
    color:#17243b;
}
.block-container {
    max-width:1250px;
    padding-top:1.4rem;
}
[data-testid="stHeader"] {
    background:transparent;
}
.hero {
    background:white;
    border:1px solid #e1e8f2;
    border-radius:20px;
    padding:30px 16px 12px;
    margin:12px 0 22px;
}
.brand {
    font-size:16px;
    color:#1d4ed8;
    font-weight:800;
    letter-spacing:.5px;
}
.title {
    font-size:clamp(26px,4vw,40px);
    font-weight:850;
    color:#17243b;
    text-align:center;
    margin:14px 0 8px;
}
.subtitle {
    color:#64748b;
    text-align:center;
    font-size:14px;
    margin-bottom:22px;
}
.heading {
    font-size:22px;
    font-weight:800;
    color:#17243b;
    margin:27px 0 13px;
}
[data-testid="stFileUploaderDropzone"] {
    background:#f8fbff;
    border:2px dashed #b8cbeb;
    border-radius:12px;
}
[data-testid="stExpander"] {
    background:white;
    border:1px solid #e2e8f0;
    border-radius:12px;
}
.kpi {
    background:white;
    border:1px solid #e2e8f0;
    border-radius:15px;
    padding:19px;
    min-height:100px;
}
.kpi small {
    color:#64748b;
}
.kpi strong {
    color:#2563eb;
    display:block;
    font-size:25px;
    margin-top:8px;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# 1. HÀM HIỂN THỊ
# =====================================================

def heading(text):
    st.markdown(
        f'<div class="heading">{text}</div>',
        unsafe_allow_html=True
    )


def kpi(label, value):
    st.markdown(
        f'<div class="kpi"><small>{label}</small>'
        f'<strong>{value}</strong></div>',
        unsafe_allow_html=True
    )


def nice(fig, h=370):
    fig.update_layout(
        template="plotly_white",
        height=h,
        margin=dict(l=5, r=5, t=24, b=22),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


# =====================================================
# 2. ĐỌC DỮ LIỆU
# =====================================================

def clean_name(s):
    s = unicodedata.normalize(
        "NFD", str(s).lower().strip()
    ).replace("đ", "d")

    return re.sub(
        r"[^a-z0-9]",
        "",
        "".join(
            c for c in s
            if unicodedata.category(c) != "Mn"
        )
    )


ALIASES = {
    "CustomerID": [
        "customer id", "customer", "client id",
        "mã khách hàng", "khách hàng"
    ],
    "InvoiceDate": [
        "invoice date", "order date",
        "transaction date", "purchase date",
        "date", "ngày mua", "ngày giao dịch"
    ],
    "InvoiceNo": [
        "invoice", "invoice no", "order id",
        "transaction id", "mã hóa đơn",
        "mã đơn hàng"
    ],
    "TotalAmount": [
        "total amount", "amount", "sales",
        "revenue", "total price", "thành tiền",
        "tổng tiền", "doanh thu"
    ],
    "Quantity": [
        "qty", "quantity", "số lượng"
    ],
    "UnitPrice": [
        "unit price", "price", "đơn giá", "giá bán"
    ],
    "Status": [
        "status", "order status",
        "invoice status", "trạng thái",
        "trạng thái đơn hàng"
    ],
}

LOOKUP = {
    clean_name(x): name
    for name, names in ALIASES.items()
    for x in [name, *names]
}


@st.cache_data(show_spinner=False)
def sheets_for(contents, filename):
    if filename.lower().endswith(".csv"):
        return ["CSV"]

    return pd.ExcelFile(
        BytesIO(contents)
    ).sheet_names


@st.cache_data(show_spinner=False)
def read_transactions(contents, filename, sheets):
    if filename.lower().endswith(".csv"):
        for encoding in ("utf-8-sig", "cp1258", "latin1"):
            try:
                return pd.read_csv(
                    BytesIO(contents),
                    encoding=encoding,
                    sep=None,
                    engine="python"
                )
            except UnicodeDecodeError:
                continue

        raise ValueError(
            "Không đọc được bảng CSV."
        )

    return pd.concat(
        [
            pd.read_excel(
                BytesIO(contents),
                sheet_name=x
            )
            for x in sheets
        ],
        ignore_index=True
    )


def detect(columns, name):
    return next(
        (
            str(c) for c in columns
            if LOOKUP.get(clean_name(c)) == name
        ),
        None
    )


# =====================================================
# 3. TIỀN XỬ LÝ VÀ HÓA ĐƠN HỦY
# =====================================================

@st.cache_data(show_spinner=False)
def prepare(
    df,
    mapping,
    remove_cancel,
    cancel_mode,
    statuses,
    dayfirst,
    one_row_order
):
    columns = [
        x for x in mapping.values()
        if x is not None
    ]

    if len(columns) != len(set(columns)):
        raise ValueError(
            "Không được chọn một cột "
            "cho nhiều trường khác nhau."
        )

    df = df.rename(
        columns={
            col: target
            for target, col in mapping.items()
            if col
        }
    ).copy()

    for name in ("CustomerID", "InvoiceDate"):
        if name not in df:
            raise ValueError(
                f"Thiếu {name}; hãy kiểm tra "
                "mục 'Thiết lập cột'."
            )

    if (
        "TotalAmount" not in df
        and not {"Quantity", "UnitPrice"}.issubset(
            df.columns
        )
    ):
        raise ValueError(
            "Cần 'Thành tiền' hoặc "
            "cả 'Số lượng' và 'Đơn giá'."
        )

    if "InvoiceNo" not in df and not one_row_order:
        raise ValueError(
            "Cần mã hóa đơn hoặc chọn "
            "'Mỗi dòng là một giao dịch'."
        )

    n0 = len(df)

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"],
        errors="coerce",
        dayfirst=dayfirst
    )

    df["CustomerID"] = (
        df["CustomerID"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    df["CustomerID"] = df["CustomerID"].replace(
        ["", "nan", "None", "<NA>"],
        pd.NA
    )

    if "InvoiceNo" not in df:
        df["InvoiceNo"] = np.arange(
            len(df)
        ).astype(str)
    else:
        df["InvoiceNo"] = (
            df["InvoiceNo"]
            .astype("string")
            .str.strip()
        )

        df["InvoiceNo"] = df["InvoiceNo"].replace(
            ["", "nan", "None", "<NA>"],
            pd.NA
        )

    if "TotalAmount" in df:
        df["TotalAmount"] = pd.to_numeric(
            df["TotalAmount"],
            errors="coerce"
        )
    else:
        df["Quantity"] = pd.to_numeric(
            df["Quantity"],
            errors="coerce"
        )

        df["UnitPrice"] = pd.to_numeric(
            df["UnitPrice"],
            errors="coerce"
        )

        df["TotalAmount"] = (
            df["Quantity"] * df["UnitPrice"]
        )

    needed = [
        "CustomerID", "InvoiceDate",
        "InvoiceNo", "TotalAmount"
    ]

    n_missing = int(
        df[needed].isna().any(axis=1).sum()
    )

    df = df.dropna(
        subset=needed
    ).copy()

    n_cancel = 0

    if remove_cancel:
        if cancel_mode == "Theo cột trạng thái":
            if "Status" not in df:
                raise ValueError(
                    "Cần chọn cột trạng thái "
                    "để lọc hóa đơn hủy."
                )

            words = {
                clean_name(x)
                for x in statuses.split(",")
                if x.strip()
            }

            if not words:
                raise ValueError(
                    "Hãy nhập ít nhất "
                    "một trạng thái hủy."
                )

            bad = (
                df["Status"]
                .fillna("")
                .astype(str)
                .map(clean_name)
                .isin(words)
            )

        else:
            if mapping.get("InvoiceNo") is None:
                raise ValueError(
                    "Không có mã hóa đơn gốc "
                    "để lọc tiền tố C."
                )

            bad = (
                df["InvoiceNo"]
                .astype(str)
                .str.upper()
                .str.startswith("C")
            )

        n_cancel = int(bad.sum())
        df = df.loc[~bad].copy()

    invalid = (
        (~np.isfinite(df["TotalAmount"]))
        |
        (df["TotalAmount"] <= 0)
    )

    if "TotalAmount" not in [
        k for k, v in mapping.items()
        if v is not None
    ]:
        invalid |= (
            (df["Quantity"] <= 0)
            |
            (df["UnitPrice"] <= 0)
        )

    n_invalid = int(invalid.sum())

    df = df.loc[
        ~invalid
    ].copy()

    n_duplicate = int(
        df.duplicated().sum()
    )

    df = df.drop_duplicates().copy()

    if df.empty:
        raise ValueError(
            "Không còn giao dịch hợp lệ "
            "sau khi làm sạch."
        )

    stats = {
        "Ban đầu": n0,
        "Thiếu thông tin": n_missing,
        "Hóa đơn hủy": n_cancel,
        "Giá trị không hợp lệ": n_invalid,
        "Trùng lặp": n_duplicate,
        "Hợp lệ": len(df)
    }

    return df, stats


# =====================================================
# 4. TÍNH RFM
# =====================================================

@st.cache_data(show_spinner=False)
def rfm_table(df):
    reference = (
        df["InvoiceDate"].max().normalize()
        +
        pd.Timedelta(days=1)
    )

    rfm = (
        df.groupby("CustomerID")
        .agg(
            Last=("InvoiceDate", "max"),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalAmount", "sum")
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
            "CustomerID", "Recency",
            "Frequency", "Monetary"
        ]
    ]


# =====================================================
# 5. CHUẨN HÓA VÀ K-MEANS
# =====================================================

@st.cache_data(show_spinner=False)
def make_groups(rfm, k):
    X = np.log1p(
        rfm[
            ["Recency", "Frequency", "Monetary"]
        ].astype(float)
    )

    scaler = StandardScaler()
    z = scaler.fit_transform(X)

    if (
        k >= len(z)
        or len(np.unique(z, axis=0)) < k
    ):
        raise ValueError(
            "Chưa đủ khách hàng có RFM "
            "khác nhau. Hãy giảm K."
        )

    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )

    groups = rfm.copy()

    groups["Cluster"] = (
        model.fit_predict(z)
    )

    return groups, z, scaler, model


# =====================================================
# 6. CUSTOMER EVOLUTION
# =====================================================

@st.cache_data(show_spinner=False)
def month_history(df):
    df = df[
        [
            "CustomerID", "InvoiceNo",
            "InvoiceDate", "TotalAmount"
        ]
    ].copy()

    df["Month"] = (
        df["InvoiceDate"].dt.to_period("M")
    )

    m = (
        df.groupby(["CustomerID", "Month"])
        .agg(
            Last=("InvoiceDate", "max"),
            MonthlyF=("InvoiceNo", "nunique"),
            MonthlyM=("TotalAmount", "sum")
        )
        .reset_index()
        .sort_values(["CustomerID", "Month"])
    )

    m["Frequency"] = (
        m.groupby("CustomerID")["MonthlyF"]
        .cumsum()
    )

    m["Monetary"] = (
        m.groupby("CustomerID")["MonthlyM"]
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


def evolution_result(monthly, scaler, model):
    # Không cache scaler/model để tránh lỗi hash.
    result = monthly.copy()

    features = [
        "Recency", "Frequency", "Monetary"
    ]

    X = np.log1p(
        result[features].astype(float)
    )

    z = scaler.transform(X)

    result["Cluster"] = model.predict(z)

    result["Nhóm khách hàng"] = (
        "Nhóm khách hàng "
        +
        result["Cluster"].astype(str)
    )

    return result


def transition_result(evo):
    x = evo.sort_values(
        ["CustomerID", "Month"]
    ).copy()

    x["BeforeCluster"] = (
        x.groupby("CustomerID")["Cluster"]
        .shift()
    )

    x["BeforeMonth"] = (
        x.groupby("CustomerID")["Month"]
        .shift()
    )

    diff = (
        x["Month"].dt.year * 12
        +
        x["Month"].dt.month
        -
        x["BeforeMonth"].dt.year * 12
        -
        x["BeforeMonth"].dt.month
    )

    pairs = x.loc[
        x["BeforeMonth"].notna()
        &
        (diff == 1)
    ].copy()

    if not pairs.empty:
        matrix = pd.crosstab(
            pairs["BeforeCluster"].astype(int),
            pairs["Cluster"]
        )
    else:
        matrix = pd.DataFrame()

    return matrix, pairs


# =====================================================
# 7. ĐÁNH GIÁ VÀ ELBOW
# =====================================================

@st.cache_data(show_spinner=False)
def model_diagnostics(z, k):
    rng = np.random.default_rng(42)

    indices = rng.choice(
        len(z),
        min(len(z), 1000),
        replace=False
    )

    sample = z[indices]

    unique = len(
        np.unique(sample, axis=0)
    )

    elbow = []

    for n in range(
        2,
        min(8, len(sample) - 1, unique) + 1
    ):
        km = KMeans(
            n_clusters=n,
            n_init=5,
            random_state=42
        ).fit(sample)

        elbow.append({
            "K": n,
            "Inertia": km.inertia_
        })

    scores = []

    if len(sample) > k and unique >= k:
        methods = [
            (
                "K-Means",
                KMeans(
                    n_clusters=k,
                    n_init=10,
                    random_state=42
                )
            ),
            (
                "Agglomerative",
                AgglomerativeClustering(
                    n_clusters=k
                )
            )
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
                        labels
                    ),
                    "Davies-Bouldin":
                        davies_bouldin_score(
                            sample,
                            labels
                        )
                })

    return (
        pd.DataFrame(elbow),
        pd.DataFrame(scores)
    )


# =====================================================
# 8. GIAO DIỆN CHÍNH
# =====================================================

st.markdown(
    '<div class="brand">'
    '◈ CUSTOMER INTELLIGENCE'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero">'
    '<div class="title">'
    'Phân nhóm khách hàng '
    '&amp; Customer Evolution'
    '</div>'
    '<div class="subtitle">'
    'RFM · K-Means · '
    'Phân tích hành vi khách hàng theo thời gian'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)

left, center, right = st.columns(
    [1, 2.5, 1]
)

with center:
    uploaded = st.file_uploader(
        "Tải dữ liệu khách hàng (Excel / CSV)",
        type=["csv", "xlsx", "xls"]
    )

# Chưa tải file: dừng.
if uploaded is None:
    st.stop()


# =====================================================
# 9. CHỌN DỮ LIỆU
# =====================================================

contents = uploaded.getvalue()

try:
    sheets = sheets_for(
        contents,
        uploaded.name
    )

except Exception as exc:
    st.error(
        f"Không mở được file: {exc}"
    )
    st.stop()


with center:
    sheet_options = sheets + (
        ["Tất cả các sheet"]
        if len(sheets) > 1
        else []
    )

    choice = st.selectbox(
        "Sheet dữ liệu",
        sheet_options
    )

selected_sheets = (
    tuple(sheets)
    if choice == "Tất cả các sheet"
    else (choice,)
)


try:
    with st.spinner("Đang đọc dữ liệu..."):
        raw = read_transactions(
            contents,
            uploaded.name,
            selected_sheets
        )

except Exception as exc:
    st.error(
        f"Đọc dữ liệu thất bại: {exc}"
    )
    st.stop()


# =====================================================
# 10. THIẾT LẬP CỘT
# =====================================================

labels = {
    "CustomerID": "Mã khách hàng *",
    "InvoiceDate": "Ngày giao dịch *",
    "InvoiceNo": "Mã hóa đơn",
    "TotalAmount": "Thành tiền",
    "Quantity": "Số lượng",
    "UnitPrice": "Đơn giá",
    "Status": "Trạng thái đơn hàng"
}

with center:
    with st.expander(
        "Thiết lập cột "
        "(mở nếu nhận diện chưa đúng)"
    ):
        mapping = {}

        options = (
            ["— Không có —"]
            +
            [str(c) for c in raw.columns]
        )

        for name, label in labels.items():
            guess = detect(
                raw.columns,
                name
            )

            index = (
                options.index(guess)
                if guess in options
                else 0
            )

            picked = st.selectbox(
                label,
                options,
                index=index,
                key=f"map_{name}"
            )

            mapping[name] = (
                None
                if picked == "— Không có —"
                else picked
            )

    each_row = st.checkbox(
        "Không có mã hóa đơn: "
        "mỗi dòng là một giao dịch",
        value=False,
        disabled=(
            mapping["InvoiceNo"] is not None
        )
    )

    dayfirst = st.checkbox(
        "Ngày ở dạng ngày/tháng/năm",
        value=False
    )

    cancel = st.checkbox(
        "Nếu có hóa đơn bị hủy, "
        "loại khỏi phân tích",
        value=False
    )

    mode = "Không lọc"

    status_words = (
        "cancelled, canceled, void, "
        "hủy, đã hủy"
    )

    if cancel:
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
                "Chọn mã hóa đơn hoặc cột trạng thái "
                "trong Thiết lập cột."
            )
            st.stop()

        mode = st.selectbox(
            "Cách nhận biết hóa đơn hủy",
            modes
        )

        if mode == "Theo cột trạng thái":
            status_words = st.text_input(
                "Giá trị trạng thái hủy "
                "(cách nhau bằng dấu phẩy)",
                status_words
            )
        else:
            st.caption(
                "Chỉ chọn khi mã bắt đầu bằng C "
                "thực sự có nghĩa là hủy."
            )


# =====================================================
# 11. CHUẨN BỊ RFM
# =====================================================

try:
    with st.spinner(
        "Đang làm sạch và tính RFM..."
    ):
        clean, cleaning_stats = prepare(
            raw,
            mapping,
            cancel,
            mode,
            status_words,
            dayfirst,
            each_row
        )

        rfm = rfm_table(clean)

except Exception as exc:
    st.error(
        f"Không thể chuẩn bị dữ liệu: {exc}"
    )
    st.stop()


if len(rfm) < 3:
    st.error(
        "Cần ít nhất 3 khách hàng hợp lệ "
        "để phân nhóm."
    )
    st.stop()


with center:
    max_k = min(
        8,
        len(rfm) - 1
    )

    k = st.slider(
        "Số nhóm khách hàng (K)",
        2,
        max_k,
        min(4, max_k)
    )


# =====================================================
# 12. CHẠY TOÀN BỘ PHÂN TÍCH
# =====================================================

try:
    with st.spinner(
        "Đang phân nhóm và tính "
        "Customer Evolution..."
    ):
        groups, z, scaler, model = (
            make_groups(rfm, k)
        )

        profiles = (
            groups.groupby("Cluster")
            .agg(
                Khach_hang=(
                    "CustomerID", "count"
                ),
                R=("Recency", "mean"),
                F=("Frequency", "mean"),
                M=("Monetary", "mean")
            )
            .reset_index()
        )

        profiles["Nhóm khách hàng"] = (
            "Nhóm khách hàng "
            +
            profiles["Cluster"].astype(str)
        )

        evolution = evolution_result(
            month_history(clean),
            scaler,
            model
        )

        matrix, pairs = (
            transition_result(evolution)
        )

except Exception as exc:
    st.error(
        f"Phân tích thất bại: {exc}"
    )
    st.stop()


# Dashboard chỉ xuất hiện sau khi chạy thành công.

st.success(
    "✅ Phân tích hoàn tất! "
    "Kéo xuống để xem Dashboard."
)

st.divider()


# =====================================================
# 13. DASHBOARD
# =====================================================

heading("📊 Tổng quan kết quả")

metrics = [
    ("Khách hàng", len(rfm)),
    (
        "Hóa đơn",
        clean["InvoiceNo"].nunique()
    ),
    (
        "Giá trị mua (theo tiền tệ gốc)",
        clean["TotalAmount"].sum()
    ),
    ("Số nhóm khách hàng", k)
]

for column, (name, value) in zip(
    st.columns(4),
    metrics
):
    with column:
        kpi(
            name,
            f"{value:,.0f}"
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
            "Khach_hang": "Số khách hàng"
        }
    )

    fig.update_layout(
        showlegend=False
    )

    st.plotly_chart(
        nice(fig),
        use_container_width=True
    )

with b:
    fig = px.pie(
        profiles,
        names="Nhóm khách hàng",
        values="Khach_hang",
        hole=0.55,
        color_discrete_sequence=COLORS
    )

    st.plotly_chart(
        nice(fig),
        use_container_width=True
    )


heading("📋 Đặc điểm các nhóm khách hàng")

profile_display = (
    profiles[
        [
            "Nhóm khách hàng",
            "Khach_hang",
            "R",
            "F",
            "M"
        ]
    ]
    .rename(columns={
        "Khach_hang": "Số khách hàng",
        "R": "R trung bình (ngày)",
        "F": "F trung bình (hóa đơn)",
        "M": "M trung bình (giá trị mua)"
    })
    .round(2)
)

st.dataframe(
    profile_display,
    hide_index=True,
    use_container_width=True
)


# =====================================================
# 14. DASHBOARD CUSTOMER EVOLUTION
# =====================================================

heading("🔄 Customer Evolution")

st.info(
    "Theo dõi nhóm khách hàng theo từng "
    "tháng có giao dịch. "
    "F và M là số tích lũy; "
    "dùng chung một mô hình K-Means "
    "để so sánh theo thời gian."
)

monthly_counts = (
    evolution.groupby([
        "Tháng",
        "Nhóm khách hàng"
    ])
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
    color_discrete_sequence=COLORS
)

st.plotly_chart(
    nice(fig, 420),
    use_container_width=True
)

if not pairs.empty:
    changes = int(
        (
            pairs["BeforeCluster"]
            !=
            pairs["Cluster"]
        ).sum()
    )

    st.caption(
        f"Có {changes:,} lượt chuyển nhóm "
        f"trên {len(pairs):,} lượt đối chiếu "
        "hai tháng giao dịch liền nhau."
    )


# =====================================================
# 15. CHI TIẾT PHÂN TÍCH
# =====================================================

heading("🔍 Chi tiết phân tích")

with st.expander("1. Dữ liệu"):
    st.caption(
        f"File: {uploaded.name} · "
        f"Sheet: {', '.join(selected_sheets)}"
    )

    st.dataframe(
        raw.head(50),
        use_container_width=True,
        hide_index=True
    )


with st.expander("2. Tiền xử lý"):
    st.dataframe(
        pd.DataFrame(
            list(cleaning_stats.items()),
            columns=["Chỉ số", "Số dòng"]
        ),
        hide_index=True,
        use_container_width=True
    )


with st.expander("3. RFM"):
    st.write(
        "**R:** số ngày từ lần mua gần nhất. "
        "**F:** số hóa đơn. "
        "**M:** tổng giá trị mua hàng."
    )

    st.dataframe(
        rfm.head(100),
        hide_index=True,
        use_container_width=True
    )


with st.expander("4. Chuẩn hóa"):
    st.write(
        "Biến đổi log1p và chuẩn hóa "
        "StandardScaler trước K-Means."
    )

    st.dataframe(
        pd.DataFrame(
            z[:30],
            columns=[
                "R chuẩn hóa",
                "F chuẩn hóa",
                "M chuẩn hóa"
            ]
        ),
        hide_index=True,
        use_container_width=True
    )


with st.expander("5. K-Means"):
    st.write(
        f"Đang phân thành "
        f"**{k} nhóm khách hàng**."
    )

    if st.button("Tính Elbow"):
        elbow, _ = model_diagnostics(
            z,
            k
        )

        if not elbow.empty:
            fig = px.line(
                elbow,
                x="K",
                y="Inertia",
                markers=True
            )

            st.plotly_chart(
                nice(fig),
                use_container_width=True
            )


with st.expander(
    "6. Phân tích nhóm khách hàng"
):
    st.dataframe(
        profiles.round(2),
        hide_index=True,
        use_container_width=True
    )

    export = groups.copy()

    export["Nhóm khách hàng"] = (
        "Nhóm khách hàng "
        +
        export["Cluster"].astype(str)
    )

    st.download_button(
        "Tải bảng phân nhóm",
        export.to_csv(
            index=False
        ).encode("utf-8-sig"),
        "nhom_khach_hang.csv",
        "text/csv"
    )


with st.expander(
    "7. Đánh giá và so sánh"
):
    st.write(
        "So sánh K-Means và Agglomerative "
        "trên cùng mẫu tối đa 1.000 "
        "khách hàng bằng Silhouette "
        "và Davies-Bouldin."
    )

    if st.button("Chạy so sánh"):
        _, scores = model_diagnostics(
            z,
            k
        )

        st.dataframe(
            scores.round(4),
            use_container_width=True,
            hide_index=True
        )


# =====================================================
# 16. CUSTOMER EVOLUTION CÓ VÍ DỤ
# =====================================================

with st.expander(
    "8. Customer Evolution – "
    "Ví dụ và hành trình khách hàng"
):
    st.markdown(
        "#### Customer Evolution là gì?"
    )

    st.write(
        "Theo dõi một khách hàng được xếp "
        "vào **nhóm nào ở mỗi tháng có mua hàng**. "
        "Khi mã nhóm khác trước, đó là một "
        "lần thay đổi nhóm được ghi nhận."
    )

    st.markdown(
        "#### Ví dụ minh họa "
        "(giả định, không phải kết quả thật)"
    )

    example = pd.DataFrame({
        "Tháng": [
            "01/2025",
            "02/2025",
            "03/2025"
        ],
        "Nhóm khách hàng": [
            "Nhóm khách hàng 2",
            "Nhóm khách hàng 2",
            "Nhóm khách hàng 1"
        ],
        "R (ngày)": [12, 6, 3],
        "F (hóa đơn tích lũy)": [1, 2, 4],
        "M (giá trị tích lũy)": [
            120000,
            350000,
            850000
        ]
    })

    st.dataframe(
        example,
        hide_index=True,
        use_container_width=True
    )

    st.info(
        "Trong ví dụ: tháng 1 → tháng 2 "
        "**giữ nhóm 2**; tháng 2 → tháng 3 "
        "**chuyển từ nhóm 2 sang nhóm 1**. "
        "F và M tăng. Mã nhóm không có "
        "thứ tự tốt/xấu."
    )

    st.markdown("#### Chú thích RFM")

    rfm_guide = pd.DataFrame([
        [
            "R – Recency",
            "Số ngày từ lần mua gần nhất "
            "đến cuối tháng",
            "R nhỏ: mua gần đây hơn"
        ],
        [
            "F – Frequency",
            "Số hóa đơn tích lũy",
            "F lớn: mua nhiều lần hơn"
        ],
        [
            "M – Monetary",
            "Tổng giá trị mua tích lũy",
            "M lớn: tổng mua nhiều hơn"
        ]
    ], columns=[
        "Chỉ số",
        "Ý nghĩa",
        "Cách đọc"
    ])

    st.dataframe(
        rfm_guide,
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "M giữ nguyên đơn vị tiền trong "
        "file tải lên. Những tháng không "
        "giao dịch chưa được thể hiện "
        "trong lịch sử này."
    )

    # MA TRẬN CHUYỂN NHÓM
    st.markdown(
        "#### Ma trận chuyển nhóm "
        "từ dữ liệu thật"
    )

    st.write(
        "**Hàng** là nhóm tháng trước; "
        "**cột** là nhóm tháng sau; "
        "**ô trên đường chéo** là giữ "
        "nguyên nhóm; **ô ngoài đường chéo** "
        "là chuyển nhóm."
    )

    if matrix.empty:
        st.warning(
            "Chưa có cặp tháng giao dịch "
            "liền nhau để hiển thị ma trận."
        )

    else:
        mat = matrix.reindex(
            index=range(k),
            columns=range(k),
            fill_value=0
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
                "x": "Nhóm tháng sau",
                "y": "Nhóm tháng trước",
                "color": "Số lượt"
            }
        )

        st.plotly_chart(
            nice(fig, 410),
            use_container_width=True
        )

    # LỊCH SỬ KHÁCH HÀNG
    st.markdown(
        "#### Hành trình khách hàng "
        "(dữ liệu thật)"
    )

    customer = st.selectbox(
        "Chọn mã khách hàng",
        sorted(
            evolution["CustomerID"]
            .unique()
            .tolist()
        )
    )

    hist = (
        evolution[
            evolution["CustomerID"] == customer
        ]
        .sort_values("Month")
    )

    hist_view = (
        hist[
            [
                "Tháng",
                "Nhóm khách hàng",
                "Recency",
                "Frequency",
                "Monetary"
            ]
        ]
        .rename(columns={
            "Recency": "R – Số ngày",
            "Frequency": "F – Hóa đơn tích lũy",
            "Monetary": "M – Tổng tiền tích lũy"
        })
    )

    st.dataframe(
        hist_view.round(2),
        hide_index=True,
        use_container_width=True
    )

    if len(hist) > 1:
        fig = px.line(
            hist,
            x="Tháng",
            y="Cluster",
            markers=True
        )

        fig.update_yaxes(
            tickvals=list(range(k)),
            ticktext=[
                f"Nhóm khách hàng {i}"
                for i in range(k)
            ]
        )

        st.plotly_chart(
            nice(fig, 310),
            use_container_width=True
        )

        n_change = int(
            (
                hist["Cluster"]
                .diff()
                .iloc[1:] != 0
            ).sum()
        )

        st.write(
            f"Khách hàng **{customer}** "
            f"có **{n_change} lần khác nhóm** "
            "giữa các lần giao dịch được "
            "ghi nhận; các lần ghi nhận "
            "có thể không nằm ở tháng liền kề."
        )

    else:
        st.info(
            "Khách hàng này chỉ xuất hiện "
            "ở một tháng giao dịch."
        )

    export_columns = [
        "CustomerID",
        "Tháng",
        "Nhóm khách hàng",
        "Recency",
        "Frequency",
        "Monetary"
    ]

    st.download_button(
        "Tải lịch sử Customer Evolution",
        evolution[
            export_columns
        ].to_csv(
            index=False
        ).encode("utf-8-sig"),
        "customer_evolution.csv",
        "text/csv"
    )


# =====================================================
# 17. KẾT QUẢ
# =====================================================

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

st.divider()

st.caption(
    "CUSTOMER INTELLIGENCE · RFM · "
    "K-Means · Customer Evolution · "
    "Phân tích dữ liệu lịch sử"
)
