
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from io import BytesIO
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

# ==================================================
# 1. CẤU HÌNH ỨNG DỤNG
# ==================================================

st.set_page_config(
    page_title="Customer Evolution",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# 2. GIAO DIỆN
# ==================================================

st.markdown("""
<style>

.stApp {
    background-color: #0B1220;
    color: #E2E8F0;
}

[data-testid="stSidebar"] {
    background-color: #111C30;
    border-right: 1px solid #26374F;
}

.block-container {
    padding-top: 2rem;
    max-width: 1500px;
}

.hero {
    background: linear-gradient(
        120deg,
        #142B4C,
        #101D33,
        #202047
    );

    padding: 30px;
    border-radius: 20px;
    border: 1px solid #29415F;
    margin-bottom: 25px;
}

.hero-title {
    font-size: 34px;
    font-weight: 800;
    color: #F8FAFC;
}

.hero-subtitle {
    color: #A8BAD0;
    font-size: 15px;
    margin-top: 10px;
}

.metric-card {
    background: linear-gradient(
        145deg,
        #15253C,
        #111D30
    );

    padding: 22px;
    border-radius: 16px;
    border: 1px solid #293D58;
    min-height: 115px;
}

.metric-label {
    font-size: 13px;
    color: #9DB0C7;
}

.metric-value {
    font-size: 29px;
    font-weight: 800;
    color: white;
    margin-top: 8px;
}

.section-title {
    font-size: 21px;
    font-weight: 800;
    color: #F1F5F9;
    margin-top: 22px;
    margin-bottom: 15px;
}

.highlight-box {
    background: #142B45;
    border-left: 4px solid #38BDF8;
    padding: 18px;
    border-radius: 10px;
    margin-bottom: 20px;
}

div[data-testid="stExpander"] {
    background: #111E32;
    border: 1px solid #2A3C56;
    border-radius: 12px;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# 3. HÀM HIỂN THỊ
# ==================================================

def format_number(number):
    return f"{number:,.0f}".replace(",", ".")


def show_metric(title, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def section_title(title):
    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True
    )


def style_chart(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#CBD5E1"),
        margin=dict(l=15, r=15, t=25, b=15)
    )
    return fig


# ==================================================
# 4. ĐỌC DỮ LIỆU
# ==================================================

@st.cache_data(show_spinner=False)
def get_sheets(file_bytes, filename):

    if filename.lower().endswith(".csv"):
        return ["CSV"]

    excel = pd.ExcelFile(BytesIO(file_bytes))

    return excel.sheet_names


@st.cache_data(show_spinner=False)
def load_data(file_bytes, filename, sheets):

    datasets = []

    for sheet in sheets:

        if filename.lower().endswith(".csv"):

            try:
                df = pd.read_csv(
                    BytesIO(file_bytes),
                    low_memory=False
                )
            except UnicodeDecodeError:
                df = pd.read_csv(
                    BytesIO(file_bytes),
                    encoding="latin1",
                    low_memory=False
                )

        else:
            df = pd.read_excel(
                BytesIO(file_bytes),
                sheet_name=sheet
            )

        df.columns = [
            str(col).strip()
            for col in df.columns
        ]

        aliases = {
            "invoice": "InvoiceNo",
            "invoiceno": "InvoiceNo",
            "invoice number": "InvoiceNo",

            "invoicedate": "InvoiceDate",
            "invoice date": "InvoiceDate",

            "price": "UnitPrice",
            "unitprice": "UnitPrice",
            "unit price": "UnitPrice",

            "customer id": "CustomerID",
            "customerid": "CustomerID",

            "quantity": "Quantity"
        }

        df.columns = [
            aliases.get(
                col.lower(),
                col
            )
            for col in df.columns
        ]

        if df.columns.duplicated().any():
            raise ValueError(
                "Có tên cột bị trùng sau khi chuẩn hóa."
            )

        datasets.append(df)

    return pd.concat(
        datasets,
        ignore_index=True
    )


# ==================================================
# 5. TIỀN XỬ LÝ
# ==================================================

@st.cache_data(show_spinner=False)
def preprocess_data(df):

    required = [
        "InvoiceNo",
        "InvoiceDate",
        "Quantity",
        "UnitPrice",
        "CustomerID"
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Thiếu cột: " + ", ".join(missing)
        )

    data = df.copy()

    original_count = len(data)

    data["InvoiceDate"] = pd.to_datetime(
        data["InvoiceDate"],
        errors="coerce"
    )

    data["Quantity"] = pd.to_numeric(
        data["Quantity"],
        errors="coerce"
    )

    data["UnitPrice"] = pd.to_numeric(
        data["UnitPrice"],
        errors="coerce"
    )

    data["CustomerID"] = (
        data["CustomerID"]
        .astype("string")
        .str.strip()
        .str.replace(
            r"\.0$",
            "",
            regex=True
        )
    )

    data["CustomerID"] = data[
        "CustomerID"
    ].replace(
        {
            "": pd.NA,
            "nan": pd.NA,
            "None": pd.NA,
            "<NA>": pd.NA
        }
    )

    missing_rows = int(
        data[required].isna().any(axis=1).sum()
    )

    data = data.dropna(
        subset=required
    ).copy()

    cancelled = (
        data["InvoiceNo"]
        .astype(str)
        .str.upper()
        .str.startswith("C")
    )

    cancelled_rows = int(cancelled.sum())

    data = data.loc[
        ~cancelled
    ].copy()

    invalid = (
        (data["Quantity"] <= 0)
        |
        (data["UnitPrice"] <= 0)
    )

    invalid_rows = int(invalid.sum())

    data = data.loc[
        ~invalid
    ].copy()

    duplicate_rows = int(
        data.duplicated().sum()
    )

    data = data.drop_duplicates()

    data["TotalAmount"] = (
        data["Quantity"]
        *
        data["UnitPrice"]
    )

    data = data.loc[
        np.isfinite(data["TotalAmount"])
        &
        (data["TotalAmount"] > 0)
    ].copy()

    if data.empty:
        raise ValueError(
            "Không còn giao dịch hợp lệ."
        )

    statistics = {
        "Dữ liệu ban đầu": original_count,
        "Thiếu dữ liệu": missing_rows,
        "Hóa đơn hủy": cancelled_rows,
        "Giá trị không hợp lệ": invalid_rows,
        "Dòng trùng lặp": duplicate_rows,
        "Dữ liệu hợp lệ": len(data)
    }

    return data, statistics


# ==================================================
# 6. TÍNH RFM
# ==================================================

@st.cache_data(show_spinner=False)
def calculate_rfm(df):

    reference_date = (
        df["InvoiceDate"].max().normalize()
        +
        pd.Timedelta(days=1)
    )

    rfm = (
        df.groupby("CustomerID")
        .agg(
            LastPurchase=(
                "InvoiceDate",
                "max"
            ),
            Frequency=(
                "InvoiceNo",
                "nunique"
            ),
            Monetary=(
                "TotalAmount",
                "sum"
            )
        )
    )

    rfm["Recency"] = (
        reference_date
        -
        rfm["LastPurchase"].dt.normalize()
    ).dt.days

    return (
        rfm[
            [
                "Recency",
                "Frequency",
                "Monetary"
            ]
        ]
        .reset_index()
    )


# ==================================================
# 7. CHUẨN HÓA VÀ K-MEANS
# ==================================================

@st.cache_data(show_spinner=False)
def run_kmeans(rfm, k):

    if len(rfm) <= k:
        raise ValueError(
            "Số khách hàng phải lớn hơn K."
        )

    features = [
        "Recency",
        "Frequency",
        "Monetary"
    ]

    X = np.log1p(
        rfm[features].astype(float)
    )

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(
        X_scaled
    )

    result = rfm.copy()

    result["Cluster"] = labels

    return (
        result,
        X_scaled,
        scaler,
        model
    )


# ==================================================
# 8. PHÂN TÍCH CỤM
# ==================================================

def analyze_clusters(clustered):

    profiles = (
        clustered.groupby("Cluster")
        .agg(
            Customers=(
                "CustomerID",
                "count"
            ),
            Recency=(
                "Recency",
                "mean"
            ),
            Frequency=(
                "Frequency",
                "mean"
            ),
            Monetary=(
                "Monetary",
                "mean"
            )
        )
        .reset_index()
    )

    profiles["Score"] = (
        profiles["Recency"].rank(
            ascending=False,
            pct=True
        )
        +
        profiles["Frequency"].rank(
            pct=True
        )
        +
        profiles["Monetary"].rank(
            pct=True
        )
    )

    profiles["Nhận xét"] = (
        "Hành vi mua sắm trung gian"
    )

    best = profiles["Score"].idxmax()

    lowest = profiles["Score"].idxmin()

    profiles.loc[
        best,
        "Nhận xét"
    ] = "Mua gần đây, giao dịch nổi bật"

    profiles.loc[
        lowest,
        "Nhận xét"
    ] = "Mua ít hơn, cần theo dõi"

    return profiles.drop(
        columns=["Score"]
    )


# ==================================================
# 9. ELBOW
# ==================================================

@st.cache_data(show_spinner=False)
def calculate_elbow(X):

    # Lấy mẫu để tiết kiệm tài nguyên
    if len(X) > 3000:

        rng = np.random.default_rng(42)

        indices = rng.choice(
            len(X),
            size=3000,
            replace=False
        )

        X = X[indices]

    results = []

    for k in range(
        2,
        min(10, len(X) - 1) + 1
    ):

        model = KMeans(
            n_clusters=k,
            n_init=5,
            random_state=42
        )

        model.fit(X)

        results.append(
            {
                "K": k,
                "Inertia": model.inertia_
            }
        )

    return pd.DataFrame(results)


# ==================================================
# 10. ĐÁNH GIÁ VÀ SO SÁNH
# ==================================================

@st.cache_data(show_spinner=False)
def compare_models(X, k):

    # Agglomerative tốn bộ nhớ với dữ liệu lớn.
    # Dùng cùng một mẫu cho cả hai thuật toán.

    if len(X) > 1500:

        rng = np.random.default_rng(42)

        indices = rng.choice(
            len(X),
            size=1500,
            replace=False
        )

        X = X[indices]

    models = {
        "K-Means": KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        ),

        "Agglomerative": AgglomerativeClustering(
            n_clusters=k
        )
    }

    results = []

    for name, model in models.items():

        labels = model.fit_predict(X)

        if len(np.unique(labels)) < 2:
            continue

        results.append(
            {
                "Thuật toán": name,

                "Silhouette": silhouette_score(
                    X,
                    labels,
                    sample_size=min(
                        1000,
                        len(X)
                    ),
                    random_state=42
                ),

                "Davies-Bouldin":
                    davies_bouldin_score(
                        X,
                        labels
                    ),

                "Calinski-Harabasz":
                    calinski_harabasz_score(
                        X,
                        labels
                    )
            }
        )

    return pd.DataFrame(results)


# ==================================================
# 11. CUSTOMER EVOLUTION
# ==================================================

@st.cache_data(show_spinner=False)
def calculate_evolution(
    df,
    scaler,
    model
):

    data = df[
        [
            "CustomerID",
            "InvoiceNo",
            "InvoiceDate",
            "TotalAmount"
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
                "Month"
            ]
        )
        .agg(
            LastPurchase=(
                "InvoiceDate",
                "max"
            ),
            Invoices=(
                "InvoiceNo",
                "nunique"
            ),
            Spending=(
                "TotalAmount",
                "sum"
            )
        )
        .reset_index()
    )

    monthly = monthly.sort_values(
        [
            "CustomerID",
            "Month"
        ]
    )

    group = monthly.groupby(
        "CustomerID"
    )

    # F và M tích lũy theo thời gian
    monthly["Frequency"] = (
        group["Invoices"].cumsum()
    )

    monthly["Monetary"] = (
        group["Spending"].cumsum()
    )

    # Tính R tại cuối tháng
    month_end = (
        monthly["Month"]
        .dt.to_timestamp(how="end")
        .dt.normalize()
    )

    monthly["Recency"] = (
        month_end
        -
        monthly["LastPurchase"].dt.normalize()
    ).dt.days.clip(lower=0)

    features = [
        "Recency",
        "Frequency",
        "Monetary"
    ]

    X = np.log1p(
        monthly[features].astype(float)
    )

    X_scaled = scaler.transform(X)

    # Giữ nguyên mô hình K-Means toàn kỳ
    monthly["Cluster"] = model.predict(
        X_scaled
    )

    monthly["Tháng"] = (
        monthly["Month"].astype(str)
    )

    return monthly


# ==================================================
# 12. MA TRẬN CHUYỂN NHÓM
# ==================================================

def calculate_transitions(evolution):

    data = evolution.sort_values(
        [
            "CustomerID",
            "Month"
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

    # Chỉ so sánh hai tháng liền nhau
    current_number = (
        data["Month"].dt.year * 12
        +
        data["Month"].dt.month
    )

    previous_number = (
        data["PreviousMonth"].dt.year * 12
        +
        data["PreviousMonth"].dt.month
    )

    valid = (
        data["PreviousMonth"].notna()
        &
        ((current_number - previous_number) == 1)
    )

    changes = data.loc[valid].copy()

    if changes.empty:
        return pd.DataFrame(), changes

    matrix = pd.crosstab(
        changes["PreviousCluster"].astype(int),
        changes["Cluster"]
    )

    return matrix, changes


# ==================================================
# 13. THANH ĐIỀU KHIỂN
# ==================================================

with st.sidebar:

    st.title("📊 Customer Evolution")

    st.caption(
        "Hệ thống phân nhóm khách hàng"
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "📁 Tải dữ liệu",
        type=[
            "xlsx",
            "xls",
            "csv"
        ]
    )

    if uploaded_file:

        file_bytes = uploaded_file.getvalue()

        filename = uploaded_file.name

        try:

            sheets = get_sheets(
                file_bytes,
                filename
            )

        except Exception as error:

            st.error(
                f"Không đọc được file: {error}"
            )

            st.stop()

        if len(sheets) > 1:

            sheet_choice = st.selectbox(
                "Chọn phạm vi dữ liệu",
                [
                    "Tất cả các sheet"
                ] + sheets
            )

        else:

            sheet_choice = sheets[0]

        k = st.slider(
            "Số cụm K",
            min_value=2,
            max_value=8,
            value=4
        )

        st.success(
            "Đã nhận dữ liệu. Hệ thống sẽ tự phân tích."
        )

    st.divider()

    st.markdown("### Quy trình")

    st.caption("""
    Dữ liệu → Tiền xử lý → RFM →
    Chuẩn hóa → K-Means →
    Phân tích khách hàng →
    Đánh giá & so sánh →
    Customer Evolution →
    Kết quả
    """)


# ==================================================
# 14. TRANG CHỦ
# ==================================================

st.markdown("""
<div class="hero">

<div style="color:#7DD3FC;font-weight:700">
CUSTOMER INTELLIGENCE PLATFORM
</div>

<div class="hero-title">
Phân nhóm khách hàng
<br>
& Customer Evolution
</div>

<div class="hero-subtitle">
Hệ thống phân tích hành vi mua sắm
bằng mô hình RFM và thuật toán K-Means
</div>

</div>
""", unsafe_allow_html=True)


if uploaded_file is None:

    st.info(
        "👈 Hãy tải file Excel hoặc CSV ở thanh bên trái để xem Dashboard."
    )

    a, b, c = st.columns(3)

    with a:
        show_metric(
            "Mô hình",
            "RFM"
        )

    with b:
        show_metric(
            "Thuật toán",
            "K-Means"
        )

    with c:
        show_metric(
            "Tính năng",
            "Evolution"
        )

    st.stop()


# ==================================================
# 15. TỰ ĐỘNG PHÂN TÍCH
# ==================================================

try:

    with st.spinner(
        "Đang xử lý dữ liệu và xây dựng mô hình..."
    ):

        if sheet_choice == "Tất cả các sheet":
            selected_sheets = sheets
        else:
            selected_sheets = [sheet_choice]

        raw = load_data(
            file_bytes,
            filename,
            selected_sheets
        )

        clean, statistics = preprocess_data(
            raw
        )

        rfm = calculate_rfm(
            clean
        )

        clustered, X_scaled, scaler, model = (
            run_kmeans(
                rfm,
                k
            )
        )

        profiles = analyze_clusters(
            clustered
        )

        evolution = calculate_evolution(
            clean,
            scaler,
            model
        )

        transition_matrix, transition_data = (
            calculate_transitions(
                evolution
            )
        )

except Exception as error:

    st.error(
        f"Lỗi phân tích dữ liệu: {error}"
    )

    st.stop()


# ==================================================
# 16. DASHBOARD TỔNG QUAN
# ==================================================

section_title(
    "📌 Tổng quan kết quả"
)

a, b, c, d = st.columns(4)

with a:
    show_metric(
        "👥 Khách hàng",
        format_number(len(rfm))
    )

with b:
    show_metric(
        "🧾 Hóa đơn",
        format_number(
            clean["InvoiceNo"].nunique()
        )
    )

with c:
    show_metric(
        "💰 Tổng giá trị mua",
        format_number(
            clean["TotalAmount"].sum()
        )
    )

with d:
    show_metric(
        "🎯 Số cụm",
        str(k)
    )


# ==================================================
# 17. BIỂU ĐỒ PHÂN NHÓM
# ==================================================

left, right = st.columns(
    [1.2, 1]
)

with left:

    section_title(
        "📊 Phân bố khách hàng"
    )

    distribution = profiles.copy()

    distribution["Cụm"] = (
        "Cụm "
        +
        distribution["Cluster"].astype(str)
    )

    fig = px.bar(
        distribution,
        x="Cụm",
        y="Customers",
        color="Cụm",
        text="Customers",
        color_discrete_sequence=[
            "#38BDF8",
            "#818CF8",
            "#34D399",
            "#FBBF24",
            "#F472B6",
            "#22D3EE",
            "#C084FC",
            "#FB7185"
        ]
    )

    fig.update_traces(
        textposition="outside"
    )

    fig.update_layout(
        showlegend=False
    )

    st.plotly_chart(
        style_chart(fig),
        use_container_width=True
    )


with right:

    section_title(
        "👥 Đặc điểm từng nhóm"
    )

    display_profiles = profiles[
        [
            "Cluster",
            "Customers",
            "Recency",
            "Frequency",
            "Monetary",
            "Nhận xét"
        ]
    ].copy()

    display_profiles.columns = [
        "Cụm",
        "Khách hàng",
        "R trung bình",
        "F trung bình",
        "M trung bình",
        "Nhận xét"
    ]

    st.dataframe(
        display_profiles,
        hide_index=True,
        use_container_width=True
    )


# ==================================================
# 18. THÔNG TIN NỔI BẬT
# ==================================================

section_title(
    "💡 Thông tin nổi bật"
)

largest = profiles.sort_values(
    "Customers",
    ascending=False
).iloc[0]

recent = profiles.sort_values(
    "Recency"
).iloc[0]

highest = profiles.sort_values(
    "Monetary",
    ascending=False
).iloc[0]

c1, c2, c3 = st.columns(3)

with c1:
    show_metric(
        "Cụm đông khách nhất",
        f"Cụm {int(largest['Cluster'])}"
    )

with c2:
    show_metric(
        "Cụm mua gần đây hơn",
        f"Cụm {int(recent['Cluster'])}"
    )

with c3:
    show_metric(
        "Cụm chi tiêu trung bình cao",
        f"Cụm {int(highest['Cluster'])}"
    )


# ==================================================
# 19. CUSTOMER EVOLUTION
# ==================================================

section_title(
    "🔄 Customer Evolution"
)

st.markdown("""
<div class="highlight-box">

<b>Theo dõi sự thay đổi nhóm khách hàng theo thời gian</b>

<br><br>

Mô hình sử dụng dữ liệu RFM tích lũy theo tháng
và một mô hình K-Means thống nhất
để quan sát sự dịch chuyển giữa các nhóm.

</div>
""", unsafe_allow_html=True)


evolution_count = (
    evolution.groupby(
        [
            "Tháng",
            "Cluster"
        ]
    )
    .size()
    .reset_index(
        name="Khách hàng"
    )
)

evolution_count["Cụm"] = (
    "Cụm "
    +
    evolution_count["Cluster"].astype(str)
)

fig_evolution = px.line(
    evolution_count,
    x="Tháng",
    y="Khách hàng",
    color="Cụm",
    markers=True
)

st.plotly_chart(
    style_chart(fig_evolution),
    use_container_width=True
)

if not transition_data.empty:

    moved = (
        transition_data["PreviousCluster"]
        !=
        transition_data["Cluster"]
    ).sum()

    st.info(
        f"Có {format_number(moved)} lượt chuyển "
        "nhóm giữa những tháng liền kề có giao dịch."
    )


# ==================================================
# 20. CHI TIẾT PHÂN TÍCH
# ==================================================

section_title(
    "🔍 Chi tiết phân tích"
)

st.caption(
    "Bấm vào từng mục để xem các bước xử lý."
)


# ---------- BƯỚC 1 ----------

with st.expander(
    "1. Dữ liệu",
    expanded=False
):

    st.write(
        f"**Tên file:** {filename}"
    )

    st.write(
        f"**Số dòng:** {format_number(len(raw))}"
    )

    st.dataframe(
        raw.head(30),
        use_container_width=True
    )


# ---------- BƯỚC 2 ----------

with st.expander(
    "2. Tiền xử lý",
    expanded=False
):

    st.write(
        "Loại bỏ dữ liệu thiếu, hóa đơn hủy, "
        "giá trị không hợp lệ và dòng trùng lặp."
    )

    stats_df = pd.DataFrame(
        list(statistics.items()),
        columns=[
            "Tiêu chí",
            "Số dòng"
        ]
    )

    st.dataframe(
        stats_df,
        hide_index=True,
        use_container_width=True
    )

    st.dataframe(
        clean.head(30),
        use_container_width=True
    )


# ---------- BƯỚC 3 ----------

with st.expander(
    "3. Mô hình RFM",
    expanded=False
):

    st.markdown("""
    **Recency:** Số ngày từ lần mua gần nhất.

    **Frequency:** Số hóa đơn của khách hàng.

    **Monetary:** Tổng giá trị mua hàng.
    """)

    st.dataframe(
        rfm.head(100),
        use_container_width=True
    )


# ---------- BƯỚC 4 ----------

with st.expander(
    "4. Chuẩn hóa dữ liệu",
    expanded=False
):

    st.write(
        "Sử dụng log1p kết hợp StandardScaler "
        "để đưa R, F và M về thang đo phù hợp."
    )

    standardized = pd.DataFrame(
        X_scaled[:30],
        columns=[
            "R chuẩn hóa",
            "F chuẩn hóa",
            "M chuẩn hóa"
        ]
    )

    st.dataframe(
        standardized,
        use_container_width=True
    )


# ---------- BƯỚC 5 ----------

with st.expander(
    "5. K-Means",
    expanded=False
):

    st.write(
        f"Số cụm hiện tại: K = {k}"
    )

    with st.spinner(
        "Đang tính Elbow..."
    ):

        elbow = calculate_elbow(
            X_scaled
        )

    fig_elbow = px.line(
        elbow,
        x="K",
        y="Inertia",
        markers=True
    )

    st.plotly_chart(
        style_chart(fig_elbow),
        use_container_width=True
    )

    # Biểu đồ 3D, lấy mẫu nếu dữ liệu lớn
    plot_data = clustered.sample(
        n=min(
            1800,
            len(clustered)
        ),
        random_state=42
    ).copy()

    plot_data["Cụm"] = (
        "Cụm "
        +
        plot_data["Cluster"].astype(str)
    )

    fig_3d = px.scatter_3d(
        plot_data,
        x="Recency",
        y="Frequency",
        z="Monetary",
        color="Cụm",
        opacity=0.7
    )

    st.plotly_chart(
        style_chart(fig_3d),
        use_container_width=True
    )


# ---------- BƯỚC 6 ----------

with st.expander(
    "6. Phân tích khách hàng",
    expanded=False
):

    st.write(
        "Thống kê các đặc trưng RFM "
        "trung bình của từng cụm."
    )

    st.dataframe(
        profiles,
        use_container_width=True
    )

    st.caption(
        "Các tên nhóm là nhận xét tương đối "
        "theo RFM, không phải nhãn đã xác minh."
    )

    st.download_button(
        "⬇️ Tải bảng phân nhóm",
        clustered.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="customer_clusters.csv",
        mime="text/csv"
    )


# ---------- BƯỚC 7 ----------

with st.expander(
    "7. Đánh giá và so sánh thuật toán",
    expanded=False
):

    st.write(
        "So sánh K-Means và Agglomerative "
        "trên cùng một mẫu khách hàng."
    )

    with st.spinner(
        "Đang đánh giá mô hình..."
    ):

        evaluation = compare_models(
            X_scaled,
            k
        )

    st.dataframe(
        evaluation,
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "Silhouette và Calinski-Harabasz: "
        "cao hơn thường tốt hơn. "
        "Davies-Bouldin: thấp hơn thường tốt hơn."
    )


# ---------- BƯỚC 8 ----------

with st.expander(
    "8. Customer Evolution",
    expanded=False
):

    st.subheader(
        "Ma trận chuyển nhóm"
    )

    if transition_matrix.empty:

        st.warning(
            "Chưa có đủ dữ liệu tháng liền kề."
        )

    else:

        matrix = transition_matrix.reindex(
            index=range(k),
            columns=range(k),
            fill_value=0
        )

        fig_heatmap = px.imshow(
            matrix,
            text_auto=True,
            color_continuous_scale="Blues",
            labels={
                "x": "Cụm tháng sau",
                "y": "Cụm tháng trước",
                "color": "Lượt"
            }
        )

        st.plotly_chart(
            style_chart(fig_heatmap),
            use_container_width=True
        )

    st.subheader(
        "Hành trình từng khách hàng"
    )

    customers = sorted(
        evolution["CustomerID"].unique()
    )

    selected_customer = st.selectbox(
        "Chọn khách hàng",
        customers
    )

    history = evolution[
        evolution["CustomerID"]
        ==
        selected_customer
    ].sort_values("Month")

    st.dataframe(
        history[
            [
                "Tháng",
                "Cluster",
                "Recency",
                "Frequency",
                "Monetary"
            ]
        ],
        hide_index=True,
        use_container_width=True
    )

    if len(history) > 1:

        fig_history = px.line(
            history,
            x="Tháng",
            y="Cluster",
            markers=True
        )

        fig_history.update_yaxes(
            dtick=1
        )

        st.plotly_chart(
            style_chart(fig_history),
            use_container_width=True
        )

    export_columns = [
        "CustomerID",
        "Tháng",
        "Cluster",
        "Recency",
        "Frequency",
        "Monetary"
    ]

    st.download_button(
        "⬇️ Tải Customer Evolution",
        evolution[
            export_columns
        ].to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="customer_evolution.csv",
        mime="text/csv"
    )


# ---------- BƯỚC 9 ----------

with st.expander(
    "9. Streamlit và kết quả",
    expanded=False
):

    st.write(
        "Ứng dụng trực quan hóa kết quả "
        "phân nhóm và Customer Evolution."
    )

    st.write(
        f"Khách hàng: {format_number(len(rfm))}"
    )

    st.write(
        f"Số cụm: {k}"
    )

    st.write(
        "Các biểu đồ cho phép quan sát "
        "đặc điểm và chuyển nhóm theo thời gian."
    )


# ==================================================
# 21. KẾT LUẬN
# ==================================================

section_title(
    "✅ Kết quả phân tích"
)

st.success(
    f"""
    Đã phân tích {format_number(len(rfm))}
    khách hàng và phân thành {k} cụm
    bằng thuật toán K-Means kết hợp RFM.

    Customer Evolution giúp quan sát
    sự thay đổi nhóm khách hàng theo tháng.
    """
)

st.caption(
    "Hệ thống phân tích hồi cứu, "
    "không phải mô hình dự báo tương lai."
)
