import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Evolution",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📊 Customer Segmentation")

st.subheader(
    "Customer Evolution – Theo dõi sự thay đổi nhóm khách hàng theo thời gian"
)

st.caption(
    "RFM + K-Means + Customer Evolution"
)


# ============================================================
# SESSION STATE
# ============================================================

if "raw_df" not in st.session_state:
    st.session_state.raw_df = None

if "clean_df" not in st.session_state:
    st.session_state.clean_df = None

if "rfm_df" not in st.session_state:
    st.session_state.rfm_df = None

if "scaled_rfm" not in st.session_state:
    st.session_state.scaled_rfm = None

if "kmeans_model" not in st.session_state:
    st.session_state.kmeans_model = None

if "clustered_rfm" not in st.session_state:
    st.session_state.clustered_rfm = None

if "evolution_df" not in st.session_state:
    st.session_state.evolution_df = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Thiết lập")

    uploaded_file = st.file_uploader(
        "📂 Upload Online Retail II",
        type=["xlsx", "xls", "csv"]
    )

    st.divider()

    k_value = st.slider(
        "Số cụm K",
        min_value=2,
        max_value=10,
        value=4
    )

    st.divider()

    st.info(
        "Dataset khuyến nghị:\n\n"
        "UCI Online Retail II"
    )


# ============================================================
# LOAD DATA
# ============================================================

if uploaded_file is None:

    st.info(
        "👆 Hãy upload file Online Retail II để bắt đầu."
    )

    st.markdown(
        """
        ### Quy trình hệ thống

        **1. Dữ liệu**

        ↓

        **2. Tiền xử lý**

        ↓

        **3. RFM**

        ↓

        **4. Chuẩn hóa**

        ↓

        **5. K-Means**

        ↓

        **6. Phân tích khách hàng**

        ↓

        **7. Đánh giá & so sánh**

        ↓

        **8. Customer Evolution**
        """
    )

    st.stop()


# ============================================================
# READ FILE
# ============================================================

try:

    if uploaded_file.name.lower().endswith(".csv"):

        df = pd.read_csv(uploaded_file)

    else:

        excel = pd.ExcelFile(uploaded_file)

        sheets = excel.sheet_names

        if len(sheets) > 1:

            selected_sheet = st.selectbox(
                "Chọn sheet",
                sheets
            )

        else:

            selected_sheet = sheets[0]

        df = pd.read_excel(
            uploaded_file,
            sheet_name=selected_sheet
        )

except Exception as e:

    st.error(
        f"❌ Không thể đọc file: {e}"
    )

    st.stop()


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
)


required_columns = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country"
]


missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]


if missing_columns:

    st.error(
        "Dataset thiếu các cột:\n\n"
        + ", ".join(missing_columns)
    )

    st.stop()


st.session_state.raw_df = df.copy()


# ============================================================
# TABS
# ============================================================

tabs = st.tabs([
    "📁 Dữ liệu",
    "🧹 Tiền xử lý",
    "📐 RFM",
    "⚙️ K-Means",
    "👥 Phân tích khách hàng",
    "📊 Đánh giá & so sánh",
    "🔄 Customer Evolution",
    "📋 Kết quả"
])


# ============================================================
# TAB 1 - DATA
# ============================================================

with tabs[0]:

    st.header("1. Dữ liệu nghiên cứu")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Số dòng",
        f"{len(df):,}"
    )

    c2.metric(
        "Số cột",
        f"{len(df.columns):,}"
    )

    c3.metric(
        "Khách hàng",
        f"{df['CustomerID'].nunique():,}"
    )

    c4.metric(
        "Quốc gia",
        f"{df['Country'].nunique():,}"
    )

    st.subheader("Xem dữ liệu")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    st.subheader("Thuộc tính")

    attribute_df = pd.DataFrame({
        "Thuộc tính": df.columns,
        "Kiểu dữ liệu": [
            str(df[c].dtype)
            for c in df.columns
        ],
        "Missing": [
            int(df[c].isna().sum())
            for c in df.columns
        ]
    })

    st.dataframe(
        attribute_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# TAB 2 - PREPROCESSING
# ============================================================

with tabs[1]:

    st.header("2. Tiền xử lý dữ liệu")

    work_df = df.copy()

    original_rows = len(work_df)

    # Convert types

    work_df["InvoiceDate"] = pd.to_datetime(
        work_df["InvoiceDate"],
        errors="coerce"
    )

    work_df["Quantity"] = pd.to_numeric(
        work_df["Quantity"],
        errors="coerce"
    )

    work_df["UnitPrice"] = pd.to_numeric(
        work_df["UnitPrice"],
        errors="coerce"
    )

    work_df["CustomerID"] = pd.to_numeric(
        work_df["CustomerID"],
        errors="coerce"
    )

    # Missing important data

    before = len(work_df)

    work_df = work_df.dropna(
        subset=[
            "CustomerID",
            "InvoiceDate",
            "InvoiceNo",
            "Quantity",
            "UnitPrice"
        ]
    )

    removed_missing = before - len(work_df)

    # Cancellation

    invoice_text = (
        work_df["InvoiceNo"]
        .astype(str)
        .str.upper()
    )

    cancellation_mask = invoice_text.str.startswith("C")

    cancelled = int(cancellation_mask.sum())

    work_df = work_df[
        ~cancellation_mask
    ]

    # Quantity

    before = len(work_df)

    work_df = work_df[
        work_df["Quantity"] > 0
    ]

    removed_quantity = before - len(work_df)

    # Price

    before = len(work_df)

    work_df = work_df[
        work_df["UnitPrice"] > 0
    ]

    removed_price = before - len(work_df)

    # Duplicate

    before = len(work_df)

    work_df = work_df.drop_duplicates()

    removed_duplicate = before - len(work_df)

    # Total amount

    work_df["TotalAmount"] = (
        work_df["Quantity"]
        * work_df["UnitPrice"]
    )

    st.session_state.clean_df = work_df.copy()

    final_rows = len(work_df)

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Dữ liệu ban đầu",
        f"{original_rows:,}"
    )

    c2.metric(
        "Sau tiền xử lý",
        f"{final_rows:,}"
    )

    c3.metric(
        "Đã loại",
        f"{original_rows - final_rows:,}"
    )

    st.subheader("Chi tiết xử lý")

    cleaning_table = pd.DataFrame({
        "Công đoạn": [
            "Thiếu dữ liệu quan trọng",
            "Giao dịch hủy",
            "Quantity ≤ 0",
            "UnitPrice ≤ 0",
            "Dòng trùng"
        ],
        "Số dòng loại": [
            removed_missing,
            cancelled,
            removed_quantity,
            removed_price,
            removed_duplicate
        ]
    })

    st.dataframe(
        cleaning_table,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Dữ liệu sau xử lý")

    st.dataframe(
        work_df.head(20),
        use_container_width=True
    )

    st.success(
        "✅ Tiền xử lý hoàn thành."
    )


# ============================================================
# TAB 3 - RFM
# ============================================================

with tabs[2]:

    st.header("3. Mô hình RFM")

    clean_df = st.session_state.clean_df

    if clean_df is None:

        st.warning(
            "Chưa có dữ liệu sau tiền xử lý."
        )

        st.stop()

    max_date = clean_df["InvoiceDate"].max()

    reference_date = (
        max_date
        + pd.Timedelta(days=1)
    )

    rfm = (
        clean_df
        .groupby("CustomerID")
        .agg(
            Recency=(
                "InvoiceDate",
                lambda x:
                (reference_date - x.max()).days
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
        .reset_index()
    )

    # Remove invalid values

    rfm = rfm[
        (rfm["Frequency"] > 0)
        & (rfm["Monetary"] > 0)
    ].copy()

    st.session_state.rfm_df = rfm.copy()

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Khách hàng",
        f"{len(rfm):,}"
    )

    c2.metric(
        "Frequency TB",
        f"{rfm['Frequency'].mean():.2f}"
    )

    c3.metric(
        "Monetary TB",
        f"{rfm['Monetary'].mean():,.2f}"
    )

    st.subheader("Bảng RFM")

    st.dataframe(
        rfm.head(30),
        use_container_width=True
    )

    st.subheader("Phân phối RFM")

    col1, col2 = st.columns(2)

    with col1:

        fig_r = px.histogram(
            rfm,
            x="Recency",
            title="Phân phối Recency"
        )

        st.plotly_chart(
            fig_r,
            use_container_width=True
        )

    with col2:

        fig_f = px.histogram(
            rfm,
            x="Frequency",
            title="Phân phối Frequency"
        )

        st.plotly_chart(
            fig_f,
            use_container_width=True
        )

    fig_m = px.histogram(
        rfm,
        x="Monetary",
        title="Phân phối Monetary"
    )

    st.plotly_chart(
        fig_m,
        use_container_width=True
    )


# ============================================================
# TAB 4 - KMEANS
# ============================================================

with tabs[3]:

    st.header("4. Chuẩn hóa & K-Means")

    rfm = st.session_state.rfm_df

    if rfm is None:

        st.warning(
            "Chưa có dữ liệu RFM."
        )

        st.stop()

    features = [
        "Recency",
        "Frequency",
        "Monetary"
    ]

    # Log transform for skewed RFM

    rfm_model = rfm.copy()

    rfm_model["Recency"] = np.log1p(
        rfm_model["Recency"]
    )

    rfm_model["Frequency"] = np.log1p(
        rfm_model["Frequency"]
    )

    rfm_model["Monetary"] = np.log1p(
        rfm_model["Monetary"]
    )

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        rfm_model[features]
    )

    st.session_state.scaled_rfm = X_scaled

    st.subheader("Elbow Method")

    inertias = []

    k_range = range(2, 11)

    for k in k_range:

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        model.fit(X_scaled)

        inertias.append(
            model.inertia_
        )

    elbow_df = pd.DataFrame({
        "K": list(k_range),
        "Inertia": inertias
    })

    fig_elbow = px.line(
        elbow_df,
        x="K",
        y="Inertia",
        markers=True,
        title="Elbow Method"
    )

    st.plotly_chart(
        fig_elbow,
        use_container_width=True
    )

    # KMeans

    kmeans = KMeans(
        n_clusters=k_value,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(
        X_scaled
    )

    clustered = rfm.copy()

    clustered["Cluster"] = labels

    st.session_state.kmeans_model = kmeans
    st.session_state.clustered_rfm = clustered

    st.success(
        f"✅ K-Means đã phân nhóm thành {k_value} cụm."
    )

    st.subheader("Số lượng khách hàng mỗi cụm")

    cluster_count = (
        clustered["Cluster"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    cluster_count.columns = [
        "Cluster",
        "Customers"
    ]

    fig_count = px.bar(
        cluster_count,
        x="Cluster",
        y="Customers",
        text="Customers",
        title="Customer Count by Cluster"
    )

    st.plotly_chart(
        fig_count,
        use_container_width=True
    )

    # 3D visualization

    fig_3d = px.scatter_3d(
        clustered,
        x="Recency",
        y="Frequency",
        z="Monetary",
        color="Cluster",
        title="K-Means Customer Segmentation"
    )

    st.plotly_chart(
        fig_3d,
        use_container_width=True
    )


# ============================================================
# TAB 5 - CUSTOMER ANALYSIS
# ============================================================

with tabs[4]:

    st.header("5. Phân tích khách hàng")

    clustered = st.session_state.clustered_rfm

    if clustered is None:

        st.warning(
            "Hãy chạy K-Means trước."
        )

        st.stop()

    summary = (
        clustered
        .groupby("Cluster")
        .agg(
            Customers=("CustomerID", "count"),
            Avg_Recency=("Recency", "mean"),
            Avg_Frequency=("Frequency", "mean"),
            Avg_Monetary=("Monetary", "mean"),
            Total_Revenue=("Monetary", "sum")
        )
        .reset_index()
    )

    st.subheader(
        "Đặc trưng từng nhóm khách hàng"
    )

    st.dataframe(
        summary.round(2),
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Tên nhóm tham khảo"
    )

    # Determine group labels by relative characteristics

    cluster_labels = {}

    for _, row in summary.iterrows():

        cluster = int(row["Cluster"])

        recency = row["Avg_Recency"]
        frequency = row["Avg_Frequency"]
        monetary = row["Avg_Monetary"]

        if (
            frequency >= summary["Avg_Frequency"].median()
            and monetary >= summary["Avg_Monetary"].median()
            and recency <= summary["Avg_Recency"].median()
        ):

            name = "Khách hàng giá trị cao"

        elif (
            recency <= summary["Avg_Recency"].median()
            and frequency >= summary["Avg_Frequency"].median()
        ):

            name = "Khách hàng trung thành"

        elif recency >= summary["Avg_Recency"].median():

            name = "Khách hàng cần quan tâm"

        else:

            name = "Khách hàng tiềm năng"

        cluster_labels[cluster] = name

    summary["Nhóm tham khảo"] = (
        summary["Cluster"]
        .map(cluster_labels)
    )

    st.dataframe(
        summary.round(2),
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "Tên nhóm chỉ mang tính diễn giải dựa trên đặc trưng RFM; "
        "cluster của K-Means là kết quả định lượng."
    )


# ============================================================
# TAB 6 - EVALUATION
# ============================================================

with tabs[5]:

    st.header("6. Đánh giá & so sánh")

    rfm = st.session_state.rfm_df
    X_scaled = st.session_state.scaled_rfm

    if rfm is None or X_scaled is None:

        st.warning(
            "Hãy hoàn thành RFM và K-Means trước."
        )

        st.stop()

    # --------------------------------------------------------
    # KMEANS
    # --------------------------------------------------------

    kmeans = KMeans(
        n_clusters=k_value,
        random_state=42,
        n_init=10
    )

    kmeans_labels = kmeans.fit_predict(
        X_scaled
    )

    kmeans_silhouette = silhouette_score(
        X_scaled,
        kmeans_labels
    )

    kmeans_db = davies_bouldin_score(
        X_scaled,
        kmeans_labels
    )

    kmeans_ch = calinski_harabasz_score(
        X_scaled,
        kmeans_labels
    )

    # --------------------------------------------------------
    # AGGLOMERATIVE
    # --------------------------------------------------------

    agglomerative = AgglomerativeClustering(
        n_clusters=k_value
    )

    agg_labels = agglomerative.fit_predict(
        X_scaled
    )

    agg_silhouette = silhouette_score(
        X_scaled,
        agg_labels
    )

    agg_db = davies_bouldin_score(
        X_scaled,
        agg_labels
    )

    agg_ch = calinski_harabasz_score(
        X_scaled,
        agg_labels
    )

    comparison = pd.DataFrame({

        "Thuật toán": [
            "K-Means",
            "Agglomerative Clustering"
        ],

        "Silhouette": [
            kmeans_silhouette,
            agg_silhouette
        ],

        "Davies-Bouldin": [
            kmeans_db,
            agg_db
        ],

        "Calinski-Harabasz": [
            kmeans_ch,
            agg_ch
        ]

    })

    st.subheader(
        "So sánh thuật toán"
    )

    st.dataframe(
        comparison.round(4),
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        """
        ### Ý nghĩa các chỉ số

        **Silhouette**
        - Càng cao càng tốt.
        - Đánh giá mức độ tách biệt giữa các cụm.

        **Davies-Bouldin**
        - Càng thấp càng tốt.
        - Đánh giá độ tương đồng trong và giữa các cụm.

        **Calinski-Harabasz**
        - Giá trị cao thường thể hiện các cụm tách biệt tốt hơn.
        """
    )

    fig_compare = px.bar(
        comparison,
        x="Thuật toán",
        y="Silhouette",
        text="Silhouette",
        title="So sánh Silhouette Score"
    )

    st.plotly_chart(
        fig_compare,
        use_container_width=True
    )


# ============================================================
# TAB 7 - CUSTOMER EVOLUTION
# ============================================================

with tabs[6]:

    st.header(
        "7. Customer Evolution"
    )

    st.write(
        """
        Theo dõi sự thay đổi nhóm khách hàng theo thời gian.

        Hệ thống sử dụng cùng mô hình K-Means đã huấn luyện
        trên toàn bộ dữ liệu RFM để dự đoán cluster cho từng
        giai đoạn. Điều này giúp cluster giữa các giai đoạn
        có cùng ý nghĩa và có thể theo dõi sự chuyển dịch.
        """
    )

    clean_df = st.session_state.clean_df
    kmeans_model = st.session_state.kmeans_model

    if clean_df is None or kmeans_model is None:

        st.warning(
            "Hãy hoàn thành K-Means trước."
        )

        st.stop()

    evolution_base = clean_df.copy()

    evolution_base["Month"] = (
        evolution_base["InvoiceDate"]
        .dt.to_period("M")
        .astype(str)
    )

    months = sorted(
        evolution_base["Month"].unique()
    )

    if len(months) < 2:

        st.warning(
            "Dữ liệu cần ít nhất 2 giai đoạn thời gian."
        )

        st.stop()

    selected_months = st.multiselect(
        "Chọn các tháng",
        months,
        default=months[-min(6, len(months)):]
    )

    if len(selected_months) < 2:

        st.info(
            "Chọn ít nhất 2 tháng để xem Customer Evolution."
        )

        st.stop()

    evolution_records = []

    # --------------------------------------------------------
    # MONTHLY RFM
    # --------------------------------------------------------

    for month in selected_months:

        monthly = evolution_base[
            evolution_base["Month"] == month
        ].copy()

        if monthly.empty:
            continue

        period_end = monthly["InvoiceDate"].max()

        reference = (
            period_end
            + pd.Timedelta(days=1)
        )

        monthly_rfm = (
            monthly
            .groupby("CustomerID")
            .agg(
                Recency=(
                    "InvoiceDate",
                    lambda x:
                    (reference - x.max()).days
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
            .reset_index()
        )

        monthly_rfm = monthly_rfm[
            (monthly_rfm["Frequency"] > 0)
            & (monthly_rfm["Monetary"] > 0)
        ].copy()

        if monthly_rfm.empty:
            continue

        model_features = monthly_rfm[
            [
                "Recency",
                "Frequency",
                "Monetary"
            ]
        ].copy()

        model_features["Recency"] = np.log1p(
            model_features["Recency"]
        )

        model_features["Frequency"] = np.log1p(
            model_features["Frequency"]
        )

        model_features["Monetary"] = np.log1p(
            model_features["Monetary"]
        )

        # Fit a period-specific scaler would make the global
        # KMeans model incompatible, so use the same scaler
        # learned globally from the RFM data.

        global_rfm = st.session_state.rfm_df.copy()

        global_features = global_rfm[
            [
                "Recency",
                "Frequency",
                "Monetary"
            ]
        ].copy()

        global_features["Recency"] = np.log1p(
            global_features["Recency"]
        )

        global_features["Frequency"] = np.log1p(
            global_features["Frequency"]
        )

        global_features["Monetary"] = np.log1p(
            global_features["Monetary"]
        )

        global_scaler = StandardScaler()

        global_scaler.fit(
            global_features
        )

        monthly_scaled = global_scaler.transform(
            model_features
        )

        monthly_rfm["Cluster"] = (
            kmeans_model.predict(
                monthly_scaled
            )
        )

        monthly_rfm["Month"] = month

        evolution_records.append(
            monthly_rfm
        )

    if not evolution_records:

        st.warning(
            "Không tạo được dữ liệu Evolution."
        )

        st.stop()

    evolution_df = pd.concat(
        evolution_records,
        ignore_index=True
    )

    st.session_state.evolution_df = evolution_df

    # --------------------------------------------------------
    # CLUSTER DISTRIBUTION BY MONTH
    # --------------------------------------------------------

    distribution = (
        evolution_df
        .groupby(
            ["Month", "Cluster"]
        )
        .size()
        .reset_index(
            name="Customers"
        )
    )

    st.subheader(
        "Phân bố nhóm khách hàng theo thời gian"
    )

    fig_evolution = px.line(
        distribution,
        x="Month",
        y="Customers",
        color="Cluster",
        markers=True,
        title="Customer Evolution"
    )

    st.plotly_chart(
        fig_evolution,
        use_container_width=True
    )

    # --------------------------------------------------------
    # CUSTOMER TRANSITIONS
    # --------------------------------------------------------

    st.subheader(
        "Chuyển dịch giữa các nhóm"
    )

    pivot = (
        evolution_df
        .pivot_table(
            index="CustomerID",
            columns="Month",
            values="Cluster",
            aggfunc="first"
        )
    )

    st.dataframe(
        pivot.head(30),
        use_container_width=True
    )

    # --------------------------------------------------------
    # TRANSITION MATRIX
    # --------------------------------------------------------

    if len(selected_months) >= 2:

        month_a = selected_months[-2]
        month_b = selected_months[-1]

        transition_df = evolution_df[
            evolution_df["Month"].isin(
                [month_a, month_b]
            )
        ]

        transition_pivot = (
            transition_df
            .pivot_table(
                index="CustomerID",
                columns="Month",
                values="Cluster",
                aggfunc="first"
            )
            .dropna()
        )

        if (
            month_a in transition_pivot.columns
            and month_b in transition_pivot.columns
        ):

            transition_matrix = pd.crosstab(
                transition_pivot[month_a],
                transition_pivot[month_b]
            )

            st.subheader(
                f"Ma trận chuyển dịch: {month_a} → {month_b}"
            )

            st.dataframe(
                transition_matrix,
                use_container_width=True
            )

            fig_transition = px.imshow(
                transition_matrix,
                text_auto=True,
                title=(
                    f"Cluster Transition: "
                    f"{month_a} → {month_b}"
                )
            )

            st.plotly_chart(
                fig_transition,
                use_container_width=True
            )

    # --------------------------------------------------------
    # RFM TREND
    # --------------------------------------------------------

    st.subheader(
        "Xu hướng RFM theo thời gian"
    )

    rfm_trend = (
        evolution_df
        .groupby("Month")
        .agg(
            Recency=("Recency", "mean"),
            Frequency=("Frequency", "mean"),
            Monetary=("Monetary", "mean")
        )
        .reset_index()
    )

    fig_rfm_trend = go.Figure()

    fig_rfm_trend.add_trace(
        go.Scatter(
            x=rfm_trend["Month"],
            y=rfm_trend["Recency"],
            mode="lines+markers",
            name="Recency"
        )
    )

    fig_rfm_trend.add_trace(
        go.Scatter(
            x=rfm_trend["Month"],
            y=rfm_trend["Frequency"],
            mode="lines+markers",
            name="Frequency"
        )
    )

    fig_rfm_trend.add_trace(
        go.Scatter(
            x=rfm_trend["Month"],
            y=rfm_trend["Monetary"],
            mode="lines+markers",
            name="Monetary"
        )
    )

    fig_rfm_trend.update_layout(
        title="RFM Trend"
    )

    st.plotly_chart(
        fig_rfm_trend,
        use_container_width=True
    )


# ============================================================
# TAB 8 - RESULTS
# ============================================================

with tabs[7]:

    st.header(
        "8. Kết quả hệ thống"
    )

    clustered = st.session_state.clustered_rfm

    if clustered is None:

        st.info(
            "Hoàn thành các bước trước để xem kết quả."
        )

    else:

        st.success(
            "🎯 Hệ thống đã hoàn thành phân nhóm khách hàng."
        )

        result_summary = (
            clustered
            .groupby("Cluster")
            .agg(
                Customers=("CustomerID", "count"),
                Avg_Recency=("Recency", "mean"),
                Avg_Frequency=("Frequency", "mean"),
                Avg_Monetary=("Monetary", "mean"),
                Revenue=("Monetary", "sum")
            )
            .reset_index()
        )

        st.dataframe(
            result_summary.round(2),
            use_container_width=True,
            hide_index=True
        )

        st.markdown(
            """
            ### Pipeline hoàn thành

            ✅ Dữ liệu

            → ✅ Tiền xử lý

            → ✅ RFM

            → ✅ Chuẩn hóa

            → ✅ K-Means

            → ✅ Phân tích khách hàng

            → ✅ Đánh giá & so sánh

            → ✅ Customer Evolution

            ### Key của đề tài

            ⭐ **Customer Evolution – Theo dõi sự thay đổi nhóm khách hàng theo thời gian**
            """
        )

        st.caption(
            "Lưu ý: kết quả phân cụm phụ thuộc vào dữ liệu, "
            "tiền xử lý và giá trị K được lựa chọn."
        )
