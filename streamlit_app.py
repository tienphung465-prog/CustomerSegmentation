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


# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="Customer Evolution",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Segmentation")
st.caption(
    "Customer Evolution – Theo dõi sự thay đổi nhóm khách hàng theo thời gian"
)


# =========================================================
# FUNCTIONS
# =========================================================

def normalize_columns(df):
    """
    Đưa tên cột của Online Retail II về tên chuẩn
    để các bước phía sau sử dụng thống nhất.
    """

    df = df.copy()

    rename_map = {
        "Invoice": "InvoiceNo",
        "Price": "UnitPrice",
        "Customer ID": "CustomerID"
    }

    df.rename(columns=rename_map, inplace=True)

    return df


def preprocess_data(df):

    df = df.copy()

    # Chuẩn hóa tên cột
    df = normalize_columns(df)

    required_columns = [
        "InvoiceNo",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
        "CustomerID"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        return None, missing

    # -----------------------------------------------------
    # Chuyển kiểu dữ liệu
    # -----------------------------------------------------

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"],
        errors="coerce"
    )

    df["Quantity"] = pd.to_numeric(
        df["Quantity"],
        errors="coerce"
    )

    df["UnitPrice"] = pd.to_numeric(
        df["UnitPrice"],
        errors="coerce"
    )

    # -----------------------------------------------------
    # Xóa dữ liệu thiếu
    # -----------------------------------------------------

    df = df.dropna(
        subset=[
            "InvoiceNo",
            "Quantity",
            "InvoiceDate",
            "UnitPrice",
            "CustomerID"
        ]
    )

    # -----------------------------------------------------
    # Loại hóa đơn hủy
    # Online Retail II: Invoice bắt đầu bằng C
    # -----------------------------------------------------

    df["InvoiceNo"] = df["InvoiceNo"].astype(str)

    df = df[
        ~df["InvoiceNo"].str.upper().str.startswith("C")
    ]

    # -----------------------------------------------------
    # Chỉ lấy giao dịch hợp lệ
    # -----------------------------------------------------

    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]

    # -----------------------------------------------------
    # Xóa duplicate
    # -----------------------------------------------------

    df = df.drop_duplicates()

    # -----------------------------------------------------
    # Tính tổng tiền
    # -----------------------------------------------------

    df["TotalAmount"] = (
        df["Quantity"] * df["UnitPrice"]
    )

    return df, []


def create_rfm(df):

    if df.empty:
        return pd.DataFrame()

    reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recency=(
            "InvoiceDate",
            lambda x: (reference_date - x.max()).days
        ),

        Frequency=(
            "InvoiceNo",
            "nunique"
        ),

        Monetary=(
            "TotalAmount",
            "sum"
        )
    ).reset_index()

    rfm = rfm[
        (rfm["Frequency"] > 0) &
        (rfm["Monetary"] > 0)
    ]

    return rfm


def prepare_features(rfm):

    features = rfm[
        [
            "Recency",
            "Frequency",
            "Monetary"
        ]
    ].copy()

    # Log transform giảm ảnh hưởng giá trị quá lớn
    features_log = np.log1p(features)

    scaler = StandardScaler()

    scaled = scaler.fit_transform(features_log)

    scaled_df = pd.DataFrame(
        scaled,
        columns=[
            "Recency",
            "Frequency",
            "Monetary"
        ],
        index=rfm.index
    )

    return scaled_df, scaler


def cluster_kmeans(features, k):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(features)

    return model, labels


def cluster_labels(rfm_cluster):

    summary = rfm_cluster.groupby("Cluster")[
        ["Recency", "Frequency", "Monetary"]
    ].mean()

    result = {}

    # Xếp hạng dựa trên RFM
    # R thấp = tốt
    # F cao = tốt
    # M cao = tốt

    score = (
        -summary["Recency"].rank(pct=True)
        + summary["Frequency"].rank(pct=True)
        + summary["Monetary"].rank(pct=True)
    )

    for cluster in summary.index:

        value = score.loc[cluster]

        if value >= score.quantile(0.75):
            name = "Khách hàng giá trị cao"

        elif value >= score.quantile(0.50):
            name = "Khách hàng tiềm năng"

        elif value >= score.quantile(0.25):
            name = "Khách hàng trung bình"

        else:
            name = "Khách hàng cần quan tâm"

        result[cluster] = name

    return result


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙️ Thiết lập")

uploaded_file = st.sidebar.file_uploader(
    "Chọn file dữ liệu",
    type=["xlsx", "xls", "csv"]
)


# =========================================================
# MAIN
# =========================================================

if uploaded_file is None:

    st.info(
        "👆 Hãy tải file Online Retail II (.xlsx) lên để bắt đầu."
    )

    st.markdown("""
    ### Dataset hỗ trợ

    File Online Retail II có thể có các cột:

    - Invoice / InvoiceNo
    - StockCode
    - Description
    - Quantity
    - InvoiceDate
    - Price / UnitPrice
    - Customer ID / CustomerID
    - Country

    Hệ thống sẽ tự động chuẩn hóa tên cột.
    """)

    st.stop()


# =========================================================
# LOAD FILE
# =========================================================

try:

    if uploaded_file.name.lower().endswith(".csv"):

        df = pd.read_csv(uploaded_file)

        sheet_name = "CSV"

    else:

        excel = pd.ExcelFile(uploaded_file)

        sheets = excel.sheet_names

        selected_sheet = st.sidebar.selectbox(
            "📄 Chọn Sheet",
            sheets
        )

        df = pd.read_excel(
            uploaded_file,
            sheet_name=selected_sheet
        )

        sheet_name = selected_sheet

except Exception as e:

    st.error(f"Không thể đọc file: {e}")

    st.stop()


# =========================================================
# NORMALIZE
# =========================================================

df = normalize_columns(df)


# =========================================================
# CHECK COLUMNS
# =========================================================

required_columns = [
    "InvoiceNo",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.error(
        "Dataset thiếu các cột cần thiết: "
        + ", ".join(missing_columns)
    )

    st.write("### Các cột hiện có:")

    st.write(list(df.columns))

    st.stop()


# =========================================================
# K SELECTION
# =========================================================

st.sidebar.markdown("---")

k = st.sidebar.slider(
    "🔢 Số cụm K",
    min_value=2,
    max_value=10,
    value=4,
    step=1
)


# =========================================================
# TABS
# =========================================================

tabs = st.tabs([
    "📁 Dữ liệu",
    "🧹 Tiền xử lý",
    "📊 RFM",
    "🤖 K-Means",
    "👥 Phân tích khách hàng",
    "📈 Đánh giá & so sánh",
    "🔄 Customer Evolution",
    "🏆 Kết quả"
])


# =========================================================
# TAB 1 - DATA
# =========================================================

with tabs[0]:

    st.subheader("📁 Dữ liệu nghiên cứu")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Số dòng",
        f"{len(df):,}"
    )

    col2.metric(
        "Số cột",
        len(df.columns)
    )

    col3.metric(
        "Số khách hàng",
        f"{df['CustomerID'].nunique():,}"
    )

    col4.metric(
        "Sheet",
        sheet_name
    )

    st.write("### Các thuộc tính")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    st.write("### Tên cột")

    st.write(list(df.columns))


# =========================================================
# TAB 2 - PREPROCESSING
# =========================================================

with tabs[1]:

    st.subheader("🧹 Tiền xử lý dữ liệu")

    processed_df, errors = preprocess_data(df)

    if errors:

        st.error(
            "Không thể xử lý dữ liệu. Thiếu: "
            + ", ".join(errors)
        )

        st.stop()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Trước xử lý",
        f"{len(df):,}"
    )

    col2.metric(
        "Sau xử lý",
        f"{len(processed_df):,}"
    )

    col3.metric(
        "Đã loại",
        f"{len(df) - len(processed_df):,}"
    )

    col4.metric(
        "Khách hàng",
        f"{processed_df['CustomerID'].nunique():,}"
    )

    st.success(
        "✓ Tiền xử lý hoàn tất"
    )

    st.write("### Dữ liệu sau xử lý")

    st.dataframe(
        processed_df.head(20),
        use_container_width=True
    )

    st.write("### Các bước đã thực hiện")

    st.markdown("""
    - Chuyển kiểu dữ liệu
    - Xử lý giá trị thiếu
    - Loại hóa đơn hủy
    - Loại Quantity ≤ 0
    - Loại UnitPrice ≤ 0
    - Loại dữ liệu trùng
    - Tính TotalAmount
    """)


# =========================================================
# RFM
# =========================================================

rfm = create_rfm(processed_df)


# =========================================================
# TAB 3 - RFM
# =========================================================

with tabs[2]:

    st.subheader("📊 Mô hình RFM")

    if rfm.empty:

        st.warning("Không tạo được RFM.")

    else:

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Customers",
            f"{len(rfm):,}"
        )

        col2.metric(
            "Frequency TB",
            f"{rfm['Frequency'].mean():.2f}"
        )

        col3.metric(
            "Monetary TB",
            f"{rfm['Monetary'].mean():,.2f}"
        )

        st.write("### Bảng RFM")

        st.dataframe(
            rfm.head(30),
            use_container_width=True
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            fig = px.histogram(
                rfm,
                x="Recency",
                title="Recency"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with col2:

            fig = px.histogram(
                rfm,
                x="Frequency",
                title="Frequency"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with col3:

            fig = px.histogram(
                rfm,
                x="Monetary",
                title="Monetary"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# =========================================================
# PREPARE FEATURES
# =========================================================

if not rfm.empty:

    features, scaler = prepare_features(rfm)

    model, labels = cluster_kmeans(
        features,
        k
    )

    rfm_clustered = rfm.copy()

    rfm_clustered["Cluster"] = labels


# =========================================================
# TAB 4 - KMEANS
# =========================================================

with tabs[3]:

    st.subheader("🤖 K-Means Clustering")

    if rfm.empty:

        st.warning("Không có dữ liệu RFM.")

    else:

        st.info(
            f"K-Means đang chia khách hàng thành **{k} cụm**."
        )

        # -------------------------------------------------
        # ELBOW
        # -------------------------------------------------

        st.write("### Elbow Method")

        inertias = []

        k_values = range(2, 11)

        for current_k in k_values:

            temp_model = KMeans(
                n_clusters=current_k,
                random_state=42,
                n_init=10
            )

            temp_model.fit(features)

            inertias.append(
                temp_model.inertia_
            )

        elbow_df = pd.DataFrame({
            "K": list(k_values),
            "Inertia": inertias
        })

        fig = px.line(
            elbow_df,
            x="K",
            y="Inertia",
            markers=True,
            title="Elbow Method"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # -------------------------------------------------
        # CLUSTER DISTRIBUTION
        # -------------------------------------------------

        st.write("### Phân bố cụm")

        cluster_count = (
            rfm_clustered["Cluster"]
            .value_counts()
            .sort_index()
            .reset_index()
        )

        cluster_count.columns = [
            "Cluster",
            "Customers"
        ]

        fig = px.bar(
            cluster_count,
            x="Cluster",
            y="Customers",
            title=f"Phân bố {k} cụm khách hàng"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # -------------------------------------------------
        # 3D
        # -------------------------------------------------

        st.write("### Không gian RFM")

        fig = px.scatter_3d(
            rfm_clustered,
            x="Recency",
            y="Frequency",
            z="Monetary",
            color="Cluster",
            hover_data=["CustomerID"],
            title="K-Means Customer Segmentation"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# =========================================================
# TAB 5 - CUSTOMER ANALYSIS
# =========================================================

with tabs[4]:

    st.subheader("👥 Phân tích khách hàng")

    if rfm.empty:

        st.warning("Không có dữ liệu.")

    else:

        summary = (
            rfm_clustered
            .groupby("Cluster")
            [
                [
                    "Recency",
                    "Frequency",
                    "Monetary"
                ]
            ]
            .mean()
            .round(2)
        )

        summary["Customers"] = (
            rfm_clustered["Cluster"]
            .value_counts()
            .sort_index()
        )

        labels_dict = cluster_labels(
            rfm_clustered
        )

        summary["Nhóm khách hàng"] = [
            labels_dict.get(
                cluster,
                "Chưa xác định"
            )
            for cluster in summary.index
        ]

        st.dataframe(
            summary,
            use_container_width=True
        )

        st.write("### Diễn giải")

        for cluster, row in summary.iterrows():

            st.markdown(
                f"""
                **Cluster {cluster} – {row['Nhóm khách hàng']}**

                - Recency trung bình: {row['Recency']:.2f}
                - Frequency trung bình: {row['Frequency']:.2f}
                - Monetary trung bình: {row['Monetary']:.2f}
                - Số khách hàng: {int(row['Customers']):,}
                """
            )


# =========================================================
# TAB 6 - EVALUATION
# =========================================================

with tabs[5]:

    st.subheader("📈 Đánh giá & so sánh")

    if rfm.empty:

        st.warning("Không có dữ liệu.")

    else:

        # KMeans
        kmeans_silhouette = silhouette_score(
            features,
            rfm_clustered["Cluster"]
        )

        kmeans_db = davies_bouldin_score(
            features,
            rfm_clustered["Cluster"]
        )

        kmeans_ch = calinski_harabasz_score(
            features,
            rfm_clustered["Cluster"]
        )

        # Agglomerative
        agg_model = AgglomerativeClustering(
            n_clusters=k
        )

        agg_labels = agg_model.fit_predict(
            features
        )

        agg_silhouette = silhouette_score(
            features,
            agg_labels
        )

        agg_db = davies_bouldin_score(
            features,
            agg_labels
        )

        agg_ch = calinski_harabasz_score(
            features,
            agg_labels
        )

        comparison = pd.DataFrame({

            "Phương pháp": [
                "K-Means",
                "Agglomerative"
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

        st.dataframe(
            comparison.round(4),
            use_container_width=True
        )

        st.markdown("""
        ### Ý nghĩa các chỉ số

        **Silhouette**
        - Càng cao càng tốt.

        **Davies-Bouldin**
        - Càng thấp càng tốt.

        **Calinski-Harabasz**
        - Càng cao càng tốt.

        Các chỉ số được sử dụng để đánh giá chất lượng phân cụm,
        không chỉ dựa vào trực quan hóa.
        """)


# =========================================================
# TAB 7 - CUSTOMER EVOLUTION
# =========================================================

with tabs[6]:

    st.subheader(
        "🔄 Customer Evolution"
    )

    st.markdown("""
    ### Theo dõi sự thay đổi nhóm khách hàng theo thời gian

    Đây là điểm khác biệt chính của đề tài.

    Thay vì chỉ phân nhóm khách hàng tại một thời điểm,
    hệ thống theo dõi khách hàng chuyển từ Cluster này
    sang Cluster khác qua từng tháng.
    """)

    if processed_df.empty:

        st.warning(
            "Không có dữ liệu để thực hiện Customer Evolution."
        )

    else:

        evolution_df = processed_df.copy()

        evolution_df["Month"] = (
            evolution_df["InvoiceDate"]
            .dt.to_period("M")
            .astype(str)
        )

        months = sorted(
            evolution_df["Month"].unique()
        )

        if len(months) < 2:

            st.warning(
                "Dataset cần ít nhất 2 tháng để theo dõi Evolution."
            )

        else:

            # -------------------------------------------------
            # GLOBAL MODEL
            # -------------------------------------------------

            global_rfm = create_rfm(
                evolution_df
            )

            global_features, global_scaler = (
                prepare_features(global_rfm)
            )

            global_model = KMeans(
                n_clusters=k,
                random_state=42,
                n_init=10
            )

            global_model.fit(
                global_features
            )

            # -------------------------------------------------
            # MONTHLY RFM
            # -------------------------------------------------

            monthly_rfm_list = []

            for month in months:

                month_df = evolution_df[
                    evolution_df["Month"] == month
                ].copy()

                if month_df.empty:
                    continue

                reference_date = (
                    month_df["InvoiceDate"].max()
                    + pd.Timedelta(days=1)
                )

                monthly_rfm = (
                    month_df
                    .groupby("CustomerID")
                    .agg(
                        Recency=(
                            "InvoiceDate",
                            lambda x:
                            (
                                reference_date - x.max()
                            ).days
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
                    (monthly_rfm["Frequency"] > 0) &
                    (monthly_rfm["Monetary"] > 0)
                ]

                if monthly_rfm.empty:
                    continue

                monthly_features = (
                    np.log1p(
                        monthly_rfm[
                            [
                                "Recency",
                                "Frequency",
                                "Monetary"
                            ]
                        ]
                    )
                )

                monthly_scaled = (
                    global_scaler.transform(
                        monthly_features
                    )
                )

                monthly_rfm["Cluster"] = (
                    global_model.predict(
                        monthly_scaled
                    )
                )

                monthly_rfm["Month"] = month

                monthly_rfm_list.append(
                    monthly_rfm
                )

            if monthly_rfm_list:

                evolution = pd.concat(
                    monthly_rfm_list,
                    ignore_index=True
                )

                # -------------------------------------------------
                # MONTHLY DISTRIBUTION
                # -------------------------------------------------

                st.write(
                    "### Phân bố khách hàng theo tháng"
                )

                monthly_count = (
                    evolution
                    .groupby(
                        ["Month", "Cluster"]
                    )
                    .size()
                    .reset_index(
                        name="Customers"
                    )
                )

                fig = px.line(
                    monthly_count,
                    x="Month",
                    y="Customers",
                    color="Cluster",
                    markers=True,
                    title="Customer Evolution"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                # -------------------------------------------------
                # CUSTOMER CLUSTER HISTORY
                # -------------------------------------------------

                st.write(
                    "### Lịch sử Cluster của khách hàng"
                )

                pivot = evolution.pivot_table(
                    index="CustomerID",
                    columns="Month",
                    values="Cluster",
                    aggfunc="first"
                )

                st.dataframe(
                    pivot.head(50),
                    use_container_width=True
                )

                # -------------------------------------------------
                # MONTH SELECTION
                # -------------------------------------------------

                st.write(
                    "### Phân tích chuyển cụm"
                )

                if len(months) >= 2:

                    col1, col2 = st.columns(2)

                    with col1:

                        month_from = st.selectbox(
                            "Tháng trước",
                            months[:-1]
                        )

                    valid_next_months = [
                        m for m in months
                        if m > month_from
                    ]

                    with col2:

                        month_to = st.selectbox(
                            "Tháng sau",
                            valid_next_months
                        )

                    from_data = evolution[
                        evolution["Month"] == month_from
                    ][
                        [
                            "CustomerID",
                            "Cluster"
                        ]
                    ].rename(
                        columns={
                            "Cluster": "FromCluster"
                        }
                    )

                    to_data = evolution[
                        evolution["Month"] == month_to
                    ][
                        [
                            "CustomerID",
                            "Cluster"
                        ]
                    ].rename(
                        columns={
                            "Cluster": "ToCluster"
                        }
                    )

                    transitions = from_data.merge(
                        to_data,
                        on="CustomerID"
                    )

                    transition_matrix = pd.crosstab(
                        transitions["FromCluster"],
                        transitions["ToCluster"]
                    )

                    st.write(
                        f"**{month_from} → {month_to}**"
                    )

                    st.dataframe(
                        transition_matrix,
                        use_container_width=True
                    )

                    # -------------------------------------------------
                    # TRANSITION HEATMAP
                    # -------------------------------------------------

                    fig = px.imshow(
                        transition_matrix,
                        text_auto=True,
                        title=(
                            f"Ma trận chuyển cụm: "
                            f"{month_from} → {month_to}"
                        )
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                # -------------------------------------------------
                # RFM TREND
                # -------------------------------------------------

                st.write(
                    "### Xu hướng RFM theo thời gian"
                )

                rfm_monthly = (
                    evolution
                    .groupby("Month")
                    [
                        [
                            "Recency",
                            "Frequency",
                            "Monetary"
                        ]
                    ]
                    .mean()
                    .reset_index()
                )

                fig = px.line(
                    rfm_monthly,
                    x="Month",
                    y=[
                        "Recency",
                        "Frequency",
                        "Monetary"
                    ],
                    markers=True,
                    title="RFM Trend"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )


# =========================================================
# TAB 8 - RESULTS
# =========================================================

with tabs[7]:

    st.subheader("🏆 Kết quả tổng hợp")

    if rfm.empty:

        st.warning("Chưa có kết quả.")

    else:

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Khách hàng",
            f"{len(rfm_clustered):,}"
        )

        col2.metric(
            "Số cụm K",
            k
        )

        col3.metric(
            "Silhouette",
            f"{kmeans_silhouette:.3f}"
        )

        col4.metric(
            "Doanh thu",
            f"{processed_df['TotalAmount'].sum():,.0f}"
        )

        st.success(
            "Phân tích Customer Segmentation đã hoàn tất."
        )

        st.markdown("""
        ### Pipeline

        **Dữ liệu**
        ↓

        **Tiền xử lý**
        ↓

        **RFM**
        ↓

        **Chuẩn hóa**
        ↓

        **K-Means**
        ↓

        **Phân tích khách hàng**
        ↓

        **Đánh giá & so sánh**
        ↓

        **Customer Evolution**
        ↓

        **Streamlit**
        """)

        st.markdown("""
        ### Key của đề tài

        ⭐ **Customer Evolution – Theo dõi sự thay đổi nhóm khách hàng theo thời gian**
        """)
