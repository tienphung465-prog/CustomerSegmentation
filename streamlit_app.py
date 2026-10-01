import streamlit as st
import pandas as pd

# =========================
# CẤU HÌNH TRANG
# =========================
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="📊",
    layout="wide"
)

# =========================
# HEADER
# =========================
st.title("Customer Segmentation")

st.subheader(
    "Customer Evolution – Theo dõi sự thay đổi nhóm khách hàng theo thời gian"
)

st.info(
    "Bước 1 – Dữ liệu: tải lên bộ dữ liệu Online Retail II "
    "để kiểm tra và khám phá dữ liệu."
)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("📂 Dữ liệu")

    uploaded_file = st.file_uploader(
        "Chọn file Online Retail II",
        type=["xlsx", "xls", "csv"]
    )

    st.divider()

    st.caption("Phiên bản: Bước 1 – Dữ liệu")


# =========================
# CHƯA CÓ FILE
# =========================
if uploaded_file is None:

    st.warning(
        "Chưa có dữ liệu. Hãy tải file Online Retail II ở thanh bên trái."
    )

    st.markdown("""
    ### File cần chuẩn bị

    Dữ liệu nên có các thuộc tính:

    - `InvoiceNo`
    - `StockCode`
    - `Description`
    - `Quantity`
    - `InvoiceDate`
    - `UnitPrice`
    - `CustomerID`
    - `Country`
    """)

    st.stop()


# =========================
# ĐỌC FILE
# =========================
try:

    file_name = uploaded_file.name

    if file_name.lower().endswith(".csv"):

        df = pd.read_csv(uploaded_file)

    else:

        excel = pd.ExcelFile(uploaded_file)

        sheet_names = excel.sheet_names

        # Nếu có nhiều sheet
        if len(sheet_names) > 1:

            selected_sheet = st.selectbox(
                "Chọn sheet dữ liệu",
                sheet_names
            )

        else:

            selected_sheet = sheet_names[0]

        df = pd.read_excel(
            uploaded_file,
            sheet_name=selected_sheet
        )

except Exception as e:

    st.error(f"Không thể đọc file: {e}")

    st.stop()


# =========================
# THÔNG BÁO LOAD THÀNH CÔNG
# =========================
st.success(
    f"Đã tải dữ liệu thành công: **{file_name}**"
)


# =========================
# THỐNG KÊ CƠ BẢN
# =========================
rows, cols = df.shape

missing_total = int(
    df.isna().sum().sum()
)

duplicate_total = int(
    df.duplicated().sum()
)


# =========================
# 4 CARD THỐNG KÊ
# =========================
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Số dòng",
    f"{rows:,}"
)

col2.metric(
    "Số cột",
    f"{cols:,}"
)

col3.metric(
    "Giá trị thiếu",
    f"{missing_total:,}"
)

col4.metric(
    "Dòng trùng",
    f"{duplicate_total:,}"
)


# =========================
# CÁC TAB
# =========================
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "👀 Xem dữ liệu",
        "🧩 Thuộc tính",
        "⚠️ Giá trị thiếu",
        "🔎 Thống kê"
    ]
)


# =========================
# TAB 1: XEM DỮ LIỆU
# =========================
with tab1:

    st.subheader("Xem trước dữ liệu")

    number_rows = st.slider(
        "Số dòng hiển thị",
        min_value=5,
        max_value=100,
        value=10
    )

    st.dataframe(
        df.head(number_rows),
        use_container_width=True
    )


# =========================
# TAB 2: THUỘC TÍNH
# =========================
with tab2:

    st.subheader("Các thuộc tính chính")

    info_df = pd.DataFrame({

        "Thuộc tính": df.columns,

        "Kiểu dữ liệu": [
            str(df[column].dtype)
            for column in df.columns
        ],

        "Số giá trị khác nhau": [
            df[column].nunique(dropna=True)
            for column in df.columns
        ]

    })

    st.dataframe(
        info_df,
        use_container_width=True
    )


# =========================
# TAB 3: GIÁ TRỊ THIẾU
# =========================
with tab3:

    st.subheader("Kiểm tra giá trị thiếu")

    missing_df = pd.DataFrame({

        "Thuộc tính": df.columns,

        "Số giá trị thiếu": [
            int(df[column].isna().sum())
            for column in df.columns
        ],

        "Tỷ lệ thiếu (%)": [
            round(
                float(df[column].isna().mean() * 100),
                2
            )
            for column in df.columns
        ]

    })

    missing_df = missing_df.sort_values(
        "Số giá trị thiếu",
        ascending=False
    )

    st.dataframe(
        missing_df,
        use_container_width=True
    )


# =========================
# TAB 4: THỐNG KÊ
# =========================
with tab4:

    st.subheader("Thống kê dữ liệu")

    st.dataframe(
        df.describe(include="all").T,
        use_container_width=True
    )


# =========================
# KIỂM TRA CẤU TRÚC DATASET
# =========================
st.divider()

st.subheader(
    "🔗 Kiểm tra cấu trúc Online Retail II"
)


expected_columns = [

    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country"

]


found_columns = [

    column
    for column in expected_columns
    if column in df.columns

]


missing_columns = [

    column
    for column in expected_columns
    if column not in df.columns

]


# =========================
# KẾT QUẢ KIỂM TRA
# =========================
if len(found_columns) == len(expected_columns):

    st.success(
        "Đã tìm thấy đầy đủ 8 thuộc tính chính "
        "của Online Retail II."
    )

else:

    st.warning(
        f"Đã tìm thấy "
        f"{len(found_columns)}/{len(expected_columns)} "
        f"thuộc tính chính."
    )

    if missing_columns:

        st.write(
            "Chưa tìm thấy:",
            ", ".join(missing_columns)
        )


# =========================
# BƯỚC TIẾP THEO
# =========================
st.divider()

st.subheader("➡️ Bước tiếp theo")

st.write(
    "Sau khi xác nhận dữ liệu, bước tiếp theo là "
    "**Tiền xử lý**: kiểm tra dữ liệu không hợp lệ, "
    "giao dịch hủy, giá trị thiếu, dữ liệu trùng lặp "
    "và tạo các biến cần thiết cho RFM."
)
