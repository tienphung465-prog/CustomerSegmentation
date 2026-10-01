profile_table = profiles[["Nhóm khách hàng", "Customers", "Recency", "Frequency", "Monetary"]].copy()
profile_table.columns = ["Nhóm khách hàng", "Số khách hàng", "R trung bình", "F trung bình", "M trung bình"]
st.dataframe(profile_table.round(2), hide_index=True, use_container_width=True)

section("Thông tin nổi bật")
largest = profiles.loc[profiles["Customers"].idxmax()]
recent = profiles.loc[profiles["Recency"].idxmin()]
valuable = profiles.loc[profiles["Monetary"].idxmax()]
a, b, c = st.columns(3)
with a:
    metric_card("Nhóm đông khách nhất", largest["Nhóm khách hàng"])
with b:
    metric_card("Nhóm có R trung bình thấp nhất", recent["Nhóm khách hàng"])
with c:
    metric_card("Nhóm có M trung bình cao nhất", valuable["Nhóm khách hàng"])

section("Customer Evolution")
st.markdown(
    '<div class="hint"><b>Theo dõi sự thay đổi nhóm khách hàng theo thời gian</b><br>'
    'Sử dụng RFM tích lũy và cùng một mô hình K-Means. Biểu đồ chỉ tính khách hàng '
    'có phát sinh giao dịch trong tháng được hiển thị.</div>',
    unsafe_allow_html=True,
)
counts = evolution.groupby(["Tháng", "Nhóm khách hàng"]).size().reset_index(name="Số khách hàng")
fig = px.line(
    counts, x="Tháng", y="Số khách hàng", color="Nhóm khách hàng",
    markers=True, color_discrete_sequence=PALETTE,
)
st.plotly_chart(format_chart(fig, 420), use_container_width=True)
if not transition_pairs.empty:
    moved = int((transition_pairs["PreviousCluster"] != transition_pairs["Cluster"]).sum())
    a, b = st.columns(2)
    with a:
        metric_card("Lượt chuyển nhóm khách hàng", num(moved), "Giữa hai tháng có giao dịch liên tiếp")
    with b:
        metric_card("Lượt đối chiếu", num(len(transition_pairs)), "Không phải số khách hàng riêng biệt")
else:
    st.info("Chưa đủ giao dịch qua các tháng liên tiếp để tính số lượt chuyển nhóm.")


# ============================================================
# 9. CHI TIẾT PHÂN TÍCH: ẨN SAU EXPANDER
# ============================================================
section("Chi tiết phân tích")
st.caption("Mở từng mục để xem quy trình; Dashboard phía trên luôn hiển thị kết quả chính.")

with st.expander("1. Dữ liệu"):
    st.write("Tên file:", uploaded.name)
    st.write("Sheet:", ", ".join(selected_sheets))
    st.dataframe(raw.head(50), hide_index=True, use_container_width=True)

with st.expander("2. Tiền xử lý"):
    st.dataframe(pd.DataFrame(list(statistics.items()), columns=["Tiêu chí", "Số dòng"]), hide_index=True, use_container_width=True)
    st.dataframe(cleaned.head(30), hide_index=True, use_container_width=True)

with st.expander("3. RFM"):
    st.markdown("**Recency:** số ngày từ lần mua gần nhất. **Frequency:** số hóa đơn. **Monetary:** tổng giá trị mua.")
    st.dataframe(rfm.head(100), hide_index=True, use_container_width=True)
    st.download_button("Tải bảng RFM", rfm.to_csv(index=False).encode("utf-8-sig"), "rfm.csv", "text/csv")

with st.expander("4. Chuẩn hóa"):
    st.write("Áp dụng log1p và StandardScaler trước K-Means.")
    st.dataframe(pd.DataFrame(X_scaled[:40], columns=["R chuẩn hóa", "F chuẩn hóa", "M chuẩn hóa"]).round(3), hide_index=True)

with st.expander("5. K-Means"):
    st.write("Số nhóm khách hàng K:", k)
    if st.button("Tính biểu đồ Elbow", key="elbow"):
        elbow = elbow_table(X_scaled)
        if not elbow.empty:
            st.plotly_chart(format_chart(px.line(elbow, x="K", y="Inertia", markers=True)), use_container_width=True)
        else:
            st.info("Dữ liệu không đủ đa dạng để tính Elbow.")
    sample = grouped.sample(min(len(grouped), 1200), random_state=42).copy()
    sample["Nhóm khách hàng"] = "Nhóm khách hàng " + sample["Cluster"].astype(str)
    chart3d = px.scatter_3d(sample, x="Recency", y="Frequency", z="Monetary", color="Nhóm khách hàng", color_discrete_sequence=PALETTE)
    st.plotly_chart(format_chart(chart3d, 460), use_container_width=True)

with st.expander("6. Phân tích nhóm khách hàng"):
    st.dataframe(profile_table.round(2), hide_index=True, use_container_width=True)
    result_download = grouped.copy()
    result_download["Nhóm khách hàng"] = "Nhóm khách hàng " + result_download["Cluster"].astype(str)
    st.download_button("Tải kết quả phân nhóm", result_download.to_csv(index=False).encode("utf-8-sig"), "customer_groups.csv", "text/csv")

with st.expander("7. Đánh giá và so sánh thuật toán"):
    st.write("So sánh K-Means và Agglomerative trên cùng một mẫu tối đa 1.200 khách hàng.")
    st.caption("Silhouette và Calinski-Harabasz: càng cao thường càng tốt. Davies-Bouldin: càng thấp thường càng tốt.")
    if st.button("Chạy so sánh", key="compare"):
        with st.spinner("Đang tính chỉ số đánh giá..."):
            scores = compare_models(X_scaled, k)
        if scores.empty:
            st.info("Không đủ dữ liệu phù hợp để đánh giá.")
        else:
            st.dataframe(scores.round(4), hide_index=True, use_container_width=True)

with st.expander("8. Customer Evolution"):
    st.subheader("Ma trận chuyển nhóm khách hàng")
    if transition_matrix.empty:
        st.info("Chưa có dữ liệu chuyển nhóm giữa hai tháng liền nhau.")
    else:
        matrix = transition_matrix.reindex(index=range(k), columns=range(k), fill_value=0)
        matrix.index = [f"Nhóm khách hàng {i}" for i in matrix.index]
        matrix.columns = [f"Nhóm khách hàng {i}" for i in matrix.columns]
        heatmap = px.imshow(matrix, text_auto=True, aspect="auto", color_continuous_scale="Blues",
                            labels={"x": "Nhóm tháng sau", "y": "Nhóm tháng trước", "color": "Lượt"})
        st.plotly_chart(format_chart(heatmap, 430), use_container_width=True)
    st.subheader("Lịch sử từng khách hàng")
    customer = st.selectbox("Chọn khách hàng", sorted(evolution["CustomerID"].unique().tolist()))
    history = evolution.loc[evolution["CustomerID"] == customer].sort_values("Month")
    st.dataframe(history[["Tháng", "Nhóm khách hàng", "Recency", "Frequency", "Monetary"]], hide_index=True, use_container_width=True)
    if len(history) > 1:
        path = px.line(history, x="Tháng", y="Cluster", markers=True)
        path.update_yaxes(tickvals=list(range(k)), ticktext=[f"Nhóm khách hàng {i}" for i in range(k)])
        st.plotly_chart(format_chart(path), use_container_width=True)
    evo_csv = evolution[["CustomerID", "Tháng", "Nhóm khách hàng", "Recency", "Frequency", "Monetary"]].to_csv(index=False).encode("utf-8-sig")
    st.download_button("Tải Customer Evolution", evo_csv, "customer_evolution.csv", "text/csv")

with st.expander("9. Streamlit và kết quả"):
    st.write("Khách hàng hợp lệ:", num(len(rfm)))
    st.write("Số nhóm khách hàng:", k)
    st.write("Thời gian dữ liệu:", cleaned["InvoiceDate"].min().strftime("%d/%m/%Y"), "đến", cleaned["InvoiceDate"].max().strftime("%d/%m/%Y"))
    st.caption("Dữ liệu → Tiền xử lý → RFM → Chuẩn hóa → K-Means → Phân tích khách hàng → Đánh giá & so sánh → Customer Evolution → Streamlit/Kết quả")

st.divider()
st.caption("CUSTOMER INTELLIGENCE · RFM · K-Means · Customer Evolution")
