# Bộ câu hỏi phản biện và trả lời — Nhóm 01 (608807)

Tài liệu chuẩn bị cho buổi bảo vệ. Mỗi câu gồm: **ý trả lời cốt lõi** (1–2 câu) và **dẫn chứng** trong báo cáo.
Nguyên tắc: trả lời đúng phạm vi đã làm, không suy diễn quá mức; nếu chưa biết thì nói rõ "nhóm chưa khảo sát".

---

## A. Bài toán & dữ liệu

**1. Vì sao gọi đây là bài toán hồi quy đa đầu ra mà không phải hai bài toán độc lập?**
Vì hai đầu ra có tương quan rất cao (r = 0,976) và cùng đơn vị kWh/m², cùng chia sẻ cơ chế vật lý. Nếu coi là
hai bài toán độc lập, ta bỏ qua thông tin này và không thể so sánh các chiến lược học chung/học riêng.
Nhóm vẫn báo cáo **metric riêng cho từng đầu ra** để không che mất khác biệt (Y2 khó hơn Y1).
→ Mục 1.1, 3.4, Bảng 10.

**2. 768 mẫu có đủ để kết luận không? Có cần thu thập thêm không?**
Đủ để huấn luyện và CV, nhưng **không đủ để ngoại suy**. Learning Curve bão hoà khi dùng ~60% dữ liệu train,
nghĩa là thêm mẫu **cùng 12 thiết kế** không cải thiện. Nút thắt là **đa dạng hình học**, không phải số mẫu.
→ Mục 4.3, 4.7.

**3. Vì sao X1, X2, X3, X4, X5 "trông liên tục" nhưng thực ra rời rạc?**
Vì chúng được sinh từ 12 thiết kế hình học cố định; X1 có 12 giá trị, X4 có 4 giá trị, X5 chỉ có 2 giá trị
(3,5 m và 7,0 m). Đây là lý do mô hình không thể nội suy theo chiều cao và là gốc của vấn đề ở mục 4.7.
→ Mục 2.1, Bảng 2.2.

**4. Vì sao X8 = 0 chỉ có 48 mẫu và điều đó ảnh hưởng gì?**
X8 = 0 tương ứng **không có kính** (X7 = 0), không phải một kiểu phân bố kính. Vì vậy khi phân tích ANOVA
cho X8, nhóm **loại 48 mẫu này** để không trộn hiệu ứng "không kính" vào hiệu ứng "kiểu phân bố kính".
→ Mục 2.4.

**5. Đa cộng tuyến hoàn hảo (VIF ~10^15) sao vẫn dự đoán được?**
Vì đa cộng tuyến làm **hệ số** không xác định duy nhất, nhưng không nhất thiết làm **dự đoán** sai. Ridge
"san phẳng" nghiệm về nghiệm có chuẩn nhỏ nhất; dự đoán vẫn ổn định. Nhóm không diễn giải hệ số, chỉ diễn
giải qua độ quan trọng.
→ Mục 2.2.

---

## B. Tiền xử lý & feature engineering

**6. Vì sao dùng `ColumnTransformer` thay vì scale tay trước khi chia dữ liệu?**
Vì mọi bước học tham số (median, mean/std, one-hot vocabulary) phải được fit **trong** fold huấn luyện.
Scale tay trước khi chia → rò rỉ thông tin từ test vào train, điểm số bị lạc quan một cách âm thầm.
→ Mục 3.2, 3.3.

**7. Vì sao mã hoá X6 bằng sin/cos mà không one-hot?**
Hướng nhà có tính **chu kỳ** (4 hướng cách nhau 90°). One-hot coi 4 hướng độc lập; sin/cos giữ được quan hệ
"gần nhau" giữa các hướng và chỉ tốn 2 chiều. Thực nghiệm cho thấy chênh lệch < 0,001 R², nên nhóm chọn theo
**ngữ nghĩa**, không theo điểm số.
→ Mục 2.6, Bảng 7.

**8. Feature engineering có thực sự giúp không, hay chỉ làm phức tạp?**
Kết quả **không đồng nhất**: giúp mô hình tuyến tính rất nhiều (+0,072), gần như không đổi với Ridge, và
**giảm nhẹ** với cây (−0,004 đến −0,005), k-NN (−0,035). Nhóm báo cáo đúng như vậy và giải thích: cây tự tạo
được tương tác nên thêm biến chỉ làm loãng lựa chọn. Nhóm vẫn giữ F1 vì pipeline thống nhất và khả năng
diễn giải, và ghi rõ đây là đánh đổi.
→ Mục 2.6, Bảng 8, 9.

**9. Vậy nếu biết F0 tốt hơn cho XGBoost, sao không dùng F0?**
Vì tiêu chí của nhóm là **một pipeline thống nhất**, và chênh lệch chỉ ~0,004 (trong vùng nhiễu). Đây là
hướng phát triển: tối ưu bộ đặc trưng theo từng họ mô hình. Nhóm không che giấu kết quả bất lợi này.
→ Mục 2.6, 5.2.

**10. Vì sao bỏ ngoại lai lại không cải thiện?**
Vì theo IQR và |z| > 3 **không có ngoại lai nào**; Isolation Forest chỉ gắn cờ nhóm X7 = 0 (nhà không kính)
— một vùng thiết kế **hợp lệ**, không phải nhiễu. Loại bỏ làm mất 16 mẫu và CV giảm 0,0012.
→ Mục 2.5, Bảng 5.

---

## C. Mô hình & đánh giá

**11. Vì sao chọn XGBoost mà không phải Stacking, khi Stacking có CV R² cao hơn ở bảng CV?**
Hai lý do. (i) Ở bảng CV **trước tuning**, Stacking hơn XGBoost mặc định 0,0003 — nhỏ hơn độ lệch chuẩn
giữa các fold, không phải khác biệt thực chất. (ii) Stacking **không được tuning** (chi phí), còn XGBoost
sau tuning đạt 0,9929, vượt Stacking 0,9915. Nhóm đặt quy tắc tie-break **trước** khi xem test: chênh < 0,002
thì chọn mô hình đơn giản hơn, nhanh hơn. Đây là hạn chế được ghi rõ.
→ Mục 4.1, 4.2, 5.2.

**12. Vì sao CV R² của XGBoost trong bảng CV là 0,9912 nhưng trong bảng cuối là 0,9929?**
Vì bảng CV dùng **siêu tham số mặc định**, còn bảng cuối dùng **sau tuning**. Ngoài ra cách lấy trung bình
hơi khác (trung bình R² từng đầu ra so với tính R² trên cả hai cột dự đoán). Nhóm ghi chú rõ dưới bảng 14.
→ Mục 4.1, 4.2.

**13. R² = 0,9935 có phải là "quá tốt để tin"?**
Không, vì dữ liệu là **mô phỏng** (Ecotect) nên quan hệ rất trơn, và 768 mẫu chỉ có 12 thiết kế. Nhưng
chính vì vậy nhóm nhấn mạnh **RMSE** (0,706 kWh/m²) và **kiểm tra GroupKFold** — nơi R² sụp mạnh. R² cao
không đồng nghĩa mô hình dùng được cho công trình có hình dạng mới.
→ Mục 4.4, 4.7.

**14. Vì sao dùng R², RMSE, MAE, MAPE mà không chỉ một metric?**
R² chuẩn hoá nhưng không phản ánh sai số tuyệt đối; RMSE cùng đơn vị nhưng phạt nặng lỗi lớn; MAE ít nhạy
ngoại lai; MAPE dễ hiểu nhưng bất ổn khi giá trị nhỏ (Y1 min = 6,01). Nhóm không dùng MAPE để chọn mô hình.
→ Mục 3.7.

**15. Vì sao K = 5 mà không phải 10?**
Chi phí của XGBoost/Stacking cao; K = 5 đủ để ước lượng độ lệch chuẩn và phù hợp giới hạn thời gian.
Hạn chế: với 5 fold, Wilcoxon không đủ lực (p nhỏ nhất 0,0625), nên nhóm **không** khẳng định khác biệt
có ý nghĩa thống kê.
→ Mục 3.6, 4.2.

**16. Làm sao biết mô hình không bị overfit?**
Ba bằng chứng: (i) Validation Curve cho thấy Ridge train ≈ validation; (ii) Learning Curve của XGBoost bão
hoà, khoảng hở nhỏ; (iii) CV và test rất sát nhau (chênh < 0,01). Ngược lại, Random Forest có khoảng hở
train–validation lớn → có dấu hiệu overfit và bị xếp sau.
→ Mục 4.3, Hình 7, 8.

---

## D. Diễn giải & hạn chế

**17. Vì sao ΔR² của X1 lớn hơn 1? Có phải lỗi không?**
Không. Permutation Importance đo **mức giảm R²**; nếu xáo trộn X1 làm R² từ 0,99 xuống −0,8 thì mức giảm
là 1,8. Điều này xảy ra vì X1 là biến chia chính ở hầu hết các nút. Nó **không** có nghĩa X1 là biến duy
nhất quan trọng — X2/X4/X5 chia sẻ thông tin (substitution effect).
→ Mục 4.6, Bảng 18.

**18. Vì sao X5 tương quan 0,89 với Y1 nhưng permutation importance ≈ 0?**
Vì hiệu ứng thay thế: khi X5 bị xáo trộn, mô hình vẫn dùng X1/X2 (tương quan chặt với X5) nên không mất
năng lực. Đây là lỗi diễn giải phổ biến nhất của permutation importance trên dữ liệu đa cộng tuyến.
→ Mục 4.6.

**19. Vì sao GroupKFold làm R² sụp, thậm chí Ridge xuống −2,37?**
Vì GroupKFold buộc mô hình **ngoại suy** sang hình dạng chưa từng gặp. Ridge ngoại suy tuyến tính trong khi
quan hệ thật phi tuyến → sai số bùng nổ; cây không ngoại suy nên sai số bị chặn (Random Forest còn 0,81;
XGBoost 0,35). Kết luận: không mô hình nào đủ tin cậy ngoài miền 12 thiết kế.
→ Mục 4.7, Bảng 19.

**20. Nhóm có thể triển khai mô hình này trong thực tế không?**
Có, nhưng **có điều kiện**: chỉ dùng cho công trình nằm trong miền thiết kế đã học, và cần cảnh báo khi
dự đoán rơi vào vùng tải cao (MAE tăng gấp ~5 lần so với nhóm tải thấp). Với công trình mới hoàn toàn, cần
thu thập thêm dữ liệu hoặc dùng mô hình lai vật lý + ML.
→ Mục 4.5, 4.7, 5.3.

---

## E. Câu hỏi kỹ thuật nhanh (chuẩn bị 30 giây)

| Câu hỏi | Trả lời ngắn |
|---|---|
| `random_state` để làm gì? | Cố định chia dữ liệu, KFold, RandomizedSearch → tái lập kết quả. |
| Vì sao lưu 2 file model? | `best_model.joblib` fit trên 614 mẫu train (tái lập số báo cáo); `best_model_full_data.joblib` fit 768 mẫu (demo). |
| Vì sao `drop='first'` khi one-hot? | Tránh bẫy dummy variable (đa cộng tuyến hoàn hảo giữa các cột one-hot). |
| Vì sao cây không cần StandardScaler? | Cây chỉ so sánh ngưỡng, bất biến với phép biến đổi đơn điệu. Nhóm vẫn để scaler trong pipeline chung cho đơn giản. |
| Vì sao `handle_unknown='ignore'`? | Để mô hình không sập khi gặp giá trị X8 lạ trong triển khai. |
| Tuning bao nhiêu tổ hợp? | 7 + 6 + 28 + 48 (Grid) và 20 + 20 + 20 + 30 (Random). |
| Metric nào dùng để chọn mô hình? | CV R² trên **train** (không dùng test). Test chỉ báo cáo một lần. |
| Mô hình chạy nhanh không? | Fit < 1 s, dự đoán ~12 ms cho 154 mẫu → phù hợp duyệt thiết kế tương tác. |
| Nếu có thêm dữ liệu thì làm gì? | Ưu tiên **đa dạng hình học** (X1–X5), không phải thêm mẫu lặp cùng thiết kế. |
| Sai số lớn nhất ở đâu? | Nhóm tải cao (Q4) và vùng X1 = 0,79, X5 = 7,0, X7 = 0,25–0,40, X8 = 4; Y2 bị dự đoán thấp. |

---

*Ghi chú: các số trong tài liệu này khớp với `03_SourceCode/results/`. Nếu chạy lại notebook, kiểm tra lại
`final_model_comparison.csv` và `robustness_groupkfold.csv` trước khi bảo vệ.*
