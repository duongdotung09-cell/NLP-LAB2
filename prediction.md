# Prediction - trước experiment

Dòng "Sau experiment" được điền sau khi chạy experiments.ipynb.

## Prediction 1 - Unigram → bigram → trigram: vocabulary có tăng không?

- Prediction: Không, vocabulary giữ nguyên.
- Reason: Vocabulary là tập word types, không phụ thuộc n; chỉ không gian n-gram tăng (tối đa |V|^n).
- Confidence: 95%.
- Sau experiment: Đúng. V = 48 963 cho cả 3 mô hình.

## Prediction 2 - Số lượng n-gram thay đổi thế nào?

- Prediction: Số n-gram types tăng mạnh theo n, tỉ lệ n-gram chỉ xuất hiện 1 lần cũng tăng.
- Reason: Tổ hợp từ càng dài càng ít lặp lại.
- Confidence: 90%.
- Sau experiment: Đúng. Unique: 105K → 1.22M → 2.80M; singleton: 47% → 70% → 86%.

## Prediction 3 - Mô hình nào gặp zero probability nhiều hơn?

- Prediction: Trigram MLE > bigram MLE > unigram (≈ 0 khi có `<unk>`).
- Reason: Trigram có nhiều n-gram types nhất, mỗi context có ít count nhất.
- Confidence: 90%.
- Sau experiment: Đúng. Tỉ lệ token P = 0 trên test: 0% / 19.9% / 58.2%.

## Prediction 4 - Mô hình nào có perplexity thấp hơn trên training set?

- Prediction: Trigram MLE < bigram MLE < unigram MLE.
- Reason: Context dài hơn làm phân phối nhọn hơn, MLE khớp chính dữ liệu train.
- Confidence: 85%.
- Sau experiment: Đúng với MLE: 1 078 / 104 / 9.7. Với Laplace thì ngược lại (1 080 / 1 943 / 10 248) vì add-one lấy nhiều mass khi V lớn.

## Prediction 5 - Corpus rất nhỏ, trigram có chắc chắn tốt hơn bigram?

- Prediction: Không, trên validation/test trigram có thể tệ hơn bigram.
- Reason: Corpus nhỏ thì phần lớn trigram ở test chưa từng gặp.
- Confidence: 75%.
- Sau experiment: Đúng. Test PPL (Laplace): bigram 2 191 < trigram 16 247 (2K docs: 4 346 < 23 836); MLE đều ∞. Top-1 accuracy: bigram 17.3% > trigram 15.8%.
