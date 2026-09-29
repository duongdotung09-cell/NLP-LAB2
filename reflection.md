# Reflection

## Câu 1. Nếu tăng n, mô hình nhận thêm thông tin gì?

Thêm một từ đứng trước trong context (n−1 thay vì n−2 từ), nên bắt được cụm cố định và phụ thuộc cục bộ dài hơn (vd P(to | in order) = 0.895 so với P(to | order) = 0.463).

## Câu 2. Tại sao tăng n lại làm sparsity tăng?

Số context có thể có tăng theo |V|^(n−1) trong khi dữ liệu không đổi, nên mỗi n-gram xuất hiện ít hơn và nhiều n-gram hợp lệ không bao giờ gặp. Trong lab: tỉ lệ token unseen trên test 0% → 20% → 58%.

## Câu 3. Tại sao smoothing cần thiết?

MLE gán P = 0 cho n-gram chưa thấy, nên một n-gram lạ làm P(câu) = 0 và perplexity = ∞. Smoothing chuyển bớt mass từ n-gram đã thấy sang n-gram chưa thấy để mô hình dùng được trên dữ liệu mới.

## Câu 4. Perplexity đo điều gì?

PP = exp(−(1/N) Σ log P(w_i | context)): nghịch đảo xác suất của dữ liệu, chuẩn hóa theo số token, tương đương số lựa chọn "đều nhau" mà mô hình phân vân ở mỗi bước. PP thấp nghĩa là mô hình gán xác suất cao cho dữ liệu thật.

## Câu 5. Perplexity thấp hơn có luôn tạo văn bản tốt hơn với con người không?

Không. PP chỉ đo khả năng dự đoán token trung bình trên một tập cụ thể, không đo tính mạch lạc, đúng sự thật hay hữu ích. Vd trigram MLE có train PPL 9.7 do thuộc lòng train; PP cũng giảm khi vocabulary nhỏ hơn (nhiều `<unk>`) mà văn bản không tốt hơn.

## Câu 6. N-gram LM thất bại ở đâu so với cách con người hiểu ngôn ngữ?

- Chỉ nhìn n−1 từ, không có phụ thuộc xa.
- Từ là ký hiệu rời rạc, không có tương đồng: biết "the dog had" không giúp gì cho "the cat had".
- Không hiểu nghĩa: "banana computer quickly" và "studies language models" có điểm gần như nhau.
- Không tổng quát hóa cho tổ hợp mới, phụ thuộc count và domain của corpus.

## Câu 7. Context dài 100 từ, trigram có dùng được thông tin của 97 từ đầu không?

Không. Trigram chỉ dùng P(w_t | w_{t−2}, w_{t−1}), 97 từ đầu bị bỏ. Tăng n lên 100 cũng không khả thi vì sparsity, nên cần neural LM biểu diễn lịch sử dài bằng vector dense.
