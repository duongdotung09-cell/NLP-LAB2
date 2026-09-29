# LAB 02 - N-gram Language Models

## Cấu trúc

| File | Nội dung |
|---|---|
| calculations.md | Bài 1-4, mục 9, 11, 18 (tính tay) |
| prediction.md | Prediction 1-5 (Prediction / Reason / Confidence, kèm kiểm chứng sau experiment) |
| ngram_lm.py | Implementation: build_vocabulary, count_ngrams, train_unigram/bigram/trigram, class NGramLanguageModel (fit, probability, sentence_probability, sentence_log_probability, next_word_distribution, perplexity) |
| experiments.ipynb | Experiment 1-3, log probability, context length, next-word prediction, sentence ranking, error analysis |
| results.csv | Perplexity (MLE/Laplace × uni/bi/tri × train/valid/test), corpus size, next-word predictions, top-1/top-5 accuracy |
| error_analysis.md | 2 prediction đúng và 2 prediction sai, kèm nguyên nhân |
| reflection.md | Câu 1-7 |

## Chạy lại

```bash
# corpus đặt ở thư mục cha: ../c4-train.00000-of-01024-30K.json.gz
pip install numpy pandas matplotlib jupyter
python ngram_lm.py                      # sanity check trên corpus Bài 1
jupyter nbconvert --to notebook --execute --inplace experiments.ipynb   # ~2.5 phút, cần khoảng 3 GB RAM
```

## Thiết lập

- Dữ liệu: 10 000 documents đầu của C4 (c4-train.00000-of-01024-30K.json.gz), split theo document 80/10/10 (seed 42). Train có 166 737 câu, khoảng 3.5M token.
- Preprocessing: tách câu theo xuống dòng và dấu . ! ?; lowercase; token là từ/số (giữ dấu ', vd don't) hoặc dấu câu.
- Boundary: mỗi câu được thêm (n−1) `<s>` ở đầu và `</s>` ở cuối. `</s>` được dự đoán và tính vào N của perplexity.
- Vocabulary: các từ có count ≥ 2 trong train, cộng `<unk>` và `</s>`, tổng V = 48 963. OOV được map thành `<unk>`.
- MLE: P(w|h) = C(h,w)/C(h), với C(h) = Σ_w C(h,w). Laplace: (C(h,w)+1)/(C(h)+V). Context chưa thấy: MLE cho 0, Laplace cho 1/V.

## Kết quả chính (perplexity)

| Model | Train | Valid | Test |
|---|---|---|---|
| Unigram MLE | 1 078.2 | 941.6 | 922.3 |
| Bigram MLE | 104.3 | ∞ (20.4% token P=0) | ∞ (19.8%) |
| Trigram MLE | 9.7 | ∞ (58.6%) | ∞ (58.2%) |
| Unigram Laplace | 1 079.6 | 944.8 | 925.7 |
| Bigram Laplace | 1 943.0 | 2 241.6 | 2 190.6 |
| Trigram Laplace | 10 248.0 | 16 200.3 | 16 247.1 |

Top-1 / top-5 next-word accuracy trên test (MLE): unigram 4.6% / 19.6%, bigram 17.2% / 33.9%, trigram 15.8% / 25.8%.

Kết luận: n lớn hơn chỉ tốt hơn trên train. Trên dữ liệu mới, sparsity cùng với add-one (V lớn) làm bigram và trigram tệ hơn unigram. Phân tích chi tiết nằm trong notebook.

## AI assistance statement

Tool: Claude.

Purpose: debug và tối ưu code trong ngram_lm.py, experiments.ipynb sau khi bản cài đặt cơ bản đã hoàn thành; trình bày lại thành văn bản các phần đã tự tính toán, tự dự đoán và tự phân tích trước.

What was generated: bản cài đặt cơ bản của các hàm build_vocabulary, count_ngrams, train_unigram/bigram/trigram và class NGramLanguageModel trong ngram_lm.py, cùng các context dùng cho next-word prediction và candidate cho sentence ranking, được tự viết và tự chọn trước. AI hỗ trợ debug và tối ưu lại pipeline thực nghiệm trong experiments.ipynb (thống kê corpus, log probability, so sánh MLE và Laplace, perplexity, context length) sau khi đã chạy được bản cơ bản. Riêng calculations.md, prediction.md, phần phân tích kết quả trong notebook, error_analysis.md và reflection.md: kết quả tính toán, prediction, 2 case đúng/2 case sai và câu trả lời cho reflection được làm trước, sau đó đưa cho AI để trình bày lại thành văn bản theo đúng định dạng đề yêu cầu.

What was modified: dựa trên góp ý debug của AI, các hàm trong ngram_lm.py được sửa lại nhiều lần (regex tokenizer để giữ chữ cái có dấu, tham số exclude cho next_word_distribution, cách xử lý phân phối Laplace khi context chưa thấy), vocabulary và số liệu trong các file được cập nhật lại sau mỗi lần chạy lại notebook. Với calculations.md, prediction.md, error_analysis.md và reflection.md, bản AI trình bày lại được đối chiếu với kết quả làm trước đó, sửa những chỗ AI diễn đạt sai ý hoặc thêm nội dung ngoài phần đã cung cấp, và chỉnh lại cho đúng định dạng đề yêu cầu (không in đậm/in nghiêng, không dùng dấu gạch ngang dài).

How the result was verified: chạy python ngram_lm.py và đối chiếu kết quả với calculations.md (Bài 1, Bài 2); kiểm tra Σ_w P(w|h) = 1 cho cả MLE và Laplace ngay trong notebook; chạy lại toàn bộ experiments.ipynb từ đầu bằng jupyter nbconvert --execute và đối chiếu số liệu trong notebook, README.md, prediction.md, error_analysis.md với results.csv để đảm bảo khớp nhau; đọc lại toàn bộ các file .md để kiểm tra nội dung đúng trọng tâm từng bài tập trong W2.pdf.
