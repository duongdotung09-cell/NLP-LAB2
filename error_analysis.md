# Error analysis

Mô hình: Trigram MLE (so với Bigram MLE), train trên 8 000 docs C4. Expected là từ thật đứng sau context trong test set. Số liệu: experiments.ipynb, mục 7 và 9. Top-1 accuracy trên 65 262 vị trí của test: unigram 4.6%, bigram 17.3%, trigram 15.8%.

## Prediction đúng

### 1. "in order" → "to"

- Context: in order (C = 563)
- Model prediction: to 0.895, for 0.043, that 0.014
- Expected: to
- Probability: P_tri(to | in order) = 0.895; P_bi(to | order) = 0.463
- Nguyên nhân: collocation cố định, context có đủ dữ liệu. Trigram hơn bigram vì "order" đứng một mình có nhiều tiếp nối ("order .", "order for").

### 2. "at the same" → "time"

- Context: at the same (trigram chỉ dùng "the same", C = 1 365)
- Model prediction: time 0.138, "." 0.053, way 0.040
- Expected: time
- Probability: P_tri = 0.138; P_bi(time | same) = 0.124
- Nguyên nhân: đúng nhưng không tự tin do context quá ngắn: trigram bỏ mất "at", trong khi C(at the same time)/C(at the same) = 161/179 = 0.899.

## Prediction sai

### 3. "natural language" → "processing"

- Context: natural language (C = 2)
- Model prediction: "." 1.000 (bigram: "," 0.141, "." 0.124, and 0.063)
- Expected: processing
- Probability: P_tri = 0, P_bi(processing | language) = 0 ("processing" có trong vocab)
- Nguyên nhân: domain mismatch, unseen n-gram, insufficient training data. C4 là web text chung; "natural language" chỉ xuất hiện 2 lần, cả hai trong cùng một document (trích dẫn "truth-value gaps in natural language."), nên MLE gán P = 1 cho ".".

### 4. "the cat" → "had"

- Context: the cat (C = 17, trong 7 document)
- Model prediction: queen 0.294, house 0.118, '"' 0.059 (bigram: house 0.116, "," 0.087, to 0.064)
- Expected: had
- Probability: P_tri = 0, P_bi(had | cat) = 0
- Nguyên nhân: sparsity, insufficient training data. Cả 5 lần "the cat queen" nằm trong một document duy nhất; MLE coi đặc thù của một văn bản là quy luật chung và gán 0 cho tiếp nối bình thường "had".

## Tổng kết

| Case | Kết quả | Nguyên nhân chính |
|---|---|---|
| in order → to | đúng | collocation, đủ dữ liệu |
| at the same → time | đúng (P thấp) | context quá ngắn |
| natural language → processing | sai | domain mismatch, unseen n-gram |
| the cat → had | sai | sparsity, insufficient data |
