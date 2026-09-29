# Calculations - tính tay

Quy ước: không thêm `<s>`, `</s>` (đúng công thức trong đề).

## Bài 1 - Unigram

Corpus: "the cat eats fish", "the cat likes fish", "the dog eats meat".

1. Vocabulary = {the, cat, eats, fish, likes, dog, meat}, |V| = 7.
2. N = 12.
3. P(w) = C(w)/N:

| w | C(w) | P(w) |
|---|---|---|
| the | 3 | 3/12 = 0.25 |
| cat | 2 | 2/12 ≈ 0.1667 |
| fish | 2 | 2/12 ≈ 0.1667 |
| dog | 1 | 1/12 ≈ 0.0833 |

4. Σ P(w) = (3 + 2 + 2 + 2 + 1 + 1 + 1)/12 = 12/12 = 1.

## Bài 2 - Bigram

C(the) = 3, C(the cat) = 2, C(the dog) = 1, C(cat) = 2, C(cat eats) = 1, C(cat likes) = 1.

- P(cat | the) = 2/3 ≈ 0.6667
- P(dog | the) = 1/3 ≈ 0.3333
- P(eats | cat) = 1/2 = 0.5
- P(likes | cat) = 1/2 = 0.5

Vì sao Σ_w P(w | the) = 1: mỗi lần "the" làm context thì theo sau đúng một từ, nên Σ_w C(the, w) = C(the) ⇒ Σ_w C(the, w)/C(the) = 1 (2/3 + 1/3 = 1). Cần C(h) đếm số lần h đứng làm context và vocabulary có `</s>` cho từ cuối câu.

## Bài 3 - Xác suất câu

P(fish | eats) = C(eats fish)/C(eats) = 1/2.

P(the cat eats fish) = 0.25 × 2/3 × 1/2 × 1/2 = 1/24 ≈ 0.0417.

Thêm một từ vào cuối câu thì xác suất không tăng: mỗi thừa số thêm vào ≤ 1 (vd P(the cat eats) = 1/12 > P(the cat eats fish) = 1/24). Nếu có `</s>` thì có thể tăng: P(`</s>` | eats) = 0 nên P(the cat eats `</s>`) = 0 < P(the cat eats fish `</s>`). Vì vậy không so sánh trực tiếp P của các câu khác độ dài mà dùng log P / token hoặc perplexity.

## Bài 4 - Sentence ranking

- P(S1) = P(the) · P(cat|the) · P(eats|cat) · P(fish|eats) = 0.25 × 2/3 × 1/2 × 1/2 = 1/24
- P(S2) = P(the) · P(dog|the) · P(eats|dog) · P(fish|eats) = 0.25 × 1/3 × 1 × 1/2 = 1/24

Dự đoán: P(S1) = P(S2) ≈ 0.0417. P(dog|the) nhỏ hơn P(cat|the) một nửa nhưng P(eats|dog) = 1 gấp đôi P(eats|cat) = 1/2.

## Mục 9 - Suy luận trước smoothing

Corpus: "I like NLP", "I like AI", "I study NLP".

1. C(study AI) = 0.
2. P_MLE(AI | study) = 0/1 = 0.
3. Câu chứa "study AI" (vd "I study AI") có P(S) = 0, perplexity = ∞.
4. Không. Mô hình chỉ chưa quan sát bigram này trong 3 câu; "I study AI" vẫn là câu hợp lệ. Không quan sát thấy ≠ xác suất thật bằng 0.

## Mục 11 - Laplace smoothing

C(cat) = 10, V = 5.

- C(cat eats) = 0: P_Laplace(eats | cat) = (0 + 1)/(10 + 5) = 1/15 ≈ 0.0667 (MLE: 0).
- C(cat eats) = 3: P_Laplace(eats | cat) = (3 + 1)/(10 + 5) = 4/15 ≈ 0.2667 (MLE: 0.3).

Ảnh hưởng đến bigram khác: mọi bigram bắt đầu bằng "cat" đổi từ c/10 thành (c + 1)/15:

| c | MLE | Laplace |
|---|---|---|
| 0 | 0 | 0.0667 ↑ |
| 1 | 0.1 | 0.1333 ↑ |
| 2 | 0.2 | 0.2 = |
| 3 | 0.3 | 0.2667 ↓ |
| 5 | 0.5 | 0.4 ↓ |

Bigram hiếm/chưa thấy tăng, bigram phổ biến giảm: smoothing thay đổi toàn bộ phân phối.

## Mục 18 - Perplexity

N = 3.

- P(W) = 0.5 × 0.25 × 0.5 = 0.0625; PP(W) = 0.0625^(−1/3) = 16^(1/3) ≈ 2.52.
- P(w2|w1) = 0.1: P(W) = 0.5 × 0.1 × 0.5 = 0.025; PP(W) = 40^(1/3) ≈ 3.42.

Vì sao: log PP = −(1/N) Σ log P, và −log p → ∞ khi p → 0. Một thừa số giảm 2.5 lần (0.25 → 0.1) làm PP tăng 2.5^(1/3) ≈ 1.36 lần; p = 0 thì PP = ∞.
