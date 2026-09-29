"""LAB 02 - N-gram Language Model implemented from scratch.

Conventions
-----------
- Tokens are lowercased words / numbers / punctuation marks.
- Each sentence is padded with (n-1) ``<s>`` on the left and one ``</s>`` on the
  right. ``</s>`` is predicted (counted in N for perplexity), ``<s>`` is not.
- Words with count < ``min_count`` in the training set are mapped to ``<unk>``.
- Vocabulary V contains ``<unk>`` and ``</s>`` but not ``<s>`` (``<s>`` is never
  predicted), so every conditional distribution P(. | h) is over V words.
"""

import copy
import gzip
import json
import math
import random
import re
from collections import Counter

BOS, EOS, UNK = "<s>", "</s>", "<unk>"

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
# word = letters/digits of any script (café, naïve), optionally with inner ' (don't); else 1 punctuation mark
_TOKEN = re.compile(r"[^\W_]+(?:'[^\W_]+)*|[^\w\s]|_")


# ---------------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------------
def load_documents(path, n_docs=None):
    """Read the ``text`` field of the first ``n_docs`` documents of a C4 .json.gz file."""
    docs = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if n_docs is not None and i >= n_docs:
                break
            docs.append(json.loads(line)["text"])
    return docs


def split_sentences(text):
    sentences = []
    for line in text.split("\n"):
        for s in _SENT_SPLIT.split(line.strip()):
            if s:
                sentences.append(s)
    return sentences


def tokenize(sentence):
    return _TOKEN.findall(sentence.lower().replace("’", "'"))


def preprocess(docs):
    """list[str] -> list of documents, each document = list of tokenized sentences."""
    return [[toks for s in split_sentences(d) if (toks := tokenize(s))] for d in docs]


def train_valid_test_split(items, ratios=(0.8, 0.1, 0.1), seed=42):
    idx = list(range(len(items)))
    random.Random(seed).shuffle(idx)
    n_train = int(ratios[0] * len(items))
    n_valid = int(ratios[1] * len(items))
    parts = (idx[:n_train], idx[n_train:n_train + n_valid], idx[n_train + n_valid:])
    return tuple([items[i] for i in part] for part in parts)


def flatten(docs):
    return [s for d in docs for s in d]


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------
def build_vocabulary(sentences, min_count=1):
    counts = Counter(w for s in sentences for w in s)
    vocab = {w for w, c in counts.items() if c >= min_count}
    vocab.update({UNK, EOS})
    return vocab


def pad_sentence(sentence, n):
    return [BOS] * (n - 1) + list(sentence) + [EOS]


def ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def count_ngrams(sentences, n, pad=True):
    """Counter of n-gram tuples. ``pad=False`` counts raw n-grams (used for hand calculations)."""
    counts = Counter()
    for s in sentences:
        counts.update(ngrams(pad_sentence(s, n) if pad else list(s), n))
    return counts


def train_ngram(sentences, n, pad=True):
    """MLE table {context: {word: P(word | context)}}, context = tuple of n-1 words."""
    ngram_counts = count_ngrams(sentences, n, pad)
    context_counts = Counter()
    for g, c in ngram_counts.items():
        context_counts[g[:-1]] += c
    table = {}
    for g, c in ngram_counts.items():
        table.setdefault(g[:-1], {})[g[-1]] = c / context_counts[g[:-1]]
    return table


def train_unigram(sentences, pad=True):
    """{word: P(word)} = C(w) / N."""
    return train_ngram(sentences, 1, pad)[()]


def train_bigram(sentences, pad=True):
    return train_ngram(sentences, 2, pad)


def train_trigram(sentences, pad=True):
    return train_ngram(sentences, 3, pad)


# ---------------------------------------------------------------------------
# Language model
# ---------------------------------------------------------------------------
class NGramLanguageModel:
    def __init__(self, n, smoothing="mle", min_count=2):
        if smoothing not in ("mle", "laplace"):
            raise ValueError("smoothing must be 'mle' or 'laplace'")
        self.n = n
        self.smoothing = smoothing
        self.min_count = min_count

    @property
    def name(self):
        order = {1: "Unigram", 2: "Bigram", 3: "Trigram"}.get(self.n, f"{self.n}-gram")
        return f"{order} {'MLE' if self.smoothing == 'mle' else 'Laplace'}"

    # ----- training -----
    def fit(self, sentences, vocabulary=None):
        self.vocab = set(vocabulary) if vocabulary is not None else build_vocabulary(sentences, self.min_count)
        self.V = len(self.vocab)
        counts = count_ngrams((self.map_oov(s) for s in sentences), self.n)
        self.followers = {}          # context -> {word: C(context, word)}
        self.context_counts = Counter()  # context -> C(context) = sum_w C(context, word)
        for g, c in counts.items():
            self.followers.setdefault(g[:-1], {})[g[-1]] = c
            self.context_counts[g[:-1]] += c
        self.num_ngrams = len(counts)
        self.num_tokens = sum(counts.values())
        return self

    def with_smoothing(self, smoothing):
        """Same counts, different estimator (avoids re-counting)."""
        other = copy.copy(self)
        other.smoothing = smoothing
        return other

    def map_oov(self, sentence):
        return [w if w in self.vocab else UNK for w in sentence]

    def _context(self, context):
        if self.n == 1:
            return ()
        h = [BOS] * (self.n - 1) + [w if w in self.vocab or w == BOS else UNK for w in context]
        return tuple(h[-(self.n - 1):])

    # ----- probabilities -----
    def probability(self, context, word):
        h = self._context(context)
        w = word if word in self.vocab else UNK
        c_hw = self.followers.get(h, {}).get(w, 0)
        c_h = self.context_counts.get(h, 0)
        if self.smoothing == "laplace":
            return (c_hw + 1) / (c_h + self.V)
        return c_hw / c_h if c_h else 0.0

    def token_probabilities(self, sentence):
        """P(w_t | context_t) for every predicted token of the padded sentence (incl. </s>)."""
        toks = pad_sentence(self.map_oov(sentence), self.n)
        return [self.probability(toks[i - self.n + 1:i], toks[i]) for i in range(self.n - 1, len(toks))]

    def sentence_log_probability(self, sentence):
        lp = 0.0
        for p in self.token_probabilities(sentence):
            if p == 0.0:
                return float("-inf")
            lp += math.log(p)
        return lp

    def sentence_probability(self, sentence):
        return math.exp(self.sentence_log_probability(sentence))

    def continuation_log_probability(self, context, continuation):
        """log P(continuation | context), without </s>."""
        hist = list(context)
        lp = 0.0
        for w in continuation:
            p = self.probability(hist, w)
            if p == 0.0:
                return float("-inf")
            lp += math.log(p)
            hist.append(w)
        return lp

    def next_word_distribution(self, context, k=None, exclude=()):
        """Top-k (word, probability) after ``context``, sorted by probability.

        MLE: only words seen after the context (unseen context -> [], every P = 0).
        Laplace: seen words first, then unseen words, which all share (0 + 1) / (C(h) + V).
        """
        h = self._context(context)
        seen = self.followers.get(h, {})
        dist = [(w, self.probability(context, w)) for w in seen if w not in exclude]
        dist.sort(key=lambda x: (-x[1], x[0]))
        if self.smoothing == "laplace" and (k is None or len(dist) < k):
            p_unseen = 1 / (self.context_counts.get(h, 0) + self.V)
            dist += [(w, p_unseen) for w in sorted(self.vocab) if w not in seen and w not in exclude]
        return dist[:k] if k else dist

    # ----- evaluation -----
    def evaluate(self, sentences):
        total_lp, n_tokens, n_zero = 0.0, 0, 0
        for s in sentences:
            for p in self.token_probabilities(s):
                n_tokens += 1
                if p == 0.0:
                    n_zero += 1
                else:
                    total_lp += math.log(p)
        ppl = float("inf") if n_zero else math.exp(-total_lp / n_tokens)
        return {"perplexity": ppl, "tokens": n_tokens, "zero_prob_tokens": n_zero,
                "zero_prob_rate": n_zero / n_tokens}

    def perplexity(self, sentences):
        return self.evaluate(sentences)["perplexity"]

    def unseen_ngram_stats(self, sentences):
        """How many n-grams of ``sentences`` never occur in the training data."""
        test_counts = count_ngrams((self.map_oov(s) for s in sentences), self.n)
        unseen = {g: c for g, c in test_counts.items() if g[-1] not in self.followers.get(g[:-1], {})}
        return {"test_ngram_types": len(test_counts),
                "unseen_types": len(unseen),
                "unseen_type_rate": len(unseen) / len(test_counts),
                "unseen_token_rate": sum(unseen.values()) / sum(test_counts.values())}


if __name__ == "__main__":
    toy = [s.split() for s in ["the cat eats fish", "the cat likes fish", "the dog eats meat"]]
    uni = train_unigram(toy, pad=False)
    big = train_bigram(toy, pad=False)
    print({w: round(uni[w], 4) for w in ["the", "cat", "fish", "dog"]}, "sum =", sum(uni.values()))
    print("P(cat|the) =", big[("the",)]["cat"], " P(eats|cat) =", big[("cat",)]["eats"])
