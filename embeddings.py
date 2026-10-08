"""Count-based word embeddings (PPMI + truncated SVD), no gensim needed.
Same idea as Word2Vec: words used in similar contexts get similar vectors.
(Levy & Goldberg 2014 showed SGNS Word2Vec implicitly factorizes a shifted PMI matrix.)"""
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


class WordVectors:
    def __init__(self, sentences, vector_size=100, window=5, min_count=5, seed=42):
        from collections import Counter
        counts = Counter(t for s in sentences for t in s)
        self.index_to_key = [w for w, c in counts.most_common() if c >= min_count]
        self.key_to_index = {w: i for i, w in enumerate(self.index_to_key)}
        V = len(self.index_to_key)

        ids, doc = [], []
        for d, s in enumerate(sentences):
            for t in s:
                i = self.key_to_index.get(t)
                if i is not None:
                    ids.append(i); doc.append(d)
        ids, doc = np.array(ids), np.array(doc)

        rows, cols, vals = [], [], []
        for k in range(1, window + 1):                      # co-occurrence within window
            same = doc[:-k] == doc[k:]
            a, b = ids[:-k][same], ids[k:][same]
            rows += [a, b]; cols += [b, a]
            vals += [np.full(len(a), 1.0 / k)] * 2          # closer words count more
        C = sparse.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(V, V)).tocsr().tocoo()

        total = C.data.sum()
        row_sum = np.asarray(C.sum(axis=1)).ravel()
        ctx = np.asarray(C.sum(axis=0)).ravel() ** 0.75      # context smoothing
        ctx = ctx / ctx.sum()
        pmi = np.log((C.data / total) / ((row_sum[C.row] / total) * ctx[C.col]) + 1e-12)
        keep = pmi > 0                                       # positive PMI
        P = sparse.coo_matrix((pmi[keep], (C.row[keep], C.col[keep])), shape=(V, V)).tocsr()

        svd = TruncatedSVD(n_components=vector_size, random_state=seed)
        U = svd.fit_transform(P)
        U = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-9)   # unit length -> dot = cosine
        self.vectors = U.astype(np.float32)

    # ---- gensim-like API so the app code stays simple ----
    @property
    def wv(self):
        return self

    def __contains__(self, w):
        return w in self.key_to_index

    def __len__(self):
        return len(self.index_to_key)

    def __getitem__(self, w):
        return self.vectors[self.key_to_index[w]]

    def similarity(self, a, b):
        return float(self[a] @ self[b])

    def most_similar(self, w, topn=10):
        sims = self.vectors @ self[w]
        order = np.argsort(-sims)
        return [(self.index_to_key[i], float(sims[i])) for i in order if self.index_to_key[i] != w][:topn]
