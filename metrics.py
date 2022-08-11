import numpy as np

def precision_at_k(r, k):
    """Score is precision @ k. Relevance is binary (nonzero is relevant).

    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def average_precision(r):
    """Score is average precision (area under PR curve). Relevance is binary (nonzero is relevant).

    Args:
        r: List/array of relevance scores in rank order
    Returns:
        Average precision
    """

    try:
        delta_r = 1. / sum(r)
        avg_p = np.mean([precision_at_k(r, i+1)
                         for i, rel in enumerate(r) if rel])
    except ZeroDivisionError:
        return 0

    return avg_p


def mean_average_precision(rs):
    """Score is mean average precision. Relevance is binary (nonzero is relevant).

    Args:
        rs: Iterator of lists/arrays of relevance scores in rank order 
    Returns:
        Mean average precision
    """

    m_avg_p = np.mean([average_precision(r) for r in rs])
    return m_avg_p


def reciprocal_rank(r):
    """Score is reciprocal rank of first relevant document. Relevance is binary (nonzero is relevant).

    Args:
        r: List/array of relevance scores in rank order
    Returns:
        Reciprocal rank of first relevant document
    """
    try:
        k = r.index(1) + 1
        score = 1/k
    except ValueError:
        score = 0
    return score


def mean_reciprocal_rank(rs):
    """Score is mean reciprocal rank. Relevance is binary (nonzero is relevant).

    Args:
        r: List/array of relevance scores in rank order
    Returns:
        Mean reciprocal rank
    """
    mrr = np.mean([reciprocal_rank(r) for r in rs])
    return mrr


def get_relevance_scores(pred, actual):
    num_retr = len(pred)
    r = np.zeros((num_retr), dtype=int)
    for doc in actual:
        try:
            k = pred.index(doc)
            r[k] = 1
        except ValueError:
            continue
    return r.tolist()

def get_ranking(doc_scores, top_K = None):
    ranking = sorted(doc_scores.items(),
                key=lambda item: item[1], reverse=True)
    if top_K == None:
        return ranking
    else:
        return ranking[:top_K]
    
def cosinesimilarity(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
