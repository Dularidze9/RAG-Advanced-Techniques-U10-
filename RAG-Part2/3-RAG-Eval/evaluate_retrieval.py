# retrieval_metrics.py

def assess_retrieval(query, candidate_indices, candidate_texts):
    """
    Evaluate retrieval performance: compute hit rate, recall, and precision.

    Args:
        query (str): User question.
        candidate_indices (list[int]): Indices of retrieved chunks.
        candidate_texts (list[str]): Texts of retrieved chunks.

    Returns:
        dict: Contains hit rate, recall, precision, and keyword match score.
    """

    # Reference dataset (ground truth)
    reference_set = [
        {
            "query": "What is an AI Agent?",
            "gold_indices": [0],
            "gold_keywords": ["autonomous", "agent", "automation"]
        },
        {
            "query": "What is a planning function?",
            "gold_indices": [5],
            "gold_keywords": ["strategic", "planning"]
        },
        {
            "query": "What is a self-improvement function?",
            "gold_indices": [7],
            "gold_keywords": ["adaptive", "self-improvement", "self-learning"]
        }
    ]

    # Default output
    metrics = {
        "query": query,
        "hit_rate": 0.0,
        "recall": 0.0,
        "precision": 0.0,
        "keyword_match": 0.0,
        "is_match": False
    }

    # Match query to its reference entry
    for item in reference_set:
        if item["query"] == query:
            true_indices = set(item["gold_indices"])
            retrieved = set(candidate_indices)

            # Overlap between ground truth and retrieved indices
            overlap = true_indices.intersection(retrieved)

            # Metrics based on index matching
            metrics["hit_rate"] = 1.0 if overlap else 0.0
            metrics["recall"] = len(overlap) / len(true_indices) if true_indices else 0.0
            metrics["precision"] = len(overlap) / len(retrieved) if retrieved else 0.0
            metrics["is_match"] = bool(overlap)

            # Keyword match scoring
            if candidate_texts:
                matched = 0
                for kw in item["gold_keywords"]:
                    if any(kw.lower() in txt.lower() for txt in candidate_texts):
                        matched += 1
                metrics["keyword_match"] = matched / len(item["gold_keywords"]) if item["gold_keywords"] else 0.0

            break

    return metrics


# Example usage
if __name__ == "__main__":
    query = "What is an AI Agent?"
    candidate_indices = [1, 2, 3]
    candidate_texts = [
        "An AI Agent is an autonomous system capable of acting independently.",
        "Self-improvement features include adaptive optimization.",
        "Unrelated content."
    ]

    result = assess_retrieval(query, candidate_indices, candidate_texts)
    print(f"Query: {query}")
    print(f"Retrieved indices: {candidate_indices}")
    print(f"Sample texts: {[t[:50] + '...' for t in candidate_texts]}")
    print(f"Evaluation result: {result}")
