import collections
import re
import string


def normalize_text(s):
    """Normalize text for evaluation."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def calculate_token_f1(prediction, ground_truth):
    """Calculate token-level F1 score."""
    prediction_tokens = normalize_text(prediction).split()
    ground_truth_tokens = normalize_text(ground_truth).split()

    common = collections.Counter(prediction_tokens) & collections.Counter(
        ground_truth_tokens
    )
    num_same = sum(common.values())

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def calculate_evidence_recall(evidence, ground_truth_snippet):
    """Verify if the ground truth evidence snippet exists within the extracted evidence."""
    norm_ev = normalize_text(evidence)
    norm_gt = normalize_text(ground_truth_snippet)

    # Ground truth should be a SUBSET of the extracted evidence sentence
    if norm_gt in norm_ev:
        return 1.0

    # Partial recall (overlap)
    ev_tokens = set(norm_ev.split())
    gt_tokens = set(norm_gt.split())
    if not gt_tokens:
        return 0.0

    overlap = len(ev_tokens.intersection(gt_tokens))
    return overlap / len(gt_tokens)


if __name__ == "__main__":
    # Test cases
    pred = "Quantum Solutions is a Critical Data Processor under Article 4."
    gt = "Quantum Solutions qualifies as a Critical Data Processor according to Article 4."

    f1 = calculate_token_f1(pred, gt)
    print(f"Token F1: {f1:.4f}")

    ev = "Quantum Solutions is a Critical Data Processor."
    gt_ctx = "Quantum Solutions is a Critical Data Processor based in the EU. They process millions of records."
    recall = calculate_evidence_recall(ev, gt_ctx)
    print(f"Evidence Recall: {recall:.4f}")
