# tests/test_reranker_normalization.py
from backend.services.reranker import logit_to_confidence_percentage

def test_sigmoid_normalization_edge_cases():
    # 1. Zero logit -> 50% confidence
    assert logit_to_confidence_percentage(0.0) == 50.0

    # 2. Strong positive logit -> high confidence
    assert logit_to_confidence_percentage(5.0) == 99.33

    # 3. Strong negative logit (from user prompt) -> low confidence
    assert logit_to_confidence_percentage(-9.8564) == 0.01

    # 4. Extreme overflow protection
    assert logit_to_confidence_percentage(1000.0) == 100.0
    assert logit_to_confidence_percentage(-1000.0) == 0.0

    print("All sigmoid normalization edge cases passed!")

if __name__ == "__main__":
    test_sigmoid_normalization_edge_cases()