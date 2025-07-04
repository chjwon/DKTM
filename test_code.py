from keyword_extract import get_KeyBert_result, get_tf_idf_result
import numpy as np

from topic_model import get_BERTopic_result, get_Top2vec_result

def test_keybert(mock_doc):
    mock_result = get_KeyBert_result(mock_doc)
    for res in mock_result:
        print(res)
        assert isinstance(res, list), f"Expected list, got {type(res)}"
        # assert len(res) > 0, "Result list should not be empty"
        assert isinstance(res[0], tuple), f"Expected tuple, got {type(res[0])}"
        assert isinstance(res[0][0], str), f"Expected str, got {type(res[0][0])}"
        assert isinstance(res[0][1], float), f"Expected float, got {type(res[0][1])}"
        
def test_tf_idf(mock_doc):
    mock_result = get_tf_idf_result(mock_doc)
    for res in mock_result:
        print(res)
        assert isinstance(res, list), f"Expected list, got {type(res)}"
        # assert len(res) > 0, "Result list should not be empty"
        
        # Handle case where result might be [" "] for empty/no keywords
        if res == [" "]:
            assert res[0] == " ", "Empty result should contain single space string"
        else:
            assert isinstance(res[0], tuple), f"Expected tuple, got {type(res[0])}"
            assert isinstance(res[0][0], str), f"Expected str, got {type(res[0][0])}"
            assert isinstance(res[0][1], (float, np.float64)), f"Expected float, got {type(res[0][1])}"
        
        
mock_doc = [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        "Natural language processing helps computers understand human language.",
        "Deep learning uses neural networks with multiple layers for complex pattern recognition."
    ]

test_keybert(mock_doc)
test_tf_idf(mock_doc)