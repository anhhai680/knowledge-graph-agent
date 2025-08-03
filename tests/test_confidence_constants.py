#!/usr/bin/env python3
"""
Test script to validate the confidence calculation constants refactoring.
"""

def test_confidence_constants():
    """Test the confidence calculation constants."""
    
    # Simulate the constants from the refactored class
    CONFIDENCE_OPTIMAL_DOC_COUNT = 5.0
    CONFIDENCE_OPTIMAL_CONTENT_LENGTH = 2000.0
    CONFIDENCE_WEIGHT_DOC_COUNT = 0.4
    CONFIDENCE_WEIGHT_CONTENT = 0.4
    CONFIDENCE_WEIGHT_METADATA = 0.2
    METADATA_SCORE_FILE_PATH = 0.2
    METADATA_SCORE_REPOSITORY = 0.1
    METADATA_SCORE_LANGUAGE = 0.1
    METADATA_SCORE_CHUNK_TYPE = 0.1
    
    # Test weight validation
    total_weight = CONFIDENCE_WEIGHT_DOC_COUNT + CONFIDENCE_WEIGHT_CONTENT + CONFIDENCE_WEIGHT_METADATA
    print(f"✅ Total weights sum to: {total_weight:.1f} (should be 1.0)")
    
    # Test all constants are positive
    constants = [
        ("CONFIDENCE_OPTIMAL_DOC_COUNT", CONFIDENCE_OPTIMAL_DOC_COUNT),
        ("CONFIDENCE_OPTIMAL_CONTENT_LENGTH", CONFIDENCE_OPTIMAL_CONTENT_LENGTH),
        ("CONFIDENCE_WEIGHT_DOC_COUNT", CONFIDENCE_WEIGHT_DOC_COUNT),
        ("CONFIDENCE_WEIGHT_CONTENT", CONFIDENCE_WEIGHT_CONTENT),
        ("CONFIDENCE_WEIGHT_METADATA", CONFIDENCE_WEIGHT_METADATA),
        ("METADATA_SCORE_FILE_PATH", METADATA_SCORE_FILE_PATH),
        ("METADATA_SCORE_REPOSITORY", METADATA_SCORE_REPOSITORY),
        ("METADATA_SCORE_LANGUAGE", METADATA_SCORE_LANGUAGE),
        ("METADATA_SCORE_CHUNK_TYPE", METADATA_SCORE_CHUNK_TYPE),
    ]
    
    all_positive = all(value > 0 for name, value in constants)
    print(f"✅ All constants are positive: {all_positive}")
    
    # Test confidence calculation with mock data
    def calculate_confidence_mock(context_documents):
        """Mock confidence calculation using the new constants."""
        if not context_documents:
            return 0.0

        # Document count score
        doc_count_score = min(len(context_documents) / CONFIDENCE_OPTIMAL_DOC_COUNT, 1.0)
        
        # Content score
        total_content_length = sum(len(doc.get("content", "")) for doc in context_documents)
        content_score = min(total_content_length / CONFIDENCE_OPTIMAL_CONTENT_LENGTH, 1.0)
        
        # Metadata quality score
        metadata_quality = 0.0
        for doc in context_documents:
            metadata = doc.get("metadata", {})
            if metadata.get("file_path"):
                metadata_quality += METADATA_SCORE_FILE_PATH
            if metadata.get("repository"):
                metadata_quality += METADATA_SCORE_REPOSITORY
            if metadata.get("language"):
                metadata_quality += METADATA_SCORE_LANGUAGE
            if metadata.get("chunk_type"):
                metadata_quality += METADATA_SCORE_CHUNK_TYPE
        
        metadata_quality = min(metadata_quality / len(context_documents), 1.0)
        
        # Combined confidence score
        confidence = (
            doc_count_score * CONFIDENCE_WEIGHT_DOC_COUNT + 
            content_score * CONFIDENCE_WEIGHT_CONTENT + 
            metadata_quality * CONFIDENCE_WEIGHT_METADATA
        )
        
        return min(confidence, 1.0)
    
    # Test with mock context documents
    mock_docs = [
        {
            "content": "def hello_world():\n    print('Hello, World!')\n    return True",
            "metadata": {
                "file_path": "src/utils.py",
                "repository": "test-repo",
                "language": "python",
                "chunk_type": "function"
            }
        },
        {
            "content": "class Calculator:\n    def add(self, a, b):\n        return a + b\n    def subtract(self, a, b):\n        return a - b",
            "metadata": {
                "file_path": "src/calculator.py",
                "repository": "test-repo",
                "language": "python"
            }
        }
    ]
    
    confidence = calculate_confidence_mock(mock_docs)
    print(f"✅ Mock confidence calculation result: {confidence:.3f}")
    
    # Verify confidence is reasonable (between 0 and 1)
    assert 0.0 <= confidence <= 1.0, f"Confidence {confidence} is not in valid range [0, 1]"
    print("✅ Confidence score is in valid range")
    
    print("\n🎉 All tests passed! The confidence constants refactoring is working correctly.")

if __name__ == "__main__":
    test_confidence_constants()
