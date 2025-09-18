"""Basic tests"""
import pytest
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

def test_imports():
    try:
        from src.data.preprocessor import DiabetesDataPreprocessor
        from src.models.gradient_boosting import DiabetesGradientBoosting
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

def test_synthetic_data():
    from scripts.generate_synthetic_data import generate_synthetic_diabetes_data
    df = generate_synthetic_diabetes_data(50)
    assert len(df) == 50
    assert 'Resultado' in df.columns

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
