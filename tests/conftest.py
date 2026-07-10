import pytest
import numpy as np

@pytest.fixture
def p():
    """Standard ATM parameters. S=100, K=100, T=0.5yr, r=5%, σ=20%."""
    return dict(S=100.0, K=100.0, T=0.5, r=0.05, sigma=0.20, q=0.0)

@pytest.fixture
def hull():
    """Hull (2018) Table 15.1 case. BSM call ≈ $4.76, put ≈ $0.81."""
    return dict(S=42.0, K=40.0, T=0.5, r=0.10, sigma=0.20, q=0.0,
                call_expected=4.76, put_expected=0.81, tol=0.02)