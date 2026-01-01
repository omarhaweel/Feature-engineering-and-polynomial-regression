# writing a test using pytest

import numpy as np
from lab_utils_multi import run_gradient_descent_feng


def test_gradient_descent_feng():
    """Test 1: Verify gradient descent finds correct w and b values."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([7, 8, 9])
    w, b = run_gradient_descent_feng(X, y, iterations=1000, alpha=0.01)
    expected_w = np.array([-1.63049253, 2.23678227])
    expected_b = 3.8673
    assert np.allclose(w, expected_w), f"w mismatch: got {w}, expected {expected_w}"
    assert np.allclose(b, expected_b), f"b mismatch: got {b}, expected {expected_b}"


if __name__ == "__main__":
    test_gradient_descent_feng()