"""
Validation suite: bond pricer checked against known, independently-verifiable examples.
Run: python -m pytest tests/test_bond_pricer.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from bond_pricer import Bond


def test_par_bond():
    """A bond priced at par should have YTM == coupon rate."""
    b = Bond(face_value=100, coupon_rate=0.05, years_to_maturity=5, freq=2)
    price = b.clean_price(0.05)
    assert abs(price - 100.0) < 1e-6


def test_textbook_example_fabozzi():
    """10Y, 6% semi-annual coupon, priced to yield 8%. Textbook value ~86.41."""
    b = Bond(face_value=100, coupon_rate=0.06, years_to_maturity=10, freq=2)
    price = b.clean_price(0.08)
    assert abs(price - 86.41) < 0.01  # within 1 cent


def test_zero_coupon_bond():
    """Zero-coupon bond: price = FV / (1+y/f)^n. 10Y, 5% yield, semi-annual compounding."""
    b = Bond(face_value=100, coupon_rate=0.0, years_to_maturity=10, freq=2)
    price = b.clean_price(0.05)
    expected = 100 / (1.025 ** 20)
    assert abs(price - expected) < 1e-6


def test_ytm_roundtrip():
    """Solving for YTM from a computed price should recover the original YTM."""
    b = Bond(face_value=100, coupon_rate=0.045, years_to_maturity=7, freq=2)
    true_ytm = 0.052
    price = b.clean_price(true_ytm)
    recovered = b.ytm_from_price(price)
    assert abs(recovered - true_ytm) < 1e-8


def test_duration_less_than_maturity():
    """Macaulay duration of a coupon bond must be strictly less than its maturity."""
    b = Bond(face_value=100, coupon_rate=0.06, years_to_maturity=10, freq=2)
    assert b.macaulay_duration(0.08) < 10


def test_zero_coupon_duration_equals_maturity():
    """For a zero-coupon bond, Macaulay duration == maturity exactly."""
    b = Bond(face_value=100, coupon_rate=0.0, years_to_maturity=10, freq=2)
    assert abs(b.macaulay_duration(0.05) - 10.0) < 1e-9


def test_annual_vs_semiannual_consistency():
    """An annual-pay bond and equivalent semi-annual bond should have close (not identical) prices."""
    annual = Bond(face_value=100, coupon_rate=0.06, years_to_maturity=10, freq=1)
    semi = Bond(face_value=100, coupon_rate=0.06, years_to_maturity=10, freq=2)
    p_annual = annual.clean_price(0.08)
    p_semi = semi.clean_price(0.08)
    assert abs(p_annual - p_semi) < 1.0  # close but not equal, due to compounding difference


if __name__ == "__main__":
    import subprocess
    subprocess.run(["python3", "-m", "pytest", __file__, "-v"])
