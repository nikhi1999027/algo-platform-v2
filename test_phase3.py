"""
test_phase3.py — Standalone Unit Tests for Phase 3 Math
Run locally via: python test_phase3.py
"""
import pandas as pd
import numpy as np

# 1. RSI with flat-price protection (returns 50.0 instead of dividing by zero)
def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    if series.empty or series.nunique() <= 1:
        return pd.Series(50.0, index=series.index)
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    safe_loss = avg_loss.replace(0.0, 1e-6)
    rs = avg_gain / safe_loss
    return (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)

# 2. Overextension check (flags price > 1.8x ATR away from EMA20)
def is_overextended(price: float, ema20: float, atr: float, threshold: float = 1.8) -> bool:
    if atr <= 0:
        return False
    return (abs(price - ema20) / atr) > threshold

# 3. Position Sizer (honors maximum rupee risk and capital limits)
def calculate_position_size(capital: float, risk_pct: float, entry: float, stop: float) -> int:
    risk_per_share = abs(entry - stop)
    if risk_per_share <= 0 or entry <= 0:
        return 0
    max_risk_amount = capital * (risk_pct / 100.0)
    return int(min(max_risk_amount // risk_per_share, capital // entry))

# --- TEST SUITE ---
def run_all_tests():
    print("Running Mathematical Validation Tests...\n")
    
    # Test 1: Flat series check
    flat_data = pd.Series([100.0] * 30)
    rsi_val = calculate_rsi(flat_data).iloc[-1]
    assert rsi_val == 50.0, f"Expected 50.0, got {rsi_val}"
    print("✔ Test 1 Passed: Flat price correctly evaluated to neutral RSI 50.0.")

    # Test 2: Overextension veto
    # EMA is 1000, ATR is 20. Price at 1040 is 2.0x ATR away (should be vetoed)
    extended = is_overextended(price=1040.0, ema20=1000.0, atr=20.0, threshold=1.8)
    assert extended is True, "Price at 2.0x ATR should be marked overextended."
    print("✔ Test 2 Passed: 1.8x ATR overextension veto triggered accurately.")

    # Test 3: Capital sizing
    # Capital: 1,00,000 | 1% risk = 1,000 | Entry: 100, Stop: 90 (Risk/share = 10)
    # Expected shares = 1,000 / 10 = 100 shares
    shares = calculate_position_size(capital=100000.0, risk_pct=1.0, entry=100.0, stop=90.0)
    assert shares == 100, f"Expected 100 shares, got {shares}"
    print(f"✔ Test 3 Passed: Position size calculated exactly as {shares} shares.")

    print("\nAll Phase 3 calculations verified successfully!")

if __name__ == "__main__":
    run_all_tests()
