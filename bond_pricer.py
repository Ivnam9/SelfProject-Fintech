"""
Bond Pricing Engine
--------------------
Implements clean price, dirty price, accrued interest, YTM (Newton-Raphson /
Brent's method), Macaulay duration, modified duration, convexity, and DV01
from first principles (no QuantLib).

Handles semi-annual and annual coupon frequencies.
"""

import numpy as np
from scipy.optimize import brentq


class Bond:
    def __init__(self, face_value, coupon_rate, years_to_maturity, freq=2):
        """
        face_value        : e.g. 100
        coupon_rate        : annual coupon rate as decimal, e.g. 0.05 for 5%
        years_to_maturity  : in years, e.g. 9.5
        freq               : coupon payments per year (2 = semi-annual, 1 = annual)
        """
        self.face_value = face_value
        self.coupon_rate = coupon_rate
        self.years_to_maturity = years_to_maturity
        self.freq = freq
        self.n_periods = round(years_to_maturity * freq)
        self.coupon_payment = face_value * coupon_rate / freq

    def cashflows(self):
        """Returns (period_times_in_years, cashflow_amounts)."""
        periods = np.arange(1, self.n_periods + 1)
        times = periods / self.freq
        cfs = np.full(self.n_periods, self.coupon_payment)
        cfs[-1] += self.face_value
        return times, cfs

    def clean_price(self, ytm):
        """Present value of all cashflows discounted at yield-to-maturity (annualized, compounded `freq` times/yr)."""
        times, cfs = self.cashflows()
        per_rate = ytm / self.freq
        n = times * self.freq
        pv = np.sum(cfs / (1 + per_rate) ** n)
        return pv

    def accrued_interest(self, frac_period_elapsed):
        """
        frac_period_elapsed: fraction of current coupon period that has elapsed (0 to 1).
        """
        return self.coupon_payment * frac_period_elapsed

    def dirty_price(self, ytm, frac_period_elapsed=0.0):
        return self.clean_price(ytm) + self.accrued_interest(frac_period_elapsed)

    def ytm_from_price(self, market_price, is_dirty=False, frac_period_elapsed=0.0,
                        bracket=(1e-6, 2.0)):
        """
        Solve for yield-to-maturity given an observed market price using Brent's method
        (robust bracketed root-finder; used in place of hand-rolled Newton-Raphson for
        numerical stability, same underlying problem: f(ytm) = price(ytm) - market_price = 0).
        """
        target = market_price
        if is_dirty:
            def f(y):
                return self.dirty_price(y, frac_period_elapsed) - target
        else:
            def f(y):
                return self.clean_price(y) - target

        return brentq(f, bracket[0], bracket[1], xtol=1e-10)

    def macaulay_duration(self, ytm):
        times, cfs = self.cashflows()
        per_rate = ytm / self.freq
        n = times * self.freq
        disc_cfs = cfs / (1 + per_rate) ** n
        price = np.sum(disc_cfs)
        mac_dur = np.sum(times * disc_cfs) / price
        return mac_dur

    def modified_duration(self, ytm):
        mac_dur = self.macaulay_duration(ytm)
        return mac_dur / (1 + ytm / self.freq)

    def convexity(self, ytm):
        times, cfs = self.cashflows()
        per_rate = ytm / self.freq
        n = times * self.freq
        disc_cfs = cfs / (1 + per_rate) ** n
        price = np.sum(disc_cfs)
        # Convexity in annual terms (standard formula, using per-period discounting)
        conv = np.sum(disc_cfs * n * (n + 1)) / (price * (1 + per_rate) ** 2 * self.freq ** 2)
        return conv

    def dv01(self, ytm, bump=1e-4):
        """Dollar value of a 1bp change in yield, via central finite difference."""
        p_up = self.clean_price(ytm + bump)
        p_down = self.clean_price(ytm - bump)
        return (p_down - p_up) / 2  # price change for +1bp move (bump=1bp here)


if __name__ == "__main__":
    # --- Validation against a known textbook bond example ---
    # 10-year, 6% semi-annual coupon bond, priced to yield 8% -> classic Fabozzi textbook example
    # Expected clean price ~= 86.41 (per 100 face value)
    b = Bond(face_value=100, coupon_rate=0.06, years_to_maturity=10, freq=2)
    price = b.clean_price(0.08)
    print(f"Test bond (6% coupon, 10Y, 8% YTM) clean price: {price:.4f}  [expected ~86.41]")

    ytm_est = b.ytm_from_price(price)
    print(f"Recovered YTM from price: {ytm_est:.6f}  [expected 0.08]")

    print(f"Macaulay duration: {b.macaulay_duration(0.08):.4f} years")
    print(f"Modified duration: {b.modified_duration(0.08):.4f}")
    print(f"Convexity: {b.convexity(0.08):.4f}")
    print(f"DV01 (per 100 face): {b.dv01(0.08):.6f}")
