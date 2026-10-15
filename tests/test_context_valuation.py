import unittest

import numpy as np
import pandas as pd

from scripts import audit_context_valuation as audit


class ValuationTests(unittest.TestCase):
    def marks(self):
        return pd.DataFrame({"event_ms": np.arange(10)*audit.MINUTE, "open": 100., "close": 100., "high": 102., "low": 98.})

    def test_funding_sign_and_price_proxy(self):
        events = [{"fundingTime": 4*audit.MINUTE, "fundingRate": .001, "markPrice": ""}]
        marks = self.marks().set_index("event_ms")
        long = audit.funding_flows("long", audit.MINUTE, 8*audit.MINUTE, 100, events, marks)[0]
        short = audit.funding_flows("short", audit.MINUTE, 8*audit.MINUTE, 100, events, marks)[0]
        self.assertAlmostEqual(long["return_fraction"], -audit.RATIO*1.02*.001)
        self.assertAlmostEqual(short["return_fraction"], audit.RATIO*.98*.001)
        self.assertTrue(long["ohlc_proxy"])

    def test_exact_mark_and_negative_rate(self):
        events = [{"fundingTime": 4*audit.MINUTE, "fundingRate": -.001, "markPrice": "101"}]
        flow = audit.funding_flows("long", audit.MINUTE, 8*audit.MINUTE, 100, events, self.marks().set_index("event_ms"))[0]
        self.assertAlmostEqual(flow["return_fraction"], audit.RATIO*1.01*.001)
        self.assertFalse(flow["ohlc_proxy"])

    def test_boundary_keeps_charge_excludes_credit(self):
        events = [{"fundingTime": audit.MINUTE, "fundingRate": .001, "markPrice": "100"}]
        marks = self.marks().set_index("event_ms")
        self.assertEqual(audit.funding_flows("short", audit.MINUTE, 8*audit.MINUTE, 100, events, marks), [])
        self.assertTrue(audit.funding_flows("long", audit.MINUTE, 8*audit.MINUTE, 100, events, marks)[0]["uncertain_timing"])

    def trade(self):
        return pd.DataFrame([{"entry_timestamp": audit.MINUTE,
                              "exit_time_utc": pd.Timestamp(8*audit.MINUTE, unit="ms", tz="UTC").isoformat(),
                              "action": "long", "entry_price": 100., "return_pct": .002, "return_r": .4}])

    def test_empty_and_zero_funding_reconciliation(self):
        self.assertEqual(audit.audit_trades(pd.DataFrame(), [], self.marks())["trades"], 0)
        result = audit.audit_trades(self.trade(), [], self.marks())
        self.assertAlmostEqual(result["original_return_pct"], .2)
        self.assertAlmostEqual(result["return_delta_pp"], 0)
        self.assertLess(result["sampled_mark_drawdown_pct"], 0)

    def test_missing_minute_fails_and_exit_minute_excluded(self):
        with self.assertRaises(ValueError):
            audit.audit_trades(self.trade(), [], self.marks().drop(index=3))
        marks = self.marks()
        original = audit.audit_trades(self.trade(), [], marks)
        marks.loc[7:, "close"] = 1
        self.assertEqual(original, audit.audit_trades(self.trade(), [], marks))


if __name__ == "__main__":
    unittest.main()
