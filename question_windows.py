"""Print how long recent tournament questions stayed open (diagnostic, read-only).

Tells us how often the bot must run to avoid missing questions. Never raises.
"""
import os
from datetime import datetime

import requests


def _t(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def report(tournaments=("33121", "minibench"), limit=40):
    token = os.getenv("METACULUS_TOKEN")
    for tid in tournaments:
        try:
            r = requests.get(
                "https://www.metaculus.com/api/posts/",
                params={"tournaments": tid, "limit": limit, "order_by": "-open_time", "with_cp": "false"},
                headers={"Authorization": f"Token {token}"} if token else {},
                timeout=30,
            )
            r.raise_for_status()
            rows = []
            for p in r.json().get("results", []):
                q = p.get("question") or {}
                o = _t(q.get("open_time") or p.get("open_time"))
                c = _t(q.get("scheduled_close_time") or p.get("scheduled_close_time"))
                if o and c:
                    rows.append((o, (c - o).total_seconds() / 3600, p.get("status"), bool(p.get("forecasts_count") is not None)))
            print(f"[windows] tournament {tid}: {len(rows)} recent questions")
            for o, hours, status, _ in rows[:25]:
                print(f"[windows]   opened {o:%Y-%m-%d %H:%M} UTC  open for {hours:6.1f} h  status {status}")
            if rows:
                hrs = sorted(h for _, h, _, _ in rows)
                print(f"[windows]   open-window hours: shortest {hrs[0]:.1f}, median {hrs[len(hrs) // 2]:.1f}, longest {hrs[-1]:.1f}")
        except Exception as e:  # diagnostic only
            print(f"[windows] tournament {tid}: could not read ({type(e).__name__}: {str(e)[:150]})")


if __name__ == "__main__":
    report()
