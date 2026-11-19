"""Forecast each GTA VI trophy's Exophase earn rate at a case's target.

A rate is split into the two things that move it, the players Exophase
tracks and the players among them who earned the trophy:

    rate_i(t) = earners_i(t) / players(t)

A read gives players_tracked and every rate, so earners_i = rate_i *
players / 100 is known at every read. The page only moves when Exophase
refreshes it, so consecutive identical reads collapse into one state.

Players: the tracked population grows with time since release. Where the
history has enough distinct states, log players is fitted against
log(1 + hours since release) over the last day of states and read at the
target (with the implied daily growth capped). Where the page has been
frozen (a single state), the population is assumed to double about every
two hours since the frozen state was first read: a launch-night
population that has not been refreshed in hours is a backlog, and the
next refresh releases it.

Earners: where the history has enough distinct states, each trophy's
rate is forecast three ways and the three are averaged on the logit
scale: earners against players (a power law), the logit rate against
log(1 + hours), and a saturating curve of the rate against hours (a
ceiling each trophy approaches). Where the page is frozen, new players
are assumed to earn each trophy at a small fraction of the rate the
tracked players show, and the trophy every tracked player holds stays
at 100%.

Spread: the quantiles sit on the logit scale around the median, with a
width built from the fit residuals, the plausible range of the tracked
population at the target, and a term that grows with the horizon.

Run as `python3 -P -s main.py` (Python 3.13, standard library only, no
network). Reads every *.in in $MAKEFUN_INPUT_DIR (else /input) and writes
one <id>.out per case to $MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Launch-night prior: the tracked population doubles about this often
# while the page is frozen.
DOUBLING_HOURS = 2.0
# Plausible doubling times either side of it, for the spread.
DOUBLING_HOURS_LO = 1.25
DOUBLING_HOURS_HI = 3.5
# Fraction of a trophy's current rate at which new players earn it
# while the page is frozen.
FROZEN_CONVERSION = 0.06
# Floor conversion for a trophy nobody holds yet.
FROZEN_CONVERSION_FLOOR = 0.002
# Prior power-law slope of earners against players, used when the
# history cannot identify one.
BETA_PRIOR = 0.5
# Multiplier on the total logit spread (tuned on the practice cases).
SPREAD_SCALE = 0.8


def parse_time(text):
    """ISO 8601 UTC timestamp -> epoch seconds."""
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def release_epoch(case):
    day = case["game"]["release_date"]
    return datetime.fromisoformat(day + "T00:00:00+00:00").timestamp()


def logit(pct):
    p = min(99.99, max(0.01, pct)) / 100.0
    return math.log(p / (1.0 - p))


def logistic(x):
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def norm_ppf(p):
    """Standard normal quantile (Acklam's approximation), good to ~1e-9."""
    a = [-3.969683028665376e01, 2.209460984245205e02,
         -2.759285984469534e02, 1.383577518672690e02,
         -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02,
         -1.556989798598866e02, 6.680131188771972e01,
         -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e00, -2.549732539343734e00,
         4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e00, 3.754408661907416e00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as ints."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def least_squares(xs, ys):
    """Least-squares line y ~ a + b*x; returns (a, b, residual std)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2))
    else:
        resid = 0.0
    return a, b, resid


def distinct_states(case):
    """Collapse consecutive identical reads into states.

    Returns [{at, hours, players, earners, rates}] where hours is the
    time since release and earners the per-trophy counts.
    """
    release = release_epoch(case)
    states = []
    for read in case["history"]:
        players = read["players_tracked"]
        rates = read["rates"]
        key = (players, tuple(rates))
        if states and states[-1]["key"] == key:
            continue
        earners = []
        for r in rates:
            if r is None:
                earners.append(None)
            else:
                earners.append(r * players / 100.0)
        states.append({
            "key": key,
            "at": parse_time(read["at"]),
            "hours": max(0.0, (parse_time(read["at"]) - release) / 3600.0),
            "players": players,
            "earners": earners,
            "rates": rates,
        })
    return states


def fit_window(states):
    """The states a fit uses: the last day of them, or all when fewer."""
    last_at = states[-1]["at"]
    window = [s for s in states if s["at"] >= last_at - 24 * 3600]
    if len(window) < 3:
        window = states
    return window


def population_forecast(states, target_epoch, release, horizon_hours):
    """Forecast players tracked at the target.

    Returns (players, spread): the central estimate and the log-scale
    half-width of the plausible range (about one sigma). The central
    estimate is a power law (log players against log(1 + hours)) fitted
    on the last day of states, whose growth decelerates with time; the
    spread is the disagreement between that fit, a power law over all
    states, and an exponential fit, so it reflects how uncertain the
    extrapolation is rather than how smooth the fit looks.
    """
    last = states[-1]
    p_last = float(last["players"])
    if len(states) >= 3:
        window = fit_window(states)
        target_hours = max(0.0, (target_epoch - release) / 3600.0)
        estimates = []
        xs = [math.log1p(s["hours"]) for s in window]
        ys = [math.log(s["players"]) for s in window]
        a, b, _resid = least_squares(xs, ys)
        estimates.append(a + b * math.log1p(target_hours))
        xs = [math.log1p(s["hours"]) for s in states]
        ys_all = [math.log(s["players"]) for s in states]
        a, b, _resid = least_squares(xs, ys_all)
        estimates.append(a + b * math.log1p(target_hours))
        xs = [s["hours"] for s in window]
        a, b, _resid = least_squares(xs, ys)
        estimates.append(a + b * target_hours)
        players = math.exp(estimates[0])
        # Do not let the fit run away: cap the implied daily growth.
        cap = p_last * (2.0 ** (horizon_hours / 24.0))
        players = min(players, cap)
        players = max(players, p_last)
        spread = 0.5 * (max(estimates) - min(estimates))
        spread = min(0.8, max(0.15, spread))
        return players, spread
    # Frozen page: the population doubles about every DOUBLING_HOURS
    # since the frozen state was first read.
    frozen_hours = max(0.0, (target_epoch - states[0]["at"]) / 3600.0)
    players = p_last * (2.0 ** (frozen_hours / DOUBLING_HOURS))
    players = min(players, p_last * 12.0)
    p_lo = p_last * (2.0 ** (frozen_hours / DOUBLING_HOURS_HI))
    p_hi = p_last * (2.0 ** (frozen_hours / DOUBLING_HOURS_LO))
    spread = 0.25 * math.log(p_hi / p_lo) if p_hi > p_lo else 0.10
    return players, spread


def rate_forecast(states, idx, p_star, target_hours):
    """Forecast one trophy's earn rate (percent) at the target.

    Three views of the same history are averaged on the logit scale:
    earners against players (a power law), the logit rate against
    log(1 + hours) (a trend that keeps rising), and a saturating curve
    of the rate against hours (a ceiling each trophy approaches).
    Returns (rate, fit_spread).
    """
    last = states[-1]
    e_last = last["earners"][idx]
    if e_last is None:
        e_last = 0.0
    r_last = last["rates"][idx]
    if r_last is None:
        r_last = 0.0
    p_last = float(last["players"])
    window = fit_window(states)
    logits = []
    spreads = []
    # View 1: earners against players.
    xs = []
    ys = []
    for s in window:
        e = s["earners"][idx]
        if e is None:
            continue
        xs.append(math.log(s["players"]))
        ys.append(math.log(e + 0.5))
    if len(xs) >= 3 and (max(xs) - min(xs)) >= 0.2 and len(set(ys)) >= 2:
        a, b, resid = least_squares(xs, ys)
        beta = min(1.5, max(-0.5, b))
        my = sum(ys) / len(ys)
        mx = sum(xs) / len(xs)
        earners = math.exp(my + beta * (math.log(p_star) - mx)) - 0.5
        earners = max(0.0, min(p_star, earners))
        logits.append(logit(100.0 * earners / p_star))
        spreads.append(max(resid, 0.15))
    elif len(xs) >= 1:
        earners = e_last * ((p_star / p_last) ** BETA_PRIOR)
        earners = max(0.0, min(p_star, earners))
        logits.append(logit(100.0 * earners / p_star))
        spreads.append(0.30)
    # View 2: logit rate against log(1 + hours).
    xs = []
    ys = []
    for s in window:
        r = s["rates"][idx]
        if r is None:
            continue
        xs.append(math.log1p(s["hours"]))
        ys.append(logit(r))
    if len(xs) >= 3 and (max(xs) - min(xs)) >= 0.2:
        a, b, resid = least_squares(xs, ys)
        b = min(3.0, max(-1.0, b))
        my = sum(ys) / len(ys)
        mx = sum(xs) / len(xs)
        logits.append(my + b * (math.log1p(target_hours) - mx))
        spreads.append(max(resid, 0.15))
    # View 3: saturating rate against hours, r = c * (1 - exp(-h/tau)).
    pts = []
    for s in window:
        r = s["rates"][idx]
        if r is None:
            continue
        pts.append((s["hours"], r / 100.0))
    if len(pts) >= 3:
        best = None
        for tau in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048):
            num = 0.0
            den = 0.0
            for h, r in pts:
                x = 1.0 - math.exp(-h / tau)
                num += r * x
                den += x * x
            if den <= 0:
                continue
            c = num / den
            sse = 0.0
            for h, r in pts:
                x = 1.0 - math.exp(-h / tau)
                sse += (r - c * x) ** 2
            if best is None or sse < best[0]:
                best = (sse, tau, c)
        if best is not None:
            _sse, tau, c = best
            c = min(1.0, max(0.0, c))
            rate3 = c * (1.0 - math.exp(-target_hours / tau))
            logits.append(logit(100.0 * rate3))
            resid3 = 0.0
            for h, r in pts:
                fit = c * (1.0 - math.exp(-h / tau))
                resid3 += (logit(100.0 * max(r, 0.0001)) - logit(100.0 * max(fit, 0.0001))) ** 2
            resid3 = math.sqrt(resid3 / len(pts))
            spreads.append(max(resid3, 0.15))
    if not logits:
        return max(0.01, r_last), 0.30
    median_logit = sum(logits) / len(logits)
    rate = logistic(median_logit)
    return min(99.99, max(0.01, rate)), max(spreads)


def forecast_case(case):
    if not case["history"]:
        # No reads at all: no signal; a wide band around 1%.
        rows = []
        for trophy in case["trophies"]:
            rows.append({"id": trophy["id"], "q": [0.01, 0.1, 1.0, 5.0, 20.0]})
        return {"trophies": rows}
    release = release_epoch(case)
    states = distinct_states(case)
    target_epoch = parse_time(case["target"])
    horizon_hours = case["horizon_minutes"] / 60.0
    target_hours = max(0.0, (target_epoch - release) / 3600.0)
    p_star, s_pop = population_forecast(states, target_epoch, release, horizon_hours)
    s_time = 0.12 * math.sqrt(case["horizon_minutes"] / 1440.0)
    levels = case["levels"]
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        always_full = all(
            s["rates"][i] is not None and s["rates"][i] >= 100.0 for s in states
        )
        if always_full:
            median = 99.99
            spread = 0.02
        elif len(states) >= 3:
            median, s_fit = rate_forecast(states, i, p_star, target_hours)
            spread = math.sqrt(s_fit ** 2 + s_pop ** 2 + s_time ** 2)
            spread = min(1.2, max(0.15, spread * SPREAD_SCALE))
        else:
            # Frozen page: new players earn at a small fraction of the
            # rate the tracked players show; the trophy everyone holds
            # stays at 100%.
            last = states[-1]
            r_last = last["rates"][i]
            if r_last is None:
                r_last = 0.0
            if r_last >= 100.0:
                median = 99.99
                spread = 0.02
            else:
                e_last = last["earners"][i]
                if e_last is None:
                    e_last = 0.0
                p_last = float(last["players"])
                conv = max(r_last / 100.0, FROZEN_CONVERSION_FLOOR) * FROZEN_CONVERSION
                earners = e_last + conv * (p_star - p_last)
                earners = max(0.0, min(p_star, earners))
                median = min(99.99, max(0.01, 100.0 * earners / p_star))
                spread = math.sqrt(0.30 ** 2 + s_pop ** 2 + s_time ** 2)
                spread = min(1.2, max(0.15, spread * SPREAD_SCALE))
        centre = logit(median)
        qs = []
        prev = 0.01
        for tau in levels:
            q = logistic(centre + norm_ppf(tau) * spread)
            q = min(99.99, max(0.01, q))
            q = max(prev, q)
            qs.append(round4(q))
            prev = qs[-1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
