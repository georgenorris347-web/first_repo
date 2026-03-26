from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

import matplotlib.pyplot as plt
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PLOTS_DIR = ROOT / "plots"
REPORTS_DIR = ROOT / "reports"


SERIES = {
    "GDPC1": {
        "label": "Real GDP",
        "freq": "quarterly",
        "source": "U.S. Bureau of Economic Analysis via FRED",
    },
    "UNRATE": {
        "label": "Unemployment Rate",
        "freq": "monthly",
        "source": "U.S. Bureau of Labor Statistics via FRED",
    },
    "PAYEMS": {
        "label": "Nonfarm Payrolls",
        "freq": "monthly",
        "source": "U.S. Bureau of Labor Statistics via FRED",
    },
    "CPIAUCSL": {
        "label": "CPI All Urban Consumers",
        "freq": "monthly",
        "source": "U.S. Bureau of Labor Statistics via FRED",
    },
    "PCEPILFE": {
        "label": "Core PCE Price Index",
        "freq": "monthly",
        "source": "U.S. Bureau of Economic Analysis via FRED",
    },
    "FEDFUNDS": {
        "label": "Effective Federal Funds Rate",
        "freq": "monthly",
        "source": "Board of Governors of the Federal Reserve via FRED",
    },
    "T10Y2Y": {
        "label": "10-Year Treasury Minus 2-Year Treasury",
        "freq": "daily",
        "source": "Board of Governors of the Federal Reserve via FRED",
    },
    "INDPRO": {
        "label": "Industrial Production Index",
        "freq": "monthly",
        "source": "Board of Governors of the Federal Reserve via FRED",
    },
    "RSAFS": {
        "label": "Retail and Food Services Sales",
        "freq": "monthly",
        "source": "U.S. Census Bureau via FRED",
    },
    "ICSA": {
        "label": "Initial Claims",
        "freq": "weekly",
        "source": "U.S. Department of Labor via FRED",
    },
    "HOUST": {
        "label": "Housing Starts",
        "freq": "monthly",
        "source": "U.S. Census Bureau via FRED",
    },
}


@dataclass
class LatestPoint:
    name: str
    date: pd.Timestamp
    value: float
    note: str


def fetch_series(series_id: str) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    raw_path = RAW_DIR / f"{series_id}.csv"
    raw_path.write_bytes(response.content)
    df = pd.read_csv(raw_path)
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna()


def latest(df: pd.DataFrame) -> tuple[pd.Timestamp, float]:
    row = df.dropna().iloc[-1]
    return pd.Timestamp(row["date"]), float(row["value"])


def annualized_three_month(series: pd.Series) -> float:
    recent = series.dropna().tail(4)
    if len(recent) < 4:
        return float("nan")
    return ((recent.iloc[-1] / recent.iloc[-4]) ** 4 - 1) * 100


def annualized_three_month_average(level_series: pd.Series) -> float:
    recent = level_series.dropna().tail(13)
    if len(recent) < 13:
        return float("nan")
    return recent.diff().tail(3).mean()


def build_monthly_frame(series_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    monthly = {}
    for series_id, df in series_frames.items():
        freq = SERIES[series_id]["freq"]
        s = df.set_index("date")["value"].sort_index()
        if freq == "daily":
            monthly[series_id] = s.resample("ME").mean()
        elif freq == "weekly":
            monthly[series_id] = s.resample("ME").mean()
        elif freq == "quarterly":
            monthly[series_id] = s.resample("ME").ffill()
        else:
            monthly[series_id] = s.resample("ME").last()
    frame = pd.DataFrame(monthly).sort_index()
    frame["gdp_yoy"] = frame["GDPC1"].pct_change(12, fill_method=None) * 100
    frame["gdp_qoq_annualized"] = ((frame["GDPC1"] / frame["GDPC1"].shift(3)) ** 4 - 1) * 100
    frame["payrolls_mom_change"] = frame["PAYEMS"].diff()
    frame["payrolls_3m_avg_change"] = frame["payrolls_mom_change"].rolling(3).mean()
    frame["cpi_yoy"] = frame["CPIAUCSL"].pct_change(12, fill_method=None) * 100
    frame["cpi_3m_ann"] = frame["CPIAUCSL"].rolling(4).apply(
        lambda x: ((x.iloc[-1] / x.iloc[0]) ** 4 - 1) * 100 if x.iloc[0] > 0 else pd.NA,
        raw=False,
    )
    frame["core_pce_yoy"] = frame["PCEPILFE"].pct_change(12, fill_method=None) * 100
    frame["core_pce_3m_ann"] = frame["PCEPILFE"].rolling(4).apply(
        lambda x: ((x.iloc[-1] / x.iloc[0]) ** 4 - 1) * 100 if x.iloc[0] > 0 else pd.NA,
        raw=False,
    )
    frame["real_retail_sales_index"] = (frame["RSAFS"] / frame["CPIAUCSL"]) * 100
    frame["real_retail_sales_yoy"] = frame["real_retail_sales_index"].pct_change(12, fill_method=None) * 100
    frame["industrial_production_yoy"] = frame["INDPRO"].pct_change(12, fill_method=None) * 100
    frame["housing_starts_yoy"] = frame["HOUST"].pct_change(12, fill_method=None) * 100
    frame["claims_4w_ma"] = frame["ICSA"].rolling(1).mean()
    frame["real_policy_rate_proxy"] = frame["FEDFUNDS"] - frame["core_pce_yoy"]
    return frame


def build_plot_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (11, 6),
            "axes.facecolor": "#faf8f2",
            "figure.facecolor": "#f4efe3",
            "axes.edgecolor": "#403a2f",
            "axes.labelcolor": "#1f1f1f",
            "xtick.color": "#2f2f2f",
            "ytick.color": "#2f2f2f",
            "grid.color": "#bdb3a3",
            "axes.titleweight": "bold",
            "font.size": 11,
        }
    )


def save_line_plot(
    df: pd.DataFrame,
    columns: list[str],
    title: str,
    ylabel: str,
    output_name: str,
    labels: list[str] | None = None,
    zero_line: bool = False,
) -> None:
    labels = labels or columns
    fig, ax = plt.subplots()
    palette = ["#355c7d", "#c06c84", "#f67280", "#6c5b7b"]
    for idx, col in enumerate(columns):
        ax.plot(df.index, df[col], label=labels[idx], linewidth=2.5, color=palette[idx % len(palette)])
    if zero_line:
        ax.axhline(0, color="#403a2f", linewidth=1.2, linestyle="--", alpha=0.75)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / output_name, dpi=160)
    plt.close(fig)


def format_month_date(ts: pd.Timestamp) -> str:
    return ts.strftime("%B %Y")


def format_quarter_date(ts: pd.Timestamp) -> str:
    quarter = (ts.month - 1) // 3 + 1
    return f"Q{quarter} {ts.year}"


def compute_latest_points(frame: pd.DataFrame, raw: dict[str, pd.DataFrame]) -> list[LatestPoint]:
    gdp_date, gdp_value = latest(raw["GDPC1"])
    unrate_date, unrate_value = latest(raw["UNRATE"])
    payroll_date, _ = latest(raw["PAYEMS"])
    payroll_3m = frame.loc[payroll_date.to_period("M").to_timestamp("M"), "payrolls_3m_avg_change"]
    cpi_date, _ = latest(raw["CPIAUCSL"])
    cpi_yoy = frame.loc[cpi_date.to_period("M").to_timestamp("M"), "cpi_yoy"]
    core_pce_date, _ = latest(raw["PCEPILFE"])
    core_pce_yoy = frame.loc[core_pce_date.to_period("M").to_timestamp("M"), "core_pce_yoy"]
    ffr_date, ffr_value = latest(raw["FEDFUNDS"])
    spread_date, spread_value = latest(raw["T10Y2Y"])
    ip_date, _ = latest(raw["INDPRO"])
    ip_yoy = frame.loc[ip_date.to_period("M").to_timestamp("M"), "industrial_production_yoy"]
    retail_date, _ = latest(raw["RSAFS"])
    retail_yoy = frame.loc[retail_date.to_period("M").to_timestamp("M"), "real_retail_sales_yoy"]

    return [
        LatestPoint("Real GDP YoY", gdp_date, frame.loc[gdp_date.to_period("M").to_timestamp("M"), "gdp_yoy"], "quarterly"),
        LatestPoint("Unemployment rate", unrate_date, unrate_value, "monthly"),
        LatestPoint("Payroll growth, 3m avg", payroll_date, payroll_3m, "thousands of jobs"),
        LatestPoint("CPI inflation YoY", cpi_date, cpi_yoy, "monthly"),
        LatestPoint("Core PCE inflation YoY", core_pce_date, core_pce_yoy, "monthly"),
        LatestPoint("Fed funds rate", ffr_date, ffr_value, "monthly average"),
        LatestPoint("10Y-2Y spread", spread_date, spread_value, "daily"),
        LatestPoint("Industrial production YoY", ip_date, ip_yoy, "monthly"),
        LatestPoint("Real retail sales YoY", retail_date, retail_yoy, "monthly"),
    ]


def generate_report(frame: pd.DataFrame, raw: dict[str, pd.DataFrame]) -> str:
    latest_points = compute_latest_points(frame, raw)
    latest_map = {point.name: point for point in latest_points}

    latest_gdp_qoq = frame["gdp_qoq_annualized"].dropna().iloc[-1]
    latest_unrate = latest_map["Unemployment rate"].value
    recent_unrate_low = frame["UNRATE"].dropna().tail(12).min()
    payroll_3m_thousands = latest_map["Payroll growth, 3m avg"].value
    payroll_3m_jobs = payroll_3m_thousands * 1000
    cpi_yoy = latest_map["CPI inflation YoY"].value
    cpi_3m = frame["cpi_3m_ann"].dropna().iloc[-1]
    core_pce_yoy = latest_map["Core PCE inflation YoY"].value
    core_pce_3m = frame["core_pce_3m_ann"].dropna().iloc[-1]
    fedfunds = latest_map["Fed funds rate"].value
    spread = latest_map["10Y-2Y spread"].value
    real_policy = frame["real_policy_rate_proxy"].dropna().iloc[-1]
    ip_yoy = latest_map["Industrial production YoY"].value
    retail_yoy = latest_map["Real retail sales YoY"].value
    housing_yoy = frame["housing_starts_yoy"].dropna().iloc[-1]
    claims = raw["ICSA"].iloc[-1]

    growth_signal = "still expanding" if latest_gdp_qoq > 1.5 else "cooling materially"
    labor_signal = "softening but not breaking" if latest_unrate <= recent_unrate_low + 0.6 else "showing clearer stress"
    inflation_signal = "disinflation is continuing" if core_pce_3m < core_pce_yoy else "inflation momentum has re-accelerated somewhat"
    rates_signal = "policy still looks meaningfully restrictive" if real_policy > 1 else "policy still leans restrictive, but not by a wide margin"
    inflation_nuance = (
        "When short-run measures run below year-over-year inflation, it usually means disinflation is still unfolding even if the headline year-over-year rate remains above target."
        if core_pce_3m < core_pce_yoy
        else "When short-run measures run above year-over-year inflation, it usually means the last few months have been firmer than the broader 12-month trend, which is a reminder that the path back to target is rarely smooth."
    )
    policy_nuance = (
        "Real short rates remain clearly positive, which is consistent with softer housing activity, a mixed industrial backdrop, and a broader economy that can keep growing while becoming more rate-sensitive."
        if real_policy > 1
        else "Real short rates remain positive, but only modestly so, which suggests monetary policy may still be restraining demand without delivering the kind of heavy drag usually associated with a sharp downturn."
    )

    report = dedent(
        f"""
        # U.S. Economy Snapshot

        Generated from current public data on {pd.Timestamp.today().strftime("%B %d, %Y")}.

        ## Executive Take

        The U.S. economy appears to be {growth_signal}, with a labor market that is {labor_signal}. Inflation has come down from its post-pandemic highs, and the latest short-run price readings suggest that {inflation_signal}. At the same time, interest rates remain elevated enough that {rates_signal}, which helps explain why growth-sensitive sectors such as housing and manufacturing are not sending an unambiguously strong signal.

        ## Current Readings

        - Real GDP was up **{latest_map["Real GDP YoY"].value:.2f}% year over year** as of **{format_quarter_date(latest_map["Real GDP YoY"].date)}**, with the latest quarterly annualized pace at **{latest_gdp_qoq:.2f}%**.
        - The unemployment rate was **{latest_unrate:.1f}%** in **{format_month_date(latest_map["Unemployment rate"].date)}**, compared with a 12-month low of **{recent_unrate_low:.1f}%**.
        - Nonfarm payrolls increased by an average of **{payroll_3m_jobs:,.0f}** jobs per month over the latest three months through **{format_month_date(latest_map["Payroll growth, 3m avg"].date)}**.
        - CPI inflation was **{cpi_yoy:.2f}% year over year** in **{format_month_date(latest_map["CPI inflation YoY"].date)}**, while the 3-month annualized pace was **{cpi_3m:.2f}%**.
        - Core PCE inflation was **{core_pce_yoy:.2f}% year over year** in **{format_month_date(latest_map["Core PCE inflation YoY"].date)}**, with a 3-month annualized pace of **{core_pce_3m:.2f}%**.
        - The effective federal funds rate averaged **{fedfunds:.2f}%** in **{format_month_date(latest_map["Fed funds rate"].date)}**. Against core PCE inflation, that implies a rough real policy rate of **{real_policy:.2f}%**.
        - The 10-year minus 2-year Treasury spread was **{spread:.2f} percentage points** on **{latest_map["10Y-2Y spread"].date.strftime("%B %d, %Y")}**.
        - Industrial production was **{ip_yoy:.2f}% year over year** in **{format_month_date(latest_map["Industrial production YoY"].date)}**.
        - Real retail sales were **{retail_yoy:.2f}% year over year** in **{format_month_date(latest_map["Real retail sales YoY"].date)}**.
        - Housing starts were **{housing_yoy:.2f}% year over year** in **{frame["housing_starts_yoy"].dropna().index[-1].strftime("%B %Y")}**.
        - Initial jobless claims were **{claims["value"]:,.0f}** in the latest available week ending **{pd.Timestamp(claims["date"]).strftime("%B %d, %Y")}**.

        ## Nuanced Read

        1. **Growth is slower, not necessarily stalled.** Real GDP is still above year-ago levels, but the most recent quarterly annualized pace is much more informative about current momentum than the year-over-year number. That leaves the economy in a cooling phase rather than a clean boom.

        2. **The labor market is losing altitude gradually.** Unemployment remains low by historical standards, but it has moved off its cycle floor. Payroll growth is still positive, yet the three-month average of roughly {payroll_3m_jobs:,.0f} jobs per month is a better gauge of trend than any single month, and it points to a slower hiring environment.

        3. **Inflation momentum needs to be read on more than one horizon.** The gap between year-over-year inflation and the 3-month annualized pace matters. {inflation_nuance}

        4. **Policy restraint is still visible, but less overwhelming than before.** A fed funds rate near {fedfunds:.2f}% with core PCE inflation at {core_pce_yoy:.2f}% means real short rates remain positive. {policy_nuance}

        5. **Consumers are not sending a recessionary signal yet.** Real retail sales matter more than nominal sales because they strip out price effects. A positive real spending trend would suggest households are still supporting the expansion; a flat or negative one would warn that consumption is no longer carrying the cycle as effectively.

        6. **The rate curve is useful, but it should not be read mechanically.** The 10Y-2Y spread at {spread:.2f} percentage points captures market expectations about future policy and growth. Its signal is more informative when paired with labor data, claims, and credit-sensitive activity rather than treated as a stand-alone recession timer.

        ## Files

        - `plots/growth_and_activity.png`
        - `plots/labor_market.png`
        - `plots/inflation.png`
        - `plots/rates_and_curve.png`
        - `plots/consumer_and_housing.png`

        ## Sources

        Public series were downloaded from FRED CSV endpoints, which mirror source-agency releases:
        - BEA: GDP and Core PCE
        - BLS: unemployment, payrolls, CPI
        - Federal Reserve: fed funds, industrial production, Treasury spread
        - Census Bureau: retail sales and housing starts
        - Department of Labor: initial claims
        """
    ).strip()

    return report


def main() -> None:
    for directory in [RAW_DIR, PROCESSED_DIR, PLOTS_DIR, REPORTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    raw = {series_id: fetch_series(series_id) for series_id in SERIES}
    monthly = build_monthly_frame(raw)
    monthly.to_csv(PROCESSED_DIR / "monthly_macro_snapshot.csv", index_label="date")

    build_plot_style()
    recent = monthly.loc[monthly.index >= (monthly.index.max() - pd.DateOffset(years=10))]

    save_line_plot(
        recent,
        ["gdp_yoy", "industrial_production_yoy"],
        "Growth And Activity",
        "Percent change",
        "growth_and_activity.png",
        labels=["Real GDP YoY", "Industrial Production YoY"],
        zero_line=True,
    )
    save_line_plot(
        recent,
        ["UNRATE", "payrolls_3m_avg_change"],
        "Labor Market",
        "Level / monthly job change",
        "labor_market.png",
        labels=["Unemployment Rate", "Payrolls 3M Avg Change"],
    )
    save_line_plot(
        recent,
        ["cpi_yoy", "core_pce_yoy", "cpi_3m_ann", "core_pce_3m_ann"],
        "Inflation",
        "Percent",
        "inflation.png",
        labels=["CPI YoY", "Core PCE YoY", "CPI 3M Ann.", "Core PCE 3M Ann."],
        zero_line=True,
    )
    save_line_plot(
        recent,
        ["FEDFUNDS", "T10Y2Y", "real_policy_rate_proxy"],
        "Rates And Curve",
        "Percent / percentage points",
        "rates_and_curve.png",
        labels=["Fed Funds", "10Y-2Y Spread", "Real Policy Rate Proxy"],
        zero_line=True,
    )
    save_line_plot(
        recent,
        ["real_retail_sales_yoy", "housing_starts_yoy"],
        "Consumer And Housing",
        "Percent change",
        "consumer_and_housing.png",
        labels=["Real Retail Sales YoY", "Housing Starts YoY"],
        zero_line=True,
    )

    report = generate_report(monthly, raw)
    (REPORTS_DIR / "us_economy_snapshot.md").write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
