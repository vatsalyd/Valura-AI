"""
Portfolio Health Check Agent — the first specialist agent.

Speaks to the MONITOR and PROTECT halves of Valura's mission.

WHAT IT DOES:
- Receives user portfolio data (does NOT fetch it)
- Fetches LIVE market data via yfinance to calculate current values
- Computes: concentration risk, performance metrics, benchmark comparison
- Produces structured output with plain-language observations
- Handles empty portfolios (BUILD mode for user_004)
- Tailors commentary to user risk profile (yield focus for retirees)

WHY THIS ARCHITECTURE:
- Computation is separated from LLM — the structured output (concentration,
  performance, benchmark) is computed LOCALLY from market data, not generated
  by the LLM. This makes the numbers trustworthy and testable.
- The LLM is used ONLY for generating the plain-language summary and
  observations — turning structured data into human-readable insight.
- This means the agent works even if the LLM fails (fallback observations).

OUT-OF-THE-BOX CHOICES:
- Sector diversification analysis — not just concentration by position
- Currency exposure breakdown for multi-currency portfolios
- Risk-profile-aware observations (conservative vs aggressive commentary)
- Annualized return calculation using actual purchase dates
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, date
from typing import Any, AsyncIterator, Optional

from src.agents.base import BaseAgent
from src.models import (
    UserProfile,
    PortfolioHealthResponse,
    ConcentrationRisk,
    PerformanceMetrics,
    BenchmarkComparison,
    DiversificationBreakdown,
    Observation,
)
from src.market_data import get_ticker_data, get_benchmark_return, BENCHMARK_TICKERS, TickerData

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "This is not investment advice. The information provided is for "
    "educational and informational purposes only. Past performance does "
    "not guarantee future results. Please consult a qualified financial "
    "advisor before making investment decisions."
)


class PortfolioHealthAgent(BaseAgent):
    """Structured assessment of the user's portfolio health."""

    @property
    def name(self) -> str:
        return "portfolio_health"

    async def run(
        self,
        query: str,
        user: UserProfile,
        entities: dict[str, Any],
        llm: Any = None,
    ) -> dict[str, Any]:
        """Execute portfolio health check and return structured output."""

        # Handle empty portfolio — BUILD mode
        if not user.positions:
            return self._empty_portfolio_response(user)

        # Fetch live market data for all positions
        ticker_data = {}
        for pos in user.positions:
            td = get_ticker_data(pos.ticker)
            ticker_data[pos.ticker] = td

        # Compute all metrics
        concentration = self._compute_concentration(user, ticker_data)
        performance = self._compute_performance(user, ticker_data)
        benchmark = self._compute_benchmark(user, performance)
        diversification = self._compute_diversification(user, ticker_data)
        observations = self._generate_observations(
            user, concentration, performance, benchmark, diversification, ticker_data
        )

        # Generate LLM summary if available
        summary = await self._generate_summary(
            user, concentration, performance, benchmark, observations, llm
        )

        response = PortfolioHealthResponse(
            concentration_risk=concentration,
            performance=performance,
            benchmark_comparison=benchmark,
            diversification=diversification,
            observations=observations,
            summary=summary,
            disclaimer=DISCLAIMER,
        )

        return response.model_dump()

    async def stream(
        self,
        query: str,
        user: UserProfile,
        entities: dict[str, Any],
        llm: Any = None,
    ) -> AsyncIterator[str]:
        """Stream portfolio health response."""
        result = await self.run(query, user, entities, llm=llm)
        # Stream in logical sections for a better UX
        sections = [
            ("concentration_risk", result.get("concentration_risk", {})),
            ("performance", result.get("performance", {})),
            ("benchmark_comparison", result.get("benchmark_comparison", {})),
            ("diversification", result.get("diversification", {})),
            ("observations", result.get("observations", [])),
            ("summary", result.get("summary", "")),
            ("disclaimer", result.get("disclaimer", "")),
        ]
        for section_name, section_data in sections:
            chunk = {section_name: section_data}
            yield json.dumps(chunk, default=str)

    # ── Empty portfolio (BUILD mode) ──────────────────────────────────────

    def _empty_portfolio_response(self, user: UserProfile) -> dict[str, Any]:
        """For users with no positions — orient toward building a portfolio."""
        risk_text = {
            "conservative": (
                "Given your conservative risk profile, consider starting with a "
                "diversified bond ETF (like BND) and a broad market index fund "
                "(like VTI or VOO). A common starting point is a 60/40 or 40/60 "
                "stock-to-bond split."
            ),
            "moderate": (
                "With a moderate risk profile, a balanced portfolio might include "
                "70-80% in diversified equity index funds (like VTI or VOO) and "
                "20-30% in bonds (like BND). Consider adding international exposure "
                "with VXUS."
            ),
            "aggressive": (
                "With your aggressive risk profile, you have more flexibility for "
                "growth-oriented holdings. Consider starting with a broad market ETF "
                "(like QQQ or VTI) and adding individual positions as you learn. "
                "Even aggressive portfolios benefit from some diversification."
            ),
        }

        guidance = risk_text.get(user.risk_profile, risk_text["moderate"])

        return PortfolioHealthResponse(
            observations=[
                Observation(
                    severity="info",
                    text=(
                        f"Welcome, {user.name}! You don't have any positions yet — "
                        f"that's perfectly fine. Everyone starts somewhere."
                    ),
                ),
                Observation(
                    severity="info",
                    text=guidance,
                ),
                Observation(
                    severity="info",
                    text=(
                        "Start small, invest regularly, and focus on low-cost "
                        "diversified funds. Time in the market beats timing the market."
                    ),
                ),
            ],
            summary=(
                f"Your portfolio is currently empty. As a {user.risk_profile} "
                f"investor based in {user.country}, here's how to get started."
            ),
            disclaimer=DISCLAIMER,
        ).model_dump()

    # ── Concentration risk ────────────────────────────────────────────────

    def _compute_concentration(
        self, user: UserProfile, ticker_data: dict[str, TickerData]
    ) -> ConcentrationRisk:
        """Calculate concentration risk based on current market values."""
        position_values = []
        for pos in user.positions:
            td = ticker_data.get(pos.ticker)
            price = td.current_price if td and td.current_price > 0 else pos.avg_cost
            value = pos.quantity * price
            position_values.append((pos.ticker, value))

        total_value = sum(v for _, v in position_values)
        if total_value == 0:
            return ConcentrationRisk()

        # Sort by value descending
        position_values.sort(key=lambda x: x[1], reverse=True)

        top_pct = (position_values[0][1] / total_value) * 100 if position_values else 0
        top_3_pct = (sum(v for _, v in position_values[:3]) / total_value) * 100

        # Flag thresholds
        if top_pct >= 40:
            flag = "high"
        elif top_pct >= 25:
            flag = "moderate"
        else:
            flag = "low"

        return ConcentrationRisk(
            top_position_pct=round(top_pct, 1),
            top_3_positions_pct=round(top_3_pct, 1),
            flag=flag,
        )

    # ── Performance metrics ───────────────────────────────────────────────

    def _compute_performance(
        self, user: UserProfile, ticker_data: dict[str, TickerData]
    ) -> PerformanceMetrics:
        """Calculate total and annualized returns."""
        total_cost = 0.0
        current_value = 0.0
        earliest_date: Optional[date] = None

        for pos in user.positions:
            cost = pos.quantity * pos.avg_cost
            total_cost += cost

            td = ticker_data.get(pos.ticker)
            price = td.current_price if td and td.current_price > 0 else pos.avg_cost
            current_value += pos.quantity * price

            if pos.purchased_at:
                try:
                    pd = datetime.strptime(pos.purchased_at, "%Y-%m-%d").date()
                    if earliest_date is None or pd < earliest_date:
                        earliest_date = pd
                except ValueError:
                    pass

        if total_cost == 0:
            return PerformanceMetrics()

        total_return_pct = ((current_value - total_cost) / total_cost) * 100

        # Annualized return
        annualized = 0.0
        if earliest_date:
            days = (date.today() - earliest_date).days
            years = days / 365.25
            if years > 0 and current_value > 0 and total_cost > 0:
                annualized = ((current_value / total_cost) ** (1 / years) - 1) * 100

        return PerformanceMetrics(
            total_return_pct=round(total_return_pct, 1),
            annualized_return_pct=round(annualized, 1),
            total_cost_basis=round(total_cost, 2),
            current_value=round(current_value, 2),
        )

    # ── Benchmark comparison ──────────────────────────────────────────────

    def _compute_benchmark(
        self, user: UserProfile, performance: PerformanceMetrics
    ) -> BenchmarkComparison:
        """Compare portfolio return against the user's preferred benchmark."""
        benchmark_name = user.preferences.preferred_benchmark
        benchmark_ticker = BENCHMARK_TICKERS.get(benchmark_name, "^GSPC")

        # Find earliest purchase date for benchmark period
        earliest_date = None
        for pos in user.positions:
            if pos.purchased_at:
                try:
                    pd = datetime.strptime(pos.purchased_at, "%Y-%m-%d").date()
                    if earliest_date is None or pd < earliest_date:
                        earliest_date = pd
                except ValueError:
                    pass

        benchmark_return = None
        if earliest_date:
            benchmark_return = get_benchmark_return(
                benchmark_ticker, earliest_date.isoformat()
            )

        if benchmark_return is None:
            benchmark_return = 0.0

        alpha = performance.total_return_pct - benchmark_return

        return BenchmarkComparison(
            benchmark=benchmark_name,
            portfolio_return_pct=performance.total_return_pct,
            benchmark_return_pct=round(benchmark_return, 1),
            alpha_pct=round(alpha, 1),
        )

    # ── Diversification ───────────────────────────────────────────────────

    def _compute_diversification(
        self, user: UserProfile, ticker_data: dict[str, TickerData]
    ) -> DiversificationBreakdown:
        """Compute sector and currency diversification."""
        sector_values: dict[str, float] = {}
        currency_values: dict[str, float] = {}
        total_value = 0.0

        for pos in user.positions:
            td = ticker_data.get(pos.ticker)
            price = td.current_price if td and td.current_price > 0 else pos.avg_cost
            value = pos.quantity * price
            total_value += value

            sector = td.sector if td and td.sector != "Unknown" else "Other"
            sector_values[sector] = sector_values.get(sector, 0) + value

            currency = pos.currency
            currency_values[currency] = currency_values.get(currency, 0) + value

        if total_value == 0:
            return DiversificationBreakdown(number_of_holdings=len(user.positions))

        by_sector = {k: round((v / total_value) * 100, 1) for k, v in sector_values.items()}
        by_currency = {k: round((v / total_value) * 100, 1) for k, v in currency_values.items()}

        return DiversificationBreakdown(
            by_sector=dict(sorted(by_sector.items(), key=lambda x: x[1], reverse=True)),
            by_currency=dict(sorted(by_currency.items(), key=lambda x: x[1], reverse=True)),
            number_of_holdings=len(user.positions),
        )

    # ── Observations ──────────────────────────────────────────────────────

    def _generate_observations(
        self,
        user: UserProfile,
        concentration: ConcentrationRisk,
        performance: PerformanceMetrics,
        benchmark: BenchmarkComparison,
        diversification: DiversificationBreakdown,
        ticker_data: dict[str, TickerData],
    ) -> list[Observation]:
        """Generate plain-language observations — prioritise what matters most."""
        obs: list[Observation] = []

        # Concentration warnings
        if concentration.flag == "high":
            # Find the top position
            position_values = []
            for pos in user.positions:
                td = ticker_data.get(pos.ticker)
                price = td.current_price if td and td.current_price > 0 else pos.avg_cost
                position_values.append((pos.ticker, pos.quantity * price))
            position_values.sort(key=lambda x: x[1], reverse=True)
            top_ticker = position_values[0][0] if position_values else "unknown"

            obs.append(Observation(
                severity="warning",
                text=(
                    f"{concentration.top_position_pct:.0f}% of your portfolio is in "
                    f"{top_ticker} — that's highly concentrated. A single bad earnings "
                    f"report could significantly impact your total wealth. Consider "
                    f"diversifying into other sectors or a broad market ETF."
                ),
            ))
        elif concentration.flag == "moderate":
            obs.append(Observation(
                severity="info",
                text=(
                    f"Your top position is {concentration.top_position_pct:.0f}% of "
                    f"your portfolio. This is moderate concentration — keep an eye on it."
                ),
            ))

        # Benchmark comparison
        if benchmark.alpha_pct > 0:
            obs.append(Observation(
                severity="info",
                text=(
                    f"Your portfolio is outperforming the {benchmark.benchmark} by "
                    f"{benchmark.alpha_pct:.1f}% over the period. Nice work!"
                ),
            ))
        elif benchmark.alpha_pct < -5:
            obs.append(Observation(
                severity="warning",
                text=(
                    f"Your portfolio is underperforming the {benchmark.benchmark} by "
                    f"{abs(benchmark.alpha_pct):.1f}%. Consider reviewing whether your "
                    f"current holdings align with your investment goals."
                ),
            ))

        # Sector concentration
        if diversification.by_sector:
            top_sector = max(diversification.by_sector.items(), key=lambda x: x[1])
            if top_sector[1] > 60:
                obs.append(Observation(
                    severity="warning",
                    text=(
                        f"{top_sector[1]:.0f}% of your portfolio is in the {top_sector[0]} "
                        f"sector. Sector concentration amplifies risk — if {top_sector[0]} "
                        f"faces headwinds, your entire portfolio suffers."
                    ),
                ))

        # Currency exposure for multi-currency portfolios
        if len(diversification.by_currency) > 1:
            currencies = ", ".join(
                f"{k} ({v:.0f}%)" for k, v in diversification.by_currency.items()
            )
            obs.append(Observation(
                severity="info",
                text=f"Currency exposure: {currencies}. Multi-currency portfolios carry FX risk.",
            ))

        # Risk-profile-specific observations
        if user.risk_profile == "conservative" and performance.total_return_pct > 20:
            obs.append(Observation(
                severity="info",
                text=(
                    "Strong returns for a conservative profile! Make sure your current "
                    "allocation still matches your risk tolerance."
                ),
            ))

        # Income-focused observations for retirees
        if user.preferences.income_focus:
            dividend_tickers = []
            for pos in user.positions:
                td = ticker_data.get(pos.ticker)
                if td and td.dividend_yield and td.dividend_yield > 0.02:
                    dividend_tickers.append(
                        f"{pos.ticker} ({td.dividend_yield*100:.1f}%)"
                    )
            if dividend_tickers:
                obs.append(Observation(
                    severity="info",
                    text=(
                        f"Dividend-yielding holdings: {', '.join(dividend_tickers)}. "
                        f"These support your income-focused strategy."
                    ),
                ))

        # If no notable observations, add a positive one
        if not obs:
            obs.append(Observation(
                severity="info",
                text=(
                    "Your portfolio looks well-balanced. No significant concentration "
                    "or performance concerns detected."
                ),
            ))

        return obs

    # ── LLM summary (optional) ────────────────────────────────────────────

    async def _generate_summary(
        self,
        user: UserProfile,
        concentration: ConcentrationRisk,
        performance: PerformanceMetrics,
        benchmark: BenchmarkComparison,
        observations: list[Observation],
        llm: Any = None,
    ) -> str:
        """Generate a plain-language summary using the LLM. Falls back to template."""
        # If no LLM available (testing), use template
        if llm is None:
            return self._template_summary(user, concentration, performance, benchmark)

        try:
            from openai import OpenAI
            from src.config import get_settings

            settings = get_settings()

            # If llm is a mock, use it directly
            if not isinstance(llm, OpenAI):
                try:
                    result = llm(user, concentration, performance, benchmark, observations)
                    if isinstance(result, str):
                        return result
                except Exception:
                    pass
                return self._template_summary(user, concentration, performance, benchmark)

            obs_text = "\n".join(f"- [{o.severity}] {o.text}" for o in observations)
            prompt = f"""You are a friendly financial advisor helping {user.name}, a {user.age}-year-old {user.risk_profile} investor.

Portfolio metrics:
- Total return: {performance.total_return_pct:.1f}%
- Annualized return: {performance.annualized_return_pct:.1f}%
- Top position concentration: {concentration.top_position_pct:.1f}% ({concentration.flag} risk)
- Benchmark ({benchmark.benchmark}): {benchmark.benchmark_return_pct:.1f}% | Alpha: {benchmark.alpha_pct:.1f}%

Key observations:
{obs_text}

Write a 2-3 sentence summary for a NOVICE investor. Use plain language, no jargon.
Focus on the ONE or TWO things that matter most. Be encouraging but honest."""

            client = llm if isinstance(llm, OpenAI) else OpenAI(api_key=settings.openai_api_key)
            response = client.chat.completions.create(
                model=settings.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.warning(f"LLM summary generation failed: {e}")
            return self._template_summary(user, concentration, performance, benchmark)

    def _template_summary(
        self,
        user: UserProfile,
        concentration: ConcentrationRisk,
        performance: PerformanceMetrics,
        benchmark: BenchmarkComparison,
    ) -> str:
        """Fallback template summary when LLM is unavailable."""
        parts = []
        parts.append(
            f"Hi {user.name}! Your portfolio has returned {performance.total_return_pct:.1f}% "
            f"overall ({performance.annualized_return_pct:.1f}% annualized)."
        )
        if benchmark.alpha_pct > 0:
            parts.append(
                f"You're beating the {benchmark.benchmark} by {benchmark.alpha_pct:.1f}%."
            )
        elif benchmark.alpha_pct < -3:
            parts.append(
                f"You're trailing the {benchmark.benchmark} by {abs(benchmark.alpha_pct):.1f}%."
            )
        if concentration.flag == "high":
            parts.append(
                f"Watch out: {concentration.top_position_pct:.0f}% in a single stock is risky."
            )
        return " ".join(parts)
