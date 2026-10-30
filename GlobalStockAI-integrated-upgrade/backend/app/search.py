
"""Identity-first stock search with Yahoo Finance discovery."""

import re
from rapidfuzz import fuzz, process

from .catalog import STOCKS
from .market_data import search_market_stocks


def normalize(value: str) -> str:
    value = (value or "").strip().casefold()
    value = value.replace("_", " ")
    value = re.sub(r"[^a-z0-9.\-]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def ticker_normalize(value: str) -> str:
    value = normalize(value).replace(" ", "")
    return value.upper()


def _fields(stock):
    return (
        stock["ticker"],
        stock["company_name"],
        stock.get("aliases", [])
    )


def _response(raw, normalized, ranked, exact=False):
    results = []

    for score, match_type, stock in ranked:
        results.append({
            **stock,
            "match_score": score,
            "match_type": match_type,
            "verified_identity": stock.get("verified_identity", True)
        })

    ambiguous = len(results) > 1 and exact

    if ambiguous:
        message = (
            "This name or alias matches multiple stocks. "
            "Choose by company name and exchange."
        )
    elif not results:
        message = "No matching stock found. Try another company name or ticker."
    elif any(r["match_type"] == "market_provider_search" for r in results):
        message = (
            "Additional matches were found through the market data provider. "
            "Confirm the company and exchange before selecting."
        )
    elif not exact:
        message = "Suggestions only. Confirm the company and exchange before opening."
    else:
        message = None

    # Only one exact match from the local catalogue can be auto-selected.
    local_exact = (
        exact
        and len(results) == 1
        and results[0]["match_type"] in (
            "exact_ticker",
            "exact_company_name",
            "exact_verified_alias"
        )
    )

    return {
        "query": raw,
        "normalized_query": normalized,
        "results": results,
        "ambiguous": ambiguous,
        "auto_select": local_exact,
        "message": message
    }


def search_stocks(query: str, limit: int = 10):
    raw = (query or "").strip()

    if not raw:
        return {
            "query": query,
            "normalized_query": "",
            "results": [],
            "ambiguous": False,
            "auto_select": False,
            "message": "Enter a company name or ticker."
        }

    q_norm = normalize(raw)
    q_ticker = ticker_normalize(raw)
    candidates = []

    # 1. Exact ticker match.
    for stock in STOCKS:
        if q_ticker == ticker_normalize(stock["ticker"]):
            candidates.append((1000, "exact_ticker", stock))

    if candidates:
        return _response(raw, q_norm, candidates[:limit], exact=True)

    # 2. Exact company name.
    for stock in STOCKS:
        if q_norm == normalize(stock["company_name"]):
            candidates.append((950, "exact_company_name", stock))

    if candidates:
        return _response(raw, q_norm, candidates[:limit], exact=True)

    # 3. Exact verified alias.
    for stock in STOCKS:
        if any(
            q_norm == normalize(alias)
            for alias in stock.get("aliases", [])
        ):
            candidates.append((900, "exact_verified_alias", stock))

    if candidates:
        return _response(
            raw,
            q_norm,
            candidates[:limit],
            exact=len(candidates) == 1
        )

    # 4. Local catalogue prefix and partial matches.
    for stock in STOCKS:
        ticker, company, aliases = _fields(stock)
        fields = [ticker, company, *aliases]
        scores = []

        for field in fields:
            field_norm = normalize(field)

            if field_norm.startswith(q_norm):
                scores.append(800)
            elif q_norm in field_norm:
                scores.append(700)

        if scores:
            candidates.append((
                max(scores),
                "prefix_or_contains",
                stock
            ))

    # 5. Local fuzzy suggestions.
    all_fields = []

    for stock in STOCKS:
        for field in [
            stock["ticker"],
            stock["company_name"],
            *stock.get("aliases", [])
        ]:
            all_fields.append((normalize(field), stock))

    if all_fields:
        fuzzy = process.extract(
            q_norm,
            [item[0] for item in all_fields],
            scorer=fuzz.WRatio,
            limit=max(limit * 3, 15)
        )

        for field_norm, score, idx in fuzzy:
            if score >= 62:
                stock = all_fields[idx][1]
                candidates.append((
                    int(score),
                    "fuzzy_candidate",
                    stock
                ))

    # 6. Search Yahoo Finance for additional stocks.
    # Provider results are candidates, not independently verified identities.
    try:
        provider_results = search_market_stocks(raw, limit)

        for stock in provider_results:
            candidates.append((
                100,
                "market_provider_search",
                stock
            ))

    except Exception as error:
        print(f"Market provider search failed: {error}")

    # 7. Remove duplicate stock tickers.
    rank = {
        "exact_ticker": 5,
        "exact_company_name": 4,
        "exact_verified_alias": 3,
        "prefix_or_contains": 2,
        "fuzzy_candidate": 1,
        "market_provider_search": 1
    }

    by_ticker = {}

    for score, kind, stock in candidates:
        key = ticker_normalize(stock["ticker"])
        previous = by_ticker.get(key)

        if previous is None or (
            rank[kind], score
        ) > (
            rank[previous[1]], previous[0]
        ):
            by_ticker[key] = (score, kind, stock)

    ranked = sorted(
        by_ticker.values(),
        key=lambda item: (rank[item[1]], item[0]),
        reverse=True
    )[:limit]

    return _response(raw, q_norm, ranked, exact=False)
