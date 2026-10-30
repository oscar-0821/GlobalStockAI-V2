
try:
    import yfinance as yf
except ImportError:
    yf = None


def get_stock_metadata(symbol: str):
    """Retrieve additional listing information from Yahoo Finance."""

    metadata = {
        "market": "Unknown",
        "currency": "Unknown",
        "exchange": "Unknown"
    }

    if yf is None:
        return metadata
    try:
        stock = yf.Ticker(symbol)
        info = stock.get_info()

        if info:
            metadata["market"] = (
                info.get("country")
                or "Unknown"
            )

            metadata["currency"] = (
                info.get("currency")
                or "Unknown"
            )

            metadata["exchange"] = (
                info.get("fullExchangeName")
                or info.get("exchange")
                or "Unknown"
            )

    except Exception as error:
        print(f"Metadata lookup failed for {symbol}: {error}")

    return metadata


def search_market_stocks(query: str, limit: int = 10):
    """Search Yahoo Finance for additional stock listings."""

    query = (query or "").strip()

    if not query:
        return []

    if yf is None:
        return []
    try:
        search = yf.Search(
            query,
            max_results=limit,
            news_count=0,
            lists_count=0
        )

        results = []
        seen = set()

        for item in search.quotes:
            quote_type = item.get("quoteType", "")

            if quote_type and quote_type.upper() != "EQUITY":
                continue

            symbol = item.get("symbol")
            name = item.get("longname") or item.get("shortname")

            if not symbol or not name or symbol in seen:
                continue

            seen.add(symbol)

            metadata = get_stock_metadata(symbol)

            results.append({
                "ticker": symbol,
                "company_name": name,
                "exchange": (
                    metadata["exchange"]
                    if metadata["exchange"] != "Unknown"
                    else item.get("exchDisp")
                    or item.get("exchange")
                    or "Unknown"
                ),
                "market": metadata["market"],
                "currency": metadata["currency"],
                "aliases": [],
                "verified_identity": False,
                "match_type": "market_provider_search",
                "match_score": 100
            })

        return results

    except Exception as error:
        print(f"Yahoo Finance search error: {error}")
        return []


if __name__ == "__main__":
    print(search_market_stocks("NVIDIA", 5))
