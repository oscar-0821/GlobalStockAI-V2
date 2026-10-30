from app.search import search_stocks, normalize, ticker_normalize

def test_case_insensitive_ticker():
    assert ticker_normalize("1155.kl") == "1155.KL"
    assert ticker_normalize("Aapl") == "AAPL"

def test_exact_ticker_auto_selects_single_identity():
    result = search_stocks("aapl")
    assert result["auto_select"] is True
    assert result["results"][0]["ticker"] == "AAPL"

def test_alias_returns_company():
    result = search_stocks("Apple")
    assert result["results"][0]["company_name"] == "Apple Inc."

def test_empty_query():
    assert not search_stocks(" ")["results"]

def test_unknown_query_is_safe():
    result = search_stocks("zzzz-no-such-stock")
    assert result["results"] == []
    assert result["auto_select"] is False
