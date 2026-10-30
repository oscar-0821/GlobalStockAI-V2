import React, { useEffect, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './style.css'

const API = ''
const money = (value, currency = 'MYR') => value == null || !Number.isFinite(Number(value)) ? '—' : new Intl.NumberFormat('en-MY', { style: 'currency', currency, maximumFractionDigits: 2 }).format(Number(value))

function App() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])
  const [message, setMessage] = useState('Search a ticker or company name to begin.')
  const [loading, setLoading] = useState(false)
  const [selected, setSelected] = useState(null)
  const [apiHealth, setApiHealth] = useState(null)
  const [analysis, setAnalysis] = useState(null)
  const [analysisLoading, setAnalysisLoading] = useState(false)
  const [fx, setFx] = useState(null)
  const [capital, setCapital] = useState('10000')
  const [riskPct, setRiskPct] = useState('1')
  const [planner, setPlanner] = useState(null)
  const debounce = useRef(null)
  const requestId = useRef(0)

  useEffect(() => {
    fetch(`${API}/api/health`).then(r => r.json()).then(setApiHealth).catch(() => setApiHealth({ status: 'offline' }))
    return () => clearTimeout(debounce.current)
  }, [])

  async function search(value = query) {
    clearTimeout(debounce.current)
    const id = ++requestId.current
    if (!value.trim()) { setResults([]); setSelected(null); setMessage('Enter a ticker or company name.'); return }
    setLoading(true); setSelected(null); setAnalysis(null); setPlanner(null)
    try {
      const response = await fetch(`${API}/api/stocks/search?q=${encodeURIComponent(value.trim())}&limit=10`)
      if (!response.ok) throw new Error(`Search returned ${response.status}`)
      const data = await response.json()
      if (id !== requestId.current) return
      setResults(data.results || []); setMessage(data.message || 'Search complete.')
      if (data.auto_select && data.results?.length === 1) chooseStock(data.results[0])
    } catch (error) { if (id === requestId.current) { setResults([]); setMessage(error.message) } }
    finally { if (id === requestId.current) setLoading(false) }
  }

  async function chooseStock(stock) {
    setSelected(stock); setAnalysis(null); setPlanner(null); setFx(null); setAnalysisLoading(true)
    try {
      const [indRes, predRes] = await Promise.all([
        fetch(`${API}/api/stocks/${encodeURIComponent(stock.ticker)}/indicators?period=1y`),
        fetch(`${API}/api/stocks/${encodeURIComponent(stock.ticker)}/prediction?period=5y`)
      ])
      const indicators = indRes.ok ? await indRes.json() : { success: false, message: `Indicators unavailable (${indRes.status})` }
      const prediction = predRes.ok ? await predRes.json() : { success: false, message: `Prediction unavailable (${predRes.status})` }
      setAnalysis({ indicators, prediction })
      const currency = indicators.currency || stock.currency || 'Unknown'
      if (currency === 'MYR') setFx({ success: true, rate: 1, to_currency: 'MYR' })
      else if (currency !== 'Unknown') {
        const fxRes = await fetch(`${API}/api/fx/rate?from_currency=${encodeURIComponent(currency)}&to_currency=MYR`)
        if (fxRes.ok) setFx(await fxRes.json())
        else setFx({ success: false })
      }
    } catch (error) { setAnalysis({ indicators: { success: false }, prediction: { success: false, message: error.message } }) }
    finally { setAnalysisLoading(false) }
  }

  async function runPlanner() {
    const currency = analysis?.indicators?.currency || selected?.currency || 'MYR'
    const rate = currency === 'MYR' ? 1 : fx?.rate
    const horizon = analysis?.prediction?.results?.['5_days'] || analysis?.prediction?.results?.['1_day']
    const plan = horizon?.trade_plan
    if (!plan || !rate || !Number(capital) || Number(capital) <= 0) { setPlanner({ error: 'A valid trade plan and MYR exchange rate are required first.' }); return }
    const nativeCapital = Number(capital) / rate
    try {
      const response = await fetch(`${API}/api/profit-planner`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ capital: nativeCapital, entry_price: plan.entry_price, stop_loss: plan.stop_loss, take_profit: plan.take_profit, risk_percent: Number(riskPct), fee_percent: 0.1, fx_rate: rate }) })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'Planner failed')
      setPlanner(data)
    } catch (error) { setPlanner({ error: error.message }) }
  }

  const currency = analysis?.indicators?.currency || selected?.currency || '—'
  const rate = currency === 'MYR' ? 1 : fx?.rate
  const convert = value => value == null || !rate ? null : Number(value) * rate
  const predResults = analysis?.prediction?.results || {}

  return <main className="shell">
    <header className="topbar"><div className="brand"><span className="brand-icon">↗</span><div><strong>GlobalStock<span>AI</span></strong><small>Market intelligence workspace · MYR view</small></div></div><div className={`status ${apiHealth?.status === 'ok' ? 'online' : 'offline'}`}><span /> API {apiHealth?.status || 'checking'}</div></header>
    <section className="hero"><div className="eyebrow">GLOBAL MARKET INTELLIGENCE</div><h1>Make decisions with <em>context.</em></h1><p>Search global listings, inspect model evidence and plan risk in Malaysian ringgit.</p>
      <form className="search-wrap" onSubmit={e => { e.preventDefault(); search() }}><span className="search-icon">⌕</span><input value={query} onChange={e => { setQuery(e.target.value); clearTimeout(debounce.current); debounce.current = setTimeout(() => search(e.target.value), 500) }} placeholder="Ticker, company name, or alias…" autoComplete="off" /><button className="search-button" disabled={loading || !query.trim()}>{loading ? 'SEARCHING…' : 'SEARCH'}</button></form>
      <div className="search-hint"><span>Try</span>{['1155.kl', 'aapl', 'Maybank'].map(v => <button key={v} onClick={() => { setQuery(v); search(v) }}>{v}</button>)}</div>
    </section>
    <section className="results-section"><div className="section-heading"><div><span className="eyebrow">STOCK UNIVERSE</span><h2>Search results</h2></div><span className="count">{results.length} MATCHES</span></div><div className="notice">{message}</div><div className="results">{results.map(stock => <button key={stock.ticker} className={`stock-row ${selected?.ticker === stock.ticker ? 'selected' : ''}`} onClick={() => chooseStock(stock)}><div className="stock-mark">{stock.ticker.replace(/[^A-Z]/gi, '').slice(0, 2).toUpperCase()}</div><div className="stock-info"><strong>{stock.company_name}</strong><span>{stock.ticker} · {stock.exchange}</span></div><div className="match"><span>{stock.match_type?.replaceAll('_', ' ')}</span><small>{stock.currency}</small></div><span className="chevron">›</span></button>)}</div></section>
    {selected && <section className="dashboard"><div className="section-heading"><div><span className="eyebrow">SELECTED LISTING</span><h2>{selected.company_name}</h2><p className="subline">{selected.ticker} · {selected.exchange} · Trading currency: {currency}</p></div><span className="count">{analysisLoading ? 'ANALYSING…' : 'ANALYSIS'}</span></div>
      {analysisLoading && <div className="notice">Loading technical indicators, historical validation and currency conversion. This may take a little time.</div>}
      {analysis && <>
        <div className="metric-grid"><div className="metric"><small>LAST CLOSE · {currency}</small><strong>{money(analysis.indicators?.latest_close, currency)}</strong></div><div className="metric"><small>LAST CLOSE · MYR</small><strong>{money(convert(analysis.indicators?.latest_close), 'MYR')}</strong></div><div className="metric"><small>RSI (14)</small><strong>{analysis.indicators?.indicators?.RSI_14 ?? '—'}</strong></div><div className="metric"><small>ATR (14) · {currency}</small><strong>{money(analysis.indicators?.indicators?.ATR_14, currency)}</strong></div></div>
        {!fx?.success && currency !== 'MYR' && <div className="notice warning">MYR conversion rate is unavailable. Native-currency values remain visible; MYR values are withheld rather than guessed.</div>}
        <div className="panel"><div className="panel-title"><div><span className="eyebrow">MODEL OUTPUT</span><h3>Forecast and reliability</h3></div><span className="count">Both forecast and validation shown</span></div>
          {!analysis.prediction?.success ? <div className="notice warning">{analysis.prediction?.message || 'Prediction service unavailable.'}</div> : <div className="forecast-grid">{Object.entries(predResults).map(([key, item]) => <article className="forecast" key={key}><div className="forecast-head"><strong>{key.replace('_', ' ')}</strong><span className={`tag ${item.reliability_label === 'BOTH_MODELS_BEAT_BASELINE' ? 'good' : 'caution'}`}>{item.reliability_label || 'NOT SCORED'}</span></div><div className="forecast-price">{money(item.predicted_price, currency)}</div><div className="muted">{money(convert(item.predicted_price), 'MYR')} · predicted price</div><div className="forecast-stats"><span>Estimated return <b>{item.predicted_return_percent}%</b></span><span>Direction <b>{item.predicted_direction}</b></span><span>Regression MAE <b>{item.regression_validation?.holdout_mae_percent ?? '—'}%</b></span><span>Zero-return MAE <b>{item.regression_validation?.baseline_mae_percent ?? '—'}%</b></span><span>Direction accuracy <b>{item.direction_validation?.balanced_accuracy_percent ?? item.direction_validation?.accuracy_percent ?? '—'}%</b></span><span>Direction baseline <b>{item.direction_validation?.baseline_accuracy_percent ?? '—'}%</b></span></div><div className="probabilities">{Object.entries(item.direction_probabilities || {}).map(([name, value]) => <span key={name}>{name}: <b>{value}%</b></span>)}</div>{item.trade_plan && <div className="trade-levels"><b>ATR risk scenario · {currency}</b><span>Entry {money(item.trade_plan.entry_price, currency)}{rate ? ` · ${money(convert(item.trade_plan.entry_price), 'MYR')}` : ''}</span><span>Stop {money(item.trade_plan.stop_loss, currency)}{rate ? ` · ${money(convert(item.trade_plan.stop_loss), 'MYR')}` : ''}</span><span>Target {money(item.trade_plan.take_profit, currency)}{rate ? ` · ${money(convert(item.trade_plan.take_profit), 'MYR')}` : ''}</span><small>Illustrative risk levels, not model-predicted optimal prices.</small></div>}</article>)}</div>}
          <p className="disclaimer">Forecasts are experimental. A reliability label indicates whether the model beat a historical baseline on the held-out sample; it does not establish future accuracy. Forecast values remain visible even when reliability is low.</p>
        </div>
        <div className="panel planner"><div className="panel-title"><div><span className="eyebrow">POSITION SIZING</span><h3>MYR profit planner</h3></div></div><div className="planner-controls"><label>Available capital (MYR)<input type="number" min="1" value={capital} onChange={e => setCapital(e.target.value)} /></label><label>Risk per trade (%)<input type="number" min="0.1" max="100" step="0.1" value={riskPct} onChange={e => setRiskPct(e.target.value)} /></label><button className="search-button" onClick={runPlanner} disabled={!analysis.prediction?.success || !rate}>CALCULATE PLAN</button></div>{planner?.error && <div className="notice warning">{planner.error}</div>}{planner && !planner.error && <div className="planner-results"><div><small>SHARES</small><strong>{planner.quantity}</strong></div><div><small>CAPITAL USED</small><strong>{money(planner.capital_used, 'MYR')}</strong></div><div><small>EST. NET PROFIT AT TARGET</small><strong>{money(planner.net_profit_myr, 'MYR')}</strong></div><div><small>EST. LOSS AT STOP</small><strong>{money(planner.net_loss_at_stop_myr, 'MYR')}</strong></div><p>{planner.assumptions}</p></div>}</div>
      </>}
    </section>}
    <footer><span>GLOBALSTOCKAI · RESEARCH TOOL</span><span>Market data and FX supplied by Yahoo Finance · Not investment advice</span></footer>
  </main>
}

createRoot(document.getElementById('root')).render(<App />)
