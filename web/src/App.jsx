/*
 * The shell: sidebar, page switching, and the one piece of state every
 * page cares about — the last analysis result.
 *
 * There is no router on purpose. The whole app is seven views behind a
 * sidebar, and a result that survives a "view packet →" click is the
 * only navigation state there is.
 */
import { useEffect, useState } from 'react'
import { getHealth } from './api.js'
import { reduceMotion, ToastHost, useRipple, useToast } from './ui.jsx'
import Analyze from './pages/Analyze.jsx'
import Evidence from './pages/Evidence.jsx'
import Home from './pages/Home.jsx'
import Packet from './pages/Packet.jsx'
import Radar from './pages/Radar.jsx'
import Response from './pages/Response.jsx'
import Review from './pages/Review.jsx'

const NAV = [
  { id: 'home', label: '◈ Command center' },
  { id: 'analyze', label: '⌕ Multimodal analysis' },
  { id: 'packet', label: '▣ Media packet' },
  { id: 'evidence', label: '◎ Evidence fusion' },
  { id: 'response', label: '↗ Response studio' },
  { id: 'radar', label: '◉ Time & narrative radar' },
  { id: 'review', label: '♙ Human review' },
]

export default function App() {
  return (
    <ToastHost>
      <Shell />
    </ToastHost>
  )
}

function Shell() {
  const toast = useToast()
  const [page, setPage] = useState('home')
  const [result, setResult] = useState(null)

  useRipple()

  function go(id) {
    setPage(id)
    window.scrollTo({ top: 0, behavior: reduceMotion ? 'auto' : 'smooth' })
  }

  const pages = {
    home: <Home go={go} />,
    analyze: (
      <Analyze
        go={go}
        result={result}
        onResult={setResult}
        onReset={() => setResult(null)}
      />
    ),
    packet: <Packet go={go} result={result} />,
    evidence: <Evidence go={go} result={result} />,
    response: <Response go={go} />,
    radar: <Radar />,
    review: <Review />,
  }

  return (
    <>
      <div className="ambient" aria-hidden="true">
        <span className="a1" />
        <span className="a2" />
        <span className="a3" />
      </div>

      <div className="app">
        <aside className="side">
          <div className="brand">
            veri<span>lens</span>
          </div>
          <div className="sub">AI-03 / MULTIMODAL TRUST OPS</div>

          <div className="nav">
            {NAV.map((item) => (
              <button
                key={item.id}
                className={page === item.id ? 'active' : ''}
                onClick={() => go(item.id)}
              >
                {item.label}
              </button>
            ))}
          </div>

          <div className="safe card">
            <div className="label">SAFE MODE</div>
            <p className="small muted">
              AI signals support reviewers; they never replace evidence or human
              judgment.
            </p>
            <Health />
            <button
              className="btn ghost"
              onClick={() => toast('Privacy controls are represented in this demo')}
            >
              Privacy controls →
            </button>
          </div>
        </aside>

        <main className="main">
          {/* Keyed so React remounts on a page change and the CSS page-in
              animation runs again, the way the old markup replayed it. */}
          <div key={page}>{pages[page]}</div>

          <div className="footer">
            VeriLens · AI-03 prototype · Claim → media packet → evidence fusion
            → time check → responsible response.
          </div>
        </main>
      </div>
    </>
  )
}

/**
 * Which optional retrievers and models the server actually has loaded.
 *
 * Worth surfacing: a run that comes back `unverified` usually means a
 * retriever or a model is missing, not that the claim is unusual, and
 * this says so before the run rather than after it.
 */
function Health() {
  const [health, setHealth] = useState(null)
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    let live = true

    getHealth()
      .then((data) => live && setHealth(data))
      .catch(() => live && setFailed(true))

    return () => {
      live = false
    }
  }, [])

  if (failed) {
    return (
      <p className="small red" style={{ marginTop: 10 }}>
        API offline — start <code>python -m backend.api.flask_app</code>
      </p>
    )
  }

  if (!health) return <p className="small muted">Checking API…</p>

  const models = Object.entries(health.models || {})
  const retrievers = Object.entries(health.retrievers || {})
  const on = [...models, ...retrievers].filter(([, ready]) => ready).length
  const total = models.length + retrievers.length

  return (
    <div style={{ marginTop: 10 }}>
      <div className="signal">
        <span className="small">API</span>
        <b className="green small">online</b>
      </div>
      <div className="signal">
        <span className="small">Models &amp; retrievers</span>
        <b className={(on === total ? 'green' : 'amber') + ' small'}>
          {on}/{total} ready
        </b>
      </div>
    </div>
  )
}
