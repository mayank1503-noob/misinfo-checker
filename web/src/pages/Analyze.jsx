/*
 * Intake: pick an input type, hand it to the pipeline, show what came
 * back.
 *
 * The run itself is one POST. The step list beside it is an honest
 * affordance rather than real progress — the server reports stage
 * timings only once it answers — so the last step holds until the
 * response lands instead of pretending to finish on a timer.
 */
import { useEffect, useRef, useState } from 'react'
import { LIMITS, check, labelClass, pct } from '../api.js'
import { Bar, Card, reduceMotion, useToast } from '../ui.jsx'

const TYPES = ['text', 'link', 'image', 'video']

const STEPS = [
  'Reading input & normalising script',
  'Extracting media signals (OCR · caption · EXIF)',
  'Building Media Analysis Packet',
  'Retrieving evidence & scoring stance',
  'Fusing signals into a verdict',
]

export default function Analyze({ go, result, onResult, onReset }) {
  const toast = useToast()

  const [type, setType] = useState('text')
  const [text, setText] = useState('')
  const [url, setUrl] = useState('')
  const [caption, setCaption] = useState('')
  const [file, setFile] = useState(null)
  const [graph, setGraph] = useState(false)
  const [agentic, setAgentic] = useState(false)

  const [running, setRunning] = useState(false)
  const [step, setStep] = useState(-1)
  const [error, setError] = useState('')
  const [over, setOver] = useState(false)

  const fileInput = useRef(null)
  const previewUrl = usePreview(file)

  /* Walk the steps while the request is in flight, stopping on the last
     one until the answer arrives. */
  useEffect(() => {
    if (!running || reduceMotion) return

    const timer = setInterval(
      () => setStep((i) => Math.min(i + 1, STEPS.length - 1)),
      900
    )

    return () => clearInterval(timer)
  }, [running])

  function choose(next) {
    setType(next)
    setError('')
    toast('Input mode: ' + next)
  }

  function attach(next) {
    if (!next) return
    setFile(next)
    toast('File attached: ' + next.name)
  }

  function reset() {
    setText('')
    setUrl('')
    setCaption('')
    setFile(null)
    setError('')
    setStep(-1)
    if (fileInput.current) fileInput.current.value = ''
    onReset()
    toast('Intake cleared')
  }

  async function run() {
    if (running) return

    let path
    let payload

    if (type === 'text') {
      if (!text.trim()) return toast('Paste a message to check')
      // The server's own ceilings, injected into the page by the Flask
      // template. Saying so here costs nothing; letting the upload run
      // costs the wait and then a 413.
      if (text.trim().length > LIMITS.maxTextChars) {
        return toast(
          `Message is too long (limit ${LIMITS.maxTextChars.toLocaleString()} characters)`
        )
      }
      path = '/check/text'
      payload = { text: text.trim() }
    }

    if (type === 'link') {
      const value = url.trim()
      if (!value) return toast('Add a URL to check')
      if (!/^https?:\/\//i.test(value)) {
        return toast('URL must start with http:// or https://')
      }
      path = '/check/link'
      payload = { url: value }
    }

    if (type === 'image' || type === 'video') {
      if (!file) return toast(`Choose a ${type} file first`)
      if (file.size > LIMITS.maxUploadBytes) {
        return toast(
          `File is too large (limit ${Math.round(
            LIMITS.maxUploadBytes / (1024 * 1024)
          )} MB)`
        )
      }
      payload = new FormData()
      payload.append('file', file)
      payload.append('caption', caption.trim())
      path = `/check/${type}`
    }

    // `graph` and `agentic` are query flags the API already understands:
    // the first asks for the explanation graph in the response, the
    // second runs the agentic reviewers over the result.
    const flags = [graph && 'graph=1', agentic && 'agentic=1'].filter(Boolean)
    if (flags.length) path += '?' + flags.join('&')

    setRunning(true)
    setError('')
    setStep(0)

    try {
      const data = await check(path, payload)

      setStep(STEPS.length)
      onResult(data)
      toast('Analysis complete: ' + data.verdict.label)
    } catch (failure) {
      setStep(STEPS.length)
      setError(failure.message)
      toast('Analysis failed')
    } finally {
      setRunning(false)
    }
  }

  return (
    <section className="page">
      <div className="eyebrow">MULTIMODAL INTAKE</div>
      <div className="h1">Analyze content</div>
      <p className="muted">
        Choose an input type. The browser builds nothing itself — the text or
        file goes to the Flask API, which runs the same pipeline the bot uses.
      </p>

      <div className="grid two">
        <Card>
          <div className="row">
            <b>Input type</b>
            <span className="tag">Live pipeline</span>
          </div>

          <div className="actions">
            {TYPES.map((name) => (
              <button
                key={name}
                className={'btn' + (type === name ? ' primary' : '')}
                onClick={() => choose(name)}
              >
                {name[0].toUpperCase() + name.slice(1)}
              </button>
            ))}
          </div>

          {type === 'text' && (
            <div style={{ marginTop: 15 }}>
              <label className="label" htmlFor="claimInput">
                Message / caption / transcript
              </label>
              <textarea
                id="claimInput"
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Paste a forwarded message, caption, or transcript..."
              />
            </div>
          )}

          {type === 'link' && (
            <div style={{ marginTop: 15 }}>
              <label className="label" htmlFor="urlInput">
                Article or video URL
              </label>
              <input
                className="input"
                id="urlInput"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="https://example.com/article"
              />
            </div>
          )}

          {(type === 'image' || type === 'video') && (
            <div style={{ marginTop: 15 }}>
              <label className="label">Upload image or video</label>
              <div
                className={'drop' + (over ? ' over' : '')}
                onDragEnter={(e) => {
                  e.preventDefault()
                  setOver(true)
                }}
                onDragOver={(e) => {
                  e.preventDefault()
                  setOver(true)
                }}
                onDragLeave={() => setOver(false)}
                onDrop={(e) => {
                  e.preventDefault()
                  setOver(false)
                  attach(e.dataTransfer.files[0])
                }}
              >
                Drop a file here or select one
                <input
                  type="file"
                  ref={fileInput}
                  accept="image/*,video/*"
                  onChange={(e) => attach(e.target.files[0])}
                />
                {previewUrl && (
                  <img
                    className="preview"
                    src={previewUrl}
                    alt=""
                    style={{ display: 'block' }}
                  />
                )}
                {file && (
                  <p className="small muted">
                    {file.name} · {Math.round(file.size / 1024)} KB
                  </p>
                )}
              </div>

              <label className="label" htmlFor="captionInput">
                Optional caption
              </label>
              <textarea
                id="captionInput"
                value={caption}
                onChange={(e) => setCaption(e.target.value)}
                placeholder="What did the sender claim about this media?"
              />
            </div>
          )}

          <label className="label" style={{ display: 'block', marginTop: 15 }}>
            Language
          </label>
          <select>
            <option>Auto-detect</option>
            <option>English</option>
            <option>Hindi / Hinglish</option>
            <option>Marathi</option>
            <option>Tamil</option>
            <option>Bengali</option>
          </select>

          <div className="actions">
            <label className="pill small">
              <input
                type="checkbox"
                checked={graph}
                onChange={(e) => setGraph(e.target.checked)}
              />{' '}
              Explanation graph
            </label>
            <label className="pill small">
              <input
                type="checkbox"
                checked={agentic}
                onChange={(e) => setAgentic(e.target.checked)}
              />{' '}
              Agentic review
            </label>
          </div>

          <div className="actions">
            <button className="btn primary" onClick={run} disabled={running}>
              {running ? 'Analyzing…' : 'Build packet & analyze ↗'}
            </button>
            <button className="btn" onClick={reset} disabled={running}>
              Reset
            </button>
          </div>

          <div className="evidence small">
            Production mapping: Whisper → transcript; OCR/TrOCR → image text;
            BLIP → description; DINOv2 → image identity; EXIF → capture date;
            CLIP → image/claim consistency.
          </div>
        </Card>

        <Card className={running ? 'scanline' : ''}>
          <div className="section">Engine result</div>

          {running && <Steps current={step} />}

          {error && (
            <>
              <div className="row">
                <b>Analysis failed</b>
                <span className="tag">Error</span>
              </div>
              <p className="red">{error}</p>
              <p className="small muted">
                Start the server with{' '}
                <code>python -m backend.api.flask_app</code> and open
                http://127.0.0.1:5000 rather than the file directly.
              </p>
            </>
          )}

          {!error && !running && result && <Verdict data={result} go={go} />}

          {!error && !running && !result && (
            <p className="muted">Your multimodal assessment will appear here.</p>
          )}
        </Card>
      </div>
    </section>
  )
}

/* ---- pieces ---------------------------------------------------- */

function Steps({ current }) {
  const reached = Math.min(current, STEPS.length - 1)

  return (
    <>
      <div className="steps">
        {STEPS.map((label, i) => (
          <div
            key={label}
            className={
              'step' + (i < current ? ' done' : i === current ? ' run' : '')
            }
          >
            <span className="dot" />
            <span>{label}</span>
          </div>
        ))}
      </div>
      <Bar value={Math.round(((reached + 1) / STEPS.length) * 100)} delay={0} />
    </>
  )
}

function Verdict({ data, go }) {
  const verdict = data.verdict
  const claims = data.results || []

  return (
    <>
      <div className="row">
        <b>Verdict</b>
        <span className="tag">{Math.round(data.ms)} ms</span>
      </div>
      <h2 className={labelClass(verdict.label)}>
        {String(verdict.label).toUpperCase()}
      </h2>
      <p className="claim">{verdict.summary || ''}</p>
      <Bar value={pct(verdict.confidence)} />
      <p className="small muted">
        Confidence: {pct(verdict.confidence)}% · {claims.length} claim(s)
        checked · assistive only, confirm before acting
      </p>

      {claims.length > 0 && (
        <>
          <div className="section">Per-claim findings</div>
          <div className="list">
            {claims.map((claim, i) => (
              <div className="item" key={claim.claim_id || i}>
                <b className={labelClass(claim.label)}>{claim.label}</b> ·{' '}
                <span className="small muted">
                  confidence {pct(claim.confidence)}%
                </span>
                <p className="claim">{claim.claim || ''}</p>
                {claim.explanation && <p className="small">{claim.explanation}</p>}
                {(claim.reasons || []).length > 0 && (
                  <ul className="small muted">
                    {claim.reasons.map((reason, j) => (
                      <li key={j}>{reason}</li>
                    ))}
                  </ul>
                )}
                {claim.demo_only && <span className="tag">demo evidence</span>}
              </div>
            ))}
          </div>
        </>
      )}

      <div className="actions">
        <button className="btn primary" onClick={() => go('packet')}>
          View packet →
        </button>
        <button className="btn" onClick={() => go('evidence')}>
          View fused result
        </button>
      </div>
    </>
  )
}

/** An object URL for an image file, revoked when the file changes. */
function usePreview(file) {
  const [url, setUrl] = useState('')

  useEffect(() => {
    if (!file || !file.type.startsWith('image/')) {
      setUrl('')
      return
    }

    const next = URL.createObjectURL(file)
    setUrl(next)

    return () => URL.revokeObjectURL(next)
  }, [file])

  return url
}
