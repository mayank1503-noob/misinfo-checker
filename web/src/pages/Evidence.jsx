/*
 * What the run actually found — or, before there is a run, an
 * illustrative version of the same layout so the page is never empty.
 *
 * The stage list is the interesting half: a retriever that failed still
 * lets the pipeline finish with `unverified`, and the page says which
 * one failed rather than hiding it behind the verdict.
 */
import { labelClass, pct } from '../api.js'
import { Bar, Card } from '../ui.jsx'

export default function Evidence({ go, result }) {
  return (
    <section className="page">
      <div className="eyebrow">FACT-CHECK ENGINE</div>
      <div className="h1">Evidence fusion</div>
      <p className="muted">
        The engine combines independent signals instead of trusting a single
        model.
      </p>

      {result ? <Live data={result} go={go} /> : <Demo go={go} />}
    </section>
  )
}

function Live({ data, go }) {
  const verdict = data.verdict
  const counts = verdict.counts || {}
  const stages = Object.entries(data.stages || {})
  const open = data.open_claims || []

  return (
    <>
      <div className="grid two">
        <Card>
          <div className="section">Signal ledger</div>
          {Object.keys(counts).length ? (
            Object.entries(counts).map(([label, n]) => (
              <div className="signal" key={label}>
                <span>{label}</span>
                <b className={labelClass(label)}>{n}</b>
              </div>
            ))
          ) : (
            <p className="small muted">
              No per-label counts returned for this run.
            </p>
          )}

          {stages.length > 0 && (
            <>
              <div className="section" style={{ marginTop: 14 }}>
                Stages
              </div>
              {stages.map(([name, stage]) => (
                <div key={name}>
                  <div className="signal">
                    <span>
                      {name} {stage && stage.error && <b className="red">failed</b>}
                    </span>
                    <b className="small muted">
                      {stage && stage.ms != null
                        ? `${Math.round(stage.ms)} ms`
                        : 'ran'}
                    </b>
                  </div>
                  {stage && stage.error && (
                    <p className="small red">{String(stage.error)}</p>
                  )}
                </div>
              ))}
            </>
          )}
        </Card>

        <Card>
          <div className="section">Fused assessment</div>
          <h2 className={labelClass(verdict.label)}>
            {String(verdict.label).toUpperCase()}
          </h2>
          <p className="claim">{verdict.summary || ''}</p>
          <Bar value={pct(verdict.confidence)} />
          <p className="small muted">
            Confidence: {pct(verdict.confidence)}% · requires human confirmation
          </p>

          {open.length > 0 && (
            <div className="evidence">
              <b>Still open</b>
              <ul className="small">
                {open.map((claim, i) => (
                  <li key={i}>
                    {typeof claim === 'string'
                      ? claim
                      : claim.claim || JSON.stringify(claim)}
                  </li>
                ))}
              </ul>
            </div>
          )}

          <div className="actions">
            <button className="btn primary" onClick={() => go('response')}>
              Create responsible response →
            </button>
            <button className="btn warn" onClick={() => go('review')}>
              Escalate to human
            </button>
          </div>
        </Card>
      </div>

      {data.agents && (
        <Card style={{ marginTop: 16 }}>
          <div className="section">Agentic review</div>
          <pre
            className="small"
            style={{ whiteSpace: 'pre-wrap', margin: 0, color: '#bfe6ff' }}
          >
            {JSON.stringify(data.agents, null, 2)}
          </pre>
        </Card>
      )}
    </>
  )
}

/* The pre-run view. Every number here is invented and labelled as such;
   it exists to show the shape of a finished case. */
function Demo({ go }) {
  return (
    <>
      <div className="grid two">
        <Card>
          <div className="section">Signal ledger</div>
          <div className="signal">
            <span>Known-rumor similarity</span>
            <b className="amber">0.84 · match</b>
          </div>
          <div className="signal">
            <span>Known-image similarity</span>
            <b className="amber">0.91 · possible reuse</b>
          </div>
          <div className="signal">
            <span>Image ↔ claim consistency</span>
            <b className="red">0.28 · mismatch</b>
          </div>
          <div className="signal">
            <span>Source date</span>
            <b className="amber">Not confirmed</b>
          </div>
          <div className="signal">
            <span>NLI evidence</span>
            <b className="amber">Conflicting</b>
          </div>
          <div className="signal">
            <span>AI-generated image hint</span>
            <b className="muted">Weak signal only</b>
          </div>
        </Card>

        <Card>
          <div className="section">Fused assessment</div>
          <div className="tag">Illustrative demo</div>
          <h2 className="amber">Misleading</h2>
          <p className="claim">
            The media may be authentic, but the accompanying time or location
            claim is not sufficiently supported.
          </p>
          <Bar value={76} />
          <p className="small muted">
            Confidence: 0.76 · Requires human confirmation
          </p>
          <div className="evidence">
            <b>Time note</b>
            <p className="small">
              The image appears to match an older indexed event. Verify the
              earliest reliable appearance before publishing a correction.
            </p>
          </div>
          <div className="evidence">
            <b>Image note</b>
            <p className="small">
              Visual similarity suggests possible reuse; similarity alone does
              not prove identical origin.
            </p>
          </div>
          <div className="actions">
            <button className="btn primary" onClick={() => go('response')}>
              Create responsible response →
            </button>
            <button className="btn warn" onClick={() => go('review')}>
              Escalate to human
            </button>
          </div>
        </Card>
      </div>

      <Card style={{ marginTop: 16 }}>
        <div className="section">Evidence sources</div>
        <div className="list">
          <div className="item">
            <b>Known-image index</b>
            <p className="small muted">
              Publisher: Example fact-check archive · First seen: 2015-12-02 ·
              URL retained by backend
            </p>
          </div>
          <div className="item">
            <b>Independent corroboration</b>
            <p className="small muted">
              Two sources required; copied articles are not independent
              confirmation.
            </p>
          </div>
          <div className="item">
            <b>Authoritative record</b>
            <p className="small muted">
              For health, safety, elections, or emergencies, route to a
              specialist reviewer.
            </p>
          </div>
        </div>
      </Card>
    </>
  )
}
