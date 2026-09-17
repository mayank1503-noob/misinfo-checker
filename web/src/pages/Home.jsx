/* The command centre: what the system is, at a glance. The numbers on
   this page are illustrative, and the page says so. */
import { Bar, Card, Metric } from '../ui.jsx'
import EvidenceGraph from './EvidenceGraph.jsx'

export default function Home({ go }) {
  return (
    <section className="page">
      <div className="top">
        <div>
          <div className="eyebrow">TRUST OPERATIONS / DEMO</div>
          <div className="h1">See the claim. Trace the context.</div>
          <div className="muted">
            A multimodal workflow for text, links, images and videos—built
            around AI-03.
          </div>
        </div>
        <select className="pill" aria-label="Interface language">
          <option>English</option>
          <option>हिन्दी</option>
          <option>मराठी</option>
          <option>தமிழ்</option>
          <option>বাংলা</option>
        </select>
      </div>

      <div className="grid metrics">
        <Card>
          <div className="label">Claims screened</div>
          <Metric value={1284} />
          <span className="green">+18.6% this week</span>
        </Card>
        <Card>
          <div className="label">Multimodal cases</div>
          <Metric value={312} />
          <span className="muted">Text + media signals</span>
        </Card>
        <Card>
          <div className="label">Languages / scripts</div>
          <Metric value={12} suffix="+" />
          <span className="muted">Indic + code-mixed ready</span>
        </Card>
        <Card>
          <div className="label">Human escalations</div>
          <Metric value={24} className="amber" />
          <span className="muted">Awaiting review</span>
        </Card>
      </div>

      <div className="grid two">
        <Card>
          <div className="section">Priority case</div>
          <div className="small muted">Synthetic example · recycled media</div>
          <div className="claim">“This photo shows today’s flooding in Mumbai.”</div>
          <span className="tag">Image claim</span>
          <span className="tag">Time mismatch risk</span>
          <span className="tag">Location unverified</span>
          <div className="row" style={{ marginTop: 18 }}>
            <b className="amber">Needs evidence review</b>
            <span className="small muted">Not a final verdict</span>
          </div>
          <div style={{ marginTop: 10 }}>
            <Bar value={82} />
          </div>
          <div className="actions">
            <button className="btn primary" onClick={() => go('analyze')}>
              Analyze media →
            </button>
            <button className="btn" onClick={() => go('evidence')}>
              Open evidence
            </button>
          </div>
        </Card>

        <Card>
          <div className="section">System architecture</div>
          <div className="list">
            <div className="item">
              <b>Media analysis layer</b>
              <p className="small muted">
                Whisper · OCR/TrOCR · BLIP · DINOv2 · EXIF · keyframes
              </p>
            </div>
            <div className="item">
              <b>Fact-check engine</b>
              <p className="small muted">
                Known rumors · known images · Fact Check API · web evidence ·
                NLI · CLIP
              </p>
            </div>
            <div className="item">
              <b>Responsible response</b>
              <p className="small muted">
                Evidence trail, time note, correction draft, human escalation
              </p>
            </div>
          </div>
        </Card>
      </div>

      <Card className="scanline" style={{ marginTop: 16 }}>
        <div className="row">
          <div>
            <div className="section">Evidence graph</div>
            <p className="small muted">
              A claim is evaluated by tracing sources, contradictions, and
              corroborating context.
            </p>
          </div>
          <span className="tag">Live-style demo</span>
        </div>
        <EvidenceGraph />
      </Card>

      <div className="grid three" style={{ marginTop: 16 }}>
        <Card>
          <div className="label">Core innovation</div>
          <h3>Time-aware verification</h3>
          <p className="small muted">
            Catch old news, recycled images, and stale links shared as current.
          </p>
        </Card>
        <Card>
          <div className="label">India-first layer</div>
          <h3>Code-mixed claims</h3>
          <p className="small muted">
            Preserve original text while normalizing Hinglish and Indic-script
            variants.
          </p>
        </Card>
        <Card>
          <div className="label">Safety principle</div>
          <h3>Explain, don’t amplify</h3>
          <p className="small muted">
            Corrections lead with verified context instead of repeating harmful
            rumors.
          </p>
        </Card>
      </div>
    </section>
  )
}
