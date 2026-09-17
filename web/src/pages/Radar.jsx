/* Recurring-narrative view. Synthetic, and labelled synthetic: the
   backend does not yet aggregate across runs. */
import { Bar, Card } from '../ui.jsx'

const TRENDS = [
  {
    tag: 'Rising',
    title: 'Recycled images',
    note: 'Old flood, protest, disaster, or conflict photos relabeled as current.',
    strength: 91,
  },
  {
    tag: 'Persistent',
    title: 'Health cures',
    note: 'Miracle-treatment claims and fake medical endorsements.',
    strength: 83,
  },
  {
    tag: 'Watch',
    title: 'Code-mixed rumors',
    note: 'Indic scripts, transliteration, and spelling mutations.',
    strength: 68,
  },
]

const RULES = [
  {
    title: 'Text / link date check',
    note: 'Compare publication date, event date, and the date implied by the message.',
  },
  {
    title: 'Image date check',
    note: 'Compare EXIF metadata, earliest known appearance, and indexed image matches.',
  },
  {
    title: 'Video frame timeline',
    note: 'Inspect keyframes and frame timestamps for reused footage.',
  },
]

export default function Radar() {
  return (
    <section className="page">
      <div className="eyebrow">TEMPORAL + PATTERN INTELLIGENCE</div>
      <div className="h1">Time &amp; narrative radar</div>
      <p className="muted">
        Synthetic examples showing how the system could detect recurring
        narratives and recycled media.
      </p>

      <div className="grid three">
        {TRENDS.map((trend) => (
          <Card key={trend.title}>
            <span className="tag">{trend.tag}</span>
            <h2>{trend.title}</h2>
            <p className="small muted">{trend.note}</p>
            <Bar value={trend.strength} />
            <p className="small">Signal strength: {trend.strength}/100</p>
          </Card>
        ))}
      </div>

      <Card style={{ marginTop: 16 }}>
        <div className="section">Time-check rules</div>
        <div className="list">
          {RULES.map((rule) => (
            <div className="item" key={rule.title}>
              <b>{rule.title}</b>
              <p className="small muted">{rule.note}</p>
            </div>
          ))}
        </div>
      </Card>
    </section>
  )
}
