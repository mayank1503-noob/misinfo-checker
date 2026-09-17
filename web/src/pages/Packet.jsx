/* The Media Analysis Packet exactly as the API returned it: the
   contract between the media layer and the fact-check engine, so it is
   shown raw rather than prettified into something else. */
import { Card, useToast } from '../ui.jsx'

export default function Packet({ go, result }) {
  const toast = useToast()

  const text = result
    ? JSON.stringify(result.packet, null, 2)
    : 'No packet yet. Run an analysis first.'

  async function copy() {
    try {
      await navigator.clipboard.writeText(text)
      toast('Copied to clipboard')
    } catch {
      toast('Clipboard blocked by the browser')
    }
  }

  return (
    <section className="page">
      <div className="eyebrow">CONTRACT / MEDIA PACKET</div>
      <div className="h1">Media Analysis Packet</div>
      <p className="muted">
        This is the contract passed from the media layer to the fact-check
        engine.
      </p>

      <Card>
        <pre
          style={{
            whiteSpace: 'pre-wrap',
            overflow: 'auto',
            color: '#bfe6ff',
            margin: 0,
          }}
        >
          {text}
        </pre>
        <div className="actions">
          <button className="btn" onClick={copy}>
            Copy packet
          </button>
          <button className="btn primary" onClick={() => go('evidence')}>
            Send to evidence fusion →
          </button>
        </div>
      </Card>
    </section>
  )
}
