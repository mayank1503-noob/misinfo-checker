/* The escalation queue. High-impact, conflicting, or low-confidence
   cases stop here and wait for a person. */
import { Card, useToast } from '../ui.jsx'

export default function Review() {
  const toast = useToast()

  return (
    <section className="page">
      <div className="eyebrow">HUMAN-IN-THE-LOOP</div>
      <div className="h1">Human review queue</div>
      <p className="muted">
        High-impact, conflicting, or low-confidence cases must be escalated.
      </p>

      <Card>
        <div className="row">
          <div>
            <b>Case VL-2048</b>
            <p className="small muted">
              Image claim · Hindi/Hinglish · conflicting signals
            </p>
          </div>
          <span className="tag">Priority high</span>
        </div>

        <div className="evidence">
          Reviewer task: verify original media, consult authoritative records,
          document reasoning, and approve or reject the response draft.
        </div>

        <div className="actions">
          <button
            className="btn primary"
            onClick={() => toast('Case assigned to demo reviewer')}
          >
            Assign reviewer
          </button>
          <button
            className="btn"
            onClick={() => toast('Evidence request created in demo mode')}
          >
            Request evidence
          </button>
          <button
            className="btn warn"
            onClick={() => toast('Escalated to specialist queue')}
          >
            Escalate
          </button>
        </div>
      </Card>
    </section>
  )
}
