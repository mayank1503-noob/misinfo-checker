/* Drafting a correction. The draft is assembled in the browser from
   verified context the reviewer types: nothing here calls a model, and
   it deliberately does not repeat the original rumor. */
import { useState } from 'react'
import { Card, useToast } from '../ui.jsx'

export default function Response({ go }) {
  const toast = useToast()

  const [context, setContext] = useState('')
  const [draft, setDraft] = useState('')

  function generate() {
    if (!context.trim()) return toast('Add verified context first')
    setDraft(context.trim())
    toast('Correction draft generated')
  }

  async function copy() {
    if (!draft) return toast('Nothing to copy yet')

    try {
      await navigator.clipboard.writeText(draft)
      toast('Copied to clipboard')
    } catch {
      toast('Clipboard blocked by the browser')
    }
  }

  return (
    <section className="page">
      <div className="eyebrow">RESPONSIBLE COMMUNICATION</div>
      <div className="h1">Response studio</div>
      <p className="muted">
        Draft a correction that provides context without unnecessarily
        repeating the rumor.
      </p>

      <div className="grid two">
        <Card>
          <label className="label" htmlFor="audience">
            Audience
          </label>
          <select id="audience">
            <option>General public</option>
            <option>WhatsApp community</option>
            <option>Student audience</option>
            <option>Local-language audience</option>
          </select>

          <label
            className="label"
            style={{ display: 'block', marginTop: 15 }}
            htmlFor="context"
          >
            Verified context
          </label>
          <textarea
            id="context"
            value={context}
            onChange={(e) => setContext(e.target.value)}
            placeholder="Enter the verified facts, source, date, and recommended action..."
          />

          <div className="actions">
            <button className="btn primary" onClick={generate}>
              Generate correction draft
            </button>
          </div>
        </Card>

        <Card>
          <div className="section">Draft</div>

          {draft ? (
            <>
              <p>
                <b>Suggested correction</b>
              </p>
              <p>{draft}</p>
              <p className="small muted">
                Please verify the linked source and share this context instead
                of forwarding the original unverified claim. If evidence
                changes, update or withdraw this message.
              </p>
            </>
          ) : (
            <div className="muted">Your draft will appear here.</div>
          )}

          <div className="actions">
            <button className="btn" onClick={copy}>
              Copy draft
            </button>
            <button className="btn warn" onClick={() => go('review')}>
              Send to review
            </button>
          </div>
        </Card>
      </div>
    </section>
  )
}
