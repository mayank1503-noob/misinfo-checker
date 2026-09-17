/* The claim-to-sources diagram on the command centre.
   Hovering a source lights the edge that reaches it; hovering the claim
   in the middle lights all of them, because every edge is its own. */
import { useState } from 'react'

const NODES = [
  { id: 'n1', x: 55, y: 22, kind: 'good', title: 'Source A ✓', note: 'Corroborates', noteFill: '#a9d8c0' },
  { id: 'n2', x: 515, y: 22, kind: 'warn', title: 'Source B ⚠', note: 'Date mismatch', noteFill: '#f1d38d' },
  { id: 'n3', x: 55, y: 154, kind: 'bad', title: 'Source C ✕', note: 'Contradicts', noteFill: '#f2b2bd' },
  { id: 'n4', x: 515, y: 154, kind: 'core', title: 'Context layer', note: 'Counter-evidence', noteFill: '#a9dbe0' },
]

const EDGES = [
  { id: 'n1', x2: 150, y2: 48 },
  { id: 'n2', x2: 610, y2: 48 },
  { id: 'n3', x2: 150, y2: 180 },
  { id: 'n4', x2: 610, y2: 180 },
]

export default function EvidenceGraph() {
  const [lit, setLit] = useState(null)

  const edgeStyle = (id) =>
    lit === 'all' || lit === id
      ? { stroke: '#5ed3c6', strokeWidth: 3 }
      : undefined

  return (
    <div className="trust-graph">
      <svg
        viewBox="0 0 760 230"
        role="img"
        aria-label="Claim connected to supporting, conflicting and contextual sources"
      >
        {EDGES.map((edge) => (
          <line
            key={edge.id}
            className="edge"
            x1="380"
            y1="110"
            x2={edge.x2}
            y2={edge.y2}
            style={edgeStyle(edge.id)}
          />
        ))}

        <g
          className="node"
          onPointerEnter={() => setLit('all')}
          onPointerLeave={() => setLit(null)}
        >
          <rect className="node-core pulse" x="272" y="72" width="216" height="76" rx="16" opacity=".3" />
          <rect className="node-core" x="280" y="78" width="200" height="64" rx="14" />
          <text x="380" y="104" textAnchor="middle">USER CLAIM</text>
          <text x="380" y="123" textAnchor="middle" style={{ fontSize: 10, fill: '#9fc6d7' }}>
            Needs verification
          </text>
        </g>

        {NODES.map((node) => (
          <g
            key={node.id}
            className="node"
            onPointerEnter={() => setLit(node.id)}
            onPointerLeave={() => setLit(null)}
          >
            <rect className={`node-${node.kind}`} x={node.x} y={node.y} width="190" height="52" rx="12" />
            <text x={node.x + 95} y={node.y + 22} textAnchor="middle">
              {node.title}
            </text>
            <text
              x={node.x + 95}
              y={node.y + 37}
              textAnchor="middle"
              style={{ fontSize: 10, fill: node.noteFill }}
            >
              {node.note}
            </text>
          </g>
        ))}
      </svg>
    </div>
  )
}
