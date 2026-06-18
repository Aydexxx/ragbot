import type { GroundingStatus } from '../api/types'

interface GroundingIndicatorProps {
  status: GroundingStatus
  coverage: number
  sourceCount: number
}

/** Coverage tier -> bar color + label, for the well-grounded (answered) case. */
function answeredTier(coverage: number): { color: string; label: string } {
  if (coverage >= 0.75) return { color: 'bg-emerald-400', label: 'Strongly grounded' }
  if (coverage >= 0.5) return { color: 'bg-emerald-400', label: 'Grounded' }
  return { color: 'bg-amber-400', label: 'Loosely grounded' }
}

/** A compact confidence indicator summarizing how well an answer is grounded.
 *
 * Always visible on a turn so the user can read support at a glance: a green
 * "Grounded in N sources" with a coverage bar when answered, or an amber
 * "weak match" cue for insufficient_context. The bar width is the coverage
 * score (best-source similarity, 0..1). */
export function GroundingIndicator({
  status,
  coverage,
  sourceCount,
}: GroundingIndicatorProps) {
  const pct = Math.round(Math.min(1, Math.max(0, coverage)) * 100)

  if (status === 'insufficient_context') {
    return (
      <div className="flex items-center gap-2 text-xs text-amber-300">
        <span aria-hidden>⚠</span>
        <span>Weak match — {pct}% similarity. Verify against the sources.</span>
      </div>
    )
  }

  const { color, label } = answeredTier(coverage)
  const sourceLabel = sourceCount === 1 ? '1 source' : `${sourceCount} sources`

  return (
    <div className="flex items-center gap-2 text-xs text-emerald-300">
      <span aria-hidden>✓</span>
      <span className="whitespace-nowrap">
        {label} in {sourceLabel}
      </span>
      <div
        className="h-1 w-16 overflow-hidden rounded-full bg-slate-700"
        role="meter"
        aria-valuenow={pct}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label="Grounding coverage"
      >
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-slate-500">{pct}%</span>
    </div>
  )
}
