import { useState } from 'react'
import type { ReactNode } from 'react'
import type { Source } from '../api/types'

interface SourceListProps {
  sources: Source[]
  flashCitation: number | null
  /** The asked question, used to highlight the most relevant sentence(s). */
  query?: string
  /** Prefix for each card's DOM id, so multiple lists (e.g. one per chat
   * turn) on the same page don't collide on duplicate ids. */
  idPrefix?: string
}

/** Show the expand toggle once a passage is longer than this many characters. */
const EXPAND_THRESHOLD = 240

export function SourceList({
  sources,
  flashCitation,
  query,
  idPrefix = 'source',
}: SourceListProps) {
  if (sources.length === 0) return null

  return (
    <div className="space-y-2">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-400">
        Sources
      </h3>
      <ul className="space-y-2">
        {sources.map((source) => (
          <SourceCard
            key={source.citation}
            source={source}
            isFlashing={flashCitation === source.citation}
            query={query}
            idPrefix={idPrefix}
          />
        ))}
      </ul>
    </div>
  )
}

function SourceCard({
  source,
  isFlashing,
  query,
  idPrefix,
}: {
  source: Source
  isFlashing: boolean
  query?: string
  idPrefix: string
}) {
  const [expanded, setExpanded] = useState(false)
  const isLong = source.text.length > EXPAND_THRESHOLD
  const pct = Math.min(100, Math.max(0, Math.round(source.score * 100)))
  const passage = highlightPassage(source.text, query)

  return (
    <li
      id={`${idPrefix}-${source.citation}`}
      className={`rounded-lg border border-slate-700 bg-slate-800/40 p-3 ${
        isFlashing ? 'citation-flash' : ''
      }`}
    >
      <div className="flex items-center justify-between gap-2">
        <div className="flex min-w-0 items-center gap-2 text-sm">
          <span className="inline-flex h-5 min-w-5 shrink-0 items-center justify-center rounded-full bg-teal-500/20 px-1 text-xs font-medium text-teal-300">
            {source.citation}
          </span>
          <span className="truncate font-medium text-slate-200">
            {source.filename}
          </span>
        </div>
        <span className="shrink-0 text-xs text-slate-400">{pct}% match</span>
      </div>

      <div className="mt-1.5 flex items-center gap-2">
        <div
          className="h-1 flex-1 overflow-hidden rounded-full bg-slate-700"
          role="meter"
          aria-valuenow={pct}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label="Similarity confidence"
        >
          <div
            className={`h-full rounded-full ${confidenceColor(source.score)}`}
            style={{ width: `${pct}%` }}
          />
        </div>
        <span className="shrink-0 text-xs text-slate-500">{locator(source)}</span>
      </div>

      <p
        className={`mt-2 text-sm leading-relaxed text-slate-300 ${
          expanded ? '' : 'line-clamp-3'
        }`}
      >
        {passage}
      </p>
      {isLong && (
        <button
          type="button"
          onClick={() => setExpanded((prev) => !prev)}
          className="mt-1 text-xs text-teal-300 hover:underline"
        >
          {expanded ? 'Show less' : 'Show more'}
        </button>
      )}
    </li>
  )
}

/** A compact, human-readable location: page (when known), chunk, char range. */
function locator(source: Source): string {
  const parts: string[] = []
  if (source.page != null) parts.push(`p. ${source.page}`)
  parts.push(`chunk ${source.chunk_index}`)
  parts.push(`chars ${source.char_start}–${source.char_end}`)
  return parts.join(' · ')
}

function confidenceColor(score: number): string {
  if (score >= 0.75) return 'bg-emerald-400'
  if (score >= 0.5) return 'bg-teal-400'
  return 'bg-amber-400'
}

const STOPWORDS = new Set([
  'the', 'and', 'for', 'are', 'was', 'were', 'has', 'have', 'had', 'with',
  'that', 'this', 'from', 'into', 'about', 'what', 'which', 'who', 'whom',
  'how', 'why', 'when', 'where', 'does', 'did', 'can', 'could', 'would',
  'should', 'their', 'there', 'them', 'they', 'you', 'your', 'its', 'his',
  'her', 'our', 'out', 'not', 'but', 'all', 'any', 'use', 'used', 'using',
])

/** Content words from the query worth matching against the passage. */
function queryTokens(query: string): Set<string> {
  return new Set(
    query
      .toLowerCase()
      .split(/[^a-z0-9]+/)
      .filter((word) => word.length > 2 && !STOPWORDS.has(word)),
  )
}

/** Split into sentences while preserving every character (delimiters and
 * trailing whitespace stay attached), so the pieces re-concatenate exactly. */
function splitSentences(text: string): string[] {
  return text.match(/[^.!?]*[.!?]+\s*|[^.!?]+$/g) ?? [text]
}

/** Render the passage with the sentence(s) most relevant to the query
 * highlighted. Falls back to the plain passage when there's no query overlap
 * or sentence splitting wouldn't round-trip the original text exactly. */
function highlightPassage(text: string, query?: string): ReactNode {
  if (!query) return text
  const tokens = queryTokens(query)
  if (tokens.size === 0) return text

  const sentences = splitSentences(text)
  if (sentences.join('') !== text) return text

  const scores = sentences.map((sentence) => {
    let score = 0
    for (const word of sentence.toLowerCase().split(/[^a-z0-9]+/)) {
      if (tokens.has(word)) score += 1
    }
    return score
  })

  const max = Math.max(...scores)
  if (max === 0) return text

  return sentences.map((sentence, i) =>
    scores[i] === max ? (
      <mark
        key={i}
        className="rounded bg-amber-300/20 text-amber-100"
      >
        {sentence}
      </mark>
    ) : (
      <span key={i}>{sentence}</span>
    ),
  )
}
