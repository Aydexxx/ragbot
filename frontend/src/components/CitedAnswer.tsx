import type { ReactNode } from 'react'

interface CitedAnswerProps {
  answer: string
  validCitations: ReadonlySet<number>
  onCitationClick: (citation: number) => void
}

const CITATION_PATTERN = /\[(\d+)\]/g

/** Renders answer text with `[n]` markers replaced by clickable citation pills. */
export function CitedAnswer({
  answer,
  validCitations,
  onCitationClick,
}: CitedAnswerProps) {
  const parts: ReactNode[] = []
  let lastIndex = 0
  const regex = new RegExp(CITATION_PATTERN)
  let match: RegExpExecArray | null

  while ((match = regex.exec(answer)) !== null) {
    if (match.index > lastIndex) {
      parts.push(answer.slice(lastIndex, match.index))
    }

    const citation = Number(match[1])
    if (validCitations.has(citation)) {
      parts.push(
        <button
          key={`citation-${match.index}`}
          type="button"
          onClick={() => onCitationClick(citation)}
          className="mx-0.5 inline-flex h-5 min-w-5 -translate-y-px items-center justify-center rounded-full bg-indigo-500/20 px-1 text-xs font-medium text-indigo-300 transition hover:bg-indigo-500/40"
        >
          {citation}
        </button>,
      )
    } else {
      parts.push(match[0])
    }
    lastIndex = match.index + match[0].length
  }

  if (lastIndex < answer.length) {
    parts.push(answer.slice(lastIndex))
  }

  return (
    <p className="whitespace-pre-wrap leading-relaxed text-slate-100">{parts}</p>
  )
}
