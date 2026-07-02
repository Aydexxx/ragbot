import type { ReactNode } from 'react'
import type { AnswerResult, Source } from '../api/types'
import { Banner } from './Banner'
import { CitedAnswer } from './CitedAnswer'
import { GroundingIndicator } from './GroundingIndicator'
import { SourceList } from './SourceList'

interface ChatTurnProps {
  turnIndex: number
  question: string
  result: AnswerResult
  flashCitation: number | null
  onCitationClick: (citation: number) => void
}

/** Unique document filenames behind `sources`, in citation order.
 *
 * Prefers the sources the model actually cited (`cited`) so the summary
 * reflects what the answer is grounded in; falls back to every retrieved
 * source when nothing was cited (e.g. retrieval-only mode).
 */
function sourceDocumentFilenames(sources: Source[], cited: number[]): string[] {
  const relevant =
    cited.length > 0
      ? sources.filter((source) => cited.includes(source.citation))
      : sources

  const seen = new Set<string>()
  const filenames: string[] = []
  for (const source of relevant) {
    if (seen.has(source.document_id)) continue
    seen.add(source.document_id)
    filenames.push(source.filename)
  }
  return filenames
}

/** The user's question, rendered as a right-aligned chat bubble. */
export function QuestionBubble({
  question,
  standalone = null,
}: {
  question: string
  standalone?: string | null
}) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[85%] rounded-2xl rounded-br-md border border-teal-500/25 bg-teal-500/10 px-4 py-2.5">
        <p className="whitespace-pre-wrap text-sm leading-relaxed text-slate-100">
          {question}
        </p>
        {standalone && (
          <p className="mt-1 text-xs italic text-teal-200/70">
            Interpreted as: &ldquo;{standalone}&rdquo;
          </p>
        )}
      </div>
    </div>
  )
}

/** RAGBot's circular avatar, marking the assistant side of the thread. */
export function AssistantAvatar() {
  return (
    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-teal-400 to-emerald-500 text-sm font-bold text-slate-950 shadow-sm">
      R
    </div>
  )
}

/** Wraps assistant content (answer, grounding, sources) in a left-aligned
 * bubble with RAGBot's avatar, mirroring the user question bubble. */
function AssistantBubble({ children }: { children: ReactNode }) {
  return (
    <div className="flex gap-3">
      <AssistantAvatar />
      <div className="min-w-0 flex-1 space-y-2 rounded-2xl rounded-tl-md border border-slate-800 bg-slate-900/60 px-4 py-3">
        {children}
      </div>
    </div>
  )
}

/** One question/answer/sources exchange within the chat thread.
 *
 * Renders one of three honest grounding states (see `result.status`):
 * a normal grounded answer, an explicit "documents don't cover this" decline
 * that still shows the closest material, or a friendly no-documents prompt. */
export function ChatTurn({
  turnIndex,
  question,
  result,
  flashCitation,
  onCitationClick,
}: ChatTurnProps) {
  const validCitations = new Set(result.sources.map((s) => s.citation))
  const answerDocuments = sourceDocumentFilenames(result.sources, result.cited)
  const sourceList = (
    <SourceList
      sources={result.sources}
      flashCitation={flashCitation}
      query={result.standalone_question ?? question}
      idPrefix={`source-${turnIndex}`}
    />
  )

  return (
    <div className="turn-enter space-y-4">
      <QuestionBubble
        question={question}
        standalone={result.standalone_question}
      />

      <AssistantBubble>
        {result.status === 'no_documents' ? (
          <Banner variant="info">
            No indexed material to draw on for this question. Upload a document
            (or widen the scope), then ask again — RAGBot only answers from your
            documents.
          </Banner>
        ) : result.status === 'insufficient_context' ? (
          <>
            <Banner variant="warning">
              The documents don&rsquo;t clearly answer this. Rather than guess,
              here&rsquo;s the closest related material I found — please judge it
              yourself.
            </Banner>
            <GroundingIndicator
              status={result.status}
              coverage={result.coverage}
              sourceCount={result.sources.length}
            />
            {sourceList}
          </>
        ) : (
          <>
            {!result.generation_enabled && (
              <Banner variant="info">
                Retrieval-only mode — showing source passages (configure an LLM
                to enable answers)
              </Banner>
            )}

            {result.answer && (
              <CitedAnswer
                answer={result.answer}
                validCitations={validCitations}
                onCitationClick={onCitationClick}
              />
            )}

            {result.low_confidence && (
              <p className="text-xs text-amber-300/80">
                Uncited — this answer cites no source, so verify it against the
                passages below.
              </p>
            )}

            <GroundingIndicator
              status={result.status}
              coverage={result.coverage}
              sourceCount={result.sources.length}
            />

            {result.answer && answerDocuments.length > 0 && (
              <p className="text-xs text-slate-500">
                Answer drawn from{' '}
                {answerDocuments.length === 1
                  ? '1 document'
                  : `${answerDocuments.length} documents`}
                : {answerDocuments.join(', ')}
              </p>
            )}

            {sourceList}
          </>
        )}
      </AssistantBubble>
    </div>
  )
}
