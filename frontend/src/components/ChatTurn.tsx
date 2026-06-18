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

function QuestionLine({
  question,
  standalone,
}: {
  question: string
  standalone: string | null
}) {
  return (
    <>
      <p className="text-sm text-slate-400">
        <span className="font-medium text-slate-300">Q:</span> {question}
      </p>
      {standalone && (
        <p className="text-xs italic text-slate-500">
          Interpreted as: &ldquo;{standalone}&rdquo;
        </p>
      )}
    </>
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
    <div className="turn-enter space-y-2 border-b border-slate-800/60 pb-4 last:border-b-0 last:pb-0">
      <QuestionLine question={question} standalone={result.standalone_question} />

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
              Retrieval-only mode — showing source passages (configure an LLM to
              enable answers)
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
    </div>
  )
}
