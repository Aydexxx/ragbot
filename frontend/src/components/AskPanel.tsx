import { useEffect, useRef, useState } from 'react'
import { ApiError, askQuestion } from '../api/client'
import type { AnswerResult, DocumentRecord } from '../api/types'
import { Banner } from './Banner'
import { AssistantAvatar, ChatTurn, QuestionBubble } from './ChatTurn'
import { Spinner } from './Spinner'

interface AskPanelProps {
  documents: DocumentRecord[]
  documentsLoading?: boolean
  disabled?: boolean
  disabledReason?: string
  /** Document scope for questions, owned by the parent so the scope selector
   * can live in the sidebar. Empty means "all documents". */
  selectedDocumentIds: string[]
}

interface Turn {
  question: string
  result: AnswerResult
}

const FLASH_DURATION_MS = 1300
const NO_DOCUMENTS_REASON = 'Upload a document before asking a question.'

/** Starter prompts shown in the empty state; clicking one asks it directly. */
const EXAMPLE_QUESTIONS = [
  'Give me a concise summary of this document.',
  'What are the main takeaways?',
  'Are there any risks or limitations mentioned?',
  'Explain the key ideas in simple terms.',
]

export function AskPanel({
  documents,
  documentsLoading = false,
  disabled = false,
  disabledReason,
  selectedDocumentIds,
}: AskPanelProps) {
  const [question, setQuestion] = useState('')
  const [turns, setTurns] = useState<Turn[]>([])
  const [conversationId, setConversationId] = useState<string | null>(null)
  const [pendingQuestion, setPendingQuestion] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [flash, setFlash] = useState<{ turnIndex: number; citation: number } | null>(
    null,
  )
  const threadEndRef = useRef<HTMLDivElement>(null)

  const noDocuments = !documentsLoading && documents.length === 0
  const isDisabled = disabled || noDocuments
  const effectiveDisabledReason = disabled ? disabledReason : NO_DOCUMENTS_REASON
  const showWelcome = turns.length === 0 && !pendingQuestion

  // Keep the newest turn in view as the conversation grows.
  useEffect(() => {
    threadEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [turns.length, pendingQuestion])

  async function submitQuestion(explicit?: string) {
    const trimmed = (explicit ?? question).trim()
    if (!trimmed || loading || isDisabled) return

    setLoading(true)
    setError(null)
    setPendingQuestion(trimmed)
    setQuestion('')
    try {
      const answer = await askQuestion({
        question: trimmed,
        document_ids: selectedDocumentIds.length > 0 ? selectedDocumentIds : undefined,
        conversation_id: conversationId ?? undefined,
      })
      setConversationId(answer.conversation_id)
      setTurns((prev) => [...prev, { question: trimmed, result: answer }])
    } catch (err) {
      const message =
        err instanceof ApiError ? err.message : 'Something went wrong.'
      setError(message)
    } finally {
      setLoading(false)
      setPendingQuestion(null)
    }
  }

  function startNewConversation() {
    setTurns([])
    setConversationId(null)
    setError(null)
  }

  function scrollToSource(turnIndex: number, citation: number) {
    const target = document.getElementById(`source-${turnIndex}-${citation}`)
    if (!target) return
    target.scrollIntoView({ behavior: 'smooth', block: 'center' })
    setFlash({ turnIndex, citation })
    setTimeout(() => setFlash(null), FLASH_DURATION_MS)
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      {/* Chat header: title + reset */}
      <div className="flex items-center justify-between gap-2 border-b border-slate-800/70 px-4 py-3 sm:px-6">
        <div className="min-w-0">
          <h2 className="text-sm font-semibold text-slate-100">Conversation</h2>
          <p className="truncate text-xs text-slate-500">
            Answers are grounded in your documents, with sources.
          </p>
        </div>
        <button
          type="button"
          onClick={startNewConversation}
          disabled={turns.length === 0}
          className="shrink-0 whitespace-nowrap rounded-lg border border-slate-700 px-2.5 py-1.5 text-xs text-slate-300 transition hover:border-slate-600 hover:text-slate-100 disabled:cursor-not-allowed disabled:opacity-40"
        >
          New chat
        </button>
      </div>

      {/* Scrollable thread */}
      <div className="thin-scrollbar min-h-0 flex-1 overflow-y-auto px-4 py-6 sm:px-6">
        <div className="mx-auto w-full max-w-3xl space-y-6">
          {showWelcome ? (
            <WelcomeState
              disabled={isDisabled}
              disabledReason={effectiveDisabledReason}
              examples={EXAMPLE_QUESTIONS}
              onPick={(text) => void submitQuestion(text)}
            />
          ) : (
            turns.map((turn, index) => (
              <ChatTurn
                key={index}
                turnIndex={index}
                question={turn.question}
                result={turn.result}
                flashCitation={flash?.turnIndex === index ? flash.citation : null}
                onCitationClick={(citation) => scrollToSource(index, citation)}
              />
            ))
          )}

          {pendingQuestion && (
            <div className="turn-enter space-y-4">
              <QuestionBubble question={pendingQuestion} />
              <div className="flex items-center gap-3">
                <AssistantAvatar />
                <div className="flex items-center gap-2 rounded-2xl rounded-tl-md border border-slate-800 bg-slate-900/60 px-4 py-3 text-sm text-slate-400">
                  <Spinner size="sm" />
                  <span>Searching your documents…</span>
                </div>
              </div>
            </div>
          )}

          <div ref={threadEndRef} />
        </div>
      </div>

      {/* Sticky composer */}
      <div className="border-t border-slate-800/70 bg-slate-950/60 px-4 py-3 sm:px-6">
        <div className="mx-auto w-full max-w-3xl space-y-2">
          {isDisabled && effectiveDisabledReason && (
            <Banner variant="warning">{effectiveDisabledReason}</Banner>
          )}
          {error && <Banner variant="error">{error}</Banner>}

          <form
            onSubmit={(event) => {
              event.preventDefault()
              void submitQuestion()
            }}
            className="flex items-end gap-2 rounded-2xl border border-slate-700 bg-slate-800/60 p-2 focus-within:border-teal-400/70"
          >
            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              disabled={isDisabled}
              placeholder="Ask a question about your documents…"
              rows={1}
              className="max-h-40 flex-1 resize-none bg-transparent px-2 py-1.5 text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none disabled:opacity-50"
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault()
                  void submitQuestion()
                }
              }}
            />
            <button
              type="submit"
              disabled={isDisabled || loading || !question.trim()}
              className="flex shrink-0 items-center justify-center rounded-xl bg-teal-500 px-4 py-2 text-sm font-medium text-slate-950 transition hover:bg-teal-400 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {loading ? <Spinner size="sm" /> : 'Ask'}
            </button>
          </form>

          <p className="px-1 text-center text-[11px] text-slate-600">
            RAGBot only answers from your documents — press Enter to send, Shift +
            Enter for a new line.
          </p>
        </div>
      </div>
    </div>
  )
}

/** Friendly intro shown before the first question, with clickable starter
 * prompts (enabled only once documents are ready to query). */
function WelcomeState({
  disabled,
  disabledReason,
  examples,
  onPick,
}: {
  disabled: boolean
  disabledReason?: string
  examples: string[]
  onPick: (text: string) => void
}) {
  return (
    <div className="flex flex-col items-center py-8 text-center">
      <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-teal-400 to-emerald-500 text-2xl font-bold text-slate-950 shadow-lg shadow-teal-500/20">
        R
      </div>
      <h3 className="mt-4 text-lg font-semibold text-slate-100">
        Ask anything about your documents
      </h3>
      <p className="mt-2 max-w-md text-sm leading-relaxed text-slate-400">
        Upload a document and ask anything about it — I&rsquo;ll answer only from
        your documents and show my sources.
      </p>

      {disabled ? (
        disabledReason && (
          <div className="mt-6 w-full max-w-md">
            <Banner variant="warning">{disabledReason}</Banner>
          </div>
        )
      ) : (
        <div className="mt-6 flex w-full max-w-lg flex-wrap justify-center gap-2">
          {examples.map((example) => (
            <button
              key={example}
              type="button"
              onClick={() => onPick(example)}
              className="rounded-full border border-slate-700 bg-slate-800/50 px-3.5 py-1.5 text-xs text-slate-300 transition hover:border-teal-500/50 hover:bg-teal-500/10 hover:text-teal-200"
            >
              {example}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
