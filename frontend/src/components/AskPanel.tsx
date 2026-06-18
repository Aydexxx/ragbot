import { useState } from 'react'
import { ApiError, askQuestion } from '../api/client'
import type { AnswerResult, DocumentRecord } from '../api/types'
import { Banner } from './Banner'
import { ChatTurn } from './ChatTurn'
import { DocumentScopeSelector } from './DocumentScopeSelector'
import { Spinner } from './Spinner'

interface AskPanelProps {
  documents: DocumentRecord[]
  documentsLoading?: boolean
  disabled?: boolean
  disabledReason?: string
}

interface Turn {
  question: string
  result: AnswerResult
}

const FLASH_DURATION_MS = 1300
const NO_DOCUMENTS_REASON = 'Upload a document before asking a question.'

export function AskPanel({
  documents,
  documentsLoading = false,
  disabled = false,
  disabledReason,
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
  const [selectedDocumentIds, setSelectedDocumentIds] = useState<string[]>([])

  const noDocuments = !documentsLoading && documents.length === 0
  const isDisabled = disabled || noDocuments
  const effectiveDisabledReason = disabled ? disabledReason : NO_DOCUMENTS_REASON

  async function submitQuestion() {
    const trimmed = question.trim()
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
    <div className="flex h-full flex-col gap-4">
      <p className="text-xs text-slate-500">
        RAGBot only answers from your documents and shows its sources. When the
        documents don&rsquo;t cover a question, it says so instead of guessing.
      </p>

      <div className="flex items-start justify-between gap-2">
        <DocumentScopeSelector
          documents={documents}
          selectedIds={selectedDocumentIds}
          onChange={setSelectedDocumentIds}
          disabled={isDisabled}
        />
        <button
          type="button"
          onClick={startNewConversation}
          disabled={turns.length === 0}
          className="shrink-0 whitespace-nowrap text-xs text-slate-400 transition hover:text-slate-200 disabled:cursor-not-allowed disabled:opacity-40"
        >
          New conversation
        </button>
      </div>

      <form
        onSubmit={(event) => {
          event.preventDefault()
          void submitQuestion()
        }}
        className="flex gap-2"
      >
        <textarea
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          disabled={isDisabled}
          placeholder="Ask a question about your documents…"
          rows={2}
          className="flex-1 resize-none rounded-lg border border-slate-700 bg-slate-800/60 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-500 focus:border-indigo-400 focus:outline-none disabled:opacity-50"
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
          className="flex shrink-0 items-center justify-center rounded-lg bg-indigo-500 px-4 py-2 text-sm font-medium text-white transition hover:bg-indigo-400 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {loading ? <Spinner size="sm" /> : 'Ask'}
        </button>
      </form>

      {isDisabled && effectiveDisabledReason && (
        <Banner variant="warning">{effectiveDisabledReason}</Banner>
      )}

      {error && <Banner variant="error">{error}</Banner>}

      <div className="flex-1 space-y-4 overflow-y-auto pb-2">
        {!isDisabled && turns.length === 0 && !pendingQuestion && (
          <p className="text-sm text-slate-500">Ask a question to get started.</p>
        )}

        {turns.map((turn, index) => (
          <ChatTurn
            key={index}
            turnIndex={index}
            question={turn.question}
            result={turn.result}
            flashCitation={flash?.turnIndex === index ? flash.citation : null}
            onCitationClick={(citation) => scrollToSource(index, citation)}
          />
        ))}

        {pendingQuestion && (
          <div className="space-y-2">
            <p className="text-sm text-slate-400">
              <span className="font-medium text-slate-300">Q:</span>{' '}
              {pendingQuestion}
            </p>
            <Spinner size="sm" />
          </div>
        )}
      </div>
    </div>
  )
}
