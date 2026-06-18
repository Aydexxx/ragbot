import { useState } from 'react'
import type { DocumentRecord, DocumentStatus } from '../api/types'
import { Spinner } from './Spinner'

interface DocumentListProps {
  documents: DocumentRecord[]
  loading: boolean
  error: string | null
  deletingIds: ReadonlySet<string>
  onDelete: (id: string) => void
}

const STATUS_CLASSES: Record<DocumentStatus, string> = {
  pending: 'bg-slate-700 text-slate-300',
  processing: 'bg-amber-500/20 text-amber-300',
  ready: 'bg-emerald-500/20 text-emerald-300',
  failed: 'bg-rose-500/20 text-rose-300',
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  })
}

export function DocumentList({
  documents,
  loading,
  error,
  deletingIds,
  onDelete,
}: DocumentListProps) {
  const [confirmId, setConfirmId] = useState<string | null>(null)

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-sm text-slate-400">
        <Spinner size="sm" />
        Loading documents…
      </div>
    )
  }

  if (error) {
    return <p className="text-sm text-rose-300">{error}</p>
  }

  if (documents.length === 0) {
    return (
      <p className="rounded-lg border border-dashed border-slate-700 px-4 py-6 text-center text-sm text-slate-500">
        No documents indexed yet — upload one to get started.
      </p>
    )
  }

  return (
    <ul className="space-y-2">
      {documents.map((doc) => {
        const isDeleting = deletingIds.has(doc.id)
        const isConfirming = confirmId === doc.id

        return (
          <li
            key={doc.id}
            className="flex items-center justify-between gap-3 rounded-lg border border-slate-700 bg-slate-800/40 px-3 py-2"
          >
            <div className="min-w-0">
              <p className="truncate text-sm font-medium text-slate-200">
                {doc.filename}
              </p>
              <p className="mt-0.5 text-xs text-slate-500">
                {doc.num_chunks} {doc.num_chunks === 1 ? 'chunk' : 'chunks'} ·{' '}
                {formatDate(doc.uploaded_at)}
              </p>
            </div>

            <div className="flex shrink-0 items-center gap-2">
              <span
                className={`rounded-full px-2 py-0.5 text-xs capitalize ${STATUS_CLASSES[doc.status]}`}
              >
                {doc.status}
              </span>

              {isConfirming ? (
                <div className="flex items-center gap-1.5">
                  <button
                    type="button"
                    onClick={() => setConfirmId(null)}
                    className="rounded-md px-2 py-1 text-xs text-slate-300 hover:bg-slate-700"
                  >
                    Cancel
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setConfirmId(null)
                      onDelete(doc.id)
                    }}
                    disabled={isDeleting}
                    className="rounded-md bg-rose-500/20 px-2 py-1 text-xs font-medium text-rose-300 hover:bg-rose-500/30 disabled:opacity-50"
                  >
                    {isDeleting ? <Spinner size="sm" /> : 'Confirm'}
                  </button>
                </div>
              ) : (
                <button
                  type="button"
                  onClick={() => setConfirmId(doc.id)}
                  disabled={isDeleting}
                  aria-label={`Delete ${doc.filename}`}
                  className="rounded-md p-1.5 text-slate-400 transition hover:bg-rose-500/10 hover:text-rose-300 disabled:opacity-50"
                >
                  {isDeleting ? <Spinner size="sm" /> : <TrashIcon />}
                </button>
              )}
            </div>
          </li>
        )
      })}
    </ul>
  )
}

function TrashIcon() {
  return (
    <svg viewBox="0 0 20 20" fill="currentColor" className="h-4 w-4">
      <path
        fillRule="evenodd"
        d="M8.75 1A2.75 2.75 0 0 0 6 3.75v.443c-.795.077-1.584.176-2.365.298a.75.75 0 1 0 .23 1.482l.149-.022.841 10.518A2.75 2.75 0 0 0 7.596 19h4.807a2.75 2.75 0 0 0 2.742-2.53l.841-10.52.149.023a.75.75 0 0 0 .23-1.482 41.03 41.03 0 0 0-2.365-.298V3.75A2.75 2.75 0 0 0 11.25 1h-2.5ZM10 4c.84 0 1.673.025 2.5.075V3.75c0-.69-.56-1.25-1.25-1.25h-2.5c-.69 0-1.25.56-1.25 1.25v.325C8.327 4.025 9.16 4 10 4ZM8.58 7.72a.75.75 0 0 0-1.5.06l.3 7.5a.75.75 0 1 0 1.5-.06l-.3-7.5Zm4.34.06a.75.75 0 1 0-1.5-.06l-.3 7.5a.75.75 0 1 0 1.5.06l.3-7.5Z"
        clipRule="evenodd"
      />
    </svg>
  )
}
