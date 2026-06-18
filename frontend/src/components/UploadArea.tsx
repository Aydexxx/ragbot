import { useId, useRef, useState } from 'react'
import type { ChangeEvent, DragEvent } from 'react'
import type { UploadTask } from '../hooks/useDocuments'
import { Banner } from './Banner'

interface UploadAreaProps {
  tasks: UploadTask[]
  onFilesSelected: (files: File[]) => void
  onDismissTask: (taskId: string) => void
  disabled?: boolean
  disabledReason?: string
  batchError?: string | null
  onDismissBatchError?: () => void
}

const ACCEPTED_EXTENSIONS = '.pdf,.txt,.md,.markdown'

const PHASE_LABEL: Record<UploadTask['phase'], string> = {
  uploading: 'Uploading…',
  indexing: 'Indexing…',
  done: 'Indexed',
  error: 'Failed',
}

export function UploadArea({
  tasks,
  onFilesSelected,
  onDismissTask,
  disabled = false,
  disabledReason,
  batchError,
  onDismissBatchError,
}: UploadAreaProps) {
  const [isDragging, setIsDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const inputId = useId()

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault()
    setIsDragging(false)
    if (disabled) return
    const files = Array.from(event.dataTransfer.files)
    if (files.length > 0) onFilesSelected(files)
  }

  function handleInputChange(event: ChangeEvent<HTMLInputElement>) {
    const files = event.target.files ? Array.from(event.target.files) : []
    if (files.length > 0) onFilesSelected(files)
    event.target.value = ''
  }

  return (
    <div className="space-y-3">
      <div
        onDragOver={(event) => {
          event.preventDefault()
          if (!disabled) setIsDragging(true)
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        className={`rounded-xl border-2 border-dashed px-6 py-8 text-center transition ${
          disabled
            ? 'border-slate-700 bg-slate-800/30 opacity-60'
            : isDragging
              ? 'border-indigo-400 bg-indigo-500/10'
              : 'border-slate-700 bg-slate-800/40 hover:border-slate-600'
        }`}
      >
        <p className="text-sm text-slate-300">
          Drag and drop a PDF, TXT, or Markdown file here
        </p>
        <p className="mt-1 text-xs text-slate-500">or</p>
        <label htmlFor={inputId}>
          <span
            className={`mt-2 inline-flex cursor-pointer items-center rounded-md px-3 py-1.5 text-sm font-medium transition ${
              disabled
                ? 'cursor-not-allowed bg-slate-700 text-slate-400'
                : 'bg-indigo-500 text-white hover:bg-indigo-400'
            }`}
          >
            Browse files
          </span>
        </label>
        <input
          ref={inputRef}
          id={inputId}
          type="file"
          multiple
          accept={ACCEPTED_EXTENSIONS}
          disabled={disabled}
          onChange={handleInputChange}
          className="sr-only"
        />
        {disabled && disabledReason ? (
          <p className="mt-3 text-xs text-amber-300">{disabledReason}</p>
        ) : (
          <p className="mt-3 text-xs text-slate-500">PDF, TXT, or MD — any size up to the server limit</p>
        )}
      </div>

      {batchError && (
        <Banner
          variant="warning"
          action={
            onDismissBatchError && (
              <button
                type="button"
                onClick={onDismissBatchError}
                className="text-xs underline hover:text-slate-100"
              >
                Dismiss
              </button>
            )
          }
        >
          {batchError}
        </Banner>
      )}

      {tasks.length > 0 && (
        <ul className="space-y-2">
          {tasks.map((task) => (
            <li
              key={task.id}
              className="rounded-lg border border-slate-700 bg-slate-800/50 px-3 py-2 text-sm"
            >
              <div className="flex items-center justify-between gap-2">
                <span className="truncate text-slate-200">{task.fileName}</span>
                <span
                  className={`shrink-0 text-xs ${
                    task.phase === 'error'
                      ? 'text-rose-300'
                      : task.phase === 'done'
                        ? 'text-emerald-300'
                        : 'text-slate-400'
                  }`}
                >
                  {PHASE_LABEL[task.phase]}
                </span>
              </div>

              {task.phase === 'uploading' && (
                <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-slate-700">
                  <div
                    className="h-full rounded-full bg-indigo-400 transition-[width]"
                    style={{ width: `${task.percent}%` }}
                  />
                </div>
              )}

              {task.phase === 'indexing' && (
                <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-slate-700">
                  <div className="h-full w-1/3 animate-pulse rounded-full bg-indigo-400" />
                </div>
              )}

              {task.phase === 'error' && (
                <div className="mt-1 flex items-center justify-between gap-2">
                  <span className="text-xs text-rose-300">{task.error}</span>
                  <button
                    type="button"
                    onClick={() => onDismissTask(task.id)}
                    className="text-xs text-slate-400 underline hover:text-slate-200"
                  >
                    Dismiss
                  </button>
                </div>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
