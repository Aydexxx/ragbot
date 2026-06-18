import type { DocumentRecord } from '../api/types'

interface DocumentScopeSelectorProps {
  documents: DocumentRecord[]
  selectedIds: string[]
  onChange: (ids: string[]) => void
  disabled?: boolean
}

const PILL_BASE =
  'max-w-[180px] truncate rounded-full px-2.5 py-1 text-xs font-medium transition disabled:cursor-not-allowed disabled:opacity-50'
const PILL_ACTIVE = 'bg-indigo-500/30 text-indigo-200'
const PILL_INACTIVE = 'bg-slate-800 text-slate-400 hover:bg-slate-700'

/** Scopes a question to all indexed documents (default, empty selection) or
 * a chosen subset. Only `ready` documents are offered — others have no
 * queryable chunks yet. */
export function DocumentScopeSelector({
  documents,
  selectedIds,
  onChange,
  disabled = false,
}: DocumentScopeSelectorProps) {
  const queryable = documents.filter((doc) => doc.status === 'ready')
  if (queryable.length === 0) return null

  const allSelected = selectedIds.length === 0

  function toggle(id: string) {
    onChange(
      selectedIds.includes(id)
        ? selectedIds.filter((existing) => existing !== id)
        : [...selectedIds, id],
    )
  }

  return (
    <div className="space-y-1.5">
      <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
        Scope
      </p>
      <div className="flex flex-wrap gap-1.5">
        <button
          type="button"
          disabled={disabled}
          onClick={() => onChange([])}
          className={`${PILL_BASE} ${allSelected ? PILL_ACTIVE : PILL_INACTIVE}`}
        >
          All documents ({queryable.length})
        </button>
        {queryable.map((doc) => {
          const isSelected = selectedIds.includes(doc.id)
          return (
            <button
              key={doc.id}
              type="button"
              disabled={disabled}
              onClick={() => toggle(doc.id)}
              title={doc.filename}
              className={`${PILL_BASE} ${isSelected ? PILL_ACTIVE : PILL_INACTIVE}`}
            >
              {doc.filename}
            </button>
          )
        })}
      </div>
    </div>
  )
}
