import { useState } from 'react'
import { AskPanel } from './components/AskPanel'
import { DocumentList } from './components/DocumentList'
import { DocumentScopeSelector } from './components/DocumentScopeSelector'
import { ProviderStatusBadge } from './components/ProviderStatusBadge'
import { UploadArea } from './components/UploadArea'
import { useDocuments } from './hooks/useDocuments'
import { useHealth } from './hooks/useHealth'

const EMBEDDINGS_DISABLED_REASON =
  'Embeddings are disabled on the server. Configure PROVIDER=ollama or PROVIDER=openai in the backend .env to enable uploads and questions.'

/** RAGBot's teal signature mark. */
function BrandMark({ className = '' }: { className?: string }) {
  return (
    <span
      className={`flex items-center justify-center rounded-xl bg-gradient-to-br from-teal-400 to-emerald-500 font-bold text-slate-950 shadow-sm ${className}`}
    >
      R
    </span>
  )
}

function App() {
  const health = useHealth()
  const documents = useDocuments(health.data?.limits)
  const [selectedDocumentIds, setSelectedDocumentIds] = useState<string[]>([])
  const [sidebarOpen, setSidebarOpen] = useState(false)

  const embeddingsDisabled = health.data
    ? !health.data.providers.embeddings_enabled
    : false

  return (
    <div className="flex h-screen overflow-hidden bg-slate-950 text-slate-100">
      {/* Mobile backdrop */}
      {sidebarOpen && (
        <button
          type="button"
          aria-label="Close sidebar"
          onClick={() => setSidebarOpen(false)}
          className="fixed inset-0 z-30 bg-black/50 lg:hidden"
        />
      )}

      {/* Sidebar: document management + provider status */}
      <aside
        className={`fixed inset-y-0 left-0 z-40 flex w-80 max-w-[85vw] flex-col border-r border-slate-800 bg-slate-900/80 backdrop-blur transition-transform duration-200 lg:static lg:z-auto lg:w-96 lg:translate-x-0 ${
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <div className="flex items-center justify-between gap-2 border-b border-slate-800 px-4 py-4">
          <div className="flex items-center gap-3">
            <BrandMark className="h-9 w-9 text-lg" />
            <div>
              <h1 className="text-base font-semibold text-white">RAGBot</h1>
              <p className="text-xs text-slate-400">Grounded document Q&amp;A</p>
            </div>
          </div>
          <button
            type="button"
            aria-label="Close sidebar"
            onClick={() => setSidebarOpen(false)}
            className="rounded-md p-1.5 text-slate-400 transition hover:bg-slate-800 hover:text-slate-200 lg:hidden"
          >
            <CloseIcon />
          </button>
        </div>

        <div className="thin-scrollbar flex-1 space-y-6 overflow-y-auto px-4 py-5">
          <ProviderStatusBadge
            health={health.data}
            loading={health.loading}
            error={health.error}
            onRetry={health.refresh}
          />

          <section className="space-y-3">
            <h2 className="text-xs font-semibold uppercase tracking-wide text-slate-400">
              Upload a document
            </h2>
            <UploadArea
              tasks={documents.tasks}
              onFilesSelected={documents.uploadFiles}
              onDismissTask={documents.dismissTask}
              disabled={embeddingsDisabled}
              disabledReason={EMBEDDINGS_DISABLED_REASON}
              batchError={documents.batchError}
              onDismissBatchError={documents.dismissBatchError}
            />
          </section>

          <section className="space-y-3">
            <div className="flex items-center justify-between">
              <h2 className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                Indexed documents
              </h2>
              <button
                type="button"
                onClick={() => void documents.refresh()}
                className="text-xs text-slate-400 transition hover:text-slate-200"
              >
                Refresh
              </button>
            </div>
            <DocumentList
              documents={documents.documents}
              loading={documents.loading}
              error={documents.error}
              deletingIds={documents.deletingIds}
              onDelete={(id) => void documents.removeDocument(id)}
            />
          </section>

          <section>
            <DocumentScopeSelector
              documents={documents.documents}
              selectedIds={selectedDocumentIds}
              onChange={setSelectedDocumentIds}
              disabled={embeddingsDisabled}
            />
          </section>
        </div>
      </aside>

      {/* Main chat column */}
      <div className="flex min-w-0 flex-1 flex-col">
        {/* Mobile top bar */}
        <div className="flex items-center gap-3 border-b border-slate-800 bg-slate-900/60 px-4 py-3 lg:hidden">
          <button
            type="button"
            aria-label="Open sidebar"
            onClick={() => setSidebarOpen(true)}
            className="rounded-md p-1.5 text-slate-300 transition hover:bg-slate-800 hover:text-white"
          >
            <MenuIcon />
          </button>
          <BrandMark className="h-7 w-7 text-sm" />
          <span className="text-sm font-semibold text-white">RAGBot</span>
        </div>

        <AskPanel
          documents={documents.documents}
          documentsLoading={documents.loading}
          disabled={embeddingsDisabled}
          disabledReason={EMBEDDINGS_DISABLED_REASON}
          selectedDocumentIds={selectedDocumentIds}
        />
      </div>
    </div>
  )
}

function MenuIcon() {
  return (
    <svg viewBox="0 0 20 20" fill="currentColor" className="h-5 w-5">
      <path
        fillRule="evenodd"
        d="M3 5.75A.75.75 0 0 1 3.75 5h12.5a.75.75 0 0 1 0 1.5H3.75A.75.75 0 0 1 3 5.75Zm0 4.5A.75.75 0 0 1 3.75 9.5h12.5a.75.75 0 0 1 0 1.5H3.75A.75.75 0 0 1 3 10.25Zm0 4.5a.75.75 0 0 1 .75-.75h12.5a.75.75 0 0 1 0 1.5H3.75a.75.75 0 0 1-.75-.75Z"
        clipRule="evenodd"
      />
    </svg>
  )
}

function CloseIcon() {
  return (
    <svg viewBox="0 0 20 20" fill="currentColor" className="h-5 w-5">
      <path d="M6.28 5.22a.75.75 0 0 0-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 1 0 1.06 1.06L10 11.06l3.72 3.72a.75.75 0 1 0 1.06-1.06L11.06 10l3.72-3.72a.75.75 0 0 0-1.06-1.06L10 8.94 6.28 5.22Z" />
    </svg>
  )
}

export default App
