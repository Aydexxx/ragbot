import { AskPanel } from './components/AskPanel'
import { DocumentList } from './components/DocumentList'
import { ProviderStatusBadge } from './components/ProviderStatusBadge'
import { UploadArea } from './components/UploadArea'
import { useDocuments } from './hooks/useDocuments'
import { useHealth } from './hooks/useHealth'

const EMBEDDINGS_DISABLED_REASON =
  'Embeddings are disabled on the server. Configure PROVIDER=ollama or PROVIDER=openai in the backend .env to enable uploads and questions.'

function App() {
  const health = useHealth()
  const documents = useDocuments(health.data?.limits)

  const embeddingsDisabled = health.data
    ? !health.data.providers.embeddings_enabled
    : false

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800 bg-slate-900/60">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-4">
          <div>
            <h1 className="text-lg font-semibold text-white">RAGBot</h1>
            <p className="text-xs text-slate-400">
              Document Q&amp;A powered by retrieval-augmented generation
            </p>
          </div>
          <ProviderStatusBadge
            health={health.data}
            loading={health.loading}
            error={health.error}
            onRetry={health.refresh}
          />
        </div>
      </header>

      <main className="mx-auto grid max-w-6xl gap-6 px-4 py-6 lg:grid-cols-[380px_1fr]">
        <section className="space-y-6">
          <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4">
            <h2 className="mb-3 text-sm font-semibold text-slate-200">
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
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="text-sm font-semibold text-slate-200">
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
          </div>
        </section>

        <section className="rounded-xl border border-slate-800 bg-slate-900/40 p-4">
          <h2 className="mb-3 text-sm font-semibold text-slate-200">
            Ask a question
          </h2>
          <AskPanel
            documents={documents.documents}
            documentsLoading={documents.loading}
            disabled={embeddingsDisabled}
            disabledReason={EMBEDDINGS_DISABLED_REASON}
          />
        </section>
      </main>
    </div>
  )
}

export default App
