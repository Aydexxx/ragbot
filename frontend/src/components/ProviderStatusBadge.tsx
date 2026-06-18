import type { HealthResponse } from '../api/types'
import { Spinner } from './Spinner'

interface ProviderStatusBadgeProps {
  health: HealthResponse | null
  loading: boolean
  error: string | null
  onRetry: () => void
}

export function ProviderStatusBadge({
  health,
  loading,
  error,
  onRetry,
}: ProviderStatusBadgeProps) {
  if (loading) {
    return (
      <div className="flex items-center gap-2 rounded-full border border-slate-700 bg-slate-800/60 px-3 py-1.5 text-sm text-slate-300">
        <Spinner size="sm" />
        <span>Checking provider…</span>
      </div>
    )
  }

  if (error || !health) {
    return (
      <button
        type="button"
        onClick={onRetry}
        title={error ?? 'Provider status unavailable'}
        className="flex items-center gap-2 rounded-full border border-rose-500/40 bg-rose-500/10 px-3 py-1.5 text-sm text-rose-200 transition hover:bg-rose-500/20"
      >
        <span className="h-2 w-2 rounded-full bg-rose-400" />
        Status unknown · retry
      </button>
    )
  }

  const { providers } = health
  const dotColor = providers.reachable ? 'bg-emerald-400' : 'bg-amber-400'
  const label = !providers.embeddings_enabled
    ? 'embeddings disabled'
    : providers.generation_enabled
      ? `${providers.provider} · answers on`
      : `${providers.provider} · retrieval only`

  const detail = [
    `Provider: ${providers.provider}`,
    `Embeddings: ${providers.embeddings_enabled ? providers.embed_model ?? 'enabled' : 'disabled'}`,
    `Generation: ${providers.generation_enabled ? providers.chat_model ?? 'enabled' : 'disabled'}`,
    `Backend reachable: ${providers.reachable ? 'yes' : 'no'}`,
  ].join('\n')

  return (
    <div
      title={detail}
      className="flex items-center gap-2 rounded-full border border-slate-700 bg-slate-800/60 px-3 py-1.5 text-sm text-slate-200"
    >
      <span className={`h-2 w-2 rounded-full ${dotColor}`} />
      <span>{label}</span>
    </div>
  )
}
