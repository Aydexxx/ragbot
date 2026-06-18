import { useCallback, useEffect, useState } from 'react'
import {
  ApiError,
  deleteDocument,
  listDocuments,
  uploadDocument,
} from '../api/client'
import type { DocumentRecord, UploadLimits } from '../api/types'

export interface UploadTask {
  id: string
  fileName: string
  phase: 'uploading' | 'indexing' | 'done' | 'error'
  percent: number
  error?: string
}

interface DocumentsState {
  documents: DocumentRecord[]
  loading: boolean
  error: string | null
}

/** Drops a finished/errored upload card from the queue after this long. */
const TASK_DISMISS_DELAY_MS = 3000

/** Used until `/health` has reported the server's real configured limits —
 * mirrors the backend defaults so behavior is sane even before that load. */
const FALLBACK_LIMITS: UploadLimits = {
  max_upload_size_bytes: 20 * 1024 * 1024,
  max_files_per_request: 20,
  allowed_extensions: ['.markdown', '.md', '.pdf', '.txt'],
}

function formatMB(bytes: number): string {
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function hasAllowedExtension(fileName: string, allowedExtensions: string[]): boolean {
  const lower = fileName.toLowerCase()
  return allowedExtensions.some((ext) => lower.endsWith(ext))
}

export function useDocuments(limits: UploadLimits = FALLBACK_LIMITS) {
  const [state, setState] = useState<DocumentsState>({
    documents: [],
    loading: true,
    error: null,
  })
  const [tasks, setTasks] = useState<UploadTask[]>([])
  const [deletingIds, setDeletingIds] = useState<ReadonlySet<string>>(new Set())
  const [batchError, setBatchError] = useState<string | null>(null)

  // No synchronous setState before this — only inside the .then()/.catch()
  // callbacks — so this is safe to call directly from the mount effect below.
  const load = useCallback(() => {
    return listDocuments()
      .then((documents) => setState({ documents, loading: false, error: null }))
      .catch((err: unknown) => {
        const message =
          err instanceof ApiError ? err.message : 'Could not load documents.'
        setState((prev) => ({ ...prev, loading: false, error: message }))
      })
  }, [])

  const refresh = useCallback(() => {
    setState((prev) => ({ ...prev, loading: true, error: null }))
    return load()
  }, [load])

  useEffect(() => {
    void load()
  }, [load])

  const uploadFiles = useCallback((files: File[]) => {
    let toUpload = files
    if (files.length > limits.max_files_per_request) {
      const skipped = files.length - limits.max_files_per_request
      toUpload = files.slice(0, limits.max_files_per_request)
      setBatchError(
        `Only ${limits.max_files_per_request} files can be uploaded at once. ` +
          `${skipped} file${skipped === 1 ? '' : 's'} ${skipped === 1 ? 'was' : 'were'} skipped — select ${skipped === 1 ? 'it' : 'them'} separately to upload.`,
      )
    } else {
      setBatchError(null)
    }

    for (const file of toUpload) {
      const taskId = crypto.randomUUID()
      setTasks((prev) => [
        ...prev,
        { id: taskId, fileName: file.name, phase: 'uploading', percent: 0 },
      ])

      if (!hasAllowedExtension(file.name, limits.allowed_extensions)) {
        setTasks((prev) =>
          prev.map((task) =>
            task.id === taskId
              ? {
                  ...task,
                  phase: 'error',
                  error: `Unsupported file type. Allowed: ${limits.allowed_extensions.join(', ')}.`,
                }
              : task,
          ),
        )
        continue
      }

      if (file.size > limits.max_upload_size_bytes) {
        setTasks((prev) =>
          prev.map((task) =>
            task.id === taskId
              ? {
                  ...task,
                  phase: 'error',
                  error: `File exceeds the ${formatMB(limits.max_upload_size_bytes)} upload limit.`,
                }
              : task,
          ),
        )
        continue
      }

      uploadDocument(file, (progress) => {
        setTasks((prev) =>
          prev.map((task) =>
            task.id === taskId
              ? { ...task, phase: progress.phase, percent: progress.percent }
              : task,
          ),
        )
      })
        .then((document) => {
          setTasks((prev) =>
            prev.map((task) =>
              task.id === taskId ? { ...task, phase: 'done', percent: 100 } : task,
            ),
          )
          setState((prev) => ({
            ...prev,
            documents: [document, ...prev.documents],
          }))
          setTimeout(() => {
            setTasks((prev) => prev.filter((task) => task.id !== taskId))
          }, TASK_DISMISS_DELAY_MS)
        })
        .catch((err: unknown) => {
          const message = err instanceof ApiError ? err.message : 'Upload failed.'
          setTasks((prev) =>
            prev.map((task) =>
              task.id === taskId ? { ...task, phase: 'error', error: message } : task,
            ),
          )
        })
    }
  }, [limits])

  const dismissTask = useCallback((taskId: string) => {
    setTasks((prev) => prev.filter((task) => task.id !== taskId))
  }, [])

  const dismissBatchError = useCallback(() => {
    setBatchError(null)
  }, [])

  const removeDocument = useCallback(async (id: string) => {
    setDeletingIds((prev) => new Set(prev).add(id))
    setState((prev) => ({ ...prev, error: null }))
    try {
      await deleteDocument(id)
      setState((prev) => ({
        ...prev,
        documents: prev.documents.filter((doc) => doc.id !== id),
      }))
    } catch (err) {
      const message =
        err instanceof ApiError ? err.message : 'Could not delete document.'
      setState((prev) => ({ ...prev, error: message }))
    } finally {
      setDeletingIds((prev) => {
        const next = new Set(prev)
        next.delete(id)
        return next
      })
    }
  }, [])

  return {
    documents: state.documents,
    loading: state.loading,
    error: state.error,
    refresh,
    tasks,
    uploadFiles,
    dismissTask,
    removeDocument,
    deletingIds,
    batchError,
    dismissBatchError,
  }
}
