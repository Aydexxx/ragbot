import type {
  AnswerResult,
  AskRequest,
  DocumentRecord,
  HealthResponse,
} from './types'

const API_BASE_URL: string =
  import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

export class ApiError extends Error {
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

/** FastAPI error bodies are either `{detail: string}` or, for validation
 * errors, `{detail: [{msg: string, ...}, ...]}`. Normalize both to a string. */
function formatDetail(detail: unknown): string {
  if (typeof detail === 'string') {
    return detail
  }
  if (Array.isArray(detail)) {
    return detail
      .map((entry) => {
        if (entry && typeof entry === 'object' && 'msg' in entry) {
          return String((entry as { msg: unknown }).msg)
        }
        return JSON.stringify(entry)
      })
      .join('; ')
  }
  return 'An unexpected error occurred.'
}

async function errorFromResponse(response: Response): Promise<ApiError> {
  try {
    const body: unknown = await response.json()
    const detail =
      body && typeof body === 'object' && 'detail' in body
        ? (body as { detail: unknown }).detail
        : undefined
    return new ApiError(
      response.status,
      detail === undefined
        ? `Request failed with status ${response.status}.`
        : formatDetail(detail),
    )
  } catch {
    return new ApiError(
      response.status,
      `Request failed with status ${response.status}.`,
    )
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, init)
  } catch {
    throw new ApiError(0, 'Could not reach the RAGBot backend.')
  }
  if (!response.ok) {
    throw await errorFromResponse(response)
  }
  return (await response.json()) as T
}

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>('/health')
}

export function listDocuments(): Promise<DocumentRecord[]> {
  return request<DocumentRecord[]>('/documents')
}

export async function deleteDocument(id: string): Promise<void> {
  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}/documents/${id}`, {
      method: 'DELETE',
    })
  } catch {
    throw new ApiError(0, 'Could not reach the RAGBot backend.')
  }
  if (!response.ok) {
    throw await errorFromResponse(response)
  }
}

export function askQuestion(payload: AskRequest): Promise<AnswerResult> {
  return request<AnswerResult>('/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}

export interface UploadProgress {
  phase: 'uploading' | 'indexing'
  percent: number
}

/** Uses XMLHttpRequest (not fetch) because only it exposes upload progress
 * events, needed to distinguish "sending the file" from "server is indexing
 * it" in the UI. */
export function uploadDocument(
  file: File,
  onProgress?: (progress: UploadProgress) => void,
): Promise<DocumentRecord> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${API_BASE_URL}/documents`)

    xhr.upload.onprogress = (event) => {
      if (!event.lengthComputable) return
      const percent = Math.round((event.loaded / event.total) * 100)
      onProgress?.({ phase: 'uploading', percent })
      if (percent >= 100) {
        onProgress?.({ phase: 'indexing', percent: 100 })
      }
    }

    xhr.onload = () => {
      let body: unknown
      try {
        body = xhr.responseText ? JSON.parse(xhr.responseText) : null
      } catch {
        body = null
      }

      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(body as DocumentRecord)
        return
      }

      const detail =
        body && typeof body === 'object' && 'detail' in body
          ? (body as { detail: unknown }).detail
          : undefined
      reject(
        new ApiError(
          xhr.status,
          detail === undefined
            ? `Request failed with status ${xhr.status}.`
            : formatDetail(detail),
        ),
      )
    }

    xhr.onerror = () => {
      reject(new ApiError(0, 'Could not reach the RAGBot backend.'))
    }

    const formData = new FormData()
    formData.append('file', file)
    xhr.send(formData)
  })
}
