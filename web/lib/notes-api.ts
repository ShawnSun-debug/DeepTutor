import { apiFetch, apiUrl } from '@/lib/api'
import type { components } from '@/contracts/generated/api'
import type { CoWriterDocument, CoWriterDocumentSummary } from '@/lib/co-writer-api'

export interface NoteBlock {
  id: string
  type: string
  props: Record<string, unknown>
  content?: unknown
  children?: NoteBlock[]
}
export interface NotePage extends CoWriterDocument {
  blocks: NoteBlock[] | null
  revision: number
  parent_id: string | null
  icon: string
  cover: string
  pinned: boolean
}
export interface NoteSummary extends CoWriterDocumentSummary {
  parent_id: string | null
  icon: string
  cover?: string
  pinned: boolean
  revision: number
}
export interface NoteSuggestion {
  id: string
  operation: 'insert' | 'replace' | 'delete' | 'comment'
  block_id: string
  markdown: string
  explanation: string
  original: NoteBlock | null
  anchors?: { id: string; side: 'before' | 'after' }[]
  preserve_children?: boolean
  blocks: NoteBlock[] | null
  status: 'pending' | 'accepted' | 'dismissed'
}
export async function noteRequest<T>(path: string, body?: unknown, method = 'POST'): Promise<T> {
  const response = await apiFetch(
    apiUrl(`/api/documents${path}`),
    body === undefined
      ? { cache: 'no-store' }
      : {
          method,
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        }
  )
  if (!response.ok) {
    const detail = await response.json().catch(() => null)
    throw new Error(detail?.detail || `Request failed (${response.status})`)
  }
  return response.json()
}

export async function exportNoteFlashcards(docId?: string): Promise<Blob> {
  const path = docId
    ? `/api/documents/${encodeURIComponent(docId)}/flashcards/export`
    : `/api/documents/flashcards/export`
  const response = await apiFetch(apiUrl(path), { cache: 'no-store' })
  if (!response.ok) {
    const detail = await response.json().catch(() => null)
    const message =
      typeof detail?.detail === 'string' ? detail.detail : `Request failed (${response.status})`
    throw new Error(message)
  }
  return response.blob()
}

export const getNote = (id: string) => noteRequest<NotePage>(`/${id}`)
export const listNotes = () => noteRequest<{ documents: NoteSummary[] }>('')
export const saveNote = (id: string, patch: Partial<NotePage> & { expected_revision: number }) =>
  noteRequest<NotePage>(`/${id}`, patch, 'PUT')
export const createNote = (title = '', parent_id: string | null = null, content = '') =>
  noteRequest<NotePage>('', { title, parent_id, content, ...(content ? {} : { blocks: [] }) })
export const getSuggestions = (id: string) =>
  noteRequest<{ suggestions: NoteSuggestion[] }>(`/${id}/suggestions`)
export const reviewSuggestion = (
  id: string,
  suggestionId: string,
  accept: boolean,
  blocks?: NoteBlock[]
) => noteRequest<NotePage>(`/${id}/suggestions/${suggestionId}`, { accept, blocks })

/** A conversation held at one line of the page. The session keeps the
 *  transcript; the page keeps where it belongs. */
export interface NoteThread {
  session_id: string
  block_id: string
  quote: string
  origin: 'ask' | 'generate'
  created_at: number
  title: string
  status: 'idle' | 'running' | 'completed' | 'failed' | 'cancelled' | 'rejected'
  updated_at: number
  message_count: number
  last_message: string
}
export const listThreads = (id: string) =>
  noteRequest<{ threads: NoteThread[] }>(`/${id}/threads`)
export const anchorThread = (
  id: string,
  thread: Pick<NoteThread, 'session_id' | 'block_id'> & Partial<Pick<NoteThread, 'quote' | 'origin'>>
) => noteRequest<NoteThread>(`/${id}/threads`, thread)
export const deleteThread = (id: string, sessionId: string) =>
  noteRequest<{ deleted: boolean }>(`/${id}/threads/${encodeURIComponent(sessionId)}`, {}, 'DELETE')


/** One file outcome from the shared import endpoint (generated contract). */
export type NoteImportFileResult = components['schemas']['ImportFileResult']
export type NoteImportResult = components['schemas']['DocumentImportResponse']
export type NoteImportGroup = components['schemas']['ImportDestinationGroup']

/**
 * Where a drop goes.
 *
 * `auto` lets the server pick a store per file: HTML, and Markdown or plain
 * text when no collection is named, become notes; Markdown and plain text with
 * named collections become records; and the formats only a knowledge base can
 * index go there — but only when `target` names one, because the server never
 * picks a knowledge base on the user's behalf.
 */
export type ImportDestination = 'auto' | 'authored_page' | 'collection' | 'knowledge_base'

export interface ImportFilesOptions {
  /** Defaults to `auto`: the server routes each file by type. */
  destination?: ImportDestination
  /** The page new notes hang under; omitted means the top level. */
  parentId?: string | null
  /** Collections to receive records. Naming one outranks the automatic choice. */
  collectionIds?: string[]
  /** The knowledge base's name, verbatim; required by the knowledge_base destination. */
  target?: string | null
}

/**
 * Send files to the shared import endpoint and get one outcome per file.
 *
 * One bad file never fails the batch: the response carries one outcome per file
 * plus a per-destination roll-up, so the caller can say where each file went,
 * which ones did not land, and why.
 */
export async function importFiles(
  files: File[],
  options: ImportFilesOptions = {}
): Promise<NoteImportResult> {
  const body = new FormData()
  body.append('destination', options.destination ?? 'auto')
  if (options.parentId) body.append('parent_id', options.parentId)
  for (const collectionId of options.collectionIds ?? [])
    body.append('collection_ids', collectionId)
  if (options.target) body.append('target', options.target)
  for (const file of files) body.append('files', file, file.name)
  const response = await apiFetch(apiUrl('/api/imports/documents'), {
    method: 'POST',
    body,
  })
  if (!response.ok) {
    const detail = await response.json().catch(() => null)
    throw new Error(detail?.detail || `Request failed (${response.status})`)
  }
  return response.json()
}

/** Import files as ordinary pages: the notes-only shortcut over `importFiles`. */
export async function importMarkdownNotes(
  files: File[],
  parentId: string | null = null
): Promise<NoteImportResult> {
  return importFiles(files, { destination: 'authored_page', parentId })
}

