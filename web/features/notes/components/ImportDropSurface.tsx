'use client'

import { useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { FileUp, X } from 'lucide-react'
import type { ImportDestination } from '@/lib/notes-api'
import { IMPORT_DESTINATIONS, IMPORT_DESTINATION_LABEL } from '../model/import-destinations'

/**
 * The extensions the note importer can turn into a page or a collection entry.
 *
 * A knowledge-base import deliberately has no counterpart here: that engine
 * decides for itself what it can read, and a second copy of its list in a file
 * picker would drift away from it.
 */
export const NOTE_IMPORT_ACCEPT = '.md,.markdown,.mdown,.txt,.html,.htm'

/** One collection (a notebook behind the endpoint) the panel can send records to. */
export interface ImportCollectionChoice {
  id: string
  name: string
}

/**
 * The single place files enter the notes page.
 *
 * It stays closed until something opens it, so the notes page keeps the top edge
 * it has always had: the toolbar's Import action opens it, and a drag anywhere
 * over the page stages the files here and opens it too. A drop is not imported
 * straight away — the files wait while the user picks where they go, then one
 * button sends the batch. Automatic lets the server route each file by type,
 * while the three named destinations send the whole batch to one store. The two
 * stores that cannot be inferred need different things: a knowledge base is
 * named in a text field (automatic routing asks for the same name when it has to
 * place a non-note file), while a collection is chosen from the real ones,
 * because its id is internal and cannot be typed.
 */
export default function ImportDropSurface({
  open,
  onOpen,
  onClose,
  destination,
  onDestination,
  target,
  onTarget,
  collections,
  collectionId,
  onCollection,
  onFiles,
  busy,
}: {
  open: boolean
  /** Called when a drop stages files and the panel has to show them. */
  onOpen: () => void
  onClose: () => void
  destination: ImportDestination
  onDestination: (next: ImportDestination) => void
  target: string
  onTarget: (next: string) => void
  collections: ImportCollectionChoice[]
  collectionId: string
  onCollection: (next: string) => void
  onFiles: (files: File[]) => void
  busy: boolean
}) {
  const { t } = useTranslation()
  const [dragging, setDragging] = useState(false)
  // Files wait here after a pick or a drop: the destination is still being
  // chosen, so nothing has been handed to the importer yet.
  const [staged, setStaged] = useState<File[]>([])
  const picker = useRef<HTMLInputElement>(null)
  const depth = useRef(0)
  // The window listeners are bound once, so they read the callbacks through
  // refs instead of closing over the first render's props.
  const openPanel = useRef(onOpen)
  const closePanel = useRef(onClose)
  useEffect(() => {
    openPanel.current = onOpen
    closePanel.current = onClose
  })

  // Listening on the window is what makes "drop it anywhere" true, including
  // over the editor; the depth counter keeps child elements from flickering the
  // hint off while the pointer crosses them.
  useEffect(() => {
    const carriesFiles = (event: DragEvent) => Boolean(event.dataTransfer?.types?.includes('Files'))
    const enter = (event: DragEvent) => {
      if (!carriesFiles(event)) return
      depth.current += 1
      setDragging(true)
    }
    const over = (event: DragEvent) => {
      if (carriesFiles(event)) event.preventDefault()
    }
    const leave = (event: DragEvent) => {
      if (!carriesFiles(event)) return
      depth.current = Math.max(0, depth.current - 1)
      if (!depth.current) setDragging(false)
    }
    const dropped = (event: DragEvent) => {
      if (!carriesFiles(event)) return
      depth.current = 0
      setDragging(false)
      const files = [...(event.dataTransfer?.files ?? [])]
      if (!files.length) return
      // Taking the drop keeps the browser from navigating to the file, and the
      // files stage into the panel instead of importing with a guessed
      // destination.
      event.preventDefault()
      setStaged(files)
      openPanel.current()
    }
    window.addEventListener('dragenter', enter)
    window.addEventListener('dragover', over)
    window.addEventListener('dragleave', leave)
    window.addEventListener('drop', dropped)
    return () => {
      window.removeEventListener('dragenter', enter)
      window.removeEventListener('dragover', over)
      window.removeEventListener('dragleave', leave)
      window.removeEventListener('drop', dropped)
    }
  }, [])

  // Esc closes the panel, the way the app's other overlays do. Closing is a
  // cancel: the staged batch is dropped instead of coming back as a stale
  // pending import the next time the panel opens.
  useEffect(() => {
    if (!open) return
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return
      setStaged([])
      closePanel.current()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open])

  // Closed renders nothing resident; the hint only exists while a drag is in
  // flight, which no layout depends on.
  if (!open) {
    if (!dragging) return null
    return (
      <div className="dt-note-import-overlay" aria-hidden>
        <div className="dt-note-import-hint">
          <FileUp size={28} />
          <span>{t('Import')}</span>
        </div>
      </div>
    )
  }

  const accept = destination === 'knowledge_base' ? undefined : NOTE_IMPORT_ACCEPT
  const needsCollection = destination === 'collection' && !collectionId
  const start = () => {
    if (!staged.length || needsCollection) return
    onFiles(staged)
    onClose()
  }
  const dismiss = () => {
    setStaged([])
    onClose()
  }

  return (
    <div className="dt-note-import-scrim">
      <div
        className="dt-note-import-panel"
        role="dialog"
        aria-modal="true"
        aria-label={t('Import')}
      >
        <div className="dt-note-import-head">
          <FileUp size={15} />
          <span>{t('Import')}</span>
          <button
            type="button"
            className="dt-note-icon-button"
            aria-label={t('Cancel')}
            onClick={dismiss}
          >
            <X size={16} strokeWidth={1.7} />
          </button>
        </div>
        <div role="group" aria-label={t('Import')} className="dt-note-import-destinations">
          {IMPORT_DESTINATIONS.map(option => (
            <button
              key={option}
              type="button"
              className={`dt-note-button ${destination === option ? 'is-active' : ''}`}
              aria-pressed={destination === option}
              disabled={busy}
              onClick={() => onDestination(option)}
            >
              {t(IMPORT_DESTINATION_LABEL[option])}
            </button>
          ))}
        </div>
        {destination === 'collection' &&
          (collections.length ? (
            <label className="dt-note-import-field">
              <span>{t(IMPORT_DESTINATION_LABEL.collection)}</span>
              <select
                className="dt-note-import-select"
                aria-label={t(IMPORT_DESTINATION_LABEL.collection)}
                value={collectionId}
                disabled={busy}
                onChange={event => onCollection(event.target.value)}
              >
                <option value="">{t('Choose a collection')}</option>
                {collections.map(collection => (
                  <option key={collection.id} value={collection.id}>
                    {collection.name}
                  </option>
                ))}
              </select>
            </label>
          ) : (
            <p className="dt-note-import-empty">{t('You have no collections yet.')}</p>
          ))}
        {(destination === 'auto' || destination === 'knowledge_base') && (
          <label className="dt-note-import-field">
            <span>{t(IMPORT_DESTINATION_LABEL.knowledge_base)}</span>
            <input
              className="dt-note-import-target"
              aria-label={t(IMPORT_DESTINATION_LABEL.knowledge_base)}
              value={target}
              disabled={busy}
              onChange={event => onTarget(event.target.value)}
            />
          </label>
        )}
        {staged.length > 0 && (
          <p className="dt-note-import-staged" role="status">
            {staged.map(file => file.name).join(', ')}
          </p>
        )}
        <div className="dt-note-import-actions">
          <button
            type="button"
            className="dt-note-button"
            disabled={busy}
            onClick={() => picker.current?.click()}
          >
            <FileUp size={14} /> {t('Choose file')}
          </button>
          <button
            type="button"
            className="dt-note-button"
            disabled={busy || !staged.length || needsCollection}
            onClick={start}
          >
            {t('Import')}
          </button>
        </div>
        <input
          ref={picker}
          type="file"
          accept={accept}
          multiple
          className="hidden"
          aria-hidden
          tabIndex={-1}
          onChange={event => {
            const picked = [...(event.target.files ?? [])]
            event.target.value = ''
            if (picked.length) setStaged(picked)
          }}
        />
      </div>
    </div>
  )
}
