'use client'

import { useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { FileUp } from 'lucide-react'
import type { ImportDestination } from '@/lib/notes-api'
import { IMPORT_DESTINATIONS, IMPORT_DESTINATION_LABEL } from '../model/import-destinations'

/**
 * The single place files enter the notes page.
 *
 * A drag anywhere over the page opens one overlay, and a drop imports with the
 * destination picked here: automatic lets the server route each file by type,
 * while the three named destinations send the whole drop to one store. The two
 * stores that cannot infer their instance (a collection, a knowledge base) take
 * a name in the field beside the buttons; leaving it empty is not a client-side
 * error — the request goes out and the per-file reasons say what was missing.
 */
export default function ImportDropSurface({
  destination,
  onDestination,
  target,
  onTarget,
  onFiles,
  busy,
}: {
  destination: ImportDestination
  onDestination: (next: ImportDestination) => void
  target: string
  onTarget: (next: string) => void
  onFiles: (files: File[]) => void
  busy: boolean
}) {
  const { t } = useTranslation()
  const [dragging, setDragging] = useState(false)
  const picker = useRef<HTMLInputElement>(null)
  const depth = useRef(0)
  const drop = useRef(onFiles)
  useEffect(() => {
    drop.current = onFiles
  })

  // Listening on the window is what makes "drop it anywhere" true, including
  // over the editor; the depth counter keeps child elements from flickering the
  // overlay off while the pointer crosses them.
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
      event.preventDefault()
      drop.current(files)
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

  return (
    <>
      <div className="dt-note-import-bar flex flex-wrap items-center gap-2 px-3 py-2">
        <div
          role="group"
          aria-label={t('Import')}
          className="flex flex-wrap items-center gap-1"
        >
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
        {(destination === 'collection' || destination === 'knowledge_base') && (
          <label className="flex items-center gap-2 text-xs">
            <span className="opacity-60">{t(IMPORT_DESTINATION_LABEL[destination])}</span>
            <input
              className="dt-note-import-target"
              aria-label={t(IMPORT_DESTINATION_LABEL[destination])}
              value={target}
              disabled={busy}
              onChange={event => onTarget(event.target.value)}
            />
          </label>
        )}
        <button
          type="button"
          className="dt-note-button ml-auto"
          disabled={busy}
          onClick={() => picker.current?.click()}
        >
          <FileUp size={14} /> {t('Import')}
        </button>
        <input
          ref={picker}
          type="file"
          multiple
          className="hidden"
          aria-hidden
          tabIndex={-1}
          onChange={event => {
            const files = [...(event.target.files ?? [])]
            event.target.value = ''
            if (files.length) onFiles(files)
          }}
        />
      </div>
      {dragging && (
        <div
          className="dt-note-import-overlay fixed inset-0 z-50 flex items-center justify-center"
          aria-hidden
        >
          <div className="flex flex-col items-center gap-2">
            <FileUp size={28} />
            <span>{t('Import')}</span>
          </div>
        </div>
      )}
    </>
  )
}
