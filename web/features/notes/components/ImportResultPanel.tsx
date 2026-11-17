'use client'

import { useTranslation } from 'react-i18next'
import type { NoteImportResult } from '@/lib/notes-api'
import { groupImportResult, ungroupedFiles } from '../model/import-destinations'

/**
 * The result of a drop, grouped by the store each file reached.
 *
 * Every section is its own destination with its own totals, because "four of
 * five imported" says nothing about which store took what. Files that reached no
 * store have no section, so they are listed under their own heading — they are
 * exactly the ones whose reasons the user needs.
 */
export default function ImportResultPanel({
  result,
  onDismiss,
}: {
  result: NoteImportResult
  onDismiss: () => void
}) {
  const { t } = useTranslation()
  const groups = groupImportResult(result)
  const orphans = ungroupedFiles(result)
  return (
    <div className="dt-note-banner" role="status">
      <div className="flex-1 space-y-1">
        {groups.map(group => (
          <div key={group.destination} className="dt-note-import-group">
            <span className="font-semibold">
              {group.labelKey ? t(group.labelKey) : group.destination}
            </span>
            <span className="opacity-70">
              {' '}
              {group.imported}/{group.files.length}
            </span>
            {group.files
              // An imported file can still carry a reason ("Not added to the
              // collection"), and dropping it would report a partial landing as
              // a clean success.
              .filter(file => file.outcome !== 'imported' || Boolean(file.reason))
              .map(file => (
                <p key={`${group.destination}:${file.filename}`} className="text-xs opacity-70">
                  {file.filename}
                  {' — '}
                  {file.reason ?? file.outcome}
                </p>
              ))}
          </div>
        ))}
        {orphans.map(file => (
          <p key={`none:${file.filename}`} className="text-xs opacity-70">
            {file.filename}
            {' — '}
            {file.reason ?? file.outcome}
          </p>
        ))}
      </div>
      <button className="dt-note-button" onClick={onDismiss}>
        {t('Dismiss')}
      </button>
    </div>
  )
}
