import { render, screen } from '@testing-library/react'
import { expect, it, vi } from 'vitest'
import ImportResultPanel from '@/features/notes/components/ImportResultPanel'
import type { NoteImportResult } from '@/lib/notes-api'

vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }))

function result(overrides: Partial<NoteImportResult> = {}): NoteImportResult {
  return {
    destination: 'collection',
    imported: 0,
    skipped: 0,
    failed: 0,
    files: [],
    groups: {},
    ...overrides,
  } as NoteImportResult
}

it('keeps the reason a file was imported from but not added to the collection', () => {
  render(
    <ImportResultPanel
      result={result({
        imported: 1,
        files: [
          {
            filename: 'notes.md',
            destination: 'collection',
            outcome: 'imported',
            reason: 'Not added to: ab12cd34. The record was not added.',
          } as NoteImportResult['files'][number],
        ],
        groups: {
          collection: {
            destination: 'collection',
            files: 1,
            imported: 1,
            skipped: 0,
            failed: 0,
            task_id: null,
          },
        },
      })}
      onDismiss={vi.fn()}
    />
  )
  expect(screen.getByText(/Not added to: ab12cd34/)).toBeTruthy()
})

it('still reports the files that reached no store', () => {
  render(
    <ImportResultPanel
      result={result({
        failed: 1,
        files: [
          {
            filename: 'payload.exe',
            destination: '',
            outcome: 'failed',
            reason: 'unsupported extension (supported: .md)',
          } as NoteImportResult['files'][number],
        ],
      })}
      onDismiss={vi.fn()}
    />
  )
  expect(screen.getByText(/unsupported extension/)).toBeTruthy()
})
