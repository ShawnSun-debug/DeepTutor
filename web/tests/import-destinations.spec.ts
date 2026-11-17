import { describe, expect, it } from 'vitest'
import en from '../locales/en/app.json'
import zh from '../locales/zh/app.json'
import {
  IMPORT_DESTINATIONS,
  IMPORT_DESTINATION_LABEL,
  groupImportResult,
  importRequestFor,
  ungroupedFiles,
} from '@/features/notes/model/import-destinations'
import type { NoteImportFileResult, NoteImportResult } from '@/lib/notes-api'

const EN = en as Record<string, string>
const ZH = zh as Record<string, string>

function file(
  overrides: Partial<NoteImportFileResult> & { filename: string }
): NoteImportFileResult {
  return {
    destination: 'authored_page',
    outcome: 'imported',
    reason: null,
    ...overrides,
  } as NoteImportFileResult
}

function response(overrides: Partial<NoteImportResult>): NoteImportResult {
  return {
    destination: 'auto',
    imported: 0,
    skipped: 0,
    failed: 0,
    files: [],
    groups: {},
    ...overrides,
  } as NoteImportResult
}

describe('groupImportResult', () => {
  it('keeps the server section order and counts every file once', () => {
    const result = response({
      imported: 3,
      skipped: 1,
      failed: 1,
      groups: {
        authored_page: {
          destination: 'authored_page',
          files: 2,
          imported: 1,
          skipped: 1,
          failed: 0,
          task_id: null,
        },
        collection: {
          destination: 'collection',
          files: 1,
          imported: 1,
          skipped: 0,
          failed: 0,
          task_id: null,
        },
        knowledge_base: {
          destination: 'knowledge_base',
          files: 1,
          imported: 1,
          skipped: 0,
          failed: 0,
          task_id: 'kb_task',
        },
      },
      files: [
        file({ filename: 'notes.md', destination: 'collection' }),
        file({ filename: 'page.html', destination: 'authored_page' }),
        file({ filename: 'paper.pdf', destination: 'knowledge_base' }),
        file({
          filename: 'page-copy.html',
          destination: 'authored_page',
          outcome: 'skipped',
          reason: 'duplicate of "page.html": only the first copy was kept.',
        }),
        file({
          filename: 'payload.exe',
          destination: '',
          outcome: 'failed',
          reason: 'unsupported extension (supported: .md)',
        }),
      ],
    })

    const groups = groupImportResult(result)
    expect(groups.map(group => group.destination)).toEqual([
      'authored_page',
      'collection',
      'knowledge_base',
    ])

    const byDestination = Object.fromEntries(groups.map(group => [group.destination, group]))
    expect(byDestination.authored_page.files.map(file => file.filename)).toEqual([
      'page.html',
      'page-copy.html',
    ])
    expect(byDestination.authored_page.imported).toBe(1)
    expect(byDestination.authored_page.skipped).toBe(1)
    expect(byDestination.authored_page.failed).toBe(0)
    expect(byDestination.collection.imported).toBe(1)
    expect(byDestination.knowledge_base.imported).toBe(1)
    // The destination that queued the indexing job carries it.
    expect(byDestination.knowledge_base.taskId).toBe('kb_task')

    // A file no store accepted is reported on its own, never inside a section.
    expect(groups.flatMap(group => group.files.map(file => file.filename))).not.toContain(
      'payload.exe'
    )
    expect(ungroupedFiles(result).map(file => file.filename)).toEqual(['payload.exe'])
  })

  it('derives the totals from the files so a section cannot contradict itself', () => {
    const result = response({
      files: [
        file({ filename: 'a.md' }),
        file({ filename: 'b.md', outcome: 'failed', reason: 'The page could not be created.' }),
      ],
    })

    const [group] = groupImportResult(result)
    expect(group.destination).toBe('authored_page')
    expect([group.imported, group.skipped, group.failed, group.files.length]).toEqual([1, 0, 1, 2])
  })

  it('opens a section for a destination the response did not roll up', () => {
    const result = response({
      groups: {
        authored_page: {
          destination: 'authored_page',
          files: 1,
          imported: 1,
          skipped: 0,
          failed: 0,
          task_id: null,
        },
      },
      files: [file({ filename: 'a.md' }), file({ filename: 'b.md', destination: 'collection' })],
    })

    expect(groupImportResult(result).map(group => group.destination)).toEqual([
      'authored_page',
      'collection',
    ])
  })

  it('renders an unknown destination by its raw name', () => {
    const result = response({ files: [file({ filename: 'a.md', destination: 'blog_post' })] })
    const [group] = groupImportResult(result)
    expect(group.destination).toBe('blog_post')
    expect(group.labelKey).toBeNull()
  })

  it('reports nothing for an empty response', () => {
    expect(groupImportResult(response({}))).toEqual([])
  })
})

describe('import destinations', () => {
  it('offers automatic first and names the stores with words the app already ships', () => {
    expect(IMPORT_DESTINATIONS).toEqual([
      'auto',
      'authored_page',
      'collection',
      'knowledge_base',
    ])
    expect(IMPORT_DESTINATION_LABEL).toEqual({
      auto: 'Auto',
      authored_page: 'Note',
      collection: 'Collections',
      knowledge_base: 'Knowledge base',
    })
  })
})

describe('importRequestFor', () => {
  it('names the knowledge base for automatic routing, which asks for one', () => {
    expect(importRequestFor('auto', { target: '  papers  ', collectionId: '' })).toEqual({
      destination: 'auto',
      collectionIds: [],
      target: 'papers',
    })
  })

  it('names the knowledge base for the knowledge base destination', () => {
    expect(importRequestFor('knowledge_base', { target: 'papers', collectionId: '' }).target).toBe(
      'papers'
    )
  })

  it('sends the picked collection id and never the text beside it', () => {
    expect(importRequestFor('collection', { target: 'typed noise', collectionId: 'ab12cd34' })).toEqual(
      {
        destination: 'collection',
        collectionIds: ['ab12cd34'],
        target: null,
      }
    )
  })

  it('says nothing it was not told', () => {
    expect(importRequestFor('collection', { target: '', collectionId: '' }).collectionIds).toEqual([])
    expect(importRequestFor('authored_page', { target: 'papers', collectionId: 'ab12cd34' })).toEqual({
      destination: 'authored_page',
      collectionIds: [],
      target: null,
    })
    expect(importRequestFor('auto', { target: '   ', collectionId: '' }).target).toBeNull()
  })
})

describe('destination labels in the catalogs', () => {
  it('names each store with the word the English catalog shows', () => {
    expect(EN[IMPORT_DESTINATION_LABEL.auto]).toMatch(/auto/i)
    expect(EN[IMPORT_DESTINATION_LABEL.authored_page]).toMatch(/note/i)
    expect(EN[IMPORT_DESTINATION_LABEL.collection]).toMatch(/collection/i)
    expect(EN[IMPORT_DESTINATION_LABEL.knowledge_base]).toMatch(/knowledge/i)
  })

  it('shows a Chinese reader the word for a notebook, not for an explanation', () => {
    expect(ZH[IMPORT_DESTINATION_LABEL.auto]).toBe('自动')
    expect(ZH[IMPORT_DESTINATION_LABEL.authored_page]).toContain('笔记')
    expect(ZH[IMPORT_DESTINATION_LABEL.authored_page]).not.toBe('说明')
    expect(ZH[IMPORT_DESTINATION_LABEL.collection]).toContain('资料')
    expect(ZH[IMPORT_DESTINATION_LABEL.knowledge_base]).toContain('知识库')
  })
})
