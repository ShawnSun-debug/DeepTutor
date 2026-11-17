import type { ImportDestination, NoteImportFileResult, NoteImportResult } from '@/lib/notes-api'

/** The destinations the drop surface offers, in the order it shows them. */
export const IMPORT_DESTINATIONS: readonly ImportDestination[] = [
  'auto',
  'authored_page',
  'collection',
  'knowledge_base',
] as const

/**
 * Locale keys for the destination labels.
 *
 * These are words the app already ships, reused on purpose: a destination is not
 * a new concept to translate, it is the store's own name.
 */
export const IMPORT_DESTINATION_LABEL: Record<ImportDestination, string> = {
  auto: 'Auto',
  authored_page: 'Notes',
  collection: 'Collections',
  knowledge_base: 'Knowledge base',
}

/** One destination's share of a drop, ready to render as a section. */
export interface ImportGroup {
  /** The store the files landed in, as the server reported it. */
  destination: string
  /** The locale key for that store's name, or null for a destination this build does not know. */
  labelKey: string | null
  files: NoteImportFileResult[]
  imported: number
  skipped: number
  failed: number
  /** The background job this destination queued, when it queued one. */
  taskId: string | null
}

function labelFor(destination: string): string | null {
  return (IMPORT_DESTINATION_LABEL as Record<string, string>)[destination] ?? null
}

function emptyGroup(destination: string, taskId: string | null): ImportGroup {
  return {
    destination,
    labelKey: labelFor(destination),
    files: [],
    imported: 0,
    skipped: 0,
    failed: 0,
    taskId,
  }
}

/**
 * Group a response per destination.
 *
 * The counts are derived from the per-file outcomes instead of copied from the
 * server's roll-up, so a section can never claim a total its own list
 * contradicts; the roll-up supplies the section order and the queued job. A file
 * with no destination reached no store at all, so it belongs to no section.
 */
export function groupImportResult(result: NoteImportResult): ImportGroup[] {
  const groups = new Map<string, ImportGroup>()
  for (const [destination, summary] of Object.entries(result.groups ?? {}))
    groups.set(destination, emptyGroup(destination, summary.task_id ?? null))
  for (const file of result.files) {
    if (!file.destination) continue
    const group = groups.get(file.destination) ?? emptyGroup(file.destination, null)
    groups.set(file.destination, group)
    group.files.push(file)
    if (file.outcome === 'imported') group.imported += 1
    else if (file.outcome === 'skipped') group.skipped += 1
    else group.failed += 1
  }
  return [...groups.values()]
}

/** Files no store accepted: they have no destination, so no section lists them. */
export function ungroupedFiles(result: NoteImportResult): NoteImportFileResult[] {
  return result.files.filter(file => !file.destination)
}
