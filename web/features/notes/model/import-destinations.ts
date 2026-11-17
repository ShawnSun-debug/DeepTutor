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
 * Each key is an existing word the app already ships, picked by reading both
 * catalogs instead of by guessing from the English key. `Notes` translates to
 * "说明" in Chinese, so it names an explanation rather than the notebook;
 * `Notebook` is already taken for the favorites store ("收藏夹"); and
 * `Notebook pages` is "Pages" in English and "笔记页面" in Chinese, which do not
 * agree. `Note` is "Note" / "笔记" everywhere, so both catalogs name where the
 * files actually land.
 */
export const IMPORT_DESTINATION_LABEL: Record<ImportDestination, string> = {
  auto: 'Auto',
  authored_page: 'Note',
  collection: 'Collections',
  knowledge_base: 'Knowledge base',
}

/** What the import endpoint needs for one batch, given the picked destination. */
export interface ImportRequest {
  destination: ImportDestination
  collectionIds: string[]
  target: string | null
}

/**
 * Build the request for one import batch.
 *
 * The two stores that cannot be inferred need different data: a knowledge base
 * is addressed by its name, a collection by its id. Automatic routing is in the
 * same position as the knowledge-base destination — it only sends a file to a
 * knowledge base when `target` names one, and its own per-file reason asks for
 * that field — so it carries the name too. A collection id is never guessed from
 * free text: unselected sends no ids, and the server reports each file as failed
 * instead of the page inventing an id.
 */
export function importRequestFor(
  destination: ImportDestination,
  { target, collectionId }: { target: string; collectionId: string }
): ImportRequest {
  const name = target.trim()
  return {
    destination,
    collectionIds: destination === 'collection' && collectionId ? [collectionId] : [],
    target: (destination === 'auto' || destination === 'knowledge_base') && name ? name : null,
  }
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
