import { describe, expect, test } from 'vitest'
import { FAVORITE_LABELS, favoriteKind } from '@/lib/favorite-content'
import { favoriteIcon, favoriteLabel } from '@/components/notebook/favorite-kinds'
import { getTypeColor } from '@/lib/notebook-selection-types'

/**
 * Imported material is stored under its own record type, so every surface that
 * turns a type into a label or an icon has to know about it. A missing entry
 * does not crash: it silently falls back to the generic "Favorite content"
 * badge, which is exactly the stale-enumeration bug worth pinning.
 */
describe('an imported collection record', () => {
  test('keeps its own type instead of being folded into a chat kind', () => {
    expect(favoriteKind('import')).toBe('import')
  })

  test('has a label of its own, not the generic fallback', () => {
    expect(FAVORITE_LABELS.import).toBe('Imported')
    expect(favoriteLabel('import')).toBe('Imported')
  })

  test('has an icon and a tint of its own', () => {
    const unknown = favoriteIcon('__no_such_kind__')
    expect(favoriteIcon('import')).not.toBe(unknown)
    expect(getTypeColor('import')).not.toBe(getTypeColor('__no_such_kind__'))
  })
})
