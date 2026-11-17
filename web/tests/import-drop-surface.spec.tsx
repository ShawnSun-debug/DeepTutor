import fs from 'node:fs'
import path from 'node:path'
import { useState } from 'react'
import { act, fireEvent, render, screen } from '@testing-library/react'
import { expect, it, vi } from 'vitest'
import ImportDropSurface, {
  NOTE_IMPORT_ACCEPT,
} from '@/features/notes/components/ImportDropSurface'
import type { ImportDestination } from '@/lib/notes-api'

vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }))

const NOTE_FORMATS = ['.md', '.markdown', '.mdown', '.txt', '.html', '.htm']

function props(overrides: Partial<React.ComponentProps<typeof ImportDropSurface>> = {}) {
  return {
    open: true,
    onOpen: vi.fn(),
    onClose: vi.fn(),
    destination: 'auto' as ImportDestination,
    onDestination: vi.fn(),
    target: '',
    onTarget: vi.fn(),
    collections: [] as { id: string; name: string }[],
    collectionId: '',
    onCollection: vi.fn(),
    onFiles: vi.fn(),
    busy: false,
    ...overrides,
  }
}

function fileInput(): HTMLInputElement {
  const input = document.querySelector('input[type="file"]')
  expect(input).not.toBeNull()
  return input as HTMLInputElement
}

function drop(files: File[]) {
  const event = new Event('drop', { bubbles: true, cancelable: true })
  Object.defineProperty(event, 'dataTransfer', { value: { types: ['Files'], files } })
  act(() => {
    window.dispatchEvent(event)
  })
}

function source(rel: string): string {
  return fs.readFileSync(path.join(process.cwd(), rel), 'utf8')
}

it('renders nothing resident while the panel is closed', () => {
  render(<ImportDropSurface {...props({ open: false })} />)
  expect(screen.queryByRole('dialog')).toBeNull()
  expect(document.querySelector('input[type="file"]')).toBeNull()
  expect(screen.queryAllByRole('button')).toHaveLength(0)
})

it('offers the four destinations once it is open', () => {
  render(<ImportDropSurface {...props()} />)
  expect(screen.getByRole('dialog', { name: 'Import' })).toBeTruthy()
  for (const label of ['Auto', 'Note', 'Collections', 'Knowledge base'])
    expect(screen.getByRole('button', { name: label })).toBeTruthy()
})

it('asks for a knowledge base name on automatic too, because auto routing wants one', () => {
  const surface = props({ destination: 'auto' })
  render(<ImportDropSurface {...surface} />)
  fireEvent.change(screen.getByRole('textbox', { name: 'Knowledge base' }), {
    target: { value: 'papers' },
  })
  expect(surface.onTarget).toHaveBeenCalledWith('papers')
})

it('asks for the name on the knowledge base destination as well', () => {
  render(<ImportDropSurface {...props({ destination: 'knowledge_base' })} />)
  expect(screen.getByRole('textbox', { name: 'Knowledge base' })).toBeTruthy()
})

it('picks a collection by id instead of asking for one to be typed', () => {
  const surface = props({
    destination: 'collection',
    collections: [{ id: 'ab12cd34', name: 'Fourier readings' }],
  })
  render(<ImportDropSurface {...surface} />)
  fireEvent.change(screen.getByRole('combobox', { name: 'Collections' }), {
    target: { value: 'ab12cd34' },
  })
  expect(surface.onCollection).toHaveBeenCalledWith('ab12cd34')
  expect(screen.getByRole('option', { name: 'Fourier readings' })).toBeTruthy()
  // An eight-character internal id is not something a user can type.
  expect(screen.queryByRole('textbox')).toBeNull()
})

it('says so instead of showing an empty collection picker', () => {
  render(<ImportDropSurface {...props({ destination: 'collection', collections: [] })} />)
  expect(screen.getByText('You have no collections yet.')).toBeTruthy()
  expect(screen.queryByRole('combobox')).toBeNull()
})

it('will not send a collection import until a collection is picked', () => {
  // Controlled, so picking a collection really changes what the panel sees.
  function Surface() {
    const [collectionId, setCollectionId] = useState('')
    return (
      <ImportDropSurface
        {...props({
          destination: 'collection',
          collections: [{ id: 'ab12cd34', name: 'Fourier readings' }],
          collectionId,
          onCollection: setCollectionId,
        })}
      />
    )
  }
  render(<Surface />)
  fireEvent.change(fileInput(), { target: { files: [new File(['x'], 'x.md')] } })
  expect((screen.getByRole('button', { name: 'Import' }) as HTMLButtonElement).disabled).toBe(true)

  fireEvent.change(screen.getByRole('combobox', { name: 'Collections' }), {
    target: { value: 'ab12cd34' },
  })
  expect((screen.getByRole('button', { name: 'Import' }) as HTMLButtonElement).disabled).toBe(false)
})

it('hands the staged files to the importer and closes itself once asked', () => {
  const surface = props({ destination: 'authored_page' })
  render(<ImportDropSurface {...surface} />)
  const picked = new File(['# hello'], 'hello.md', { type: 'text/markdown' })

  fireEvent.change(fileInput(), { target: { files: [picked] } })
  fireEvent.click(screen.getByRole('button', { name: 'Import' }))

  expect(surface.onFiles).toHaveBeenCalledWith([picked])
  expect(surface.onClose).toHaveBeenCalledOnce()
})

it('drops a staged batch when the panel is dismissed', () => {
  const surface = props({ destination: 'authored_page' })
  render(<ImportDropSurface {...surface} />)
  fireEvent.change(fileInput(), { target: { files: [new File(['x'], 'x.md')] } })
  expect(screen.getByRole('status').textContent).toContain('x.md')

  fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
  expect(screen.queryByRole('status')).toBeNull()
  expect(surface.onFiles).not.toHaveBeenCalled()
})

it('stages a dropped file, opens the panel and imports nothing yet', () => {
  const surface = props({ open: false })
  const { rerender } = render(<ImportDropSurface {...surface} />)
  const dropped = new File(['# hello'], 'dropped.md', { type: 'text/markdown' })

  drop([dropped])

  expect(surface.onOpen).toHaveBeenCalledOnce()
  expect(surface.onFiles).not.toHaveBeenCalled()

  // The files are waiting in the panel the user is about to pick a store in.
  rerender(<ImportDropSurface {...surface} open />)
  expect(screen.getByRole('status').textContent).toContain('dropped.md')
})

it('closes on Escape', () => {
  const surface = props()
  render(<ImportDropSurface {...surface} />)
  fireEvent.keyDown(window, { key: 'Escape' })
  expect(surface.onClose).toHaveBeenCalledOnce()
})

it('offers every importable note format in the file picker', () => {
  render(<ImportDropSurface {...props({ destination: 'authored_page' })} />)
  expect(fileInput().accept.split(',')).toEqual(expect.arrayContaining(NOTE_FORMATS))
})

it('keeps the note format list for the note destinations but not for a knowledge base', () => {
  for (const destination of ['auto', 'authored_page', 'collection'] as ImportDestination[]) {
    const surface = render(<ImportDropSurface {...props({ destination })} />)
    expect(fileInput().accept).toBe(NOTE_IMPORT_ACCEPT)
    surface.unmount()
  }

  render(<ImportDropSurface {...props({ destination: 'knowledge_base' })} />)
  // Not a silent empty string: the attribute is absent, so the picker shows
  // every file and the engine decides what it can read.
  expect(fileInput().hasAttribute('accept')).toBe(false)
})

it('gives every import class it renders a style', () => {
  const css = source('features/notes/components/note-workspace.css')
  const used = new Set<string>()
  for (const file of [
    'features/notes/components/ImportDropSurface.tsx',
    'features/notes/components/ImportResultPanel.tsx',
  ])
    for (const match of source(file).matchAll(/dt-note-import-[a-z-]+/g)) used.add(match[0])

  expect(used.size).toBeGreaterThan(0)
  const undefinedClasses = [...used].filter(name => !new RegExp(`\\.${name}(?![\\w-])`).test(css))
  expect(undefinedClasses).toEqual([])
})
