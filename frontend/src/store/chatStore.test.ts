// @ts-nocheck

import { useChatStore } from './chatStore'

describe('chatStore document view state', () => {
  const initialState = useChatStore.getState()

  const resetStore = () => {
    useChatStore.setState(
      {
        ...initialState,
        messages: [],
        sessionId: '',
        isHistoryLoading: false,
        historyRefreshKey: 0,
        activeView: 'chat',
        selectedDocumentId: null,
      },
      true
    )
  }

  beforeEach(() => {
    resetStore()
  })

  it('initializes with chat view and no selected document', () => {
    const state = useChatStore.getState()
    expect(state.activeView).toBe('chat')
    expect(state.selectedDocumentId).toBeNull()
  })

  it('switches to document view when selectDocument is called', () => {
    useChatStore.getState().selectDocument('doc-42')
    const state = useChatStore.getState()
    expect(state.activeView).toBe('document')
    expect(state.selectedDocumentId).toBe('doc-42')
  })

  it('clears selection and returns to chat view', () => {
    useChatStore.getState().selectDocument('doc-42')
    useChatStore.getState().clearSelectedDocument()
    const state = useChatStore.getState()
    expect(state.activeView).toBe('chat')
    expect(state.selectedDocumentId).toBeNull()
  })

  it('respects manual active view changes', () => {
    useChatStore.getState().setActiveView('document')
    expect(useChatStore.getState().activeView).toBe('document')

    useChatStore.getState().setActiveView('chat')
    expect(useChatStore.getState().activeView).toBe('chat')
  })

  it('clearChat resets to chat view and clears selection', () => {
    useChatStore.getState().selectDocument('doc-42')
    useChatStore.getState().clearChat()

    const state = useChatStore.getState()
    expect(state.activeView).toBe('chat')
    expect(state.selectedDocumentId).toBeNull()
    expect(state.messages).toHaveLength(0)
  })
})
