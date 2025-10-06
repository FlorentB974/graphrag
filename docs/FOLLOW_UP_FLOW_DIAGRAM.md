# Follow-Up Questions Flow Diagram

## Architecture Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Interface (app.py)                      │
│  - Maintains chat history in st.session_state.messages              │
│  - Formats history: [{"role": "user/assistant", "content": "..."}]  │
└────────────────────────────┬────────────────────────────────────────┘
                             │ query + chat_history
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    GraphRAG Pipeline (graph_rag.py)                  │
│  - Receives query and chat_history                                   │
│  - Passes through LangGraph workflow                                 │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│              Step 1: Query Analysis (query_analysis.py)              │
│                                                                       │
│  ┌─────────────────────────────────────────────────────┐            │
│  │ 1. Receive query + chat_history                     │            │
│  └────────────┬────────────────────────────────────────┘            │
│               │                                                       │
│               ▼                                                       │
│  ┌─────────────────────────────────────────────────────┐            │
│  │ 2. _detect_follow_up_question()                     │            │
│  │    - Heuristic checks (pronouns, indicators)        │            │
│  │    - LLM-based analysis                             │            │
│  │    → Returns: is_follow_up, needs_context           │            │
│  └────────────┬────────────────────────────────────────┘            │
│               │                                                       │
│               ▼                                                       │
│  ┌─────────────────────────────────────────────────────┐            │
│  │ 3. If follow-up: _create_contextualized_query()    │            │
│  │    - Extract context from history                   │            │
│  │    - Rewrite query to be self-contained             │            │
│  │    → Returns: contextualized_query                  │            │
│  └────────────┬────────────────────────────────────────┘            │
│               │                                                       │
│               ▼                                                       │
│  ┌─────────────────────────────────────────────────────┐            │
│  │ 4. Return analysis:                                 │            │
│  │    - original_query                                 │            │
│  │    - contextualized_query                           │            │
│  │    - is_follow_up, needs_context                    │            │
│  │    - query_type, key_concepts, etc.                 │            │
│  └─────────────────────────────────────────────────────┘            │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│              Step 2: Document Retrieval (retrieval.py)               │
│                                                                       │
│  ┌─────────────────────────────────────────────────────┐            │
│  │ 1. Extract contextualized_query from analysis       │            │
│  └────────────┬────────────────────────────────────────┘            │
│               │                                                       │
│               ▼                                                       │
│  ┌─────────────────────────────────────────────────────┐            │
│  │ 2. Use contextualized_query for retrieval           │            │
│  │    - Semantic search                                │            │
│  │    - Graph expansion                                │            │
│  │    - Multi-hop reasoning                            │            │
│  └────────────┬────────────────────────────────────────┘            │
│               │                                                       │
│               ▼                                                       │
│  ┌─────────────────────────────────────────────────────┐            │
│  │ 3. Return relevant chunks                           │            │
│  └─────────────────────────────────────────────────────┘            │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│             Step 3: Graph Reasoning (graph_reasoning.py)             │
│  - Enhance chunks with graph context (unchanged)                     │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│            Step 4: Response Generation (generation.py)               │
│                                                                       │
│  ┌─────────────────────────────────────────────────────┐            │
│  │ 1. Receive query, chunks, analysis, chat_history    │            │
│  └────────────┬────────────────────────────────────────┘            │
│               │                                                       │
│               ▼                                                       │
│  ┌─────────────────────────────────────────────────────┐            │
│  │ 2. Check if is_follow_up from analysis              │            │
│  └────────────┬────────────────────────────────────────┘            │
│               │                                                       │
│               ▼                                                       │
│  ┌─────────────────────────────────────────────────────┐            │
│  │ 3. Call llm_manager.generate_rag_response()         │            │
│  │    - Include chat_history if follow-up              │            │
│  └────────────┬────────────────────────────────────────┘            │
│               │                                                       │
│               ▼                                                       │
│  ┌─────────────────────────────────────────────────────┐            │
│  │ 4. LLM Manager (llm.py)                             │            │
│  │    - Build prompt with conversation history         │            │
│  │    - Format: Previous conversation + Context + Query│            │
│  │    - Generate context-aware response                │            │
│  └────────────┬────────────────────────────────────────┘            │
│               │                                                       │
│               ▼                                                       │
│  ┌─────────────────────────────────────────────────────┐            │
│  │ 5. Return response + sources                        │            │
│  └─────────────────────────────────────────────────────┘            │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Return to User Interface                          │
│  - Display response                                                  │
│  - Update chat history                                               │
│  - Show sources and graph                                            │
└─────────────────────────────────────────────────────────────────────┘
```

## Example Conversation Flow

### Scenario: Multi-turn conversation about a person

```
┌─────────────────────────────────────────────────────────────────────┐
│ Turn 1: Initial Question (Not a follow-up)                          │
└─────────────────────────────────────────────────────────────────────┘

User Input: "Who is Florent?"

Query Analysis:
  ├─ is_follow_up: False
  ├─ needs_context: False
  ├─ original_query: "Who is Florent?"
  └─ contextualized_query: "Who is Florent?"

Retrieval:
  └─ Search for: "Who is Florent?"

Generation:
  ├─ No chat history included in prompt
  └─ Response: "Florent is a software engineer..."

Chat History Updated:
  [
    {"role": "user", "content": "Who is Florent?"},
    {"role": "assistant", "content": "Florent is a software engineer..."}
  ]

┌─────────────────────────────────────────────────────────────────────┐
│ Turn 2: Follow-up Question with Pronoun                             │
└─────────────────────────────────────────────────────────────────────┘

User Input: "Tell me more about his work experience"

Query Analysis:
  ├─ is_follow_up: True ✓
  ├─ needs_context: True ✓
  ├─ original_query: "Tell me more about his work experience"
  └─ contextualized_query: "Tell me more about Florent's work experience"

Retrieval:
  └─ Search for: "Tell me more about Florent's work experience"

Generation:
  ├─ Chat history included in prompt:
  │   "Previous conversation:
  │    User: Who is Florent?
  │    Assistant: Florent is a software engineer..."
  └─ Response: "Florent's work experience includes..."

Chat History Updated:
  [
    {"role": "user", "content": "Who is Florent?"},
    {"role": "assistant", "content": "Florent is a software engineer..."},
    {"role": "user", "content": "Tell me more about his work experience"},
    {"role": "assistant", "content": "Florent's work experience includes..."}
  ]

┌─────────────────────────────────────────────────────────────────────┐
│ Turn 3: Follow-up with "What about"                                 │
└─────────────────────────────────────────────────────────────────────┘

User Input: "What about his education?"

Query Analysis:
  ├─ is_follow_up: True ✓
  ├─ needs_context: True ✓
  ├─ original_query: "What about his education?"
  └─ contextualized_query: "What about Florent's education?"

Retrieval:
  └─ Search for: "What about Florent's education?"

Generation:
  ├─ Chat history included (last 4 messages)
  └─ Response: "Florent's education includes..."

┌─────────────────────────────────────────────────────────────────────┐
│ Turn 4: New Topic (Not a follow-up)                                 │
└─────────────────────────────────────────────────────────────────────┘

User Input: "What is the capital of France?"

Query Analysis:
  ├─ is_follow_up: False
  ├─ needs_context: False
  ├─ original_query: "What is the capital of France?"
  └─ contextualized_query: "What is the capital of France?"

Retrieval:
  └─ Search for: "What is the capital of France?"

Generation:
  ├─ No chat history included (new topic)
  └─ Response: "The capital of France is Paris..."
```

## Key Decision Points

```
                    ┌─────────────────┐
                    │  Receive Query  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Has chat_history?│
                    └────┬────────┬───┘
                         │        │
                      YES│        │NO
                         │        │
                         ▼        ▼
              ┌──────────────┐  ┌──────────────┐
              │ Run follow-up│  │ Process as   │
              │ detection    │  │ standalone   │
              └──────┬───────┘  └──────┬───────┘
                     │                  │
                     ▼                  │
              ┌──────────────┐         │
              │ is_follow_up?│         │
              └──┬────────┬──┘         │
                 │        │            │
              YES│        │NO          │
                 │        │            │
                 ▼        ▼            ▼
      ┌──────────────┐ ┌────────────────────┐
      │ Contextualize│ │ Use original query │
      │ query        │ │                    │
      └──────┬───────┘ └────────┬───────────┘
             │                  │
             └──────────┬───────┘
                        │
                        ▼
              ┌──────────────────┐
              │ Retrieve with    │
              │ final query      │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ is_follow_up?    │
              └──┬───────────┬───┘
                 │           │
              YES│           │NO
                 │           │
                 ▼           ▼
      ┌──────────────┐ ┌────────────────┐
      │ Include      │ │ Generate without│
      │ chat_history │ │ history         │
      │ in prompt    │ │                 │
      └──────┬───────┘ └────────┬────────┘
             │                  │
             └──────────┬───────┘
                        │
                        ▼
              ┌──────────────────┐
              │ Return response  │
              └──────────────────┘
```

## Token Flow for Follow-up Questions

```
Original Query: "Tell me more about his work"
                        │
                        ▼
         ┌──────────────────────────┐
         │ Follow-up Detection      │
         │ Tokens: ~100-150         │
         └──────────┬───────────────┘
                    │
                    ▼
         ┌──────────────────────────┐
         │ Query Contextualization  │
         │ Tokens: ~150-200         │
         └──────────┬───────────────┘
                    │
                    ▼
Contextualized: "Tell me more about Florent's work"
                        │
                        ▼
         ┌──────────────────────────┐
         │ Document Retrieval       │
         │ (uses contextualized)    │
         └──────────┬───────────────┘
                    │
                    ▼
         ┌──────────────────────────┐
         │ Response Generation      │
         │ Prompt includes:         │
         │ - History: ~200-400 tkns │
         │ - Context: ~2000 tkns    │
         │ - Query: ~20 tkns        │
         └──────────┬───────────────┘
                    │
                    ▼
         ┌──────────────────────────┐
         │ Final Response           │
         │ Output: ~500-1500 tkns   │
         └──────────────────────────┘

Total Additional Tokens for Follow-up: ~450-750 tokens
(compared to standalone questions)
```
