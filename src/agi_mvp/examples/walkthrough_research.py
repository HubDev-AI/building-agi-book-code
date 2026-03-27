"""
Walkthrough 1: Solving a Complex Multi-Step Problem
====================================================
From Chapter 14 (The Cognitive Loop in Practice)

User prompt: "Research the current state of quantum computing,
write a summary, and create a comparison table of the top 5
quantum computing companies."

This walkthrough demonstrates the full cognitive loop handling a
complex multi-step research task where every layer participates:
web search (L5), memory storage (L2), cross-source synthesis (L3),
structured planning (L4), entity tracking (L6), and quality
control (L7).

All code blocks below are execution traces (not runnable code).
They show the actual data flowing between layers -- concrete
messages, timing, and token costs.
"""


# ---------------------------------------------------------------
# Iteration 0: Meta-Cognitive Assessment
# ---------------------------------------------------------------
# Layer 7 sees the incoming request and makes routing decisions.

TRACE_ASSESSMENT = """
[T+0ms] Layer 7 -> MetaCognition.assess_task()

Input:  "Research the current state of quantum computing,
         write a summary, and create a comparison table
         of the top 5 quantum computing companies."

Difficulty estimate:  HARD
  Signals:
    - keyword "research"  -> EXTREME weight
    - keyword "compare"   -> MEDIUM weight
    - multi-part structure (3 sub-tasks) -> +1 level
    - combined estimate: HARD (capped from EXTREME
      because sub-tasks are individually tractable)

Model routing:
    - Planning phase     -> DeepSeek-R1
    - Web search queries -> Qwen 2.5-72B
    - Synthesis          -> DeepSeek-R1
    - Table formatting   -> Qwen 2.5-72B

Strategy selected: "search_synthesize_verify"
  Source: default for task_type="research"

Resource budget:
    - Max tokens:    25,000 (generation)
    - Max time:      120 seconds
    - Max tool calls: 15
"""
# Cost: ~200 tokens, 150ms. Uses Qwen for routing (classification task).


# ---------------------------------------------------------------
# Iteration 1: Planning
# ---------------------------------------------------------------
# The Planner queries episodic memory for past research tasks,
# finds none, and decomposes.

TRACE_PLANNING = """
[T+350ms] Layer 4 -> Planner.decompose()
  Model: DeepSeek-R1

Generated task tree:
  Goal: "QC research + summary + comparison table"
  +-- Task 1: "Search for current state of QC" [30s]
  |   +-- 1a: "Search for QC 2025-2026 breakthroughs"
  |   +-- 1b: "Search for QC industry landscape"
  |   +-- 1c: "Search for top QC companies"
  +-- Task 2: "Synthesize findings into summary" [15s]
  |   +-- 2a: "Identify key themes across results"
  |   +-- 2b: "Write 3-4 paragraph summary"
  +-- Task 3: "Create comparison table" [15s]
      +-- 3a: "Define comparison dimensions"
      +-- 3b: "Populate data for each company"
      +-- 3c: "Format as markdown table"

  Dependencies: Task 2 depends on Task 1.
                Task 3 depends on Task 1.
                Tasks 2 and 3 are independent.
"""
# Token cost: ~2,800 tokens (R1 reasoning for decomposition), 4.2s.


# ---------------------------------------------------------------
# Iterations 2-4: Information Gathering (Tasks 1a-1c)
# ---------------------------------------------------------------
# The three search sub-tasks are independent, so the planner
# dispatches them in parallel.

TRACE_SEARCH = """
[T+4.5s] Layer 5 -> web_search("QC breakthroughs 2025 2026")
[T+4.5s] Layer 5 -> web_search("QC industry landscape")
[T+4.5s] Layer 5 -> web_search("top QC companies comparison")

[T+7.2s] All three searches return.
"""


# Layer 6 (World Model) starts tracking entities as they are
# discovered.

TRACE_ENTITY_TRACKING = """
[T+7.5s] Layer 6 -> WorldModel.update(observations)

Entities registered:
  IBM (1121 qubits, superconducting)
  Google (105 qubits, superconducting)
  IonQ (35, trapped-ion)
  Quantinuum (trapped-ion)
  PsiQuantum (photonic)
  D-Wave (5000, annealing)
  Microsoft (topological)

Relations:
  Google -> achieved -> "error correction below threshold"
  IBM -> demonstrated -> "quantum advantage in materials sim"
"""
# Layer 2 stores raw results in working memory and key facts in
# semantic memory. Token cost for search phase: ~1,200 (query
# generation via Qwen), 3 tool calls, 2.7s wall time.


# ---------------------------------------------------------------
# Iteration 5: Synthesis (Task 2)
# ---------------------------------------------------------------
# The reasoning engine receives full context: working memory,
# retrieved facts, and the world model's entity graph.

TRACE_SYNTHESIS = """
[T+8.5s] Layer 3 -> Reasoning.infer(
    query: "Synthesize a summary of the current
            state of quantum computing",
    mode: ReasoningMode.NEURAL,
    model: DeepSeek-R1
)

<reasoning_trace>
Key themes I see:
1. Error correction is the major frontier -- Google and
   IBM have both demonstrated milestones in 2024-2025.
2. Multiple hardware approaches are competing:
   superconducting, trapped-ion, photonic, topological,
   annealing. No clear winner yet.
3. Enterprise adoption is beginning in specific verticals.
4. Market is growing but still pre-revenue for most
   pure-play quantum companies.
</reasoning_trace>

Confidence: 0.78
"""
# Token cost: ~4,500 tokens (R1 reasoning + generation), 6.8s.


# ---------------------------------------------------------------
# Iteration 6: Comparison Table (Task 3)
# ---------------------------------------------------------------
# The world model provides entity data directly -- no additional
# search needed.

TRACE_TABLE = """
[T+16s] Layer 6 -> WorldModel.query(
    "What are the properties of all company entities?"
)

Returns structured data for 7 companies. The planner
selects the top 5 by relevance score.

[T+17.2s] Layer 1 -> complete(
    prompt: "Format this data as a comparison table...",
    model: Qwen 2.5-72B
)
"""
# Token cost: ~1,800 tokens (Qwen, fast), 2.1s.


# ---------------------------------------------------------------
# Iteration 7: Quality Evaluation
# ---------------------------------------------------------------

TRACE_EVALUATION = """
[T+19s] Layer 7 -> MetaCognition.evaluate_outcome(...)

Evaluation scores:
    correctness:  8/10
    completeness: 9/10
    efficiency:   7/10
    quality:      8/10
    overall:      8/10

Verdict: PASS (threshold: 7/10)
"""


# ---------------------------------------------------------------
# Full Trace Summary
# ---------------------------------------------------------------
# Phase          | Layer(s)   | Model       | Tokens | Time   | Cost Est.
# Assessment     | L7         | Qwen 2.5    |    350 |  0.2s  | $0.0003
# Planning       | L4, L2     | DeepSeek-R1 |  2,800 |  4.2s  | $0.0056
# Search (x3)    | L5, L1     | Qwen 2.5    |  1,200 |  2.7s  | $0.0010
# Entity tracking| L6         | (procedural)|      0 |  0.3s  | $0.0000
# Memory storage | L2         | (procedural)|      0 |  0.2s  | $0.0000
# Synthesis      | L3, L1     | DeepSeek-R1 |  4,500 |  6.8s  | $0.0090
# Table gen.     | L3, L1, L6 | Qwen 2.5    |  1,800 |  2.1s  | $0.0015
# Quality eval   | L7         | Qwen 2.5    |    800 |  1.2s  | $0.0007
# Episode store  | L2         | Qwen 2.5    |    500 |  0.7s  | $0.0004
# ---------------------------------------------------------------
# Total                                      ~12,000  ~18.4s   ~$0.019
#
# INSIGHT: Smart Routing Saves 75%
# Only two of eight phases required the expensive reasoning model.
# Layer 7's routing saved ~75% versus uniform R1 usage. The world
# model eliminated a fourth search call -- the comparison table
# was built from cached entity data rather than raw-text re-search.
