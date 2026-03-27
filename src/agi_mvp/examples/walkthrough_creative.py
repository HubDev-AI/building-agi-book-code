"""
Walkthrough 3: Handling a Completely Novel Task
===============================================
From Chapter 14 (The Cognitive Loop in Practice)

User prompt: "Generate a musical chord progression that follows
the emotional arc of this short story: 'A child finds a bird
with a broken wing, nurses it back to health over weeks, and
releases it -- watching it fly away with a mix of pride and
sadness.'"

The system has never mapped narrative arcs to chord theory.
No episodic memory exists. No tool handles this. It must reason
from first principles and build capability through the process.
"""


# ---------------------------------------------------------------
# Phase 1: Recognizing Novelty
# ---------------------------------------------------------------

TRACE_NOVELTY = """
[T+0ms] Layer 7 -> assess_task()
Difficulty: EXTREME
  - No matching keywords in difficulty heuristics
  - Novel cross-domain task (narrative -> music)

[T+200ms] EpisodicMemory.recall("music theory") -> 0
[T+400ms] EpisodicMemory.recall("cross-domain creative") -> 0
[T+500ms] SemanticMemory.search("chord progression") -> 0

Meta-cognitive assessment:
    Prior experience: NONE
    Strategy: "decompose_and_explore" (fallback)
    Model: DeepSeek-R1
    Resource budget: EXTENDED (30,000 tokens, 180s)
    Confidence: 0.30 (very low -- honest uncertainty)
"""
# NOTE: Honest Uncertainty
# A standard LLM agent would attempt this with no indication that
# it is operating outside its competence. Layer 7's confidence of
# 0.30 triggers a user-facing warning: "This task is outside my
# experience. The result may need expert review."


# ---------------------------------------------------------------
# Phase 2: First-Principles Decomposition
# ---------------------------------------------------------------
# With no episodic memory to guide it, the planner relies entirely
# on the reasoning model.

TRACE_DECOMPOSITION = """
[T+1s] Layer 3 -> Reasoning.infer(
    query: "How should I approach mapping a narrative
            emotional arc to a musical chord progression?",
    model: DeepSeek-R1
)

<reasoning_trace>
This is a cross-domain mapping problem. I need to:

1. ANALYZE THE NARRATIVE: Identify distinct emotional
   phases in the story.
2. MAP EMOTIONS TO MUSICAL PROPERTIES: Music theory has
   established correspondences between emotions and
   harmonic properties.
3. SELECT A KEY AND GENERATE PROGRESSIONS.
4. VERIFY MUSICAL VALIDITY.
</reasoning_trace>
"""


# ---------------------------------------------------------------
# Phase 3: Building Up Knowledge Step by Step
# ---------------------------------------------------------------
# Task 1 identifies four emotional phases:
#   - Discovery (curiosity, concern)
#   - Nurturing (tenderness, patience)
#   - Healing (hope, growing joy)
#   - Release (pride, loss, bittersweet acceptance)
#
# Layer 6 registers these as entities with temporal relations.
# Layer 2 stores them in working memory.
#
# Task 2 maps each emotional phase to musical properties:
#
# Measure | Chord                          | Emotional Note
# --------|--------------------------------|---------------------------
# 1-4     | Am -> Em -> F -> G             | Discovery: quiet concern
# 5-8     | F -> C/E -> Dm7 -> G/B         | Nurturing: gentle care
# 9-13    | C -> G -> Am -> F -> G -> C    | Healing: hope, confidence
# 14-18   | Am -> F -> C -> Gsus4 -> G -> C| Release: bittersweet peace


# ---------------------------------------------------------------
# Phase 4: Self-Verification (Limited)
# ---------------------------------------------------------------
# Musical aesthetics cannot be verified symbolically. Z3 cannot
# judge whether a chord progression is emotionally appropriate.
# The system verifies what it can.

TRACE_VERIFICATION = """
[T+30s] Layer 3 -> Symbolic verification (partial)

Verifiable properties:
  PASS  All chords are diatonic to C major / A minor
  PASS  Voice leading: no parallel fifths
  PASS  Progression resolves (ends on tonic: C major)
  PASS  Cadence structure: V -> I present

Non-verifiable:
  ?  Emotional mapping accuracy (requires human judgment)
  ?  Aesthetic quality (subjective)
  ?  Performance practicality

Verification status: PARTIAL
Confidence adjustment: 0.55 -> 0.60
"""


# ---------------------------------------------------------------
# Phase 5: Learning from a Novel Task
# ---------------------------------------------------------------

TRACE_LEARNING = """
[T+33s] EpisodicMemory.store(Episode{
    task: "Map narrative arc to chord progression",
    lesson: "Cross-domain creative tasks can be
    decomposed into (1) analyze source domain,
    (2) identify mapping rules between domains,
    (3) generate in target domain, (4) verify what
    is verifiable and flag what is not."
    importance: 0.85
})
"""

# The Memory Manager then consolidates, recognizing the lesson
# is generalizable:

TRACE_CONSOLIDATION = """
[T+34s] Extracted general fact:
    Fact{
      subject: "cross-domain creative tasks",
      predicate: "strategy",
      object: "decompose into: domain analysis,
               mapping rules, generation,
               partial verification"
    }
"""
# This fact is now available for any future cross-domain task --
# mapping emotions to colors, translating poetry styles, or
# converting visualizations to narrative.


# ---------------------------------------------------------------
# Timing and Cost Summary
# ---------------------------------------------------------------
# Phase                         | Tokens | Time  | Model
# Assessment + memory search    |    600 |  0.5s | Qwen
# First-principles decomposition|  3,500 |  7.0s | R1
# Emotional analysis            |  1,200 |  3.0s | R1
# Emotion-to-music mapping      |  4,200 | 12.0s | R1
# Progression generation        |  2,800 |  5.0s | R1
# Partial verification          |    800 |  2.0s | Z3 + Qwen
# Evaluation + learning         |  1,400 |  2.0s | Qwen
# ----------------------------------------------------------
# Total                          ~14,500  ~31.5s
#
# Almost all tokens went to R1 -- correct, because the fast model
# could not contribute to a genuinely creative task.


# ---------------------------------------------------------------
# Cross-Cutting Observations (from all three walkthroughs)
# ---------------------------------------------------------------
#
# Pattern 1: The Meta-Cognitive Tax is Worth Paying
#   Layer 7 added 1-3 seconds and 500-1,500 tokens per task.
#   In exchange: 75% model cost savings (Walkthrough 1),
#   prevented infinite retry loops (Walkthrough 2), and
#   honest uncertainty quantification (Walkthrough 3).
#
# Pattern 2: The world model prevents redundant work.
#   Entity tracking during search meant the comparison table was
#   built from cached data (W1). Emotional phase entities served
#   as structured anchors for music mapping (W3).
#
# Pattern 3: Episodic memory is the growth engine.
#   The system carries forward not just facts but strategies,
#   failure modes, and meta-level insights. A future optimization
#   task skips the failed first attempt (W2). A future cross-domain
#   task starts with a validated decomposition pattern (W3).
