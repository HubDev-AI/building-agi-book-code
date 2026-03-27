# background_processor.py

import asyncio

class BackgroundProcessor:
    """Runs during idle time -- the system's
    default mode network."""

    def __init__(self, memory: 'MemoryManager',
                 metacognition: 'MetaCognition'):
        self.memory = memory
        self.meta = metacognition
        self._running = False

    async def run(self):
        """Background loop: consolidate, reflect,
        pre-compute."""
        self._running = True
        while self._running:
            # 1. Consolidate episodic -> semantic
            await self._consolidate_memories()

            # 2. Detect inconsistencies in knowledge
            await self._check_consistency()

            # 3. Reflect on recent performance
            await self._reflect()

            # Sleep before next cycle (5 minutes)
            await asyncio.sleep(300)

    async def _consolidate_memories(self):
        """Promote important episodic memories to
        semantic facts."""
        recent = self.memory.episodic.recall(
            "important lessons", top_k=10,
        )
        for episode in recent:
            if episode.importance > 0.7:
                await self.memory._consolidate(episode)

    async def _check_consistency(self):
        """Find contradictions in semantic memory."""
        facts = self.memory.semantic.query_facts()
        # Group by subject and check for
        # conflicting predicates
        by_subject = {}
        for f in facts:
            by_subject.setdefault(
                f.subject, []
            ).append(f)
        for subject, fact_list in by_subject.items():
            if len(fact_list) > 1:
                for i, f1 in enumerate(fact_list):
                    for f2 in fact_list[i+1:]:
                        if (f1.predicate == f2.predicate
                                and f1.object != f2.object):
                            # Contradiction! Keep the
                            # more confident one
                            loser = (f1 if f1.confidence
                                     < f2.confidence
                                     else f2)
                            self.memory.semantic \
                                .correct_fact(
                                    loser.id,
                                    (f2.object if loser == f1
                                     else f1.object),
                                    reason="consistency check",
                                )

    async def _reflect(self):
        """Review recent performance and extract
        meta-lessons."""
        perf = self.meta.get_performance_summary(
            last_n=20,
        )
        if perf.get("total_tasks", 0) < 10:
            return
        # If calibration is poor, note it
        cal_error = perf.get("calibration_error", -1)
        if cal_error > 0.2:
            self.memory.working.set(
                "meta_note",
                "Calibration is poor -- being "
                "overconfident or underconfident",
                importance=0.8,
            )

    def stop(self):
        self._running = False
