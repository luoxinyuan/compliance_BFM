# Interaction steering interfaces (S0 only)

Small isolated package, not a training framework. No imports from research_notes, no simulator at import, no global configuration mutation. Tests: `python -m unittest discover -s research/interaction_steering/tests -v`.

Pipeline: typed causal observation + known Goal/mode → Guidance → versioned source adapter → frozen executor → isolated physical worker → identity-tagged evaluator. Physical worker integration is pending S1 approval. Source memory belongs to the actual executing controller.

`ProgressToLatent` is an experimental forward-only reference, not a qualified root controller. `ForwardAdmittance` is a classical reference with explicit parameters, not learned interaction understanding. `OriginalExecutor` passed archived real-action/q_PD equality. P/I backends fail closed until their own weights/observations/PD/physics are qualified. No external asset is bundled.

See research_notes/S0_PORTABILITY_AND_ARCHITECTURE.md and S1_PRE_REGISTERED_PROTOCOL.md. S0 stops at code/metadata/offline interfaces; no S1 experiment or training autorun exists.
