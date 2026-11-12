# S1 errata and accounting boundary

## Physical-step budget wording was incomplete — not a clean budget pass

The approved draft enumerated225s /45,000 physics ticks for its48 candidate/controller branches and extensions, but also required complete archival reconstruction and restore qualification. The implementation supplement classified reconstruction as separate overhead before launch; that does not retroactively authorize changing a literal **total** step cap.

Actual experimental execution:33,600 ticks,1,680 high-level commands,48 branches. At the final native initial state, archived-restoration counters are `total_prefill=15,000`, `total_steps=21,880`, `total_controls=5,470`; source_physics_total7,000 is a subset, not added again. The reused full reconstruction order also performed1,200 ticks for an allocated efc_JT cache verification and600 ticks for original backend-prefix qualification. Thus known total is **at least72,280 physics ticks**, not33,600. Simulator-internal initialization ticks were not globally instrumented; do not invent an exact all-process count.

Time and output caps remain satisfied, no new scientific parent/candidate/task was added, but under the literal total-step cap this is an overrun. Reported to Yi, physical jobs stopped, no further physical run is authorized by this result. The source/counter evidence is retained. This does not change physical results or success definitions; it does prevent a claim of fully compliant budget execution. Any next protocol must explicitly include all reset/prefix/restore/probe costs and count simulator steps across resets, rather than only trial transitions.

The inherited cache verification was on an already exposed archival state outside the7 test windows, solely to reproduce its complete-state dependency chain; it is not counted as an eighth scientific test or a positive Steering sample. Its data remain in the private run. No sealed inputs were read.

## Postprocessing import failure

First reducer and renderer failed while unpickling the own-runtime Guidance Tier enum: `ModuleNotFoundError: No module named research`. These were data-reading/render jobs with zero physical steps, not simulation failures. Original logs/output directories retained. Explicit pinned runtime import path fixes the readers; analysis02/videos02 are new outputs, no overwrite. Second reduction and3 video full-frame decodes passed. Failure costs are in the ledger (reducer failed duration only bounded, not falsely exact).

## Pre-registered coverage limitation retained

The fixed velocity0/.09/.18 control is allocated only to native-standing. The draft's stronger two-demand-parent speed-gain clause has no matched fixed-speed control on a second parent; it remains unproven. No extra state or candidate was added after seeing outcomes. No threshold, PD, reward, information permission, force cap or probability assertion was relaxed.
