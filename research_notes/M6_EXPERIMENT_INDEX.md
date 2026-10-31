# M6 experiment index

Issue27; base b5bfb0d; branch gaoe05/m6-multibehavior. Source research_notes/m6_*.py, immutable previous research files unchanged.

Run root /mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m6-multibehavior-20261031A; local mirror /Users/gaoe/Projects/compliance-bfm-diagnostics/m6-multibehavior.

The compute-ledger.json records each exact GPU/PID/start ticks/command/runtime revision and all failures. Stages: qualification (negative teachers and controls), projected (actual privileged per-state latent control, not deployment Student), train0, collect1, train1, collect2, train2, final.

Each stage uses project env and `python research_notes/m6_supervisor.py --gpu-index <live-available> --stage <stage> --execute-after-owner-gpu-notice`. Never replay into old output root. Analysis via m6_analysis.py and final audit m6_final_audit.py.

Methods: original frozen M5 Projected-z initialization for both shared methods and negative-only adaptation. Shared sample40/40/20 percent positive/negative/free; independent80/20 negative/free. Same architecture/update count/optimizer, independent different data by design. Complete historical M5 dataset reused, old weights unchanged. There is no training-time or deployment task label in the network.

FIBER reference: https://arxiv.org/html/2610.24675v1 Appendix C/D. This is small MLP shared LPP with online expert-query aggregation and true physical action projection, not full AH-BFM/FlowPlanner replication. All selected labels are global projected executable z and physical q_PD; not raw teacher alpha supervision.

Completed: qualification4, projected1, collect1/2 each13; final-attempt01 six complete then reference-grid engineering exception, final-attempt02 remaining32. Three training rounds3000/2000/2000 per each of3 methods. Full report m6_results.md; final dataset17837.

Final models train2-attempt-01/outputs/{alignment,distill,single_negative}-final.pt, SHAs in protocol/report. All original raw histories unchanged. Final reference resume uses exact200Hz state instead of50Hz-only lookup at6.65; old aligned times parity0. No completed final condition repeated.

Saved-trace analysis commands (CPU, no models): m6_analysis.py, m6_mechanism.py, m6_report_data.py, m6_plot.py; final m6_final_audit.py. Force uses force_time/active; body state usespost-step time. Boundary-inclusive recovery window can contain last-contact force sample; do not mislabel as recontact.

To reproduce a new run, use an independent checkout and a fresh run ID consistently in the M6 RUN constants and supervisor assertion, preserving the committed protocol and pinned asset/initial-model SHAs. Original exact staged runtime commands are in compute-ledger.json. Reused exact controls are explicitly listed in protocol; do not silently count them as new simulations. Existing completed RUN is intentionally not replayable or overwritten.
