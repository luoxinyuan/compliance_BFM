# Issue62 engineering failure log

control-01 / af1648b: isolated stage lacked F1 pointer-layout sidecar. Import failed before sampling/update;1 initialization physics tick. Copied exact qualified layout files from archived S1, SHA recorded; no C-array rule/tolerance change.

control-02 / af1648b:250 real transitions collected, round0 full package committed; first update raised TypeError because native CPR row metadata(time/source) entered tensor-only collator. Zero optimizer updates. Preserve source/output/package and uncommitted-stop.pt; do not call this one completed round. Fix tensor field selection, explicit held-command state for F/Q, and exact one-projection action contract before new run. No behavior result or pass claimed from affected implementation. Failed costs/ticks count.
