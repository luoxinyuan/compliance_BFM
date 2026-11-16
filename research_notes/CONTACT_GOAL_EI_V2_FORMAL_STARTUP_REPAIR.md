# Formal startup repair — immutable MuJoCo layout helper

2026-11-16. After the earlier disk-blocked preflight, attempt02 loaded the environment but stopped before formal sampling because the isolated stage lacked `prep/pointer_layout.so` and its field manifest. The failed attempt is retained: 101.603457 job seconds, 1 physical tick, zero formal rows and zero updates. This was a missing runtime dependency, not a failed optimization/behavior qualification.

Repair: copy the exact `.so` and JSON from the completed OR1 stage. Both source/destination hashes match; all 173 generated field offsets are within the 163056-byte structure. No recompilation, MuJoCo change, array-comparison relaxation or physics change. The model-specific ABI/null-pointer assertions still execute during real reset.

The first restart wrapper accidentally retained the old attempt argument while expecting the new sidecar; the immutable-output assertion rejected it before a child experiment. Its log is retained. The replacement queue takes one integer attempt argument and derives both child argument and sidecar name from it; syntax validation passed. Neither wrapper failure altered training data or weights. Wrapper/preparation costs are within the separate conservative preparation debit; experiment child costs are in the supervisor ledger.

Attempt03 uses the same frozen training runtime `646ba2531e5ef418a93d7e990bb4bf2850616e6c`. No source/optimizer/reward/split change. Initial physical prefix has begun; formal learning and physical success are not claimed by this repair note. All original failures remain.
