# Postprocessing corrections, no extra numerical attempts

1. An ad-hoc progress summary read `phases.all_contact` from a no-contact episode and raised `KeyError: all_contact`. Corrected by guarding `condition.enabled`. The mandatory complete analyzer already had this phase guard. No experiment or checkpoint changed.
2. The first full final analysis assumed all reused historical normal teacher records had `applied_wrist_wrench` and raised `KeyError: applied_wrist_wrench`. Those records predate this instrumentation. The analyzer now records whether the field exists, verifies it when present, and still requires it for every new stage record. Historical no-contact records similarly predate `connection_enabled`; zero recorded force is checked and connection flags are checked where recorded. No fields are synthesized. All records retain source SHA and common-state identity.

These are saved-log schema compatibility corrections. No training rerun, physical rerun, criterion change or checkpoint reselection occurred.
