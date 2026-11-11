# S0 failures and scope limitations

The first external audit at runtime80af844 stopped on the I-BFM upstream sphere-log antipode check. Original criterion: abs(norm(log(q,-q))−16π)<1e-5. Actual FP32 norm50.26551818847656, expected50.26548245743669, error3.573103987264403e-5; finite. This is a failed offline numeric helper criterion, not evidence of bad physical behavior.

Runtime168a91b preserves this failed result as `completed_with_numeric_helper_limitation`, allowing independent metadata/PD audit to complete. Criterion not changed, failure not converted to pass, upstream math not modified. Same-point case and32 P velocity helper cases pass. No external weights/physical model were evaluated. I-BFM remains unqualified; a future implementation must inspect non-antipodal/antipodal numerical behavior before physical use.

F1 L3's prior reset-order reconstruction failure and fix remain in F1_ERRATUM; S0 does not overwrite them or any L0/L2 conclusion.
