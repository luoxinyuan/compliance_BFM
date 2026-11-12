# Envelope V1: complete time projection (before new physics)

Approval: https://github.com/GaoE05/robot-research-bridge/issues/62#issuecomment-6827917070

No new qualification has run. This is a conservative planning estimate, not measured complete Envelope training throughput. All phases remain subject to actual hard budget accounting.

- already_charged: 34028.589 job seconds
- current_preparation_analysis_tests_reserve: 1800.000 job seconds
- qualification_4_bundles_20_branches: 2400.000 job seconds
- new_process_restore_engineering_B_C_6_jobs: 1800.000 job seconds
- B_full_128_rounds: 6501.917 job seconds
- C_full_128_rounds: 8082.872 job seconds
- five_way_dev_100_episodes: 7955.541 job seconds
- BI_probe: 600.000 job seconds
- report_publication: 900.000 job seconds
- future_failure_reserve: 600.000 job seconds

Total **17.9636h**. Chosen cumulative cap **18h**. Historical costs not reset.

Evidence: V3 whole A128 process 5334.931s includes sampling, updates, IO, fingerprints, cold initialization; new B 1.20 multiplier plus100s operator allowance. C multiplies by worst matched C/B collection ratio 1.2470; new C32k not measured. A collect p50/p90/max32.060/34.556/37.938s; update .06989/.07674/.10053s; IO1.176/2.350/5.199s. Original whole round sums alone undercount whole job.

Dev: original archived eval-C-01 actual-0→actual-1 file timestamps yield62.399s for9.6s physics, including reset/scoring/compression. Use max warm rate6.4999wall/sim-second, ×1.15, plus135s cold/exit per model. Original cold121.84s. New20-episode manifest totals194.800s/model, five models100episodes. Counts and durations read only from train/dev, not sealed.

Qualification: four shared-load processes, each branch independently complete restore; budget600s/process covers20 total branches and source replay. B19 on/off reused only after full prefix/RNG/identity verification. Restore engineering: six processes300s each. These are projections, not measured new qualified throughput. No workflow/checkpoint reductions permitted to meet cap. Failed numerical operator attempt retained. Future failure/report reserve explicit; preparation reserve1800s includes current read-only jobs, inverse retry, analysis and locking.
