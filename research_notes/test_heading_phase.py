from heading_analyze import applied_phase
s=dict(onset=4.25,ramp=7.8,release=14.05)
assert applied_phase(dict(time=4.25,force_time=4.245,active=False),s)=='pre'
assert applied_phase(dict(time=4.255,force_time=4.25,active=True),s)=='move'
assert applied_phase(dict(time=14.05,force_time=14.045,active=True),s)=='hold'
assert applied_phase(dict(time=14.055,force_time=14.05,active=False),s)=='free_recovery'
print('PASS applied-force boundary regression; no precontact/recovery concatenation')
