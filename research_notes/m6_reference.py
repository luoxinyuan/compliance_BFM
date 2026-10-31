"""Exact stored physical-time anchor reference, independent of actor cadence."""
def wrist_at_time(physics,time):
 row=min(physics,key=lambda row:abs(row['time']-time))
 if abs(row['time']-time)>1e-6:raise ValueError(f'No exact 200Hz reference at {time}; nearest {row["time"]}')
 return row['wrist']
