"""External task connection schedule; never a controller/Student input."""
def disconnected(time_s,window,physics_dt=.005):
    if window is None:return False
    tick=round(time_s/physics_dt)
    return round(window[0]/physics_dt)<=tick<round(window[1]/physics_dt)


def apply_connection(tether,time_s,window):
    # Original Tether.measure/apply remains responsible for the real wrench.
    tether.cfg.enabled=not disconnected(time_s,window)
    return tether.cfg.enabled
