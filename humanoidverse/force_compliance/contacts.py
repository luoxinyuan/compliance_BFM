"""Fixed, explicit local contact points; one maintained tether per episode."""
CONTACTS = [
    {'name': 'left_hand', 'body': 'left_wrist_yaw_link', 'point': [.07065, .0215, .00141]},
    {'name': 'right_hand', 'body': 'right_wrist_yaw_link', 'point': [.07065, -.0215, .00141]},
    {'name': 'left_upper_arm', 'body': 'left_shoulder_yaw_link', 'point': [.00817, .02203, -.03929]},
    {'name': 'right_upper_arm', 'body': 'right_shoulder_yaw_link', 'point': [.00817, -.02203, -.03929]},
    {'name': 'chest', 'body': 'torso_link', 'point': [.07327, 0., .17319]},
    {'name': 'pelvis', 'body': 'pelvis', 'point': [.0003, .06514, -.03211]},
    {'name': 'left_thigh', 'body': 'left_hip_yaw_link', 'point': [-.04227, .03331, -.09931]},
    {'name': 'right_thigh', 'body': 'right_hip_yaw_link', 'point': [-.04227, -.03331, -.09931]},
    {'name': 'left_shin', 'body': 'left_knee_link', 'point': [.02005, .02322, -.15073]},
    {'name': 'right_shin', 'body': 'right_knee_link', 'point': [.02005, -.02322, -.15073]},
]


def select_contact(cfg, rng, index=None):
    """Use a separate RNG from anchor sampling, so matching trajectories is reproducible."""
    from dataclasses import replace
    if not cfg.contact_points:
        if index is not None:
            raise ValueError('Explicit contact index requires contact_points')
        return 0, cfg.body, cfg
    index = int(rng.integers(len(cfg.contact_points))) if index is None else index
    if not 0 <= index < len(cfg.contact_points):
        raise ValueError('Contact index outside configured catalog')
    contact = cfg.contact_points[index]
    return index, contact['name'], replace(cfg, body=contact['body'], local_point=list(contact['point']))
