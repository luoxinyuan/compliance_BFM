"""Isolated short wrist task; original runtime and its checkpoints stay immutable."""
from training_live import TrainingLive
from reward_v2 import step_reward

class ShortGuidanceLive(TrainingLive):
    def step(self, action, before, z):
        first = len(self.e.trace)
        row = super().step(action, before, z)
        ticks = self.e.trace[first:]
        components = row["ctft_reward_components"]
        native = components["native_aux"].item()
        total, burden, tracking = step_reward(native, self.spec["mode"], self.e.tether.force, ticks)
        components["contact_aux"].fill_(-burden - tracking)
        row["aux_rewards"]["ctft_total_native"].fill_(total)
        self.last_reward_parts = dict(native=native, force=-burden, tracking=-tracking)
        return row
