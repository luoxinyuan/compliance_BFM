"""Use the existing saved-qpos renderer with software-only visual cost reductions.
No physics stepping, BFM execution, trajectory modification or playback speed change.
"""
import mujoco
Original=mujoco.Renderer

class SoftwareRenderer(Original):
    def __init__(self,model,*args,**kw):
        model.vis.quality.offsamples=0
        super().__init__(model,*args,**kw)
    def update_scene(self,*args,**kw):
        super().update_scene(*args,**kw)
        self.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
        self.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=False

mujoco.Renderer=SoftwareRenderer
if __name__=='__main__':
    import adapter_render
    adapter_render.main()
