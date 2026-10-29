"""Keep the original frozen BFM intact; stage only inference modules on CUDA."""
import contextlib
import hashlib

INFERENCE_MODULES = ('_actor', '_obs_normalizer', '_backward_map')


def tensor_digest(module):
    h = hashlib.sha256()
    for name, value in module.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def place_inference_modules(model, device='cuda'):
    assert not model.cfg.amp, 'Keep original FP32 inference'
    assert set(INFERENCE_MODULES) <= dict(model.named_children()).keys()
    before = {name: tensor_digest(getattr(model, name)) for name in INFERENCE_MODULES}
    calls = {}
    def unused(name):
        def check(_module, _inputs):
            calls[name] = calls.get(name, 0) + 1
            raise RuntimeError('Unexpected training-only BFM forward: ' + name)
        return check
    sizes = {}
    for name, module in model.named_children():
        if name in INFERENCE_MODULES:
            module.to(device)
        else:
            assert all(p.device.type == 'cpu' for p in module.parameters())
            module.register_forward_pre_hook(unused(name))
        sizes[name] = sum(t.numel()*t.element_size() for t in module.state_dict().values())
    # Match BaseModel.to(): inference uses the mutable runtime device.
    # cfg.device is the frozen construction/load setting; keep it unchanged.
    model.device = device
    model.eval();model.requires_grad_(False)
    after = {name: tensor_digest(getattr(model, name)) for name in INFERENCE_MODULES}
    assert before == after, 'Device staging changed frozen tensor values'
    report = dict(mode='full_original_BFM_on_CPU_inference_modules_on_CUDA',
                  inference_modules=list(INFERENCE_MODULES),module_bytes=sizes,
                  inference_tensor_hash_before=before,inference_tensor_hash_after=after,
                  inference_weights_bytes=sum(sizes[n] for n in INFERENCE_MODULES),
                  unused_forward_calls=calls,precision='original FP32',amp=False,
                  model_class=type(model).__name__,all_modules_retained=True,
                  runtime_device=model.device,load_config_device=model.cfg.device)
    return report


@contextlib.contextmanager
def cpu_load_then_stage(load_utils, reports):
    original = load_utils.load_model_from_checkpoint_dir
    def load(checkpoint_dir, device='cpu'):
        assert device == 'cuda'
        model = original(checkpoint_dir, device='cpu')
        reports.append(place_inference_modules(model, device=device))
        return model
    load_utils.load_model_from_checkpoint_dir = load
    try:
        yield
    finally:
        load_utils.load_model_from_checkpoint_dir = original
