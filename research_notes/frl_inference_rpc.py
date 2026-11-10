"""Private per-process Pipe packets for the bounded Issue58 sampling benchmark."""
class RemoteChannel:
    def __init__(self, connection, policy_sha):
        self.connection=connection;self.policy_sha=policy_sha;self.sequence=0
    def call(self, kind, payload):
        self.sequence+=1
        self.connection.send(dict(kind=kind,sequence=self.sequence,policy_sha=self.policy_sha,payload=payload))
        response=self.connection.recv()
        assert response['sequence']==self.sequence,'Inference response reordered'
        assert response['policy_sha']==self.policy_sha,'Inference policy version mismatch'
        if 'error' in response:raise RuntimeError(response['error'])
        return response['payload']

def validate_request(request, previous, policy_sha):
    assert request['policy_sha']==policy_sha,'Mixed policy versions'
    assert request['sequence']==previous+1,'Lost or reordered actual inference requests'
    assert request['kind'] in ('bfm','high'),'Unsupported inference operation'
    return request['sequence']


def preserved_row_actions(actor, observations, latents):
    # A compiled actor may return reusable output storage. Each real response
    # must retain its own command before another worker/shape is evaluated.
    return [actor(o,z,mean=True).detach().clone() for o,z in zip(observations,latents)]
