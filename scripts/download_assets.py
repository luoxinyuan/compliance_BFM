"""Download official assets excluded from Git; existing local files are preserved."""
import argparse
from pathlib import Path
import shutil
from huggingface_hub import hf_hub_download

REPOSITORY = 'LeCAR-Lab/BFM-Zero'
REVISION = '62b4206d68e026de5e5dc7efb1529bccfb95164c'
MODEL = 'new_model_for_training_code_inference'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['evaluation', 'training', 'both'], default='evaluation')
    parser.add_argument('--skip-model', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = []
    if args.dataset in ('evaluation', 'both'):
        paths.append(('data/lafan_29dof.pkl', root / 'humanoidverse/data/lafan_29dof.pkl'))
    if args.dataset in ('training', 'both'):
        paths.append(('data/lafan_29dof_10s-clipped.pkl', root / 'humanoidverse/data/lafan_29dof_10s-clipped.pkl'))
    if not args.skip_model:
        for name in ('config.json', 'checkpoint/model/config.json',
                     'checkpoint/model/init_kwargs.json', 'checkpoint/model/model.safetensors'):
            remote = f'{MODEL}/{name}'
            paths.append((remote, root / 'pretrained/official' / remote))
    for remote, destination in paths:
        if destination.exists():
            print(f'Preserving existing file: {destination}')
            continue
        cached = hf_hub_download(REPOSITORY, remote, revision=REVISION)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + '.download')
        shutil.copyfile(cached, temporary)
        temporary.replace(destination)
        print(f'Downloaded: {destination}')


if __name__ == '__main__':
    main()
