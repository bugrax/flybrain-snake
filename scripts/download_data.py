"""Download the three official MaleCNS v1.0 source tables for rebuilding."""
from pathlib import Path
from urllib.request import urlopen
import shutil
from flysnake.brain import DATA

BASE = 'https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/'
FILES = (
    'body-annotations-male-cns-v1.0-minconf-0.5.feather',
    'body-neurotransmitters-male-cns-v1.0.feather',
    'connectome-weights-male-cns-v1.0-minconf-0.5.feather',
)


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        target = DATA / filename
        if target.exists():
            print(f'Using existing {filename}')
            continue
        temporary = target.with_suffix('.part')
        try:
            print(f'Downloading {filename}', flush=True)
            with urlopen(BASE + filename, timeout=120) as response, temporary.open('wb') as out:
                shutil.copyfileobj(response, out, length=1024 * 1024)
                expected = response.headers.get('Content-Length')
            if expected is not None and temporary.stat().st_size != int(expected):
                raise IOError(f'Incomplete download: {filename}')
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    print('Ready. Rebuild with: uv run python -m flysnake.brain')


if __name__ == '__main__': main()
