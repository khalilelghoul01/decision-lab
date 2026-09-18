"""Download a pinned research release and verify its checksum before extraction."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from urllib.request import Request, urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def extract_checked(archive, destination, prefix):
    destination = Path(destination).resolve()
    target = destination / prefix
    if target.exists():
        raise FileExistsError(f'{target} already exists. Move it aside before downloading a different version.')
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            path = Path(item.filename)
            if path.is_absolute() or '..' in path.parts or not item.filename.startswith(prefix + '/'):
                raise ValueError(f'Unexpected archive entry: {item.filename}')
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Symlink entries are not supported.')
        bad = z.testzip()
        if bad:
            raise ValueError(f'Archive CRC check failed: {bad}')
        # Validate the complete archive before writing any destination files.
        destination.mkdir(parents=True, exist_ok=True)
        z.extractall(destination)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native', action='store_true', help='Download the native adapter/head instead of browser weights.')
    p.add_argument('--archive', type=Path, help='Use an already-downloaded release ZIP (still verifies SHA256).')
    args = p.parse_args()
    release = json.loads((ROOT / 'scripts/artifacts.json').read_text())
    artifact = release['native' if args.native else 'browser']
    destination = ROOT if args.native else ROOT / 'browser/public'
    if (destination / artifact['prefix']).exists():
        raise SystemExit('Model directory already exists; refusing to overwrite it.')
    with tempfile.TemporaryDirectory(prefix='decision-lab-download-') as tmp:
        path = args.archive or Path(tmp) / artifact['filename']
        if not args.archive:
            request = Request(artifact['url'], headers={'User-Agent': 'Decision-Lab-model-loader/0.1'})
            with urlopen(request, timeout=120) as response, path.open('wb') as out:
                shutil.copyfileobj(response, out, length=1024 * 1024)
        with path.open('rb') as source:
            digest = hashlib.file_digest(source, 'sha256').hexdigest()
        if digest != artifact['sha256']:
            raise SystemExit('Checksum mismatch; the model has not been extracted.')
        extract_checked(path, destination, artifact['prefix'])
    print(f'Verified and installed {destination / artifact["prefix"]}')


if __name__ == '__main__':
    main()
