"""Package a built browser demo and already-frozen evaluation records.

Run from the repository root after downloading/exporting the browser artifact
and building browser/. Raw prediction records are optional (from a local run).
"""
import argparse
import hashlib
from pathlib import Path
import zipfile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--records', type=Path)
a = p.parse_args()
out = Path('artifacts')
out.mkdir(exist_ok=True)
dist = Path('browser/dist')
assert (dist / 'model-v3-q4/decision.weights').exists(), 'Build with the exported weights installed first.'
with zipfile.ZipFile(out / 'decision-lab-browser-demo.zip', 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for file in sorted(dist.rglob('*')):
        if file.is_file():
            z.write(file, str(file.relative_to(dist)))
    z.write('LICENSE', 'SOURCE_LICENSE')
    z.write('NOTICE.md', 'NOTICE.md')
    for file in sorted(Path('licenses/runtime').rglob('*')):
        if file.is_file():
            z.write(file, str(file))
    z.writestr('README.md', '''# Decision Lab browser experiment

Run `python3 -m http.server 8000 --bind 127.0.0.1` in this directory.
Open http://localhost:8000/sdk.html (loader) or http://localhost:8000/ (playground).
Use a WebGPU-capable browser with shader-f16. No prompt is sent to an AI server.

This is an experimental 350M decision model, not the Jev service. The current
workflow regression has only 59.6% field accuracy and 19.0% exact records.
Read https://github.com/khalilelghoul01/decision-lab for the method and limitations.

Original project code: MIT (SOURCE_LICENSE). Model artifacts retain the separate
LFM Open License inside model-v3-q4. Runtime notices are in licenses/runtime.
''')
if a.records:
    with zipfile.ZipFile(out / 'decision-lab-evaluation-records.zip', 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for pattern in ['test-*.jsonl', 'q4-test-*.jsonl', 'base-test-*.jsonl', 'v2-common-test.jsonl']:
            for file in sorted(a.records.glob(pattern)):
                z.write(file, file.name)
        z.write('docs/RESULTS.md', 'RESULTS.md')
        z.write('artifacts.json', 'artifacts.json')
        z.writestr('README.md', 'Per-example predictions from the frozen V3 evaluation.\nThese are dataset IDs, labels and model scores, not a fresh independent test.\nSee RESULTS.md and the source repository for task definitions and limitations.\n')
checks = []
for file in sorted(out.iterdir()):
    if file.suffix not in ('.zip', '.tgz'):
        continue
    if file.suffix == '.zip':
        with zipfile.ZipFile(file) as z:
            assert z.testzip() is None, file
    with file.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    checks.append(f'{digest}  {file.name}')
(out / 'SHA256SUMS').write_text('\n'.join(checks) + '\n')
print('\n'.join(checks))
