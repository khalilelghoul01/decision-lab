"""Make a held-out-only payload after training and checkpoint selection finish."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--root', default='runs/v3')
a = p.parse_args()
root = Path(a.root)
if not (root / 'training-complete.json').exists():
    raise SystemExit('Finish checkpoint selection before preparing final evaluation.')
out = root / 'evaluation-data.json'
if out.exists():
    raise SystemExit('Evaluation payload already exists; refusing to overwrite it.')
data = json.loads((root / 'data.json').read_text())
metadata = dict(data['metadata'])
metadata['full_dataset_sha256'] = (root / 'data.sha256').read_text().strip()
payload = {key: data[key] for key in ['calibration', 'test']}
payload['metadata'] = metadata
out.write_text(json.dumps(payload, ensure_ascii=False))
print(f'Prepared {out}: calibration={len(payload["calibration"])}, test={len(payload["test"])}')
