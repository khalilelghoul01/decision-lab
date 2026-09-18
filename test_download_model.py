import importlib.util
import zipfile
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('download_model', Path(__file__).parent / 'scripts/download_model.py')
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


def test_extract_rejects_traversal_before_writing(tmp_path):
    archive = tmp_path / 'bad.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('model/ok.json', '{}')
        z.writestr('model/../../outside', 'bad')
    with pytest.raises(ValueError):
        download.extract_checked(archive, tmp_path / 'out', 'model')
    assert not (tmp_path / 'out').exists()


def test_extract_preserves_existing_model(tmp_path):
    archive = tmp_path / 'good.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('model/manifest.json', '{}')
    destination = tmp_path / 'out'
    download.extract_checked(archive, destination, 'model')
    with pytest.raises(FileExistsError):
        download.extract_checked(archive, destination, 'model')
