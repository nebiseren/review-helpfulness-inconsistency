"""Assemble this repository's downloaded files in a local working directory."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

ROOT = Path(__file__).resolve().parent
EXPECTED = {'analysis_derived.csv.gz': '48271f4a3459d1e05dd311e1bad22b2d144db5659fd48e89dd7e21cb80f2a836', 'source_matching_manifest.csv.gz': '645e0e7423a190db1ad56904333b27ba6dc7dee176b888617ce5493237e18308', 'development_split_manifest.csv.gz': '608727d76bca03b5324562b8f2916a333f757f17f2678bc85a17185ad400483f', 'text_model.joblib': '17b4192ebe21835e582e1e7dd7e90fa24e6d1ec6cb5563cdb8647a5405bccfae', 'analysis_outputs.zip': '941dd4914671ae57e99d80a9cfaff1b8ad24dddbc1d311222fee28f2fd0c19a6'}
DESTINATIONS = {'analysis_derived.csv.gz': 'results/analysis_derived.csv.gz', 'source_matching_manifest.csv.gz': 'results/source_matching_manifest.csv.gz', 'development_split_manifest.csv.gz': 'results/development_split_manifest.csv.gz', 'text_model.joblib': 'models/text_model.joblib'}

def main():
    for name, expected in EXPECTED.items():
        path = ROOT / name
        if not path.is_file():
            raise FileNotFoundError('Place the downloaded file next to this script: ' + name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('SHA-256 mismatch: ' + name)
    with zipfile.ZipFile(ROOT / 'analysis_outputs.zip') as archive:
        for info in archive.infolist():
            relative = Path(info.filename)
            if (relative.is_absolute() or '..' in relative.parts
                    or relative.parts[0] not in {'results', 'tables', 'figures', 'logs', 'reporting'}):
                raise ValueError('Unexpected archive member: ' + info.filename)
        archive.extractall(ROOT)
    for name, relative in DESTINATIONS.items():
        destination = ROOT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    print('Repository files assembled and SHA-256 checked.')
    print('Next: python code/03_verify_and_summarise.py --root .')

if __name__ == '__main__':
    main()
