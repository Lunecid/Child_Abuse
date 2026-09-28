"""Verify the frozen release files and reject restricted data in the release."""
from pathlib import Path
import csv
import hashlib
import json
import re

ROOT=Path(__file__).resolve().parents[1]
EXCLUDED={'.git','__pycache__','generated','.venv'}
BLOCKED_COLUMNS={'doc_id','record_id','source_file','tfidf_text','counseling_text',
                 'participant_id','child_id','counselor_id','transcript','utterance'}
SECRET=re.compile(r'gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|sk-[A-Za-z0-9]{40,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')


def release_files():
    return sorted(p for p in ROOT.rglob('*') if p.is_file() and not any(x in EXCLUDED for x in p.relative_to(ROOT).parts))


def main():
    manifest=json.loads((ROOT/'MANIFEST.json').read_text())
    files=release_files()
    actual={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files if p.name!='MANIFEST.json'}
    assert actual==manifest['files'],'Release contents differ from the frozen manifest'
    csvs=0
    for p in files:
        assert p.suffix not in {'.zip','.jsonl','.xlsx','.pkl','.pt','.pth','.hwp','.docx','.pdf'},p.name
        if p.suffix=='.csv':
            with p.open(encoding='utf-8-sig',newline='') as f:
                rows=list(csv.reader(f))
            assert not BLOCKED_COLUMNS.intersection(x.lower() for x in rows[0]),p.name
            assert len(rows)<1000,(p.name,'Unexpected large table: review record-level disclosure')
            csvs+=1
        if p.suffix in {'.py','.md','.tex','.csv','.json','.txt','.cff'}:
            text=p.read_text(encoding='utf-8-sig')
            assert ('/'+'Users'+'/') not in text,(p.name,'Personal local path')
            assert not SECRET.search(text),(p.name,'Credential-shaped text')
    print(f'Verified {len(actual)} frozen files and {csvs} aggregate CSVs; no restricted table columns, personal paths or credential patterns found.')


if __name__=='__main__':main()
