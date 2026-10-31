"""Recover matching review bodies from the provider's original category JSONL files.

No download or scraping is performed. Download the six category review files
from the provider first. Matching requires category, parent_asin and an exact
SHA-256 of the UTF-8 review body. The output contains third-party review text
and must be handled as a private research input.
"""
from pathlib import Path
import argparse,json,gzip,csv,hashlib
import pandas as pd

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--source-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();manifest=pd.read_csv(a.manifest,keep_default_na=False)
    wanted={}
    for x in manifest.itertuples(index=False):wanted.setdefault((x.category,x.parent_asin,x.text_sha256),[]).append((x.record_type,int(x.record_id)))
    total=len(manifest);found=0;a.out.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(a.out,'wt',encoding='utf-8',newline='') as out:
        writer=csv.writer(out);writer.writerow(['record_type','record_id','text'])
        for category in sorted(manifest.category.unique()):
            file=a.source_dir/(category+'.jsonl.gz');op=gzip.open
            if not file.exists():file=a.source_dir/(category+'.jsonl');op=open
            if not file.exists():raise FileNotFoundError('Missing original category file: '+category)
            with op(file,'rt',encoding='utf-8') as stream:
                for line in stream:
                    x=json.loads(line);body=x.get('text')
                    if not isinstance(body,str):continue
                    key=(category,str(x.get('parent_asin','')),hashlib.sha256(body.encode('utf-8')).hexdigest())
                    matched=wanted.pop(key,None)
                    if matched:
                        for typ,idx in matched:writer.writerow([typ,idx,body])
                        found+=len(matched)
            print(category,'matched records',found,'of',total,flush=True)
    # Each matched key may correspond to several retained review rows with the
    # same text. Report record completeness; do not substitute similar text.
    remaining=sum(len(v) for v in wanted.values())
    if remaining:raise RuntimeError(f'{remaining} records remain unmatched. Do not use an incomplete reconstruction.')
    print('All source texts matched. Output is a private text file.',flush=True)

if __name__=='__main__':main()
