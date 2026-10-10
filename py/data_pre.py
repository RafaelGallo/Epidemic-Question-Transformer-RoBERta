import os
import sys
import pandas as pd
import xml.etree.ElementTree as ET

shard_id = int(sys.argv[1])
n_shards = int(sys.argv[2])
paths_file = sys.argv[3]
out_dir = sys.argv[4]

with open(paths_file, "r", encoding="utf-8") as f:
    all_paths = [line.strip() for line in f if line.strip()]

rows = []
n_failed = 0
for xml_path in all_paths[shard_id::n_shards]:
    # parse and skip unreadable files / ler e pular arquivos ilegíveis
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        n_failed += 1
        continue

    title = ""
    abstract = ""
    doi = ""
    subject = ""
    for element in root.iter():
        if not isinstance(element.tag, str):
            continue
        tag = element.tag.split("}")[-1]
        if tag == "article-title" and not title:
            title = " ".join("".join(element.itertext()).split())
        elif tag == "abstract" and not abstract:
            abstract = " ".join("".join(element.itertext()).split())
        elif tag == "article-id" and not doi and element.attrib.get("pub-id-type") == "doi":
            doi = (element.text or "").strip()
        elif tag == "subj-group" and not subject and element.attrib.get("subj-group-type") == "hwp-journal-coll":
            # research area lives only in this subj-group type / a área de pesquisa fica só neste tipo de subj-group
            for child in element.iter():
                if isinstance(child.tag, str) and child.tag.split("}")[-1] == "subject":
                    subject = (child.text or "").strip()
                    break

    rows.append(
        {
            "article_id": os.path.splitext(os.path.basename(xml_path))[0],
            "title": title,
            "abstract": abstract,
            "subject": subject,
            "doi": doi,
        }
    )

pd.DataFrame(rows).to_parquet(os.path.join(out_dir, f"shard_{shard_id}.parquet"), index=False)
with open(os.path.join(out_dir, f"shard_{shard_id}.failed"), "w") as f:
    f.write(str(n_failed))