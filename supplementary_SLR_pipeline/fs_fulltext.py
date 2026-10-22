#!/usr/bin/env python3
"""Download open-access full texts for the screened studies and extract their text.

Needs include_list.csv in the same folder.
Run:  pip install pypdf
      python fs_fulltext.py
Output: fulltext/pdfs/*.pdf, fulltext/fulltext.jsonl.gz, fulltext/retrieval_log.csv
"""
import csv, gzip, json, os, re, time
from html import unescape
from urllib.parse import urljoin

import pandas as pd
import requests
from pypdf import PdfReader

EMAIL = "usamanazir@graduate.utm.my"
OUT = "fulltext"
PDF_DIR = os.path.join(OUT, "pdfs")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
s = requests.Session()
s.headers.update({"User-Agent": UA, "Accept": "text/html,application/pdf;q=0.9,*/*;q=0.8"})


def fetch(url):
    try:
        r = s.get(url, timeout=60, allow_redirects=True)
        return r
    except requests.RequestException:
        return None


def try_pdf(url, path, depth=0):
    if not url or depth > 2:
        return False
    r = fetch(url)
    if r is None or r.status_code != 200:
        return False
    if r.content[:5] == b"%PDF-":
        with open(path, "wb") as fh:
            fh.write(r.content)
        return True
    html = r.text if "html" in r.headers.get("Content-Type", "") else ""
    m = re.search(r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)', html, re.I) \
        or re.search(r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']citation_pdf_url', html, re.I)
    if m:
        return try_pdf(urljoin(r.url, unescape(m.group(1))), path, depth + 1)
    m = re.search(r'href=["\']([^"\']+\.pdf[^"\']*)["\']', html, re.I)
    if m:
        return try_pdf(urljoin(r.url, unescape(m.group(1))), path, depth + 1)
    return False


def unpaywall_urls(doi):
    if not isinstance(doi, str) or not doi:
        return []
    r = fetch(f"https://api.unpaywall.org/v2/{doi}?email={EMAIL}")
    if r is None or r.status_code != 200:
        return []
    js = r.json()
    urls = []
    for loc in [js.get("best_oa_location")] + (js.get("oa_locations") or []):
        if loc:
            urls += [loc.get("url_for_pdf"), loc.get("url")]
    return [u for u in dict.fromkeys(urls) if u]


def pdf_text(path, max_chars=150000):
    try:
        rd = PdfReader(path)
        txt = "\n".join((p.extract_text() or "") for p in rd.pages[:40])
        return re.sub(r"[ \t]+", " ", txt)[:max_chars]
    except Exception as e:
        return f"__EXTRACT_ERROR__ {e}"


def main():
    os.makedirs(PDF_DIR, exist_ok=True)
    df = pd.read_csv("include_list.csv")
    log, n_ok = [], 0
    out = gzip.open(os.path.join(OUT, "fulltext.jsonl.gz"), "wt", encoding="utf-8")
    for k, r in df.iterrows():
        rid, doi = r["rec_id"], r["doi"] if isinstance(r["doi"], str) else ""
        path = os.path.join(PDF_DIR, f"{rid}.pdf")
        status, src = "not retrieved", ""
        if os.path.exists(path) and os.path.getsize(path) > 10000:
            status, src = "retrieved", "existing file"
        else:
            cands = []
            if isinstance(r["oa_url"], str):
                cands.append(r["oa_url"])
            cands += unpaywall_urls(doi)
            if doi:
                cands.append(f"https://doi.org/{doi}")
            for u in dict.fromkeys(cands):
                if try_pdf(u, path):
                    status, src = "retrieved", u
                    break
                time.sleep(0.5)
        if status == "retrieved":
            txt = pdf_text(path)
            if txt.startswith("__EXTRACT_ERROR__") or len(txt) < 3000:
                status = "retrieved but unreadable"
            else:
                n_ok += 1
                out.write(json.dumps({"rec_id": rid, "doi": doi, "text": txt}) + "\n")
        log.append({"rec_id": rid, "doi": doi, "status": status, "source": src})
        print(f"{k + 1}/{len(df)}  {rid}  {status}   (readable so far: {n_ok})", flush=True)
    out.close()
    pd.DataFrame(log).to_csv(os.path.join(OUT, "retrieval_log.csv"), index=False)
    print(f"\nDone. Readable full texts: {n_ok} of {len(df)}")
    print("Upload: fulltext\\fulltext.jsonl.gz and fulltext\\retrieval_log.csv")


if __name__ == "__main__":
    main()
