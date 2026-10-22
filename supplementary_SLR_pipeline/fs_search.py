#!/usr/bin/env python3
"""Automated literature search for the FS-IoT-IDS systematic review.

Sources: OpenAlex and Semantic Scholar (free APIs, no login needed).
Output folder: search_exports/
    openalex.csv, semanticscholar.csv  raw records per source
    merged_records.csv                 deduplicated records with abstracts
    search_log.txt                     queries, dates, counts (for PRISMA)

Run:  python fs_search.py [OPENALEX_API_KEY]
"""
import csv
import datetime as dt
import os
import re
import sys
import time

import pandas as pd
import requests

OUT = "search_exports"
YEARS = (2018, 2025)
MAILTO = "usamanazir@graduate.utm.my"
OPENALEX_KEY = sys.argv[1] if len(sys.argv) > 1 else ""

OPENALEX_QUERY = (
    '("IoT" OR "Internet of Things" OR "IIoT") AND '
    '("intrusion detection" OR "attack detection" OR "anomaly detection") AND '
    '("feature selection" OR "feature reduction" OR "dimensionality reduction" OR "feature subset") AND '
    '("machine learning" OR "deep learning" OR "classifier" OR "classification")'
)
S2_QUERY = (
    '("IoT" | "Internet of Things" | "IIoT") + '
    '("intrusion detection" | "attack detection" | "anomaly detection") + '
    '("feature selection" | "feature reduction" | "dimensionality reduction" | "feature subset")'
)

session = requests.Session()
session.headers["User-Agent"] = f"FS-IoT-IDS-SLR (mailto:{MAILTO})"


def get(url, params, tries=8):
    for k in range(tries):
        try:
            r = session.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                wait = 5 * (k + 1)
                print(f"    server busy ({r.status_code}), waiting {wait}s")
                time.sleep(wait)
                continue
            print(f"    HTTP {r.status_code}: {r.text[:300]}")
            return None
        except requests.RequestException as e:
            print(f"    network error: {e}; retrying")
            time.sleep(5 * (k + 1))
    return None


def norm_doi(d):
    if not d or not isinstance(d, str):
        return ""
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", d.strip().lower())


def norm_title(t):
    return re.sub(r"[^a-z0-9]", "", str(t).lower())


def abstract_from_index(inv):
    if not inv:
        return ""
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


# --------------------------------------------------------------- OpenAlex
def search_openalex():
    print("Searching OpenAlex ...")
    url = "https://api.openalex.org/works"
    flt = (f"title_and_abstract.search:{OPENALEX_QUERY},"
           f"publication_year:{YEARS[0]}-{YEARS[1]},language:en,type:article|review")
    params = {"filter": flt, "per-page": 200, "page": 1, "mailto": MAILTO,
              "select": "id,doi,display_name,publication_year,type,primary_location,"
                        "authorships,abstract_inverted_index,open_access,cited_by_count"}
    if OPENALEX_KEY:
        params["api_key"] = OPENALEX_KEY
    rows, total = [], None
    first = get(url, params)
    if first is None:  # fallback: boolean query in the search parameter
        print("  retrying OpenAlex with the search parameter")
        params["filter"] = f"publication_year:{YEARS[0]}-{YEARS[1]},language:en,type:article|review"
        params["search"] = OPENALEX_QUERY
        first = get(url, params)
    js = first
    while True:
        if js is None:
            print("  OpenAlex search failed.")
            break
        if total is None:
            total = js["meta"]["count"]
            print(f"  OpenAlex hits: {total}")
        for w in js["results"]:
            loc = w.get("primary_location") or {}
            src = loc.get("source") or {}
            rows.append(dict(
                source_db="OpenAlex", openalex_id=w.get("id", ""), doi=norm_doi(w.get("doi")),
                title=w.get("display_name") or "", year=w.get("publication_year"),
                type=w.get("type", ""), venue=src.get("display_name", ""),
                venue_type=src.get("type", ""),
                authors="; ".join((a.get("author") or {}).get("display_name", "")
                                  for a in (w.get("authorships") or [])[:10]),
                abstract=abstract_from_index(w.get("abstract_inverted_index")),
                oa_url=(w.get("open_access") or {}).get("oa_url") or "",
                citations=w.get("cited_by_count", 0)))
        print(f"  retrieved {len(rows)}", end="\r")
        if not js["results"] or len(rows) >= total:
            break
        params["page"] += 1
        time.sleep(0.3)
        js = get(url, params)
        if js is None and not OPENALEX_KEY:
            print("\n  OpenAlex needs a free API key for more pages. Get it at "
                  "https://openalex.org/settings/api and run: python fs_search.py YOUR_KEY")
    print(f"\n  OpenAlex records retrieved: {len(rows)}")
    return rows, total


# -------------------------------------------------------- Semantic Scholar
def search_s2():
    print("Searching Semantic Scholar ...")
    url = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
    params = {"query": S2_QUERY, "year": f"{YEARS[0]}-{YEARS[1]}",
              "fields": "title,abstract,year,venue,publicationTypes,externalIds,authors,"
                        "openAccessPdf,citationCount"}
    rows, total = [], None
    while True:
        js = get(url, params)
        if js is None:
            print("  Semantic Scholar search failed.")
            break
        if total is None:
            total = js.get("total", 0)
            print(f"  Semantic Scholar hits: {total}")
        for p in js.get("data") or []:
            ext = p.get("externalIds") or {}
            pt = p.get("publicationTypes") or []
            rows.append(dict(
                source_db="SemanticScholar", openalex_id="", doi=norm_doi(ext.get("DOI")),
                title=p.get("title") or "", year=p.get("year"),
                type=";".join(pt), venue=p.get("venue") or "", venue_type="",
                authors="; ".join(a.get("name", "") for a in (p.get("authors") or [])[:10]),
                abstract=p.get("abstract") or "",
                oa_url=(p.get("openAccessPdf") or {}).get("url") or "",
                citations=p.get("citationCount") or 0))
        print(f"  retrieved {len(rows)}", end="\r")
        tok = js.get("token")
        if not tok:
            break
        params["token"] = tok
        time.sleep(1.2)
    print(f"\n  Semantic Scholar records retrieved: {len(rows)}")
    return rows, total


# ------------------------------------------------------------------ merge
def merge(frames):
    df = pd.concat(frames, ignore_index=True)
    df["title_key"] = df["title"].map(norm_title)
    df = df[df["title_key"].str.len() > 10]
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[df["year"].between(*YEARS)]
    before = len(df)
    df["has_abs"] = df["abstract"].fillna("").str.len() > 50
    df = df.sort_values(["has_abs", "source_db"], ascending=[False, True])
    keep, seen_doi, seen_title = [], set(), set()
    for _, r in df.iterrows():
        d, t = r["doi"], r["title_key"]
        if (d and d in seen_doi) or t in seen_title:
            continue
        if d:
            seen_doi.add(d)
        seen_title.add(t)
        keep.append(r)
    m = pd.DataFrame(keep).drop(columns=["title_key", "has_abs"])
    m = m.sort_values(["year", "title"]).reset_index(drop=True)
    m.insert(0, "rec_id", [f"R{i:04d}" for i in range(1, len(m) + 1)])
    return m, before - len(m)


def main():
    os.makedirs(OUT, exist_ok=True)
    started = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    oa_rows, oa_total = search_openalex()
    s2_rows, s2_total = search_s2()
    if not oa_rows and not s2_rows:
        sys.exit("Both searches failed. Check the internet connection and run again.")
    oa, s2 = pd.DataFrame(oa_rows), pd.DataFrame(s2_rows)
    oa.to_csv(os.path.join(OUT, "openalex.csv"), index=False, quoting=csv.QUOTE_ALL, encoding="utf-8-sig")
    s2.to_csv(os.path.join(OUT, "semanticscholar.csv"), index=False, quoting=csv.QUOTE_ALL, encoding="utf-8-sig")
    merged, dups = merge([x for x in (oa, s2) if len(x)])
    merged.to_csv(os.path.join(OUT, "merged_records.csv"), index=False, quoting=csv.QUOTE_ALL, encoding="utf-8-sig")
    log = [f"Search run: {started}",
           f"Years: {YEARS[0]}-{YEARS[1]}",
           f"OpenAlex query (title_and_abstract): {OPENALEX_QUERY}",
           f"OpenAlex hits: {oa_total}  retrieved: {len(oa)}",
           f"Semantic Scholar query (bulk): {S2_QUERY}",
           f"Semantic Scholar hits: {s2_total}  retrieved: {len(s2)}",
           f"Duplicates/out-of-range removed: {dups}",
           f"Unique records: {len(merged)}",
           f"Records with abstract: {(merged['abstract'].fillna('').str.len() > 50).sum()}"]
    with open(os.path.join(OUT, "search_log.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(log))
    print("\n".join(log))
    print(f"\nDone. Upload these files from the folder '{OUT}': merged_records.csv and search_log.txt")


if __name__ == "__main__":
    main()
