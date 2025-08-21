# data_preprocessing_gamescsv.py
"""
What this does:
1) Reads a raw CSV with inconsistent quoting and shifted columns.
2) Extracts price (price_overview.final) & currency either from a
   broken JSON-ish blob or from plain columns; missing -> 0.00.
3) Derives is_free (price == 0.0) and drops currency when free.
4) Cleans the languages field (strips HTML/BBCode/notes; removes
   tokens like 'strong', 'b', etc.; maps locale variants to English).
5) Normalizes release_date to YYYY-MM-DD and app_id to Int64.
6) Prints a short summary and writes a compact, merge-friendly CSV.
"""

import csv
import html
import re
from functools import lru_cache
from pathlib import Path
import pandas as pd

#-------------------- File paths (adjust if needed) --------------------
input_file = r"C:\Users\baugu\Dokumentai\GitHub\Advanced-Analytics-Group-2\data\raw data\steam-insights-main\games\games.csv"
output_file = r"C:\Users\baugu\Dokumentai\GitHub\Advanced-Analytics-Group-2\data\processed data\games_cleaned.csv"
Path(output_file).parent.mkdir(parents=True, exist_ok=True)

def clean_null(x):
    #Treat '\N', 'null', 'unset', '<unset>' and empty strings as missing.
    if x is None:
        return None
    s = str(x).strip()
    return None if s.lower() in {"\\n", "null", "unset", "<unset>", ""} else s

#-------------------- Language normalization --------------------
BR_RE       = re.compile(r"(?i)[<>\s]*/?\s*br\s*/?\s*[<>]?")
HTML_TAG_RE = re.compile(r"<[^>]+>")
BBCODE_RE   = re.compile(r"\[/?[a-zA-Z]+]")

NOISE_RES   = [re.compile(p) for p in (
    r"(?i)\blanguages?\s+with\s+full\s+audio\s+support\b",
    r"(?i)\blangues?\s+avec\s+support\s+audio\s+complet\b",
    r"(?i)\btalen\s+met\s+volledige\s+audio-ondersteuning\b",
    r"(?i)\bj[eę]zyki?\s+z\s+pe[łl]nym\s+wsparciem\s+audio\b",
    r"(?i)\binterface\s*:\s*[^,;]+", r"(?i)\bfull\s*audio\s*:\s*[^,;]+",
    r"(?i)\bsubtitles?\s*:\s*[^,;]+",
)]

SEP_MULTI_RE    = re.compile(r"[;/]+")
COMMA_TRIM_RE   = re.compile(r"\s*,\s*")
SPACE_MULTI_RE  = re.compile(r"\s+")
MULTI_COMMAS_RE = re.compile(r"(,\s*){2,}")
PUNCT_TRIM_RE   = re.compile(r"^[\W_]+|[\W_]+$", re.UNICODE)

_CANON_PATTERNS = [
    (r"(?i)(spanish\s*-\s*spain|espagnol\s*-\s*espagne|español\s*-\s*españa|espanol\s*-\s*espana|spaans\s*-\s*spanje|hiszpa(?:ński|nski)\s*-\s*hiszpania)", "Spanish - Spain"),
    (r"(?i)(portuguese\s*-\s*brazil|portugu[eê]s\s*-\s*brasil|portugais\s*du\s*br[ée]sil|portugalski\s*brazylijski)", "Portuguese - Brazil"),
    (r"(?i)(portuguese\s*-\s*portugal|portugu[eê]s\s*-\s*portugal|portugais\s*-\s*portugal)", "Portuguese - Portugal"),
    (r"(?i)(simplified\s*chinese|chinois\s*simplifi[é]|chino\s*simplificado|chinees\s*\(vereenvoudigd\)|chi(?:ń|n)ski\s*uproszczony)", "Chinese - Simplified"),
    (r"(?i)(traditional\s*chinese|chinois\s*traditionnel|chino\s*tradicional|traditioneel\s*chinees|chi(?:ń|n)ski\s*tradycyjny)", "Chinese - Traditional"),
    (r"(?i)(english|anglais|ingl[ée]s|ingles|englisch|inglese|angielski|engelsk|engels|ingilizce)", "English"),
    (r"(?i)(french|fran[çc]ais|francais|fransk|frans|francuski|francese)", "French"),
    (r"(?i)(german|allemand|deutsch|duits|niem(?:ie)?cki|tysk|deutsc?h)", "German"),
    (r"(?i)(italian|italien|italiano|italiaans|w(?:ł|l)oski|italiens?k?)", "Italian"),
    (r"(?i)(spanish|espagnol|espa[ñn]ol|spanisch|spaans|hiszpa(?:ński|nski))", "Spanish"),
    (r"(?i)(portuguese|portugu[eê]s|portugais|portugalski)", "Portuguese"),
    (r"(?i)(russian|russe|ruso|russo|rosyjski|russisch)", "Russian"),
    (r"(?i)(polish|polonais|polaco|polski|polnisch|polsk)", "Polish"),
    (r"(?i)(japanese|japonais|japon[ée]s|japans|japo(?:ński|nski))", "Japanese"),
    (r"(?i)(korean|cor[ée]en|coreano|korean(?:sk)?|korea(?:ński|nski))", "Korean"),
    (r"(?i)(dutch|n[ée]erlandais|holl[aä]ndisch|nederlands)", "Dutch"),
    (r"(?i)(czech|tch[eè]que|checo|tschechisch|czeski)", "Czech"),
]
_CANON_RES = [(re.compile(p), name) for p, name in _CANON_PATTERNS]

# extra stop tokens to drop if they survive markup stripping
_STOP_TOKENS = {"br", "amp", "lt", "gt", "strong", "/strong", "b", "/b", "i", "/i", "u", "/u"}

@lru_cache(maxsize=50000)
def _strip_and_split(text: str) -> tuple[str, ...]:
    s = text.replace("\r", ",").replace("\n", ",").replace("\t", " ").replace("\u21B5", ",")
    s = BR_RE.sub(",", s)
    s = s.replace("<", ",").replace(">", ",").replace("|", ",").replace("\\", " ")
    s = html.unescape(s).replace("&", " ")
    s = HTML_TAG_RE.sub("", s)
    s = BBCODE_RE.sub("", s)
    for rx in NOISE_RES:
        s = rx.sub("", s)
    s = SEP_MULTI_RE.sub(",", s)
    s = COMMA_TRIM_RE.sub(",", s)
    s = SPACE_MULTI_RE.sub(" ", s).strip()
    s = MULTI_COMMAS_RE.sub(",", s)
    return tuple(t for t in (p.strip() for p in s.split(",")) if t)

@lru_cache(maxsize=100000)
def _canon_token(tok: str) -> str | None:
    tok = PUNCT_TRIM_RE.sub("", tok)
    if not tok:
        return None
    for rx, name in _CANON_RES:
        if rx.search(tok):
            return name
    low = tok.casefold()
    return tok if any(c.isalpha() for c in tok) and low not in _STOP_TOKENS else None

def clean_languages(val: str | None) -> str | None:
    if val is None:
        return None
    seen, out = set(), []
    for t in _strip_and_split(str(val)):
        canon = _canon_token(t)
        if canon and canon not in seen:
            seen.add(canon); out.append(canon)
    return ", ".join(out) if out else None


# -------------------- Price parsing --------------------
FINAL_RE   = re.compile(r'["\\\']?final["\\\']?\s*:\s*(-?\d+)')
CURR_RE    = re.compile(r'["\\\']?currency["\\\']?\s*:\s*"([^"]*)"')
EURO_FF_RE = re.compile(r'["\\\']?final_formatted["\\\']?\s*:\s*"?\s*([0-9]+[,.][0-9]{2})\s*€')

def parse_price_blob(raw: str):
    """Return (final_cents:int|None, currency:str|None, final_formatted:float|None)."""
    if not raw: return None, None, None
    s = raw.strip(); b, e = s.find("{"), s.rfind("}")
    if b == -1 or e == -1 or e <= b: return None, None, None
    body = s[b+1:e].replace('\\"', '"').replace('""', '"')
    m_final = FINAL_RE.search(body); m_curr = CURR_RE.search(body); m_ff = EURO_FF_RE.search(body)
    cents = int(m_final.group(1)) if m_final else None
    curr  = m_curr.group(1) if m_curr else None
    ff    = None
    if m_ff:
        try:
            ff = float(m_ff.group(1).replace(",", "."))
            if curr is None: curr = "EUR"
        except ValueError:
            pass
    return cents, curr, ff

def extract_price_blob(tokens, start_idx):
    #Scan tokens to collect a balanced {...} block starting at start_idx.
    n, i = len(tokens), start_idx
    while i < n and "{" not in (tokens[i] or ""): i += 1
    if i >= n: return None, start_idx
    buf, opened, started = [], 0, False
    while i < n:
        t = tokens[i] or ""; buf.append(t)
        opened += t.count("{") - t.count("}")
        if t.count("{"): started = True
        i += 1
        if started and opened <= 0: break
    return ",".join(buf), i

# -------------------- Main --------------------
def main():
    #Stream raw CSV, repair rows, extract price, clean languages, and save.
    rows, scanned = [], 0

    with open(input_file, newline="", encoding="utf-8") as f:
        rdr = csv.reader(f)
        next(rdr, None)  # header

        for r in rdr:
            if not r:
                continue
            scanned += 1

            # repair fully-quoted whole line
            if len(r) == 1 and r[0]:
                inner = r[0].strip()
                if inner.startswith('"') and inner.endswith('"'):
                    inner = inner[1:-1]
                r = next(csv.reader([inner]))
                if len(r) == 1:
                    r = next(csv.reader([inner.replace('""', '"')]))

            app_id       = clean_null(r[0]) if len(r) > 0 else None
            name         = clean_null(r[1]) if len(r) > 1 else None
            release_date = clean_null(r[2]) if len(r) > 2 else None

            raw_blob, next_i = extract_price_blob(r, 4)

            if raw_blob:
                cents, currency, ff = parse_price_blob(raw_blob)
                price = (cents / 100.0) if isinstance(cents, int) else (ff if ff is not None else None)
                languages_raw = clean_null(r[next_i])     if next_i     < len(r) else None
                type_raw      = clean_null(r[next_i + 1]) if next_i + 1 < len(r) else None
            else:
                c5 = clean_null(r[4]) if len(r) > 4 else None
                c6 = clean_null(r[5]) if len(r) > 5 else None
                c7 = clean_null(r[6]) if len(r) > 6 else None
                c8 = clean_null(r[7]) if len(r) > 7 else None
                if c7 and c8 and re.fullmatch(r"\d+(?:\.\d+)?", c7) and re.fullmatch(r"[A-Z]{3}", c8):
                    languages_raw, type_raw = c5, c6
                    try: price = float(c7)
                    except Exception: price = None
                    currency = c8
                else:
                    languages_raw, type_raw = c6, c7
                    price, currency = None, None

            rows.append({
                "app_id": app_id,
                "name": name,
                "release_date": release_date,
                "languages": clean_languages(languages_raw),
                "type": type_raw,
                "price_overview.final": price,
                "price_overview.currency": currency,
            })

    df = pd.DataFrame(rows)

    # Normalize types
    df["app_id"] = pd.to_numeric(df["app_id"], errors="coerce").astype("Int64")  # keep NA if unparseable (no dropping)
    dt = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_date"] = dt.dt.strftime("%Y-%m-%d").astype("string")

    df["price_overview.final"] = (
        pd.to_numeric(df["price_overview.final"], errors="coerce").fillna(0.0).astype(float)
    )
    df["is_free"] = df["price_overview.final"].eq(0.0)
    df.loc[df["is_free"], "price_overview.currency"] = None

    # Save
    df[[
        "app_id","name","release_date","is_free",
        "languages","type",
        "price_overview.final","price_overview.currency",
    ]].to_csv(output_file, index=False, encoding="utf-8")

    # Summary
    print(f"Rows scanned: {scanned:,}")
    print(f"Final rows saved: {len(df):,}")
    print(f"Cleaned data saved to: {output_file}")

if __name__ == "__main__":
    main()
