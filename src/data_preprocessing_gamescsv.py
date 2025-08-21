# data_preprocessing_gamescsv.py
# Fast, robust cleaner for heterogeneous Steam-like CSV rows.

import csv
import html
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

# --- PATHS ---
SRC = Path(r"C:/Users/test_/Documents/GitHub/Advanced-Analytics-Group-2/data/raw data/games.csv")
OUT = Path(r"C:/Users/test_/Documents/GitHub/Advanced-Analytics-Group-2/data/processed data/games_clean.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

# --- HELPERS ---------------------------------------------------------------

def clean_null(x):
    r"""Treat '\\N', 'null', 'unset', '<unset>' and empty strings as missing."""
    if x is None:
        return None
    s = str(x).strip()
    if s.lower() in {"\\n", "null", "unset", "<unset>", ""}:
        return None
    return s

# quick ID validator (digits only)
ID_OK_RE = re.compile(r'^\d+$')

# -------------------- Language normalization --------------------

BR_RE              = re.compile(r'(?i)[<>\s]*/?\s*br\s*/?\s*[<>]?')
HTML_TAG_RE        = re.compile(r'<[^>]+>')
BBCODE_RE          = re.compile(r'\[/?[a-zA-Z]+]')
NOISE_PHRASES_RES  = [
    re.compile(p) for p in [
        r'(?i)\blanguages?\s+with\s+full\s+audio\s+support\b',
        r'(?i)\blangues?\s+avec\s+support\s+audio\s+complet\b',
        r'(?i)\btalen\s+met\s+volledige\s+audio-ondersteuning\b',
        r'(?i)\bj[eę]zyki?\s+z\s+pe[łl]nym\s+wsparciem\s+audio\b',
        r'(?i)\binterface\s*:\s*[^,;]+',
        r'(?i)\bfull\s*audio\s*:\s*[^,;]+',
        r'(?i)\bsubtitles?\s*:\s*[^,;]+',
    ]
]
SEP_MULTI_RE       = re.compile(r'[;/]+')
COMMA_TRIM_RE      = re.compile(r'\s*,\s*')
SPACE_MULTI_RE     = re.compile(r'\s+')
MULTI_COMMAS_RE    = re.compile(r'(,\s*){2,}')
PUNCT_TRIM_RE      = re.compile(r'^[\W_]+|[\W_]+$', re.UNICODE)

_CANON_PATTERNS_RES = [
    (re.compile(r'(?i)(spanish\s*-\s*spain|espagnol\s*-\s*espagne|español\s*-\s*españa|espanol\s*-\s*espana|spaans\s*-\s*spanje|hiszpa(?:ński|nski)\s*-\s*hiszpania)'), "Spanish - Spain"),
    (re.compile(r'(?i)(portuguese\s*-\s*brazil|portugu[eê]s\s*-\s*brasil|portugais\s*du\s*br[ée]sil|portugalski\s*brazylijski)'), "Portuguese - Brazil"),
    (re.compile(r'(?i)(portuguese\s*-\s*portugal|portugu[eê]s\s*-\s*portugal|portugais\s*-\s*portugal)'), "Portuguese - Portugal"),
    (re.compile(r'(?i)(simplified\s*chinese|chinois\s*simplifié|chino\s*simplificado|chinees\s*\(vereenvoudigd\)|chi[ńn]ski\s*uproszczony)'), "Chinese - Simplified"),
    (re.compile(r'(?i)(traditional\s*chinese|chinois\s*traditionnel|chino\s*tradicional|traditioneel\s*chinees|chi[ńn]ski\s*tradycyjny)'), "Chinese - Traditional"),
    (re.compile(r'(?i)(english|anglais|ingl[ée]s|ingles|englisch|inglese|angielski|engelsk|engels|ingilizce)'), "English"),
    (re.compile(r'(?i)(french|fran[çc]ais|francais|fransk|frans|francuski|francese)'), "French"),
    (re.compile(r'(?i)(german|allemand|deutsch|duits|niem(?:ie)?cki|tysk|deutsc?h)'), "German"),
    (re.compile(r'(?i)(italian|italien|italiano|italiaans|w[łl]oski|italiens?k?)'), "Italian"),
    (re.compile(r'(?i)(spanish|espagnol|espa[ñn]ol|spanisch|spaans|hiszpa(?:ński|nski))'), "Spanish"),
    (re.compile(r'(?i)(portuguese|portugu[eê]s|portugais|portugalski)'), "Portuguese"),
    (re.compile(r'(?i)(russian|russe|ruso|russo|rosyjski|russisch)'), "Russian"),
    (re.compile(r'(?i)(polish|polonais|polaco|polski|polnisch|polsk)'), "Polish"),
    (re.compile(r'(?i)(japanese|japonais|japon[ée]s|japans|japo(?:ński|nski))'), "Japanese"),
    (re.compile(r'(?i)(korean|cor[ée]en|coreano|korean(?:sk)?|korea(?:ński|nski))'), "Korean"),
    (re.compile(r'(?i)(dutch|n[ée]erlandais|holl[aä]ndisch|nederlands)'), "Dutch"),
    (re.compile(r'(?i)(czech|tch[eè]que|checo|tschechisch|czeski)'), "Czech"),
    (re.compile(r'(?i)(danish|danois|dan[ée]s|d[aä]nisch|du[ńn]ski|dansk)'), "Danish"),
    (re.compile(r'(?i)(finnish|finnois|fin[ée]s|finnisch|fi[ńn]ski)'), "Finnish"),
    (re.compile(r'(?i)(greek|grec|griego|griechisch|grecki)'), "Greek"),
    (re.compile(r'(?i)(hungarian|hongrois|h[úu]ngaro|ungarisch|w[ęe]gierski)'), "Hungarian"),
    (re.compile(r'(?i)(norwegian|norv[ée]gien|noruego|norwegisch|norweski)'), "Norwegian"),
    (re.compile(r'(?i)(swedish|su[ée]dois|sueco|schwedisch|szwedzki)'), "Swedish"),
    (re.compile(r'(?i)(thai|tha[íi]|tailand[eé]s|thail[aä]ndisch|tajs?ki)'), "Thai"),
    (re.compile(r'(?i)(turkish|turc|turco|t[üu]rkisch|turecki|t[üu]rk[çc]e)'), "Turkish"),
    (re.compile(r'(?i)(ukrainian|ukrainien|ucraniano|ukrainisch|ukrai[ńn]ski)'), "Ukrainian"),
    (re.compile(r'(?i)(romanian|roumain|rumano|rum[aä]nisch|rumu[ńn]ski)'), "Romanian"),
    (re.compile(r'(?i)(arabic|arabe|[áa]rabe|arabisch|arabski)'), "Arabic"),
    (re.compile(r'(?i)(bulgarian|bulgare|b[úu]lgaro|bulgarisch|bu[łl]garski)'), "Bulgarian"),
    (re.compile(r'(?i)(chinese|chinois|chino|chinees|chi[ńn]ski)'), "Chinese"),
]

@lru_cache(maxsize=50000)
def _strip_markup_and_split_cached(text: str) -> tuple[str, ...]:
    s = text.replace("\r", ",").replace("\n", ",").replace("\t", " ")
    s = s.replace("\u21B5", ",")
    s = BR_RE.sub(",", s)
    s = s.replace("<", ",").replace(">", ",").replace("|", ",").replace("\\", " ")
    s = s.replace("&", " ")
    s = html.unescape(s)
    s = HTML_TAG_RE.sub("", s)
    s = BBCODE_RE.sub("", s)
    for rx in NOISE_PHRASES_RES:
        s = rx.sub("", s)
    s = SEP_MULTI_RE.sub(",", s)
    s = COMMA_TRIM_RE.sub(",", s)
    s = SPACE_MULTI_RE.sub(" ", s).strip()
    s = MULTI_COMMAS_RE.sub(",", s)
    parts = [t.strip() for t in s.split(",") if t.strip()]
    return tuple(parts)

@lru_cache(maxsize=100000)
def _canon_token(t: str) -> str | None:
    t = PUNCT_TRIM_RE.sub("", t)
    if not t:
        return None
    for rx, name in _CANON_PATTERNS_RES:
        if rx.search(t):
            return name
    low = t.casefold()
    if any(ch.isalpha() for ch in t) and low not in {"br", "amp", "lt", "gt"}:
        return t
    return None

def clean_languages(val: str) -> str | None:
    if val is None:
        return None
    tokens = _strip_markup_and_split_cached(str(val))
    seen, out = set(), []
    for tok in tokens:
        canon = _canon_token(tok)
        if canon and canon not in seen:
            seen.add(canon)
            out.append(canon)
    return ", ".join(out) if out else None

# -------------------- Price parsing -----------------------------

FINAL_RE   = re.compile(r'["\\\']?final["\\\']?\s*:\s*(-?\d+)')
CURR_RE    = re.compile(r'["\\\']?currency["\\\']?\s*:\s*"([^"]*)"')
EURO_FF_RE = re.compile(r'["\\\']?final_formatted["\\\']?\s*:\s*"?\s*([0-9]+[,.][0-9]{2})\s*€')

def parse_price_blob_final_currency(raw: str):
    if not raw:
        return None, None, None
    s = raw.strip()
    b, e = s.find("{"), s.rfind("}")
    if b == -1 or e == -1 or e <= b:
        return None, None, None
    body = s[b+1:e].replace('\\"', '"').replace('""', '"')
    m_final = FINAL_RE.search(body)
    m_curr  = CURR_RE.search(body)
    final_int = int(m_final.group(1)) if m_final else None
    currency  = m_curr.group(1) if m_curr else None
    m_ff = EURO_FF_RE.search(body)
    final_formatted = None
    if m_ff:
        ff = m_ff.group(1).replace(",", ".")
        try:
            final_formatted = float(ff)
            if currency is None:
                currency = "EUR"
        except ValueError:
            pass
    return final_int, currency, final_formatted

def extract_price_blob(tokens, start_idx):
    n = len(tokens)
    i = start_idx
    while i < n and "{" not in (tokens[i] or ""):
        i += 1
    if i >= n:
        return None, start_idx
    buf, open_braces, started = [], 0, False
    while i < n:
        t = tokens[i] or ""
        buf.append(t)
        lb = t.count("{"); rb = t.count("}")
        if lb:
            started = True
        open_braces += lb - rb
        i += 1
        if started and open_braces <= 0:
            break
    return ",".join(buf), i

# --- MAIN ------------------------------------------------------------------

def main():
    rows = []
    with SRC.open(newline="", encoding="utf-8") as f:
        rdr = csv.reader(f)
        next(rdr, None)  # skip header

        for r in rdr:
            if not r:
                continue

            # skip stray header lines
            if r[0].strip().lower() == "app_id":
                continue

            # Repair: entire row in one cell -> reparse
            if len(r) == 1 and r[0]:
                inner = r[0].strip()
                if inner.startswith('"') and inner.endswith('"'):
                    inner = inner[1:-1]
                r = next(csv.reader([inner]))
                if len(r) == 1:
                    r = next(csv.reader([inner.replace('""', '"')]))

            # ---- ID sanity check (skip broken row) ----
            raw_id = r[0] if len(r) > 0 else None
            if raw_id is None or not ID_OK_RE.fullmatch(raw_id.strip()):
                # first cell contains commas / non-digits (e.g., full line)
                continue

            # Base fields
            app_id       = clean_null(raw_id)
            name         = clean_null(r[1]) if len(r) > 1 else None
            release_date = clean_null(r[2]) if len(r) > 2 else None

            # Price detection
            raw_blob, next_i = extract_price_blob(r, 4)

            price = None
            if raw_blob:
                final_cents, currency, final_formatted = parse_price_blob_final_currency(raw_blob)
                if isinstance(final_cents, int):
                    price = final_cents / 100.0
                elif final_formatted is not None:
                    price = final_formatted
                languages_raw = clean_null(r[next_i])     if next_i     < len(r) else None
                type_raw      = clean_null(r[next_i + 1]) if next_i + 1 < len(r) else None
            else:
                c5 = clean_null(r[4]) if len(r) > 4 else None
                c6 = clean_null(r[5]) if len(r) > 5 else None
                c7 = clean_null(r[6]) if len(r) > 6 else None
                c8 = clean_null(r[7]) if len(r) > 7 else None

                if c7 and c8 and re.fullmatch(r'\d+(?:\.\d+)?', c7) and re.fullmatch(r'[A-Z]{3}', c8):
                    languages_raw = c5; type_raw = c6
                    try:
                        price = float(c7)
                    except Exception:
                        price = None
                    currency = c8
                else:
                    languages_raw = c6; type_raw = c7
                    price = None; currency = None

            languages = clean_languages(languages_raw)

            rows.append({
                "app_id": app_id,
                "name": name,
                "release_date": release_date,
                "languages": languages,
                "type": type_raw,
                "price_overview.final": price,
                "price_overview.currency": currency,
            })

    df = pd.DataFrame(rows)

    # --- Final guard: drop rows where first column isn't a clean numeric ID ---
    app_id_str = df["app_id"].astype("string").str.strip()

    bad_commas = app_id_str.str.contains(",", regex=False, na=False)
    bad_nondigits = ~app_id_str.str.fullmatch(r"\d+", na=False)

    df = df.loc[~(bad_commas | bad_nondigits)].copy()

    # ---- Final coercions & drop any remaining bad ids (belt & suspenders) ----
    df["app_id"] = pd.to_numeric(df["app_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["app_id"]).reset_index(drop=True)

    dt = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_date"] = dt.dt.strftime("%Y-%m-%d").astype("string")

    df["price_overview.final"] = (
        pd.to_numeric(df["price_overview.final"], errors="coerce")
        .fillna(0.0)
        .astype(float)
    )

    df["is_free"] = df["price_overview.final"].eq(0.0)
    df.loc[df["is_free"], "price_overview.currency"] = None

    df[[
        "app_id", "name", "release_date", "is_free",
        "languages", "type",
        "price_overview.final", "price_overview.currency",
    ]].to_csv(OUT, index=False, encoding="utf-8")

if __name__ == "__main__":
    main()
