import re
import json
import random
from collections import defaultdict

PATTERNS = [
    r"according to (a |the |this )?(tweet|recent|passage|cited|analysis|text|model|article|study|paper|recent|abstract|excerpt|context|student video|interview|paragraph|document|research|authors?|content|reading|findings|data|source|2024 study|2007 study|new study|UCLA study|given|recent|latest study|provided|psychologists|simulation|one theory|resilience)",
    r"based on (the |this )?(given data|citation|passage|text|article|study|paper|abstract|excerpt|provided|description|above|information|reading|content|findings|data|source|figure)",
    r"as (mentioned|stated|described|noted|discussed|explained|reported|shown|presented|outlined|summarized|cited|clarified|explored)( in| above| below| earlier)?( the| this)?( passage| text| article| study| paper| abstract| excerpt| context| document| content| reading|comments)?",
    r"the (passage|text|article|study|studies|paper|abstract|excerpt|context|document|research|group) (specialized|states?|says?|cited|mentions?|describes?|notes?|indicates?|suggests?|reports?|shows?|explains?|discusses?|highlights?|focuses?|investigates?)",
    r"(in|within|from|for) (the |this |his |her )(passage|text|main graph|article|paper|abstract|excerpt|document|reading|findings|decribed|research|content|figure|research group|researchers|study team|scientists at|research team|simulation study|assessment described|grant lab)",
    r"the authors? (state?s?|mention?s?|suggest?s?|report?s?|note?s?|describe?s?|indicate?s?)",
    r"\b[A-Z][a-z]+ (et al\.|and colleagues)\b",
    r"\b(research group|researchers|study team|scientists at|research team)\b",
    r"(as (shown|indicated) in|refer to|see) (Figure|Table|Graph|Box|Chart|Appendix)\s+\d+",
    r"(focus|goal|objective|purpose|main (idea|topic)) of (the |this )(text|passage|study|paper|article|reading|content)",
    r"\bthe (provided|given|above|following|given|cited) (passage|text|excerpt|context|paragraph|content|information|figure|research|study|studies)\b",
    r"\bthis (passage|article|study|text|excerpt|paper|research)\b",
    r"\bthe (text|textbook|article|new study|lab|department|author|professor)\b",
    r"\bthe above (passage|text|information|context)\b",
    r"et\s?al\.",
    r"the group's primary interest",
    r"the group's (primary )?(interest|focus|research|findings|results|study)",
    r"the researchers?' (interest|focus|goal)",
    r"this group of (authors|scientists|researchers)",
    r"the study('s)? group",
    r"the research team('s)? (goal|focus|interest|findings|results|study)",
    r"the (studies|study) (identified|cited|regard|compar|reveal|assessed|report|investigate|explore|focus|highlight|suggest|indicate|describe|note|mention)",
    r"Who is the author of",
    r"Which (resource|study|paper|studies provided|authors|text sections|research institute|document|two articles|company|study explores|nursing journal|research groups|research areas|research topics|scientific publication|student-made videos)",
    r"approach the study",
    r"(Which|What)( international)? organization",
    r"(ClinicalTrials\.gov|Pennell|Leticia Carvalho|Cleveland Clinic|Linus Pauling|Zhang|Francois Serra|researcher|Tony's|Partha Kasturi|some students|Dr\. Holt|\bFigure\s+\d+|wiki|panel)",
    r"recent (research|study|studies|discovery|discoveries|scientific discoveries)",
    r"(focus of the research|group focusing on)",
    r"(paragraph|section|page|chapter|line) \d+",
    r"(wikti|ISBN|DOI)",
    r"in the study (?!of\b)(mentioned|described|cited|suggested|identified|found|conducted|reported|shown|published)?",
    r"\bthe passage\b(?! of)",
]

compiled = [re.compile(p, re.IGNORECASE) for p in PATTERNS]


def has_outside_ref(text: str) -> tuple[bool, list[str]]:
    hits = []
    for pat in compiled:
        for m in pat.finditer(text):
            hits.append(m.group(0))
    return bool(hits), hits


def extract_text_fields(sample: dict) -> str:
    parts = []
    for key, val in sample.items():
        if isinstance(val, list):
            parts.extend([str(v) for v in val])
        elif isinstance(val, dict):
            parts.extend([str(v) for v in val.values()])
        elif val:
            parts.append(str(val))
    return " ".join(parts)


def load_hf_data(repo: str, n: int, seed: int):
    from datasets import load_dataset

    print(f"Loading {repo} from HuggingFace...")
    ds = load_dataset(repo, name="L", split="train")
    data = [dict(ds[i]) for i in range(len(ds))]
    if len(data) > n:
        random.seed(seed)
        data = random.sample(data, n)
    return data


def analyze_and_clean(samples: list[dict], tag: str):
    total = len(samples)
    for pat in compiled:
        count = sum(1 for s in samples if pat.search(extract_text_fields(s)))
        if count == len(samples):
            print(f"100% match: {pat.pattern}")
    matched_samples = []
    unmatched_samples = []

    for s in samples:
        text = extract_text_fields(s)
        hit, _ = has_outside_ref(text)
        if hit:
            matched_samples.append(s)
        else:
            unmatched_samples.append(s)

    matched_count = len(matched_samples)
    match_probability = (matched_count / total) if total > 0 else 0

    return {
        "tag": tag,
        "total": total,
        "matched": matched_count,
        "clean": len(unmatched_samples),
        "removal_probability": round(match_probability, 4),
        "unmatched_samples": unmatched_samples,
        "matched_samples": matched_samples,
    }


if __name__ == "__main__":
    SEED = 42
    N_LIMIT = 2500

    all_results = []

    try:
        hf_raw = load_hf_data("ingoziegler/CRAFT-BioQA", n=N_LIMIT, seed=SEED)
        hf_res = analyze_and_clean(hf_raw, "hf")
        all_results.append(hf_res)
    except Exception as e:
        print(f"HF Loading Error: {e}")

    for local_file, tag in [
        ("outputs_bio/clean_samples_bioqa_mc.jsonl", "bioqa_mc"),
    ]:
        try:
            with open(local_file, "r", encoding="utf-8") as f:
                raw = [json.loads(line) for line in f if line.strip()]
            if len(raw) > N_LIMIT:
                random.seed(SEED)
                raw = random.sample(raw, N_LIMIT)
            res = analyze_and_clean(raw, tag)
            all_results.append(res)
        except FileNotFoundError:
            continue

    print("\n" + "=" * 80)
    print(f"{'Source Tag':<20} | {'Total':>8} | {'Matched':>8} | {'Removal Prob':>15}")
    print("-" * 80)
    for r in all_results:
        print(
            f"{r['tag']:<20} | {r['total']:>8} | {r['matched']:>8} | {r['removal_probability']:>15.2%}"
        )

        with open(f"bioqa.jsonl", "w", encoding="utf-8") as f:
            for s in r["unmatched_samples"]:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        with open(f"removed_{r['tag']}.jsonl", "w", encoding="utf-8") as f:
            for s in r["matched_samples"]:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print("=" * 80)
