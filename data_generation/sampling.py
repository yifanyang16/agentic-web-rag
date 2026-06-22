import json
import random

INPUT_FILE = "bioqa.jsonl"

SIZES = [50, 200, 500, 1000]


def read_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def write_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for x in data:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def main():
    data = read_jsonl(INPUT_FILE)

    print(f"Loaded {len(data)} samples")

    random.seed(513)

    for n in SIZES:
        if n > len(data):
            print(f"Skip {n}, not enough data")
            continue

        sampled = random.sample(data, n)

        out_file = f"bioqa_{n}.jsonl"
        write_jsonl(out_file, sampled)

        print(f"Saved {n} samples → {out_file}")


if __name__ == "__main__":
    main()
