"""Prepare the training corpus.

This repo does NOT redistribute the LotR screenplay (it's copyrighted). Drop your
own raw text file at data/raw/lotr_trilogy_movie_script.txt and run:

    python data/prepare.py

It writes a lightly-cleaned copy to data/processed/lotr.txt, which is what
training reads. Any plain-text corpus works here -- swap in Shakespeare, your own
writing, anything -- the pipeline is corpus-agnostic.
"""

import os

RAW = "data/raw/lotr_trilogy_movie_script.txt"
OUT = "data/processed/lotr.txt"


def main():
    if not os.path.exists(RAW):
        raise FileNotFoundError(
            f"Expected a raw corpus at {RAW}.\n"
            "Add a .txt file there and re-run. Any plain-text file works."
        )
    with open(RAW, encoding="utf-8") as f:
        text = f.read()

    # light cleaning: normalise line endings, strip trailing whitespace
    text = text.replace("\r\n", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {len(text):,} chars -> {OUT}")


if __name__ == "__main__":
    main()
