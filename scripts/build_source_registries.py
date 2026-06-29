from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_DIR = ROOT / "data" / "source_registry"


LITERARY_AUTHORS = [
    ("Jane Austen", "approved", "Project Gutenberg", "public-domain prose; multiple novels"),
    ("Charles Dickens", "approved", "Project Gutenberg", "public-domain prose; large corpus"),
    ("George Eliot", "approved", "Project Gutenberg", "public-domain prose; large corpus"),
    ("Thomas Hardy", "approved", "Project Gutenberg", "public-domain prose; multiple novels"),
    ("Mark Twain", "approved", "Project Gutenberg", "public-domain prose; fiction and essays"),
    ("Henry James", "approved", "Project Gutenberg", "public-domain prose; large corpus"),
    ("Edith Wharton", "approved", "Project Gutenberg", "public-domain early prose; verify each work"),
    ("Herman Melville", "approved", "Project Gutenberg", "public-domain prose"),
    ("Nathaniel Hawthorne", "approved", "Project Gutenberg", "public-domain prose"),
    ("Edgar Allan Poe", "approved", "Project Gutenberg", "public-domain short fiction and essays"),
    ("Washington Irving", "approved", "Project Gutenberg", "public-domain sketches and essays"),
    ("Louisa May Alcott", "approved", "Project Gutenberg", "public-domain prose"),
    ("Kate Chopin", "approved", "Project Gutenberg", "public-domain prose; smaller corpus"),
    ("Willa Cather", "approved", "Project Gutenberg", "public-domain early prose; verify each work"),
    ("Jack London", "approved", "Project Gutenberg", "public-domain prose"),
    ("Stephen Crane", "approved", "Project Gutenberg", "public-domain prose"),
    ("Charlotte Bronte", "approved", "Project Gutenberg", "public-domain prose"),
    ("Emily Bronte", "approved", "Project Gutenberg", "public-domain prose; smaller corpus"),
    ("Anne Bronte", "approved", "Project Gutenberg", "public-domain prose; smaller corpus"),
    ("Mary Wollstonecraft Shelley", "approved", "Project Gutenberg", "public-domain prose"),
    ("Robert Louis Stevenson", "approved", "Project Gutenberg", "public-domain prose"),
    ("Joseph Conrad", "approved", "Project Gutenberg", "public-domain prose"),
    ("H. G. Wells", "approved", "Project Gutenberg", "public-domain early prose; verify each work"),
    ("Frances Hodgson Burnett", "approved", "Project Gutenberg", "public-domain prose"),
    ("L. M. Montgomery", "approved", "Project Gutenberg", "public-domain early prose; verify each work"),
    ("Elizabeth Cleghorn Gaskell", "approved", "Project Gutenberg", "public-domain prose"),
    ("Anthony Trollope", "approved", "Project Gutenberg", "public-domain prose; large corpus"),
    ("Wilkie Collins", "approved", "Project Gutenberg", "public-domain prose"),
    ("Arthur Conan Doyle", "approved", "Project Gutenberg", "public-domain prose; verify each work"),
    ("George MacDonald", "approved", "Project Gutenberg", "public-domain prose"),
]


RHETORICAL_SPEAKERS = [
    ("George Washington", "approved", "official presidential public documents; Project Gutenberg / APP", "no campaign material"),
    ("John Adams", "approved", "official presidential public documents; Project Gutenberg / APP", "no campaign material"),
    ("Thomas Jefferson", "approved", "official presidential public documents; Project Gutenberg / APP", "no campaign material"),
    ("James Madison", "approved", "official presidential public documents; Project Gutenberg / APP", "no campaign material"),
    ("James Monroe", "approved", "official presidential public documents; Project Gutenberg / APP", "no campaign material"),
    ("John Quincy Adams", "approved", "official presidential public documents; Project Gutenberg / APP", "no campaign material"),
    ("Andrew Jackson", "approved", "official presidential public documents; Project Gutenberg / APP", "no campaign material"),
    ("Martin Van Buren", "approved", "official presidential public documents; Project Gutenberg / APP", "no campaign material"),
    ("John Tyler", "approved", "official presidential public documents; Project Gutenberg / APP", "no campaign material"),
    ("James K. Polk", "approved", "official presidential public documents; Project Gutenberg / APP", "no campaign material"),
    ("Zachary Taylor", "approved", "official presidential public documents; Project Gutenberg / APP", "no campaign material"),
    ("Millard Fillmore", "approved", "official presidential public documents; Project Gutenberg / APP", "no campaign material"),
    ("Franklin Pierce", "approved", "official presidential public documents; Project Gutenberg / APP", "no campaign material"),
    ("James Buchanan", "approved", "official presidential public documents; Project Gutenberg / APP", "no campaign material"),
    ("Abraham Lincoln", "approved", "official presidential public documents; Project Gutenberg / APP", "no campaign material"),
    ("Andrew Johnson", "approved", "official presidential public documents; Project Gutenberg / APP", "no campaign material"),
    ("Ulysses S. Grant", "approved", "official presidential public documents; Project Gutenberg / APP", "no campaign material"),
    ("Rutherford B. Hayes", "approved", "official presidential public documents; Project Gutenberg / APP", "no campaign material"),
    ("James A. Garfield", "approved", "official presidential public documents; Project Gutenberg / APP", "no campaign material"),
    ("Chester A. Arthur", "approved", "official presidential public documents; Project Gutenberg / APP", "no campaign material"),
    ("Grover Cleveland", "approved", "official presidential public documents; Project Gutenberg / APP", "no campaign material"),
    ("Benjamin Harrison", "approved", "official presidential public documents; Project Gutenberg / APP", "no campaign material"),
    ("William McKinley", "approved", "official presidential public documents; Project Gutenberg / APP", "no campaign material"),
    ("Theodore Roosevelt", "approved", "official presidential public documents; Project Gutenberg / APP", "no campaign material"),
    ("William Howard Taft", "approved", "official presidential public documents; Project Gutenberg / APP", "no campaign material"),
    ("Woodrow Wilson", "approved", "official presidential public documents; Project Gutenberg / APP", "no campaign material"),
    ("Warren G. Harding", "approved", "official presidential public documents; Project Gutenberg / APP", "no campaign material"),
    ("Calvin Coolidge", "approved", "official presidential public documents; Project Gutenberg / APP", "no campaign material"),
    ("Herbert Hoover", "approved", "official presidential public documents; Project Gutenberg / APP", "no campaign material"),
    ("Franklin D. Roosevelt", "approved", "official presidential public documents; Project Gutenberg / APP", "no campaign material"),
    ("Harry S. Truman", "approved", "official presidential public documents; Project Gutenberg / APP", "no campaign material"),
    ("Dwight D. Eisenhower", "approved", "official presidential public documents; Project Gutenberg / APP", "no campaign material"),
    ("John F. Kennedy", "approved", "official presidential public documents; APP / JFK Library", "presidential speeches only; no campaign material"),
    ("Lyndon B. Johnson", "approved", "official presidential public documents; APP", "presidential speeches only; no campaign material"),
    ("Richard Nixon", "approved", "official presidential public documents; APP", "presidential speeches only; no campaign material"),
    ("Gerald Ford", "approved", "official presidential public documents; APP", "presidential speeches only; no campaign material"),
    ("Jimmy Carter", "approved", "official presidential public documents; APP", "presidential speeches only; no campaign material"),
    ("Ronald Reagan", "approved", "official presidential public documents; APP", "presidential speeches only; no campaign material"),
    ("George H. W. Bush", "approved", "official presidential public documents; APP", "presidential speeches only; no campaign material"),
    ("Bill Clinton", "approved", "official presidential public documents; APP", "presidential speeches only; no campaign material"),
    ("George W. Bush", "approved", "official presidential public documents; APP", "presidential speeches only; no campaign material"),
    ("Barack Obama", "approved", "official presidential public documents; APP / White House archives", "presidential speeches only; no campaign material"),
    ("Donald Trump", "approved", "official presidential public documents; APP / White House archives", "presidential speeches only; no campaign material"),
    ("Joe Biden", "approved", "official presidential public documents; APP / White House archives", "presidential speeches only; no campaign material"),
]


CANDIDATES = [
    ("Winston Churchill", "rhetorical", "candidate", "public speeches exist, but UK Crown/copyright and edition rights need review before ingestion"),
    ("Martin Luther King Jr.", "rhetorical", "candidate", "speeches are strongly copyrighted by estate; do not ingest without license"),
    ("Adolf Hitler", "rhetorical", "candidate", "English speech translations and editions have unclear or copyrighted translator rights; do not ingest now"),
    ("Pope Francis", "rhetorical", "candidate", "Vatican/publication copyright terms need review; do not ingest now"),
    ("Pope Benedict XVI", "rhetorical", "candidate", "Vatican/publication copyright terms need review; do not ingest now"),
    ("Pope John Paul II", "rhetorical", "candidate", "Vatican/publication copyright terms need review; do not ingest now"),
    ("Bill Gates", "rhetorical", "candidate", "public talks/interviews are usually copyrighted by venue, publisher, or Gates Notes; do not ingest now"),
    ("Elon Musk", "rhetorical", "candidate", "interview/transcript sources are usually copyrighted; do not ingest now"),
    ("Steve Jobs", "rhetorical", "candidate", "Stanford commencement and Apple keynotes have unclear reuse rights; do not ingest now"),
    ("Nelson Mandela", "rhetorical", "candidate", "public speeches exist, but reuse rights vary by source/edition; do not ingest now"),
    ("Margaret Thatcher", "rhetorical", "candidate", "foundation archive/public speeches exist, but reuse rights need review"),
    ("Tony Blair", "rhetorical", "candidate", "official/public speeches exist, but Crown/copyright status needs review"),
    ("Boris Johnson", "rhetorical", "candidate", "official speeches exist, but modern Crown/copyright status needs review"),
    ("Keir Starmer", "rhetorical", "candidate", "official speeches exist, but modern party/government rights need review"),
    ("Theresa May", "rhetorical", "candidate", "official speeches exist, but modern Crown/copyright status needs review"),
    ("David Cameron", "rhetorical", "candidate", "official speeches exist, but modern Crown/copyright status needs review"),
    ("Angela Merkel", "rhetorical", "candidate", "English translations are likely copyrighted; original German is outside v1 English scope"),
    ("Emmanuel Macron", "rhetorical", "candidate", "English translations are likely copyrighted; original French is outside v1 English scope"),
    ("Greta Thunberg", "rhetorical", "candidate", "public speeches are modern copyrighted text; do not ingest without license"),
    ("Malala Yousafzai", "rhetorical", "candidate", "public speeches are modern copyrighted text; do not ingest without license"),
]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    write_csv(
        REGISTRY_DIR / "literary_authors.csv",
        ["author", "status", "source_family", "notes"],
        [{"author": a, "status": s, "source_family": f, "notes": n} for a, s, f, n in LITERARY_AUTHORS],
    )
    write_csv(
        REGISTRY_DIR / "rhetorical_speakers.csv",
        ["speaker", "status", "source_family", "notes"],
        [{"speaker": a, "status": s, "source_family": f, "notes": n} for a, s, f, n in RHETORICAL_SPEAKERS],
    )
    write_csv(
        REGISTRY_DIR / "candidate_public_figures.csv",
        ["name", "corpus", "status", "reason_not_ingested"],
        [{"name": a, "corpus": c, "status": s, "reason_not_ingested": n} for a, c, s, n in CANDIDATES],
    )
    print(f"Wrote registries to {REGISTRY_DIR}")


if __name__ == "__main__":
    main()

