from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_DIR = ROOT / "data" / "source_registry"


LITERARY_AUTHORS = [
    ("Jane Austen", "literary", "en", "Project Gutenberg", "original English prose"),
    ("Charles Dickens", "literary", "en", "Project Gutenberg", "original English prose"),
    ("George Eliot", "literary", "en", "Project Gutenberg", "original English prose"),
    ("Thomas Hardy", "literary", "en", "Project Gutenberg", "original English prose"),
    ("Mark Twain", "literary", "en", "Project Gutenberg", "original English prose"),
    ("Henry James", "literary", "en", "Project Gutenberg", "original English prose"),
    ("Edith Wharton", "literary", "en", "Project Gutenberg", "original English prose"),
    ("Herman Melville", "literary", "en", "Project Gutenberg", "original English prose"),
    ("Nathaniel Hawthorne", "literary", "en", "Project Gutenberg", "original English prose"),
    ("Edgar Allan Poe", "literary", "en", "Project Gutenberg", "original English prose"),
    ("Washington Irving", "literary", "en", "Project Gutenberg", "original English prose"),
    ("Louisa May Alcott", "literary", "en", "Project Gutenberg", "original English prose"),
    ("Kate Chopin", "literary", "en", "Project Gutenberg", "original English prose"),
    ("Willa Cather", "literary", "en", "Project Gutenberg", "original English prose"),
    ("Jack London", "literary", "en", "Project Gutenberg", "original English prose"),
    ("Stephen Crane", "literary", "en", "Project Gutenberg", "original English prose"),
    ("Charlotte Bronte", "literary", "en", "Project Gutenberg", "original English prose"),
    ("Emily Bronte", "literary", "en", "Project Gutenberg", "original English prose"),
    ("Anne Bronte", "literary", "en", "Project Gutenberg", "original English prose"),
    ("Mary Wollstonecraft Shelley", "literary", "en", "Project Gutenberg", "original English prose"),
    ("Robert Louis Stevenson", "literary", "en", "Project Gutenberg", "original English prose"),
    ("Joseph Conrad", "literary", "en", "Project Gutenberg", "original English prose"),
    ("H. G. Wells", "literary", "en", "Project Gutenberg", "original English prose"),
    ("Frances Hodgson Burnett", "literary", "en", "Project Gutenberg", "original English prose"),
    ("L. M. Montgomery", "literary", "en", "Project Gutenberg", "original English prose"),
    ("Elizabeth Cleghorn Gaskell", "literary", "en", "Project Gutenberg", "original English prose"),
    ("Anthony Trollope", "literary", "en", "Project Gutenberg", "original English prose"),
    ("Wilkie Collins", "literary", "en", "Project Gutenberg", "original English prose"),
    ("Arthur Conan Doyle", "literary", "en", "Project Gutenberg", "original English prose"),
    ("George MacDonald", "literary", "en", "Project Gutenberg", "original English prose"),
]


PRESIDENTIAL_SPEAKERS = [
    ("George Washington", "rhetorical", "en", "official presidential public documents / APP / Project Gutenberg", "original English public rhetoric; no campaign material"),
    ("John Adams", "rhetorical", "en", "official presidential public documents / APP / Project Gutenberg", "original English public rhetoric; no campaign material"),
    ("Thomas Jefferson", "rhetorical", "en", "official presidential public documents / APP / Project Gutenberg", "original English public rhetoric; no campaign material"),
    ("James Madison", "rhetorical", "en", "official presidential public documents / APP / Project Gutenberg", "original English public rhetoric; no campaign material"),
    ("James Monroe", "rhetorical", "en", "official presidential public documents / APP / Project Gutenberg", "original English public rhetoric; no campaign material"),
    ("John Quincy Adams", "rhetorical", "en", "official presidential public documents / APP / Project Gutenberg", "original English public rhetoric; no campaign material"),
    ("Andrew Jackson", "rhetorical", "en", "official presidential public documents / APP / Project Gutenberg", "original English public rhetoric; no campaign material"),
    ("Martin Van Buren", "rhetorical", "en", "official presidential public documents / APP / Project Gutenberg", "original English public rhetoric; no campaign material"),
    ("John Tyler", "rhetorical", "en", "official presidential public documents / APP / Project Gutenberg", "original English public rhetoric; no campaign material"),
    ("James K. Polk", "rhetorical", "en", "official presidential public documents / APP / Project Gutenberg", "original English public rhetoric; no campaign material"),
    ("Zachary Taylor", "rhetorical", "en", "official presidential public documents / APP / Project Gutenberg", "original English public rhetoric; no campaign material"),
    ("Millard Fillmore", "rhetorical", "en", "official presidential public documents / APP / Project Gutenberg", "original English public rhetoric; no campaign material"),
    ("Franklin Pierce", "rhetorical", "en", "official presidential public documents / APP / Project Gutenberg", "original English public rhetoric; no campaign material"),
    ("James Buchanan", "rhetorical", "en", "official presidential public documents / APP / Project Gutenberg", "original English public rhetoric; no campaign material"),
    ("Abraham Lincoln", "rhetorical", "en", "official presidential public documents / APP / Project Gutenberg", "original English public rhetoric; no campaign material"),
    ("Andrew Johnson", "rhetorical", "en", "official presidential public documents / APP / Project Gutenberg", "original English public rhetoric; no campaign material"),
    ("Ulysses S. Grant", "rhetorical", "en", "official presidential public documents / APP / Project Gutenberg", "original English public rhetoric; no campaign material"),
    ("Rutherford B. Hayes", "rhetorical", "en", "official presidential public documents / APP / Project Gutenberg", "original English public rhetoric; no campaign material"),
    ("James A. Garfield", "rhetorical", "en", "official presidential public documents / APP / Project Gutenberg", "original English public rhetoric; no campaign material"),
    ("Chester A. Arthur", "rhetorical", "en", "official presidential public documents / APP / Project Gutenberg", "original English public rhetoric; no campaign material"),
    ("Grover Cleveland", "rhetorical", "en", "official presidential public documents / APP / Project Gutenberg", "original English public rhetoric; no campaign material"),
    ("Benjamin Harrison", "rhetorical", "en", "official presidential public documents / APP / Project Gutenberg", "original English public rhetoric; no campaign material"),
    ("William McKinley", "rhetorical", "en", "official presidential public documents / APP / Project Gutenberg", "original English public rhetoric; no campaign material"),
    ("Theodore Roosevelt", "rhetorical", "en", "official presidential public documents / APP / Project Gutenberg", "original English public rhetoric; no campaign material"),
    ("William Howard Taft", "rhetorical", "en", "official presidential public documents / APP / Project Gutenberg", "original English public rhetoric; no campaign material"),
    ("Woodrow Wilson", "rhetorical", "en", "official presidential public documents / APP / Project Gutenberg", "original English public rhetoric; no campaign material"),
    ("Warren G. Harding", "rhetorical", "en", "official presidential public documents / APP / Project Gutenberg", "original English public rhetoric; no campaign material"),
    ("Calvin Coolidge", "rhetorical", "en", "official presidential public documents / APP / Project Gutenberg", "original English public rhetoric; no campaign material"),
    ("Herbert Hoover", "rhetorical", "en", "official presidential public documents / APP / Project Gutenberg", "original English public rhetoric; no campaign material"),
    ("Franklin D. Roosevelt", "rhetorical", "en", "official presidential public documents / APP / Project Gutenberg", "original English public rhetoric; no campaign material"),
    ("Harry S. Truman", "rhetorical", "en", "official presidential public documents / APP", "original English public rhetoric; no campaign material"),
    ("Dwight D. Eisenhower", "rhetorical", "en", "official presidential public documents / APP", "original English public rhetoric; no campaign material"),
    ("John F. Kennedy", "rhetorical", "en", "official presidential public documents / APP / JFK Library", "original English public rhetoric; no campaign material"),
    ("Lyndon B. Johnson", "rhetorical", "en", "official presidential public documents / APP", "original English public rhetoric; no campaign material"),
    ("Richard Nixon", "rhetorical", "en", "official presidential public documents / APP", "original English public rhetoric; no campaign material"),
    ("Gerald Ford", "rhetorical", "en", "official presidential public documents / APP", "original English public rhetoric; no campaign material"),
    ("Jimmy Carter", "rhetorical", "en", "official presidential public documents / APP", "original English public rhetoric; no campaign material"),
    ("Ronald Reagan", "rhetorical", "en", "official presidential public documents / APP", "original English public rhetoric; no campaign material"),
    ("George H. W. Bush", "rhetorical", "en", "official presidential public documents / APP", "original English public rhetoric; no campaign material"),
    ("Bill Clinton", "rhetorical", "en", "official presidential public documents / APP", "original English public rhetoric; no campaign material"),
    ("George W. Bush", "rhetorical", "en", "official presidential public documents / APP", "original English public rhetoric; no campaign material"),
    ("Barack Obama", "rhetorical", "en", "official presidential public documents / APP / White House archives", "original English public rhetoric; no campaign material"),
    ("Donald Trump", "rhetorical", "en", "official presidential public documents / APP / White House archives", "original English public rhetoric; no campaign material"),
    ("Joe Biden", "rhetorical", "en", "official presidential public documents / APP / White House archives", "original English public rhetoric; no campaign material"),
]


PUBLIC_FIGURE_SPEAKERS = [
    ("Winston Churchill", "rhetorical", "en", "Churchill archive / official speeches", "original English public rhetoric only"),
    ("Martin Luther King Jr.", "rhetorical", "en", "speech archive", "original English public rhetoric only"),
    ("Adolf Hitler", "rhetorical", "de", "German speech archive", "German originals only; skip if only English translation is available"),
    ("Pope Francis", "rhetorical", "it", "Vatican source texts", "Italian originals only unless the original speech was delivered in another source language"),
    ("Pope Benedict XVI", "rhetorical", "de", "Vatican source texts", "German originals only unless the original speech was delivered in another source language"),
    ("Pope John Paul II", "rhetorical", "pl", "Vatican source texts", "Polish originals only unless the original speech was delivered in another source language"),
    ("Bill Gates", "rhetorical", "en", "official essays / speeches / transcripts", "original English texts only"),
    ("Elon Musk", "rhetorical", "en", "official speeches / interviews / posts", "original English texts only"),
    ("Steve Jobs", "rhetorical", "en", "official speeches / keynotes", "original English texts only"),
    ("Nelson Mandela", "rhetorical", "en", "foundation/archive speeches", "original English texts only for v1; non-English originals require source-language handling"),
    ("Margaret Thatcher", "rhetorical", "en", "foundation/archive speeches", "original English public rhetoric only"),
    ("Tony Blair", "rhetorical", "en", "official speeches/archive", "original English public rhetoric only"),
    ("Boris Johnson", "rhetorical", "en", "official speeches/archive", "original English public rhetoric only"),
    ("Keir Starmer", "rhetorical", "en", "official speeches/archive", "original English public rhetoric only; no campaign material"),
    ("Theresa May", "rhetorical", "en", "official speeches/archive", "original English public rhetoric only"),
    ("David Cameron", "rhetorical", "en", "official speeches/archive", "original English public rhetoric only"),
    ("Angela Merkel", "rhetorical", "de", "official German speeches", "German originals only"),
    ("Emmanuel Macron", "rhetorical", "fr", "official French speeches", "French originals only"),
    ("Greta Thunberg", "rhetorical", "en", "official speeches", "original English texts only when originally delivered in English"),
    ("Malala Yousafzai", "rhetorical", "en", "official speeches", "original English texts only when originally delivered in English"),
]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rows(records: list[tuple[str, str, str, str, str]], name_field: str) -> list[dict[str, str]]:
    return [
        {
            name_field: name,
            "corpus": corpus,
            "original_language": original_language,
            "source_family": source_family,
            "notes": notes,
        }
        for name, corpus, original_language, source_family, notes in records
    ]


def main() -> None:
    fields = ["corpus", "original_language", "source_family", "notes"]
    write_csv(
        REGISTRY_DIR / "literary_authors.csv",
        ["author", *fields],
        rows(LITERARY_AUTHORS, "author"),
    )
    write_csv(
        REGISTRY_DIR / "rhetorical_speakers.csv",
        ["speaker", *fields],
        rows(PRESIDENTIAL_SPEAKERS + PUBLIC_FIGURE_SPEAKERS, "speaker"),
    )
    write_csv(
        REGISTRY_DIR / "public_figure_speakers.csv",
        ["speaker", *fields],
        rows(PUBLIC_FIGURE_SPEAKERS, "speaker"),
    )
    print(f"Wrote registries to {REGISTRY_DIR}")


if __name__ == "__main__":
    main()
