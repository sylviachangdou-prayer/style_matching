from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_DIR = ROOT / "data" / "source_registry"

FIELDS = [
    "name",
    "corpus",
    "original_language",
    "era",
    "batch",
    "source_family",
    "source_unit",
    "min_independent_sources",
    "modeling_status",
    "notes",
]


LITERARY_SEED = [
    ("Jane Austen", "en", "19c", "seed_literary_en", "Project Gutenberg", "work", 3, "collected", "original English prose"),
    ("Charles Dickens", "en", "19c", "seed_literary_en", "Project Gutenberg", "work", 3, "collected", "original English prose"),
    ("George Eliot", "en", "19c", "seed_literary_en", "Project Gutenberg", "work", 3, "collected", "original English prose"),
    ("Thomas Hardy", "en", "19c", "seed_literary_en", "Project Gutenberg", "work", 3, "collected", "original English prose"),
    ("Mark Twain", "en", "19c", "seed_literary_en", "Project Gutenberg", "work", 3, "collected", "original English prose"),
    ("Henry James", "en", "19c", "seed_literary_en", "Project Gutenberg", "work", 3, "collected", "original English prose"),
    ("Edith Wharton", "en", "20c", "seed_literary_en", "Project Gutenberg", "work", 3, "collected", "original English prose"),
    ("Herman Melville", "en", "19c", "seed_literary_en", "Project Gutenberg", "work", 3, "collected", "original English prose"),
    ("Nathaniel Hawthorne", "en", "19c", "seed_literary_en", "Project Gutenberg", "work", 3, "collected", "original English prose"),
    ("Edgar Allan Poe", "en", "19c", "seed_literary_en", "Project Gutenberg", "work", 3, "collected", "original English short fiction and essays"),
    ("Washington Irving", "en", "19c", "seed_literary_en", "Project Gutenberg", "work", 3, "collected", "original English sketches and essays"),
    ("Louisa May Alcott", "en", "19c", "seed_literary_en", "Project Gutenberg", "work", 3, "collected", "original English prose"),
    ("Kate Chopin", "en", "19c", "seed_literary_en", "Project Gutenberg", "work", 3, "collected", "original English prose"),
    ("Willa Cather", "en", "20c", "seed_literary_en", "Project Gutenberg", "work", 3, "collected", "original English prose"),
    ("Jack London", "en", "20c", "seed_literary_en", "Project Gutenberg", "work", 3, "collected", "original English prose"),
    ("Stephen Crane", "en", "19c", "seed_literary_en", "Project Gutenberg", "work", 3, "collected", "original English prose"),
    ("Charlotte Bronte", "en", "19c", "seed_literary_en", "Project Gutenberg", "work", 3, "collected", "original English prose"),
    ("Emily Bronte", "en", "19c", "seed_literary_en", "Project Gutenberg", "work", 3, "collected", "original English prose"),
    ("Anne Bronte", "en", "19c", "seed_literary_en", "Project Gutenberg", "work", 3, "collected", "original English prose"),
    ("Mary Wollstonecraft Shelley", "en", "19c", "seed_literary_en", "Project Gutenberg", "work", 3, "collected", "original English prose"),
    ("Robert Louis Stevenson", "en", "19c", "seed_literary_en", "Project Gutenberg", "work", 3, "collected", "original English prose"),
    ("Joseph Conrad", "en", "20c", "seed_literary_en", "Project Gutenberg", "work", 3, "collected", "original English prose"),
    ("H. G. Wells", "en", "20c", "seed_literary_en", "Project Gutenberg", "work", 3, "collected", "original English prose"),
    ("Frances Hodgson Burnett", "en", "20c", "seed_literary_en", "Project Gutenberg", "work", 3, "collected", "original English prose"),
    ("L. M. Montgomery", "en", "20c", "seed_literary_en", "Project Gutenberg", "work", 3, "collected", "original English prose"),
    ("Elizabeth Cleghorn Gaskell", "en", "19c", "seed_literary_en", "Project Gutenberg", "work", 3, "collected", "original English prose"),
    ("Anthony Trollope", "en", "19c", "seed_literary_en", "Project Gutenberg", "work", 3, "collected", "original English prose"),
    ("Wilkie Collins", "en", "19c", "seed_literary_en", "Project Gutenberg", "work", 3, "collected", "original English prose"),
    ("Arthur Conan Doyle", "en", "20c", "seed_literary_en", "Project Gutenberg", "work", 3, "collected", "original English prose"),
    ("George MacDonald", "en", "19c", "seed_literary_en", "Project Gutenberg", "work", 3, "collected", "original English prose"),
]


LITERARY_EXPANSION = [
    ("Virginia Woolf", "en", "20c", "expand_literary_modern_en", "curated original texts", "work", 3, "registry_only", "original English prose/essays"),
    ("James Joyce", "en", "20c", "expand_literary_modern_en", "curated original texts", "work", 3, "registry_only", "original English prose"),
    ("D. H. Lawrence", "en", "20c", "expand_literary_modern_en", "curated original texts", "work", 3, "registry_only", "original English prose"),
    ("E. M. Forster", "en", "20c", "expand_literary_modern_en", "curated original texts", "work", 3, "registry_only", "original English prose"),
    ("Katherine Mansfield", "en", "20c", "expand_literary_modern_en", "curated original texts", "work", 3, "registry_only", "original English short fiction"),
    ("George Orwell", "en", "20c", "expand_literary_modern_en", "curated original texts", "work", 3, "registry_only", "original English prose/essays"),
    ("W. E. B. Du Bois", "en", "20c", "expand_literary_modern_en", "curated original texts", "work", 3, "registry_only", "original English essays/prose"),
    ("Zora Neale Hurston", "en", "20c", "expand_literary_modern_en", "curated original texts", "work", 3, "registry_only", "original English prose"),
    ("F. Scott Fitzgerald", "en", "20c", "expand_literary_modern_en", "curated original texts", "work", 3, "registry_only", "original English prose"),
    ("Ernest Hemingway", "en", "20c", "expand_literary_modern_en", "curated original texts", "work", 3, "registry_only", "original English prose"),
    ("William Faulkner", "en", "20c", "expand_literary_modern_en", "curated original texts", "work", 3, "registry_only", "original English prose"),
    ("John Steinbeck", "en", "20c", "expand_literary_modern_en", "curated original texts", "work", 3, "registry_only", "original English prose"),
    ("Ralph Ellison", "en", "20c", "expand_literary_modern_en", "curated original texts", "work", 3, "registry_only", "original English prose/essays"),
    ("James Baldwin", "en", "20c", "expand_literary_modern_en", "curated original texts", "work", 3, "registry_only", "original English prose/essays"),
    ("Toni Morrison", "en", "20c", "expand_literary_modern_en", "curated original texts", "work", 3, "registry_only", "original English prose/essays"),
    ("Joan Didion", "en", "20c", "expand_literary_modern_en", "curated original texts", "work", 3, "registry_only", "original English essays/prose"),
    ("Ursula K. Le Guin", "en", "20c", "expand_literary_modern_en", "curated original texts", "work", 3, "registry_only", "original English prose/essays"),
    ("Kurt Vonnegut", "en", "20c", "expand_literary_modern_en", "curated original texts", "work", 3, "registry_only", "original English prose/essays"),
    ("Flannery O'Connor", "en", "20c", "expand_literary_modern_en", "curated original texts", "work", 3, "registry_only", "original English prose"),
    ("Marcel Proust", "fr", "20c", "expand_literary_multilingual", "curated French originals", "work", 3, "registry_only", "French originals only"),
    ("Albert Camus", "fr", "20c", "expand_literary_multilingual", "curated French originals", "work", 3, "registry_only", "French originals only"),
    ("Jean-Paul Sartre", "fr", "20c", "expand_literary_multilingual", "curated French originals", "work", 3, "registry_only", "French originals only"),
    ("Simone de Beauvoir", "fr", "20c", "expand_literary_multilingual", "curated French originals", "work", 3, "registry_only", "French originals only"),
    ("Colette", "fr", "20c", "expand_literary_multilingual", "curated French originals", "work", 3, "registry_only", "French originals only"),
    ("Franz Kafka", "de", "20c", "expand_literary_multilingual", "curated German originals", "work", 3, "registry_only", "German originals only"),
    ("Thomas Mann", "de", "20c", "expand_literary_multilingual", "curated German originals", "work", 3, "registry_only", "German originals only"),
    ("Bertolt Brecht", "de", "20c", "expand_literary_multilingual", "curated German originals", "work", 3, "registry_only", "German originals only"),
    ("Hannah Arendt", "de", "20c", "expand_literary_multilingual", "curated German originals", "work", 3, "registry_only", "German originals only"),
    ("Hannah Arendt", "en", "20c", "expand_literary_multilingual", "curated English originals", "work", 3, "registry_only", "English originals only"),
    ("Jorge Luis Borges", "es", "20c", "expand_literary_multilingual", "curated Spanish originals", "work", 3, "registry_only", "Spanish originals only"),
    ("Gabriel Garcia Marquez", "es", "20c", "expand_literary_multilingual", "curated Spanish originals", "work", 3, "registry_only", "Spanish originals only"),
    ("Federico Garcia Lorca", "es", "20c", "expand_literary_multilingual", "curated Spanish originals", "work", 3, "registry_only", "Spanish originals only"),
    ("Italo Calvino", "it", "20c", "expand_literary_multilingual", "curated Italian originals", "work", 3, "registry_only", "Italian originals only"),
    ("Lu Xun", "zh", "20c", "expand_literary_multilingual", "curated Chinese originals", "work", 3, "registry_only", "Chinese originals only"),
    ("Lao She", "zh", "20c", "expand_literary_multilingual", "curated Chinese originals", "work", 3, "registry_only", "Chinese originals only"),
    ("Eileen Chang", "zh", "20c", "expand_literary_multilingual", "curated Chinese originals", "work", 3, "registry_only", "Chinese originals only"),
    ("Eileen Chang", "en", "20c", "expand_literary_multilingual", "curated English originals", "work", 3, "registry_only", "English originals only"),
    ("Natsume Soseki", "ja", "20c", "expand_literary_multilingual", "curated Japanese originals", "work", 3, "registry_only", "Japanese originals only"),
    ("Ryunosuke Akutagawa", "ja", "20c", "expand_literary_multilingual", "curated Japanese originals", "work", 3, "registry_only", "Japanese originals only"),
    ("Osamu Dazai", "ja", "20c", "expand_literary_multilingual", "curated Japanese originals", "work", 3, "registry_only", "Japanese originals only"),
]


PRESIDENTIAL_SPEAKERS = [
    ("George Washington", "en", "18c", "seed_rhetorical_us_presidents", "official presidential public documents / APP / Project Gutenberg", "speech_or_document", 3, "registry_only", "original English public rhetoric; no campaign material"),
    ("John Adams", "en", "18c", "seed_rhetorical_us_presidents", "official presidential public documents / APP / Project Gutenberg", "speech_or_document", 3, "registry_only", "original English public rhetoric; no campaign material"),
    ("Thomas Jefferson", "en", "19c", "seed_rhetorical_us_presidents", "official presidential public documents / APP / Project Gutenberg", "speech_or_document", 3, "registry_only", "original English public rhetoric; no campaign material"),
    ("James Madison", "en", "19c", "seed_rhetorical_us_presidents", "official presidential public documents / APP / Project Gutenberg", "speech_or_document", 3, "registry_only", "original English public rhetoric; no campaign material"),
    ("James Monroe", "en", "19c", "seed_rhetorical_us_presidents", "official presidential public documents / APP / Project Gutenberg", "speech_or_document", 3, "registry_only", "original English public rhetoric; no campaign material"),
    ("John Quincy Adams", "en", "19c", "seed_rhetorical_us_presidents", "official presidential public documents / APP / Project Gutenberg", "speech_or_document", 3, "registry_only", "original English public rhetoric; no campaign material"),
    ("Andrew Jackson", "en", "19c", "seed_rhetorical_us_presidents", "official presidential public documents / APP / Project Gutenberg", "speech_or_document", 3, "registry_only", "original English public rhetoric; no campaign material"),
    ("Martin Van Buren", "en", "19c", "seed_rhetorical_us_presidents", "official presidential public documents / APP / Project Gutenberg", "speech_or_document", 3, "registry_only", "original English public rhetoric; no campaign material"),
    ("John Tyler", "en", "19c", "seed_rhetorical_us_presidents", "official presidential public documents / APP / Project Gutenberg", "speech_or_document", 3, "registry_only", "original English public rhetoric; no campaign material"),
    ("James K. Polk", "en", "19c", "seed_rhetorical_us_presidents", "official presidential public documents / APP / Project Gutenberg", "speech_or_document", 3, "registry_only", "original English public rhetoric; no campaign material"),
    ("Zachary Taylor", "en", "19c", "seed_rhetorical_us_presidents", "official presidential public documents / APP / Project Gutenberg", "speech_or_document", 3, "registry_only", "original English public rhetoric; no campaign material"),
    ("Millard Fillmore", "en", "19c", "seed_rhetorical_us_presidents", "official presidential public documents / APP / Project Gutenberg", "speech_or_document", 3, "registry_only", "original English public rhetoric; no campaign material"),
    ("Franklin Pierce", "en", "19c", "seed_rhetorical_us_presidents", "official presidential public documents / APP / Project Gutenberg", "speech_or_document", 3, "registry_only", "original English public rhetoric; no campaign material"),
    ("James Buchanan", "en", "19c", "seed_rhetorical_us_presidents", "official presidential public documents / APP / Project Gutenberg", "speech_or_document", 3, "registry_only", "original English public rhetoric; no campaign material"),
    ("Abraham Lincoln", "en", "19c", "seed_rhetorical_us_presidents", "official presidential public documents / APP / Project Gutenberg", "speech_or_document", 3, "registry_only", "original English public rhetoric; no campaign material"),
    ("Andrew Johnson", "en", "19c", "seed_rhetorical_us_presidents", "official presidential public documents / APP / Project Gutenberg", "speech_or_document", 3, "registry_only", "original English public rhetoric; no campaign material"),
    ("Ulysses S. Grant", "en", "19c", "seed_rhetorical_us_presidents", "official presidential public documents / APP / Project Gutenberg", "speech_or_document", 3, "registry_only", "original English public rhetoric; no campaign material"),
    ("Rutherford B. Hayes", "en", "19c", "seed_rhetorical_us_presidents", "official presidential public documents / APP / Project Gutenberg", "speech_or_document", 3, "registry_only", "original English public rhetoric; no campaign material"),
    ("James A. Garfield", "en", "19c", "seed_rhetorical_us_presidents", "official presidential public documents / APP / Project Gutenberg", "speech_or_document", 3, "registry_only", "original English public rhetoric; no campaign material"),
    ("Chester A. Arthur", "en", "19c", "seed_rhetorical_us_presidents", "official presidential public documents / APP / Project Gutenberg", "speech_or_document", 3, "registry_only", "original English public rhetoric; no campaign material"),
    ("Grover Cleveland", "en", "19c", "seed_rhetorical_us_presidents", "official presidential public documents / APP / Project Gutenberg", "speech_or_document", 3, "registry_only", "original English public rhetoric; no campaign material"),
    ("Benjamin Harrison", "en", "19c", "seed_rhetorical_us_presidents", "official presidential public documents / APP / Project Gutenberg", "speech_or_document", 3, "registry_only", "original English public rhetoric; no campaign material"),
    ("William McKinley", "en", "19c", "seed_rhetorical_us_presidents", "official presidential public documents / APP / Project Gutenberg", "speech_or_document", 3, "registry_only", "original English public rhetoric; no campaign material"),
    ("Theodore Roosevelt", "en", "20c", "seed_rhetorical_us_presidents", "official presidential public documents / APP / Project Gutenberg", "speech_or_document", 3, "registry_only", "original English public rhetoric; no campaign material"),
    ("William Howard Taft", "en", "20c", "seed_rhetorical_us_presidents", "official presidential public documents / APP / Project Gutenberg", "speech_or_document", 3, "registry_only", "original English public rhetoric; no campaign material"),
    ("Woodrow Wilson", "en", "20c", "seed_rhetorical_us_presidents", "official presidential public documents / APP / Project Gutenberg", "speech_or_document", 3, "registry_only", "original English public rhetoric; no campaign material"),
    ("Warren G. Harding", "en", "20c", "seed_rhetorical_us_presidents", "official presidential public documents / APP / Project Gutenberg", "speech_or_document", 3, "registry_only", "original English public rhetoric; no campaign material"),
    ("Calvin Coolidge", "en", "20c", "seed_rhetorical_us_presidents", "official presidential public documents / APP / Project Gutenberg", "speech_or_document", 3, "registry_only", "original English public rhetoric; no campaign material"),
    ("Herbert Hoover", "en", "20c", "seed_rhetorical_us_presidents", "official presidential public documents / APP / Project Gutenberg", "speech_or_document", 3, "registry_only", "original English public rhetoric; no campaign material"),
    ("Franklin D. Roosevelt", "en", "20c", "seed_rhetorical_us_presidents", "official presidential public documents / APP / Project Gutenberg", "speech_or_document", 3, "registry_only", "original English public rhetoric; no campaign material"),
    ("Harry S. Truman", "en", "20c", "seed_rhetorical_us_presidents", "official presidential public documents / APP", "speech_or_document", 3, "registry_only", "original English public rhetoric; no campaign material"),
    ("Dwight D. Eisenhower", "en", "20c", "seed_rhetorical_us_presidents", "official presidential public documents / APP", "speech_or_document", 3, "registry_only", "original English public rhetoric; no campaign material"),
    ("John F. Kennedy", "en", "20c", "seed_rhetorical_us_presidents", "official presidential public documents / APP / JFK Library", "speech_or_document", 3, "registry_only", "original English public rhetoric; no campaign material"),
    ("Lyndon B. Johnson", "en", "20c", "seed_rhetorical_us_presidents", "official presidential public documents / APP", "speech_or_document", 3, "registry_only", "original English public rhetoric; no campaign material"),
    ("Richard Nixon", "en", "20c", "seed_rhetorical_us_presidents", "official presidential public documents / APP", "speech_or_document", 3, "registry_only", "original English public rhetoric; no campaign material"),
    ("Gerald Ford", "en", "20c", "seed_rhetorical_us_presidents", "official presidential public documents / APP", "speech_or_document", 3, "registry_only", "original English public rhetoric; no campaign material"),
    ("Jimmy Carter", "en", "20c", "seed_rhetorical_us_presidents", "official presidential public documents / APP", "speech_or_document", 3, "registry_only", "original English public rhetoric; no campaign material"),
    ("Ronald Reagan", "en", "20c", "seed_rhetorical_us_presidents", "official presidential public documents / APP", "speech_or_document", 3, "registry_only", "original English public rhetoric; no campaign material"),
    ("George H. W. Bush", "en", "20c", "seed_rhetorical_us_presidents", "official presidential public documents / APP", "speech_or_document", 3, "registry_only", "original English public rhetoric; no campaign material"),
    ("Bill Clinton", "en", "20c", "seed_rhetorical_us_presidents", "official presidential public documents / APP", "speech_or_document", 3, "registry_only", "original English public rhetoric; no campaign material"),
    ("George W. Bush", "en", "21c", "seed_rhetorical_us_presidents", "official presidential public documents / APP", "speech_or_document", 3, "registry_only", "original English public rhetoric; no campaign material"),
    ("Barack Obama", "en", "21c", "seed_rhetorical_us_presidents", "official presidential public documents / APP / White House archives", "speech_or_document", 3, "registry_only", "original English public rhetoric; no campaign material"),
    ("Donald Trump", "en", "21c", "seed_rhetorical_us_presidents", "official presidential public documents / APP / White House archives", "speech_or_document", 3, "registry_only", "original English public rhetoric; no campaign material"),
    ("Joe Biden", "en", "21c", "seed_rhetorical_us_presidents", "official presidential public documents / APP / White House archives", "speech_or_document", 3, "registry_only", "original English public rhetoric; no campaign material"),
]


RHETORICAL_EXPANSION = [
    ("Winston Churchill", "en", "20c", "expand_rhetorical_public_figures", "Churchill archive / official speeches", "speech_or_document", 3, "registry_only", "original English public rhetoric only"),
    ("Martin Luther King Jr.", "en", "20c", "expand_rhetorical_public_figures", "speech archive", "speech_or_document", 3, "registry_only", "original English public rhetoric only"),
    ("Bill Gates", "en", "20c_21c", "expand_rhetorical_public_figures", "official essays / speeches / transcripts", "speech_or_document", 3, "registry_only", "original English texts only"),
    ("Elon Musk", "en", "21c", "expand_rhetorical_public_figures", "official speeches / interviews / posts", "speech_or_document", 3, "registry_only", "original English texts only"),
    ("Steve Jobs", "en", "20c_21c", "expand_rhetorical_public_figures", "official speeches / keynotes", "speech_or_document", 3, "registry_only", "original English texts only"),
    ("Nelson Mandela", "en", "20c_21c", "expand_rhetorical_public_figures", "foundation/archive speeches", "speech_or_document", 3, "registry_only", "collect English originals only in the English pass"),
    ("Margaret Thatcher", "en", "20c", "expand_rhetorical_uk", "foundation/archive speeches", "speech_or_document", 3, "registry_only", "original English public rhetoric only"),
    ("Tony Blair", "en", "20c_21c", "expand_rhetorical_uk", "official speeches/archive", "speech_or_document", 3, "registry_only", "original English public rhetoric only"),
    ("Boris Johnson", "en", "21c", "expand_rhetorical_uk", "official speeches/archive", "speech_or_document", 3, "registry_only", "original English public rhetoric only"),
    ("Keir Starmer", "en", "21c", "expand_rhetorical_uk", "official speeches/archive", "speech_or_document", 3, "registry_only", "original English public rhetoric only; no campaign material"),
    ("Theresa May", "en", "21c", "expand_rhetorical_uk", "official speeches/archive", "speech_or_document", 3, "registry_only", "original English public rhetoric only"),
    ("David Cameron", "en", "21c", "expand_rhetorical_uk", "official speeches/archive", "speech_or_document", 3, "registry_only", "original English public rhetoric only"),
    ("Gordon Brown", "en", "21c", "expand_rhetorical_uk", "official speeches/archive", "speech_or_document", 3, "registry_only", "original English public rhetoric only"),
    ("John Major", "en", "20c", "expand_rhetorical_uk", "official speeches/archive", "speech_or_document", 3, "registry_only", "original English public rhetoric only"),
    ("Rishi Sunak", "en", "21c", "expand_rhetorical_uk", "official speeches/archive", "speech_or_document", 3, "registry_only", "original English public rhetoric only"),
    ("Liz Truss", "en", "21c", "expand_rhetorical_uk", "official speeches/archive", "speech_or_document", 3, "registry_only", "original English public rhetoric only"),
    ("Kamala Harris", "en", "21c", "expand_rhetorical_us_modern", "official speeches / White House / Senate archive", "speech_or_document", 3, "registry_only", "original English public rhetoric; no campaign material"),
    ("Hillary Clinton", "en", "20c_21c", "expand_rhetorical_us_modern", "official speeches / State Department / Senate archive", "speech_or_document", 3, "registry_only", "original English public rhetoric; no campaign material"),
    ("Bernie Sanders", "en", "20c_21c", "expand_rhetorical_us_modern", "official congressional/senate speeches", "speech_or_document", 3, "registry_only", "original English public rhetoric; no campaign material"),
    ("Elizabeth Warren", "en", "21c", "expand_rhetorical_us_modern", "official senate speeches", "speech_or_document", 3, "registry_only", "original English public rhetoric; no campaign material"),
    ("Alexandria Ocasio-Cortez", "en", "21c", "expand_rhetorical_us_modern", "official congressional speeches", "speech_or_document", 3, "registry_only", "original English public rhetoric; no campaign material"),
    ("Ron DeSantis", "en", "21c", "expand_rhetorical_us_modern", "official governor speeches", "speech_or_document", 3, "registry_only", "original English public rhetoric; no campaign material"),
    ("Justin Trudeau", "en", "21c", "expand_rhetorical_commonwealth", "official English speeches", "speech_or_document", 3, "registry_only", "English originals only"),
    ("Justin Trudeau", "fr", "21c", "expand_rhetorical_commonwealth", "official French speeches", "speech_or_document", 3, "registry_only", "French originals only"),
    ("Jacinda Ardern", "en", "21c", "expand_rhetorical_commonwealth", "official speeches", "speech_or_document", 3, "registry_only", "original English public rhetoric only"),
    ("Anthony Albanese", "en", "21c", "expand_rhetorical_commonwealth", "official speeches", "speech_or_document", 3, "registry_only", "original English public rhetoric only"),
    ("Adolf Hitler", "de", "20c", "expand_rhetorical_multilingual", "German speech archive", "speech_or_document", 3, "registry_only", "German originals only"),
    ("Angela Merkel", "de", "21c", "expand_rhetorical_multilingual", "official German speeches", "speech_or_document", 3, "registry_only", "German originals only"),
    ("Olaf Scholz", "de", "21c", "expand_rhetorical_multilingual", "official German speeches", "speech_or_document", 3, "registry_only", "German originals only"),
    ("Emmanuel Macron", "fr", "21c", "expand_rhetorical_multilingual", "official French speeches", "speech_or_document", 3, "registry_only", "French originals only"),
    ("Charles de Gaulle", "fr", "20c", "expand_rhetorical_multilingual", "French speech archive", "speech_or_document", 3, "registry_only", "French originals only"),
    ("Francois Mitterrand", "fr", "20c", "expand_rhetorical_multilingual", "French speech archive", "speech_or_document", 3, "registry_only", "French originals only"),
    ("Pope Francis", "it", "21c", "expand_rhetorical_multilingual", "Vatican source texts", "speech_or_document", 3, "registry_only", "Italian originals unless speech delivered in another source language"),
    ("Pope Benedict XVI", "de", "21c", "expand_rhetorical_multilingual", "Vatican source texts", "speech_or_document", 3, "registry_only", "German originals unless speech delivered in another source language"),
    ("Pope John Paul II", "pl", "20c_21c", "expand_rhetorical_multilingual", "Vatican source texts", "speech_or_document", 3, "registry_only", "Polish originals unless speech delivered in another source language"),
    ("Fidel Castro", "es", "20c_21c", "expand_rhetorical_multilingual", "Spanish speech archive", "speech_or_document", 3, "registry_only", "Spanish originals only"),
    ("Salvador Allende", "es", "20c", "expand_rhetorical_multilingual", "Spanish speech archive", "speech_or_document", 3, "registry_only", "Spanish originals only"),
    ("Hugo Chavez", "es", "20c_21c", "expand_rhetorical_multilingual", "Spanish speech archive", "speech_or_document", 3, "registry_only", "Spanish originals only"),
    ("Greta Thunberg", "en", "21c", "expand_rhetorical_public_figures", "official speeches", "speech_or_document", 3, "registry_only", "collect only speeches originally delivered in English"),
    ("Malala Yousafzai", "en", "21c", "expand_rhetorical_public_figures", "official speeches", "speech_or_document", 3, "registry_only", "collect only speeches originally delivered in English"),
]


def row(name: str, corpus: str, language: str, era: str, batch: str, source_family: str, source_unit: str, min_sources: int, status: str, notes: str) -> dict[str, str | int]:
    return {
        "name": name,
        "corpus": corpus,
        "original_language": language,
        "era": era,
        "batch": batch,
        "source_family": source_family,
        "source_unit": source_unit,
        "min_independent_sources": min_sources,
        "modeling_status": status,
        "notes": notes,
    }


def rows(records: list[tuple[str, str, str, str, str, str, int, str, str]], corpus: str) -> list[dict[str, str | int]]:
    return [row(name, corpus, *rest) for name, *rest in records]


def dedupe(records: list[dict[str, str | int]]) -> list[dict[str, str | int]]:
    seen = set()
    output = []
    for record in records:
        key = (record["name"], record["corpus"], record["original_language"])
        if key not in seen:
            seen.add(key)
            output.append(record)
    return output


def write_csv(path: Path, records: list[dict[str, str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    literary = dedupe(rows(LITERARY_SEED + LITERARY_EXPANSION, "literary"))
    rhetorical = dedupe(rows(PRESIDENTIAL_SPEAKERS + RHETORICAL_EXPANSION, "rhetorical"))
    all_people = dedupe(literary + rhetorical)

    write_csv(REGISTRY_DIR / "literary_authors.csv", literary)
    write_csv(REGISTRY_DIR / "rhetorical_speakers.csv", rhetorical)
    write_csv(REGISTRY_DIR / "all_people.csv", all_people)
    write_csv(REGISTRY_DIR / "registry_expansion_batch.csv", [r for r in all_people if str(r["modeling_status"]) == "registry_only"])

    print(f"Wrote {len(literary)} literary records")
    print(f"Wrote {len(rhetorical)} rhetorical records")
    print(f"Wrote {len(all_people)} total records")


if __name__ == "__main__":
    main()
