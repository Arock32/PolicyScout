import re
import unicodedata

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "extracted_text"
OUTPUT_DIR = BASE_DIR / "preprocessed_text"
OUTPUT_DIR.mkdir(exist_ok=True)

def normalize_unicode(text):
    return unicodedata.normalize("NFC", text)

def normalize_headings(text):

    # Convert separator-based headings into normal headings
    text = re.sub(
        r"={3,}\s*(.*?)\s*={3,}",
        r"\1",
        text
    )

    return text


def remove_artifacts(text):

    cleaned_lines = []

    for line in text.splitlines():

        line = line.strip()

        # Remove standalone dots
        if line == ".":
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def normalize_whitespace(text):
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def preprocess_text(text):
    text = normalize_unicode(text)
    text = normalize_headings(text)
    text = remove_artifacts(text)
    text = normalize_whitespace(text)
    return text


def process_documents():

    files = list(INPUT_DIR.glob("*.txt"))

    if not files:

        print("No extracted text files found.")

        return

    for file_path in files:

        print(f"\nProcessing: {file_path.name}")

        original_text = file_path.read_text(
            encoding="utf-8"
        )

        cleaned_text = preprocess_text(
            original_text
        )

        output_file = OUTPUT_DIR / file_path.name

        output_file.write_text(
            cleaned_text,
            encoding="utf-8"
        )

        print(
            f"Original characters: {len(original_text)}"
        )

        print(
            f"Cleaned characters: {len(cleaned_text)}"
        )

        print(
            f"Saved: {output_file}"
        )


if __name__ == "__main__":

    process_documents()