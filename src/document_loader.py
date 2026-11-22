from pathlib import Path
from docx import Document


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "extracted_text"
OUTPUT_DIR.mkdir(exist_ok=True)

def extract_docx(file_path):

    document = Document(file_path)
    extracted_content = []

    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if text:

            style = paragraph.style.name

            # Preserve heading information
            if style.startswith("Heading"):

                extracted_content.append(
                    f"\n{'=' * 10} {text} {'=' * 10}\n"
                )

            else:

                extracted_content.append(text)

    for table_number, table in enumerate(
        document.tables, start=1
    ):

        extracted_content.append(
            f"\n[TABLE {table_number}]"
        )

        for row in table.rows:

            row_data = [
                cell.text.strip().replace("\n", " ")
                for cell in row.cells
            ]

            extracted_content.append(
                " | ".join(row_data)
            )

    return "\n\n".join(extracted_content)


def extract_txt(file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8-sig"
    ) as file:

        return file.read()


def save_extracted_text(file_name, content):

    output_file = OUTPUT_DIR / f"{file_name}.txt"

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(content)

    print(f"Saved: {output_file}")


def main():

    documents = [
        "Claims User Manual.docx",
        "Regularisiation User Manual.docx",
        "Timecard User Manual.docx",
        "Work_Week_Email.txt"
    ]

    for file_name in documents:

        file_path = DATA_DIR / file_name

        if not file_path.exists():

            print(f"File not found: {file_name}")
            continue

        print(f"\nProcessing: {file_name}")

        if file_path.suffix.lower() == ".docx":

            content = extract_docx(file_path)

        elif file_path.suffix.lower() == ".txt":

            content = extract_txt(file_path)

        else:

            print("Unsupported file format")
            continue

        output_name = file_path.stem

        save_extracted_text(
            output_name,
            content
        )

        print(
            f"Extracted characters: {len(content)}"
        )


if __name__ == "__main__":

    main()