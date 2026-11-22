import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "preprocessed_text"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "chunks.json"


MAX_CHUNK_SIZE = 1400 # Maximum chunk size in characters
CHUNK_OVERLAP = 150 # Number of characters to preserve as overlap
DOCUMENT_CONFIG = {
    "Claims User Manual": {
        "document_type": "user_manual",
        "headings": [
            "Preface:",
            "Pre-requisites for initiating a Claim:",
            "Process Workflow:",
            "Sequence of steps and actions for Claim Creation:",
            "Navigation:",
            "Claim Reimbursement Dashboard:",
            "Entering a Claim:",
        ],
    },

    "Regularisiation User Manual": {
        "document_type": "user_manual",
        "headings": [
            "Preface:",
            "ATTENDANCE REGULARIZATION OPTIONS:",
            "Regularization:",
            "Work from home:",
            "On-site:",
            "Forgot-id card:",
            "Missed to Swipe:",
            "Pre-requisites:",
            "Workflow:",
            "Sequence of steps and actions:",
            "To Apply",
            "1. Work from home:",
            "2. On-Site:",
            "3. Forgot id card:",
            "4. Missed to swipe:",
        ],
    },

    "Timecard User Manual": {
        "document_type": "user_manual",
        "headings": [
            "Preface:",
            "Pre-requisites for initiating a Timecard Submission :",
            "Sequence of steps and actions:",
            "Timecard Policy Definition:",
            "Navigation:",
            "Daily Timecard UI:",
            "Weekly Timecard UI",
            "Timecard Submission:",
            "Timecard Approvals:",
            "Mass Timecard Approval:",
        ],
    },

    "Work_Week_Email": {
        "document_type": "policy_email",
        "headings": [
            "Eligibility",
            "Mandate Attendance Regularization",
            "Less than 9 Hours Clock-in",
            "Client Site Guidelines",
            "Salary Impact",
            "Exceptions",
        ],
    },
}

def get_document_config(file_name):

    for document_name, config in DOCUMENT_CONFIG.items():

        if document_name.lower() in file_name.lower():

            return document_name, config

    return None, None

def is_heading(line, heading_list):

    cleaned_line = line.strip()

    if not cleaned_line:
        return False

    for heading in heading_list:

        if cleaned_line.lower() == heading.lower():
            return True

    # Detect numbered section headings such as:
    # A. Preface:
    # B. Pre-requisites:
    if re.match(r"^[A-Z]\.\s+.+", cleaned_line):
        return True

    return False

def extract_sections(text, heading_list):

    lines = text.splitlines()

    sections = []

    current_heading = "General Information"

    current_content = []

    for line in lines:

        if is_heading(line, heading_list):

            if current_content:

                sections.append({
                    "section": current_heading,
                    "text": "\n".join(current_content).strip()
                })

            current_heading = line.strip()

            current_content = []

        else:

            current_content.append(line)

    if current_content:

        sections.append({
            "section": current_heading,
            "text": "\n".join(current_content).strip()
        })

    return sections

def recursive_split(text, max_size=MAX_CHUNK_SIZE):

    text = text.strip()
    if len(text) <= max_size:

        return [text]

    # Try splitting using paragraphs first
    paragraphs = re.split(r"\n\s*\n", text)

    # If there are multiple paragraphs, group them
    # without exceeding the maximum size.
    if len(paragraphs) > 1:

        chunks = []

        current_chunk = ""

        for paragraph in paragraphs:

            paragraph = paragraph.strip()

            if not paragraph:
                continue

            candidate = (
                current_chunk + "\n\n" + paragraph
                if current_chunk
                else paragraph
            )

            if len(candidate) <= max_size:

                current_chunk = candidate

            else:

                if current_chunk:
                    chunks.append(current_chunk)

                if len(paragraph) > max_size:

                    chunks.extend(
                        recursive_split(paragraph, max_size)
                    )

                    current_chunk = ""

                else:

                    current_chunk = paragraph

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    # If there are no paragraph boundaries,
    # try splitting at sentence boundaries.
    sentences = re.split(r"(?<=[.!?])\s+", text)

    if len(sentences) > 1:

        chunks = []

        current_chunk = ""

        for sentence in sentences:

            candidate = (
                current_chunk + " " + sentence
                if current_chunk
                else sentence
            )

            if len(candidate) <= max_size:

                current_chunk = candidate

            else:

                if current_chunk:
                    chunks.append(current_chunk)

                if len(sentence) > max_size:

                    chunks.extend(
                        recursive_split(sentence, max_size)
                    )

                    current_chunk = ""

                else:

                    current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    # Final fallback:
    # Split at the nearest available space.
    split_position = text.rfind(" ", 0, max_size)

    if split_position == -1:
        split_position = max_size

    first_part = text[:split_position].strip()

    remaining_text = text[split_position:].strip()

    return (
        [first_part]
        + recursive_split(remaining_text, max_size)
    )

def add_overlap(chunks, overlap=CHUNK_OVERLAP):

    if len(chunks) <= 1:
        return chunks

    overlapped_chunks = []

    for index, chunk in enumerate(chunks):
        if index == 0:
            overlapped_chunks.append(chunk)
            continue

        previous_chunk = chunks[index - 1]

        sentences = re.split(r"(?<=[.!?])\s+", previous_chunk.strip())
        overlap_sentences = []
        overlap_length = 0

        for sentence in reversed(sentences):

            if not sentence.strip():
                continue

            # Account for spaces between sentences
            separator_length = 1 if overlap_sentences else 0

            sentence_length = len(sentence)
            if overlap_length + sentence_length + separator_length > overlap:
                break

            overlap_sentences.insert(0, sentence)

            overlap_length += sentence_length + separator_length

        overlap_text = " ".join(overlap_sentences)

        if overlap_text:

            combined_chunk = overlap_text + "\n" + chunk

            if len(combined_chunk) <= MAX_CHUNK_SIZE:
                overlapped_chunks.append(combined_chunk)

            else:
                overlapped_chunks.append(chunk)

        else:
            overlapped_chunks.append(chunk)

    return overlapped_chunks

def create_document_chunks(file_path):

    document_name, config = get_document_config(
        file_path.name
    )

    if config is None:

        print(f"Skipping unknown document: {file_path.name}")

        return []

    text = file_path.read_text(encoding="utf-8")

    sections = extract_sections(
        text,
        config["headings"]
    )

    document_chunks = []

    chunk_number = 1

    for section_data in sections:

        section_name = section_data["section"]

        section_text = section_data["text"]

        if not section_text:
            continue

        split_chunks = recursive_split(section_text)

        split_chunks = add_overlap(split_chunks)

        for chunk_text in split_chunks:

            chunk = {

                "chunk_id": (
                    f"{document_name.lower().replace(' ', '_')}"
                    f"_{chunk_number:03d}"
                ),

                "document_name": document_name,

                "document_type": config["document_type"],

                "section": section_name,

                "topic": section_name.replace(":", "").strip(),

                "text": chunk_text,

            }

            document_chunks.append(chunk)

            chunk_number += 1

    return document_chunks

def process_all_documents():

    all_chunks = []

    files = list(INPUT_DIR.glob("*.txt"))

    if not files:

        print("No preprocessed documents found.")

        return

    for file_path in files:

        print(f"\nProcessing: {file_path.name}")

        chunks = create_document_chunks(file_path)

        all_chunks.extend(chunks)

        print(f"Chunks generated: {len(chunks)}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

        json.dump(
            all_chunks,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n--------------------------------")

    print("CHUNKING COMPLETED")

    print(f"Total chunks: {len(all_chunks)}")

    print(f"Output saved at: {OUTPUT_FILE}")

    print("--------------------------------")

if __name__ == "__main__":

    process_all_documents()