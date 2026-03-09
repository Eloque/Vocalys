# this take text file, and splits it into chunks for processing
import re

def new_chunk_text(text, max_chunk_size=900, overlap_chars=0):
    text = re.sub(r'\s+', ' ', text).strip()  # normalize all whitespace

    # Split on punctuation + any whitespace
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

    chunks = []
    current = ""

    for sentence in sentences:
        if current and (len(current) + 1 + len(sentence) > max_chunk_size):
            chunks.append(current)

            # overlap tail from previous chunk
            tail = current[-overlap_chars:].lstrip()
            current = f"{tail} {sentence}".strip()
        else:
            current = f"{current} {sentence}".strip() if current else sentence

    if current:
        chunks.append(current)

    for c in chunks:
        print(f"Chunk ({len(c)} chars): {c[:120]}...")

    for i in range(1, len(chunks)):
        print("Overlap:")
        print("PREV TAIL:", chunks[i - 1][-80:])
        print("NEXT HEAD:", chunks[i][:80])
        print()

    return chunks

def chunk_text(text, max_chunk_size=120):
    """
    Splits the input text into chunks of approximately max_chunk_size characters,
    trying to split at sentence boundaries.

    Args:
        text (str): The input text to be chunked.
        max_chunk_size (int): The maximum size of each chunk in characters.

    Returns:
        List[str]: A list of text chunks.
    """

    ## Thing is, the text is from PDF and has a lot of newlines in the middle of sentences.
    ## So first we replace newlines with spaces
    text = text.replace('\n', ' ')

    # Remove the weird unicodes
    text = text.replace("“", '"').replace("”", '"')

    # Treat em/en dashes as a ! or ,
    text = re.sub(r"\s*[—–]\s*", "!", text)
    # text = re.sub(r"\s*[—–]\s*", ", ", text)

    # There are specific words the defy pronunciation, split them for the TTS to pronounce them better
    text = text.replace("Snowspeaker", "Snow Speaker")
    text = text.replace("Icespeaker", "Ice Speaker")

    # Split text into sentences using regex
    # sentences = re.split(r'(?<=[.!?]) +', text)

    sentences = re.split(r'(?:(?<=[.!?])|(?<=[.!?]["\”’]))\s+(?=[A-Z"“])', text)

#    for sentence in sentences:
#        print(sentence)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # If adding the next sentence exceeds the max chunk size, save the current chunk
        if len(current_chunk) + len(sentence) + 1 > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence

    # Add any remaining text as the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

#    for chunk in chunks:
#        print(f"Chunk ({len(chunk)} chars): {chunk[:120+120]}")

    return chunks


