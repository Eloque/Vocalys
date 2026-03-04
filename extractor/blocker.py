import json
import random

import re
from pprint import pprint

import numpy as np
import pdfplumber
from pdfplumber.display import PageImage
from skimage.color import rgb2lab
from enum import Enum, auto
from PIL import Image

TRANSPARENT = (0, 0, 0, 0)

class PageType(Enum):
    TITLE = auto()
    SCENARIO = auto()
    CONTINUED_SCENARIO = auto()
    UNKNOWN = auto()

def get_headers_paragraphs(words, ignore_italic=False):
    paragraph_words = []
    header_words = []

    for word in words:
        if ignore_italic and word.get("italic"):
            continue

        h = word.get("height")

        if h == 10.0:
            paragraph_words.append(word)
        elif h == 12.0:
            header_words.append(word)

    return paragraph_words, header_words

def words_with_style(page, words):
    styled = []

    for w in words:
        chars = [
            c for c in page.chars
            if c["x0"] >= w["x0"]
            and c["x1"] <= w["x1"]
            and c["top"] >= w["top"]
            and c["bottom"] <= w["bottom"]
        ]

        if not chars:
            continue

        sizes = [c["size"] for c in chars]
        fonts = [c["fontname"] for c in chars]

        styled.append({
            **w,
            "fontname": max(set(fonts), key=fonts.count),
            "size": round(sum(sizes) / len(sizes), 2),
            "bold": any("bold" in f.lower() for f in fonts),
            "italic": any("italic" in f.lower() or "oblique" in f.lower() for f in fonts),
        })

    return styled

def bbox_to_rect(b):
    return {"x0": b[0], "top": b[1], "x1": b[2], "bottom": b[3]}

def rect_to_bbox(r):
    return (r["x0"], r["top"], r["x1"], r["bottom"])

def rgb_to_lab(rgb):
    rgb_arr = np.array([[rgb]], dtype=np.uint8) / 255.0
    lab = rgb2lab(rgb_arr)
    return lab[0][0]

def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)

def color_distance(lab1, lab2):
    return np.linalg.norm(lab1 - lab2)

def cluster_bbox(words):
    return {
        "x0": min(w["x0"] for w in words),
        "x1": max(w["x1"] for w in words),
        "top": min(w["top"] for w in words),
        "bottom": max(w["bottom"] for w in words),
    }

def word_background_color(word, img, samples=5):
    """
    word: pdfplumber word dict
    img: PIL image of page
    returns: (r, g, b)
    """
    x0, x1 = int(word["x0"]), int(word["x1"])
    top, bottom = int(word["top"]), int(word["bottom"])

    # No sample the 1ste pixel to right left int middle of the word
    sample_x = x1 + 1

    y = (top + bottom) / 2
    colors = []

    sample_color = img.getpixel((sample_x, top))
    colors.append(sample_color)

    # Average sampled colors
    r = sum(c[0] for c in colors) // len(colors)
    g = sum(c[1] for c in colors) // len(colors)
    b = sum(c[2] for c in colors) // len(colors)

    return (r, g, b)

def rects_overlap(a, b):
    return not (
            a["x1"] <= b["x0"] or
            a["x0"] >= b["x1"] or
            a["bottom"] <= b["top"] or
            a["top"] >= b["bottom"]
    )

# Given a bounding box and dimensions, return a new bounding box that is the inner rectangle defined by the dimensions and offset from the original bbox
def rect_inside(bbox, dx, dy, w, h):
    x0, top, x1, bottom = bbox
    return (x0 + dx, top + dy, x0 + dx + w, top + dy + h)

# Checks if the inner rectangle is fully contained within the outer rectangle
def rect_contains(outer, inner):
    return (
            inner["x0"] >= outer["x0"] and
            inner["x1"] <= outer["x1"] and
            inner["top"] >= outer["top"] and
            inner["bottom"] <= outer["bottom"]
    )

def words_in_bbox(words, bbox):
    kept = []

    x0, top, x1, bottom = bbox

    for w in words:
        if (
                w["x0"] >= x0 and
                w["x1"] <= x1 and
                w["top"] >= top and
                w["bottom"] <= bottom
        ):
            kept.append(w)

    return kept

def merge_words_same_line(merge_words, tol=0.0, sort_within_line=False):
    merged = {}  # key -> {"first_i": int, "words": [...]}

    for i, w in enumerate(merge_words):
        top = w["top"] if tol == 0.0 else round(w["top"] / tol) * tol
        bot = w["bottom"] if tol == 0.0 else round(w["bottom"] / tol) * tol
        key = (top, bot)

        if key not in merged:
            merged[key] = {"first_i": i, "words": []}

        merged[key]["words"].append(w)

    out = []
    for (top, bot), entry in merged.items():
        group = entry["words"]

        # Preserve original order unless you explicitly want left-to-right.
        if sort_within_line:
            group = sorted(group, key=lambda x: x["x0"])

        text = " ".join(g["text"] for g in group)
        x0 = min(g["x0"] for g in group)
        x1 = max(g["x1"] for g in group)

        out.append({
            "_first_i": entry["first_i"],  # internal sort key
            "text": text,
            "x0": x0,
            "x1": x1,
            "top": top,
            "bottom": bot,
        })

    # Preserve original “first seen” line order
    out.sort(key=lambda x: x["_first_i"])

    # Cleanup internal key
    for item in out:
        del item["_first_i"]

    return out

def remove_phrase(words, phrase):
    """
    Remove a phrase from a list of words.

    Args:
        words: List of word dictionaries (each with a "text" key)
        phrase: List of strings representing the phrase to remove (e.g., ["Continued", "on", "next", "page"])

    Returns:
        List of words with the phrase removed (if found)
    """
    if not phrase or not words:
        return words

    phrase_len = len(phrase)

    # Iterate through the words looking for the phrase sequence
    for i in range(len(words) - phrase_len + 1):
        # Check if the next phrase_len words match the phrase (case-insensitive)
        match = True
        for j in range(phrase_len):
            if words[i + j]["text"].lower() != phrase[j].lower():
                match = False
                break

        if match:
            # Found the phrase, remove it

            del words[i:i + phrase_len]
            print(f"Removed phrase '{' '.join(phrase)}' from words")
            break  # Only remove the first occurrence

    return words


def split_paragraphs_into_columns(paragraphs_sections, column_list, header_words):
    """
    Split paragraphs into columns and handle headers inside columns.

    Returns the modified paragraphs_sections list.
    """
    for paragraph in paragraphs_sections:
        paragraph["columns"] = []

        for column_rect in column_list:
            paragraph_col_words = words_in_bbox(
                paragraph["words"],
                rect_to_bbox(column_rect)
            )

            if not paragraph_col_words:
                continue

            for header in header_words:
                header_rect = {
                    "x0": header["x0"],
                    "x1": header["x1"],
                    "top": header["top"],
                    "bottom": header["bottom"],
                }

                if rect_contains(cluster_bbox(paragraph_col_words), header_rect):
                    # print("Found header inside column:", header["text"])

                    dividing_y = header["top"] + (header["bottom"] - header["top"]) / 2

                    above_header_words = [
                        w for w in paragraph_col_words if w["bottom"] <= dividing_y
                    ]
                    below_header_words = [
                        w for w in paragraph_col_words if w["top"] >= dividing_y
                    ]

                    if above_header_words:
                        paragraphs_sections.append({
                            "words": above_header_words,
                            "bbox": cluster_bbox(above_header_words),
                        })

                    paragraph_col_words = below_header_words

            # the header can also be above the column.
            # Consider this, draw a line from the column straight up.
            # If that line crosses a header before it crosses another column
            # Then we consider that a split as well.
            # But only if this is not the first column in the paragraph.
            # If the line crosses no other columns or headers that it is also fine

            skip_because_header_found = False

            # don't do this for the first column
            if len(paragraph["columns"]) != 0:
                for header in header_words:
                    header_rect = {
                        "x0": header["x0"],
                        "x1": header["x1"],
                        "top": header["top"],
                        "bottom": header["bottom"],
                    }

                    check_bbox = cluster_bbox(paragraph_col_words)

                    header_above_column = (
                            header_rect["bottom"] <= check_bbox["top"] and
                            min(header_rect["x1"], check_bbox["x1"]) > max(header_rect["x0"], check_bbox["x0"])
                    )

                    if header_above_column:
                        # print("Found header above column:", header["text"])
                        distance = check_bbox["top"] - header_rect["bottom"]

                        if abs(distance - 6.668) <= 0.1:
                            # print("distance is within tolerance, column is it's own paragraph")
                            paragraphs_sections.append({
                                "words": paragraph_col_words,
                                "bbox": check_bbox,
                            })
                            skip_because_header_found = True
                            break

            # Check if found, then break
            if skip_because_header_found:
                continue

            column = {
                "words": paragraph_col_words,
                "bbox": cluster_bbox(paragraph_col_words)
            }

            paragraph["columns"].append(column)

    return paragraphs_sections

def marry_headers_to_paragraphs(header_words, paragraphs_sections, debug=False):
    """
    Match each header to the closest paragraph (by vertical distance) whose column starts below it
    and overlaps horizontally.

    Mutates paragraphs_sections by popping matched paragraphs (same behavior as your inline code).

    Returns:
        sections: list of {"header": str, "text": str}
    """
    sections = []

    for word in header_words:
        if debug:
            print("Considering header:", word["text"], word.get("top"), word.get("bottom"))

        bottom_of_header = word["bottom"]
        delta_y = float("inf")
        candidate = None

        for index, paragraph in enumerate(paragraphs_sections):
            for column in paragraph.get("columns", []):
                col_bbox = column["bbox"]

                # does this column start below the bottom of the header?
                if col_bbox["top"] >= bottom_of_header:
                    # does the column start beyond the farthest right of the header?
                    if not col_bbox["x0"] >= word["x1"]:
                        # does the column end before the farthest left of the header?
                        if not col_bbox["x1"] <= word["x0"]:
                            new_delta_y = col_bbox["top"] - bottom_of_header

                            if delta_y > new_delta_y:
                                delta_y = new_delta_y
                                candidate = index

        if candidate is not None:
            section = {
                "header": word["text"],
                "text": "",
            }

            for column in paragraphs_sections[candidate].get("columns", []):
                section["text"] += " " + words_to_text(column["words"])


            # remove the candidate so we don't match it again
            paragraphs_sections.pop(candidate)

            sections.append(section)

    return sections


# This will group words into sections based on their background color.
# It will return a list of sections, where each section is a dict with a "lab" key
# for the average LAB color and a "words" key for the list of words in that section.

def get_sections(words, pil_img):
    sections = []
    threshold = 4
    for word in words:
        bg = word_background_color(word, pil_img)
        lab = rgb_to_lab(bg)

        placed = False
        for section in sections:
            if color_distance(section["lab"], lab) < threshold:
                section["words"].append(word)
                placed = True
                break

        if not placed:
            sections.append({
                "lab": lab,
                "words": [word]
            })

    for section in sections:
        section["bbox"] = cluster_bbox(section["words"])

    return sections

def draw_sections(sections, target_img):
    for section in sections:
        bbox = section["bbox"]
        color = "#{:06x}".format(random.randint(0, 0xFFFFFF))

        target_img.draw_rect(
            bbox,
            stroke=color,
            fill=TRANSPARENT,
            stroke_width=1
        )

def filter_words(words, exclude_rects):
    kept = []

    if isinstance(exclude_rects, tuple):
        exclude_rects = [exclude_rects]

    for w in words:
        word_rect = {
            "x0": w["x0"],
            "x1": w["x1"],
            "top": w["top"],
            "bottom": w["bottom"],
        }

        if any(rects_overlap(word_rect, ex) for ex in exclude_rects):
            continue

        kept.append(w)

    return kept

def words_to_text(words):
    return " ".join(w["text"] for w in words)


# Functions specifically for Frosthaven scenario book
def find_header(images, expected_height=91.68):
    best = None
    best_width = -1
    best_hdiff = float("inf")

    for img in images:
        x0 = img["x0"]
        x1 = img["x1"]
        top = img.get("top", img["y0"])
        bottom = img.get("bottom", img["y1"])

        width = x1 - x0
        height = bottom - top
        hdiff = abs(height - expected_height)

        # widest wins, height closeness breaks ties
        if width > best_width or (width == best_width and hdiff < best_hdiff):
            best_width = width
            best_hdiff = hdiff
            best = (x0, top, x1, bottom)

    # Determine the page type, by the header height
    # Okay we need magic numbers here
    title_page_header_height = 113.07
    scenario_page_header_height = 91.68
    continued_page_header_height = 48.00

    type = PageType.UNKNOWN

    # Check if it's close, within 1 point of the heights above
    if abs((best[3] - best[1]) - title_page_header_height) < 1.0:
        type = PageType.TITLE
    elif abs((best[3] - best[1]) - scenario_page_header_height) < 1.0:
        type = PageType.SCENARIO
    elif abs((best[3] - best[1]) - continued_page_header_height) < 1.0:
        type = PageType.CONTINUED_SCENARIO
    else:
        type = PageType.UNKNOWN

    return type, best  # (x0, top, x1, bottom) or None

def get_scenario(words, header):
    title_rect = rect_inside(header, 110, 24, 430, 38)
    reference_rect = rect_inside(header, 35, 22, 70, 38)

    title_words = words_in_bbox(words, title_rect)
    reference_words = words_in_bbox(words, reference_rect)

    title_text = words_to_text(title_words)
    reference_text = words_to_text(reference_words)

    return title_text, reference_text


def get_number_location(words, header):
    reference_rect = rect_inside(header, 35, 22, 70, 38)
    reference_words = words_in_bbox(words, reference_rect)
    reference_text = words_to_text(reference_words)

    match = re.match(r"(\d+)\s*•\s*(\w+)", reference_text)
    if not match:
        return None, None

    return match.group(1), match.group(2)

# End function specifically for Frosthaven scenario book

