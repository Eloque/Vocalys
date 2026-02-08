import random
from pprint import pprint

import re
import numpy as np
import pdfplumber
from skimage.color import rgb2lab

TRANSPARENT = (0, 0, 0, 0)

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

def merge_words_same_line(merge_words, tol=0.0):
    merged = {}

    for w in merge_words:
        # optionally quantize to avoid tiny float differences
        top = w["top"] if tol == 0.0 else round(w["top"] / tol) * tol
        bot = w["bottom"] if tol == 0.0 else round(w["bottom"] / tol) * tol
        key = (top, bot)

        merged.setdefault(key, []).append(w)

    out = []
    for (top, bot), group in merged.items():
        group.sort(key=lambda x: x["x0"])  # left-to-right

        text = " ".join(g["text"] for g in group)
        x0 = min(g["x0"] for g in group)
        x1 = max(g["x1"] for g in group)

        out.append({
            "text": text,
            "x0": x0,
            "x1": x1,
            "top": top,
            "bottom": bot,
        })

    # keep output in reading order (top-to-bottom, left-to-right)
    out.sort(key=lambda x: (x["top"], x["x0"]))
    return out

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
            stroke_width=4
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

    return best  # (x0, top, x1, bottom) or None

def get_scenerario(words, header):
    title_rect = rect_inside(header, 110, 24, 430, 38)
    reference_rect = rect_inside(header, 35, 22, 70, 38)

    title_words = words_in_bbox(words, title_rect)
    reference_words = words_in_bbox(words, reference_rect)

    title_text = words_to_text(title_words)
    reference_text = words_to_text(reference_words)

    return title_text, reference_text

# End function specifically for Frosthaven scenario book
pdf = pdfplumber.open("input/page_007.pdf")  # See note below
page = pdf.pages[0]
im = page.to_image(72)
pil_img = im.original

# get the width of the page
page_width = page.width
page_height = page.height

print(f"Page width: {page_width}")

# Get All the words on the page
words = page.extract_words()
rects = page.rects
images = page.images

exclude = []

# remove the first image (it's the full page background)
images = images[1:]

for image in images:
    # convert image into to a rectangle
    x0 = image["x0"]
    x1 = image["x1"]
    top = image.get("top", image["y0"])
    bottom = image.get("bottom", image["y1"])

    r = {
        "x0": x0,
        "x1": x1,
        "top": top,
        "bottom": bottom
    }

for rect in rects:
    # im.draw_rect(rect, stroke="red", fill=None, stroke_width=5)
    exclude.append(rect)

# Cull the words, remove all that overlap with exclude rectangles
culled_words = filter_words(words, exclude)

# the analysis portion
scenario = dict()

# Find the header based on the images
header = find_header(images)
im.draw_rect(header, stroke="blue", fill=TRANSPARENT, stroke_width=5)

# Find the title and reference based on the header position and expected offsets
title = rect_inside(header, 110, 24, 430, 38)
reference = rect_inside(header, 35, 22, 70, 38)
im.draw_rect(title, stroke="red", fill=TRANSPARENT, stroke_width=5)
im.draw_rect(reference, stroke="red", fill=TRANSPARENT, stroke_width=5)

# get the columns based on the header position
col_separators = [header[0] + 215, header[0] + 210 + 190]

second_column = {
    "x0": col_separators[0],
    "x1": col_separators[1],
    "top": 0,
    "bottom": page_height}

first_column = {
    "x0": 0,
    "x1": col_separators[0],
    "top": 0,
    "bottom": page_height}

third_column = {
    "x0": col_separators[1],
    "x1": page_width,
    "top": 0,
    "bottom": page_height}

im.draw_rect(first_column, stroke="blue", fill=TRANSPARENT)
im.draw_rect(second_column, stroke="blue", fill=TRANSPARENT)
im.draw_rect(third_column, stroke="blue", fill=TRANSPARENT)

scenario["title"] = words_to_text(words_in_bbox(words, title))
reference = words_in_bbox(words, reference)

# reference is two bits of text, a number, a •, and then a code.
# need a regex to split that into the number and the code
reference = words_to_text(reference)

match = re.match(r"(\d+)\s*•\s*(\w+)", reference)
if match:
    scenario["number"] = match.group(1)
    scenario["location"] = match.group(2)

# Now, we need to find the scenario "Introduction" text
# That means find all the headers.
# First in words find the word "Introduction"

# We will exclude all words that are in the header
words = filter_words(words, [bbox_to_rect(header)])

# Divide it up into headers and paragraphs
paragraph_words = list()
header_words = list()
# find the different font sizes
for word in words:
    if word["height"] == 10.0:
        paragraph_words.append(word)

    if word["height"] == 12.0:
        header_words.append(word)

paragraphs_sections = get_sections(paragraph_words, pil_img)
draw_sections(paragraphs_sections, im)
sections_to_draw = []

scenario["sections"] = list()

for paragraph in paragraphs_sections:
    # get all the text in the paragraph that falls in the first column
    paragraph_first_col_words = words_in_bbox(paragraph["words"], rect_to_bbox(first_column))
    paragraph_second_col_words = words_in_bbox(paragraph["words"], rect_to_bbox(second_column))
    paragraph_third_col_words = words_in_bbox(paragraph["words"], rect_to_bbox(third_column))

    paragraph["columns"] = list()

    if paragraph_first_col_words:
        column = {
            "words": paragraph_first_col_words,
            "bbox": cluster_bbox(paragraph_first_col_words)
        }

        paragraph["columns"].append(column)
        sections_to_draw.append(cluster_bbox(paragraph_first_col_words))

    if paragraph_second_col_words:
        column = {
            "words": paragraph_second_col_words,
            "bbox": cluster_bbox(paragraph_second_col_words)
        }

        paragraph["columns"].append(column)
        sections_to_draw.append(cluster_bbox(paragraph_second_col_words))

    if paragraph_third_col_words:
        column = {
            "words": paragraph_third_col_words,
            "bbox": cluster_bbox(paragraph_third_col_words)
        }

        paragraph["columns"].append(column)
        sections_to_draw.append(cluster_bbox(paragraph_third_col_words))

for section in sections_to_draw:
    im.draw_rect(section, stroke="green", fill=TRANSPARENT, stroke_width=4)

headers = []
header_words = merge_words_same_line(header_words)

for word in header_words:
    print(word["text"], word["top"], word["bottom"])

    color = "#{:06x}".format(random.randint(0, 0xFFFFFF))

    im.draw_rect(
        word,
        stroke=color,
        fill=TRANSPARENT,
        stroke_width=4
    )

    bottom_of_header = word["bottom"]

    delta_y = float("inf")
    candidate = None

    for index, paragraph in enumerate(paragraphs_sections):
        for column in paragraph["columns"]:
            # # does this column start below the bottom of the header?
            if column["bbox"]["top"] >= bottom_of_header:
                # does the column start beyond the farthest right of the header?
                if not column["bbox"]["x0"] >= word["x1"]:

                    # calculate the delta_y
                    new_delta_y = column["bbox"]["top"] - bottom_of_header
                    para = words_to_text(column["words"])
                    print(new_delta_y, bottom_of_header, column["bbox"]["top"], para)

                    if delta_y > new_delta_y:
                        delta_y = new_delta_y
                        candidate = index

    if candidate is not None:
        # We have a match, and can marry the header to the paragraph
        section = dict()
        section["header"] = word["text"]
        section["text"] = ""

        for column in paragraphs_sections[candidate]["columns"]:
            paragraph_text = words_to_text(column["words"])
            section["text"] += paragraph_text

        # remove the candidate from the paragraphs_sections so we don't match it again
        paragraphs_sections.pop(candidate)

        scenario["sections"].append(section)

    print()

im.save("blocker.png")
print("Results in JSON")
pprint(scenario, width=120)

# save this scenario to a json file
import json
filename = "../input/scenarios/" + scenario["number"] + " - " + scenario["title"] + ".json"

with open(filename, "w") as f:
    json.dump(scenario, f, indent=4)
    f.close()