import random

import numpy as np
import pdfplumber
from skimage.color import rgb2lab
from click import style

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

    # Sample a horizontal line just ABOVE the text
    y = max(top - 2, 0)

    colors = []
    step = max((x1 - x0) // samples, 1)

    for x in range(x0, x1, step):
        try:
            colors.append(img.getpixel((x, y)))
        except Exception:
            pass

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

def filter_words(words, exclude_rects):
    kept = []

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

pdf = pdfplumber.open("page_004.pdf") # See note below
page = pdf.pages[0]
im = page.to_image(72)
pil_img = im.original

# get the width of the page
page_width = page.width
page_height = page.height
print(f"Page width: {page_width}")

im.debug_tablefinder({
    "explicit_vertical_lines": [ 230, 420 ],
    "explicit_horizontal_lines": [90]
})

# Get All the words on the page
words = page.extract_words()

# cut off the top and bottom bars
bottom_bar = {
        "x0": 0,
        "x1": page_width,
        "top": page_height - 60,
        "bottom": page_height
    }

top_bar = {
        "x0": 0,
        "x1": page_width,
        "top": 0,
        "bottom": 90
    }

im.draw_rect( top_bar, stroke="black", fill=None, stroke_width=2)
im.draw_rect( bottom_bar, stroke="black", fill=None, stroke_width=2)

rects = page.rects
images = page.images

exclude = []
# exclude.append(top_bar)
exclude.append(bottom_bar)

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

    # check the words, if the word "Introduction" is inside the rectangle, skip it
    has_intro = False
    for word in words:
        if word["text"].lower() == "introduction":
            if (word["x0"] >= r["x0"] and word["x1"] <= r["x1"] and
                word["top"] >= r["top"] and word["bottom"] <= r["bottom"]):
                has_intro = True
                break

    if not has_intro:
        exclude.append(r)
        im.draw_rect(r, stroke="black", fill=None, stroke_width=2)

for rect in rects:
    # im.draw_rect(rect, stroke="red", fill=None, stroke_width=5)
    exclude.append(rect)

# Cull the words, remove all that overlap with exclude rectangles
culled_words = filter_words(words, exclude)

# now consider that the page has three columns separators at x=230 and x=420
col_separators = [210, 410]

# now again, cull the words, retain only those in the first column
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

im.draw_rect(first_column, stroke="blue", fill=None, stroke_width=2)
im.draw_rect(second_column, stroke="red", fill=None, stroke_width=2)
im.draw_rect(third_column, stroke="green", fill=None, stroke_width=2)

first_column_words = filter_words(culled_words, [second_column, third_column])

words = first_column_words

clusters = []
threshold=15
for word in words:
    bg = word_background_color(word, pil_img)
    lab = rgb_to_lab(bg)

    placed = False
    for cluster in clusters:
        if color_distance(cluster["lab"], lab) < threshold:
            cluster["words"].append(word)
            placed = True
            break

    if not placed:
        clusters.append({
            "lab": lab,
            "words": [word]
        })

for c in clusters:
    print(c)

for cluster in clusters:
    color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
    for word in cluster["words"]:
        im.draw_rect(
            word,
            stroke=color,
            fill=None,
            stroke_width=2
        )

for cluster in clusters:
    cluster["bbox"] = cluster_bbox(cluster["words"])

for cluster in clusters:
    bbox = cluster["bbox"]
    color = "#{:06x}".format(random.randint(0, 0xFFFFFF))

    im.draw_rect(
        bbox,
        stroke=color,
        fill=None,
        stroke_width=4
    )

im.save("debug_page_3.png")