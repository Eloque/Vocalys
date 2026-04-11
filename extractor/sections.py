import json

import pdfplumber

from blocker import *

# get a list of all the files in the input directory
import os

input_folder = "../input/books"
input_files = [f for f in os.listdir(input_folder) if f.startswith("stripped-section-book") and f.endswith(".pdf")]

print("Input files:", input_files)
sections_book = list()

for entry in input_files:
    pdf_file = os.path.join(input_folder, entry)
    pdf = pdfplumber.open(pdf_file)  # See note below
    print("Processing file:", entry)

    page_offset = 0

    for page_number, page in enumerate(pdf.pages[page_offset:]):

        try:
            # First, get some basic data, load the page
            # Retrieve an image and save it.
            # page = pdf.pages[0]
            im = page.to_image(72)
            pil_img = im.original

            pil_img.save(f"output/processing.png")

            # get the width of the page
            page_width = page.width
            page_height = page.height

            # Get All the words on the page
            words = page.extract_words()

            rects = page.rects
            images = page.images

            if len(images) == 0:
                print("First page or no images, skipping")
                continue

            for image in images:
                # convert image into to a rectangle
                x0 = image["x0"]
                x1 = image["x1"]
                top = image.get("top", image["y0"])
                bottom = image.get("bottom", image["y1"])
                image["bbox"] = (x0, top, x1, bottom)

            # draw_images(images, im)
            # im.save(f"output/processing.png")

            header_images = list()

            # go through all images
            # Find the text inside it, see if the first word matches a format of nn.nn
            for image in images:
                bbox = image["bbox"]
                text = words_in_bbox(words, bbox)
                text = words_to_text(text)

                # check if the height is consistent with a title bar
                delta = image["bottom"] - image["top"]
                if delta < 20:
                    # use a regex to check if the first word is a number
                    if text and re.match(r'^\d+\.\d+$', text.split()[0]):
                        image["section"] = text
                        header_images.append(image)

            # each of these headers is the start of a block
            # for each header, cast a line down, until it hits either another header
            blocks = list()

            for header in header_images:
                bottom_y = header["bottom"]

                delta = float("inf")
                new_block = dict()

                # check if there are headers below this one
                # if there are, cast a line down until we hit a header or the bottom of the page

                # check if there is a header below this one
                # if there is, cast a line down until we hit a header or the bottom of the page
                for other_header in header_images:
                    if other_header["top"] > bottom_y:

                        # check if the left_most x of the other header is at greater or equal the left most x of this header
                        if other_header["x1"] >= header["x0"]:
                            # Only the smallest delta is considered
                            new_delta = other_header['top'] - bottom_y

                            if new_delta < delta:

                                # this header is under the block we are currently looking at
                                new_block["bbox"] = (header['x0'], header['top'],
                                                     header['x1'], other_header['top'])

                                new_block["section"] = header["section"]
                                new_block["header"] = header

                                delta = new_delta

                if delta == float("inf"):
                    # this header is under the block we are currently looking at
                    new_block["bbox"] = (header['x0'], header['top'],
                                         header['x1'], page_height)

                    new_block["section"] = header["section"]
                    new_block["header"] = header

                # expand the new block, give it 5 px left and right padding
                new_block["bbox"] = (new_block["bbox"][0] - 5, new_block["bbox"][1],
                                     new_block["bbox"][2] + 5, new_block["bbox"][3])

                blocks.append(new_block)

            draw_sections(blocks, im)
            im.save(f"output/processing.png")

            # At this point we got the blocks. We should go over them block by block
            # First get the min x on the blocks
            min_x = min(block["bbox"][0] for block in blocks)
            column_offset = 0
            entry = dict()
            entry["sections"] = list()

            col_separators = [min_x + 185 + column_offset, min_x + 185 + column_offset + 190]
##
            # x boundaries from left edge -> separators -> right edge
            x_edges = [0, *col_separators, page_width]

            column_list = [
                {
                    "x0": x_edges[i],
                    "x1": x_edges[i + 1],
                    "top": 0,
                    "bottom": page_height,
                }
                for i in range(len(x_edges) - 1)
            ]

            for column in column_list:
                bbox = (column["x0"], 0, column["x1"], page_height)

                im.draw_rect(bbox,
                             stroke="red",
                             fill=TRANSPARENT,
                             stroke_width=1)

            im.save(f"output/processing.png")

            for block in blocks:
                # print(block["section"])

                section = dict()
                # For now, this needs splitting
                text = block["section"]

                m = re.match(r'^([\d.]+)\s*•\s*(.*?)\s*(?:\((\d+)\))?$', text)

                section["number"] = m.group(1)
                section["title"] = m.group(2)
                section["reference"] = m.group(3)

                section_words = words_in_bbox(words, block["bbox"])

                # exclude the header from the section
                # create exclusion rect
                exclude = block["header"]["bbox"]
                exclude = bbox_to_rect(exclude)

                section_words = filter_words(section_words, [exclude])
                text = words_to_text(section_words)

                paragraph_words, header_words = get_headers_paragraphs(section_words)

                # The headers are all one-liners
                header_words = merge_words_same_line(header_words)

                # Divide into sections based on background colors
                paragraphs_sections = get_sections(paragraph_words, pil_img)

                paragraphs_sections = split_paragraphs_into_columns(
                    paragraphs_sections,
                    column_list,
                    header_words,
                )

                marriage = marry_headers_to_paragraphs(header_words, paragraphs_sections)

                # Now check if there is header called "Conclusion" if there is not, then the
                # scenario text is not added. We will need to add a header called "Continuation"
                # and add the largest remainig text to it, if there still is one.
                concluded = False

                # marriage is a list, check all the list items
                for paragraph in marriage:
                    if paragraph["header"] == "Conclusion":
                        concluded = True
                        continue

                sections = marriage
                section["sections"] = sections

                if not concluded:
                    item = dict()
                    item["header"] = "Continuation"

                    # get the largest remaining text
                    largest_text = 0
                    candidate = None

                    for paragraph in paragraphs_sections:
                        length = len(paragraph["words"])
                        if length > largest_text:
                            largest_text = length
                            candidate = paragraph

                    if candidate is not None:
                        item["text"] = ""

                        for column in candidate.get("columns", []):
                            item["text"] += " " + words_to_text(column["words"])

                        section["sections"].append(item)

                sections_book.append(section)

                im.save(f"output/processing.png")

            # save this book to a json file
            filename = "../input/scenarios/sections_book.json"

            with open(filename, "w") as f:
                json.dump(sections_book, f, indent=4, ensure_ascii=False)
                f.close()

        except Exception as e:
            print("Error processing file:", entry)
            print(page_number)
            print(e)

input_file = "../input/scenarios/sections_book.json"
book = json.load(open(input_file))

# remove all items that don't have a number key, those are wrong
book = [b for b in book if "number" in b]

# sort on number key
book.sort(key=lambda x: int(x["number"]))

# save this book to a json file
with open(input_file, "w") as f:
    json.dump(book, f, indent=4, ensure_ascii=False)
    f.close()

