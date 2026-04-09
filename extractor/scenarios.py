from blocker import *

# get a list of all the files in the input directory
import os

input_folder = "./input/books"
input_files = [f for f in os.listdir(input_folder) if f.startswith("stripped-scenario-book") and f.endswith(".pdf")]

print("Input files:", input_files)
book = list()

for entry in input_files:
    pdf_file = os.path.join(input_folder, entry)
    pdf = pdfplumber.open(pdf_file)  # See note below
    print("Processing file:", entry)

    page_offset = 0

    for page_number, page in enumerate(pdf.pages[page_offset:]):

        try:
            # First get some basic data, load the page
            # Retrieve an image and save it.
            # page = pdf.pages[0]
            im = page.to_image(72)
            pil_img = im.original

            pil_img.save(f"processing.png")

            # get the width of the page
            page_width = page.width
            page_height = page.height

            # Get All the words on the page
            words = page.extract_words()

            rects = page.rects
            images = page.images

            exclude = []

            printable_words = [w for w in words if w["text"].strip()]
            words = words_with_style(page, words)

            # remove the first image (it's the full page background)
            # images = images[1:]

            for image in images:
                # convert image into to a rectangle
                x0 = image["x0"]
                x1 = image["x1"]
                top = image.get("top", image["y0"])
                bottom = image.get("bottom", image["y1"])

            # Need to find out if this page is a regular scenario page or not.
            # This is done by finding the header
            # Find the header based on the images
            page_type, header = find_header(images)
            if page_type != PageType.UNKNOWN:
                im.draw_rect(header, stroke="black", fill=TRANSPARENT, stroke_width=2)

            # the analysis portion
            # if page_type != PageType.UNKNOWN:
            entry = dict()
            entry["sections"] = list()
            entry["type"] = page_type.name

            match page_type:
                case PageType.TITLE:

                    # first, find the title.
                    # It's is the text in center of the header
                    center_x = (header[0] + header[2]) / 2
                    width = header[2] - header[0] - 60

                    title = rect_inside(header, 60/2, 28, width, 38)
                    title_words = words_in_bbox(words, title)
                    title_text = words_to_text(title_words)
                    entry["title"] = title_text
                    im.draw_rect(title, stroke="red", fill=TRANSPARENT, stroke_width=2)

                    # Divide it up into headers and paragraphs
                    paragraph_words = list()
                    header_words = list()
                    # find the different font sizes
                    for word in words:

                        if word["height"] == 37.56820000000005 or word["height"] == 12.0 or word["height"] == 10.0:
                            paragraph_words.append(word)

                        if word["height"] == 12.0 or word["height"] == 24.0:
                            # this might be a header, but on the title page,
                            # we need to check the background color, since the text is also in size 12
                            word_color = word_background_color(word, im.original)
                            if word_color != (255, 255, 255):
                                header_words.append(word)

                    paragraphs_sections = get_sections(paragraph_words, pil_img)
                    # this page doesn't have columns
                    for paragraph in paragraphs_sections:
                        paragraph["columns"] = list()
                        column = {
                            "words": paragraph["words"],
                            "bbox": cluster_bbox(paragraph["words"])
                        }

                        paragraph["columns"].append(column)

                    header_words = merge_words_same_line(header_words)

                    # append the title to the header words
                    # header_words.append(title_words)

                    header_sections = get_sections(header_words, pil_img)
                    draw_sections(header_sections, im)

                    ##
                    for word in header_words:
                        im.draw_rect(
                            word,
                            stroke="white",
                            fill=TRANSPARENT,
                            stroke_width=1
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

                                        # does the column end before the farthest left of the header?
                                        if not column["bbox"]["x1"] <= word["x0"]:
                                            # calculate the delta_y
                                            new_delta_y = column["bbox"]["top"] - bottom_of_header
                                            para = words_to_text(column["words"])

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
                                section["text"] += " " + paragraph_text

                            # remove the candidate from the paragraphs_sections so we don't match it again
                            paragraphs_sections.pop(candidate)

                            entry["sections"].append(section)

                    im.save("blocker.png")
                    book.append(entry)
                    continue

                case PageType.SCENARIO:

                    # Find the title and reference based on the header position and expected offsets
                    title = rect_inside(header, 110, 24, 430, 38)
                    reference = rect_inside(header, 35, 22, 70, 38)
                    im.draw_rect(title, stroke="red", fill=TRANSPARENT, stroke_width=4)
                    im.draw_rect(reference, stroke="red", fill=TRANSPARENT, stroke_width=4)

                    entry["title"] = words_to_text(words_in_bbox(words, title))
                    reference = words_in_bbox(words, reference)
                    reference = words_to_text(reference)

                    # reference is two bits of text, a number, a •, and then a code.
                    # need a regex to split that into the number and the code
                    match = re.match(r"(\d+)\s*•\s*(\w+)", reference)
                    if match:
                        entry["number"] = match.group(1)
                        entry["location"] = match.group(2)

                    # get the columns based on the header position
                    # We know scenario sections are split in 3 columns

                    # we also need a page offset of 5 for even pages
                    if page_number % 2 == 0:
                        column_offset = 5
                    else:
                        column_offset = 0

                    col_separators = [header[0] + 215 + column_offset, header[0] + 215 + column_offset + 190]

                    # x boundaries from left edge -> separators -> right edge
                    x_edges = [0, *col_separators, page.width]

                    column_list = [
                        {
                            "x0": x_edges[i],
                            "x1": x_edges[i + 1],
                            "top": 0,
                            "bottom": page_height,
                        }
                        for i in range(len(x_edges) - 1)
                    ]

                    for col in column_list:
                        im.draw_rect(col, stroke="blue", fill=TRANSPARENT, stroke_width=1)

                    # We will exclude all words that are in the header
                    words = filter_words(words, [bbox_to_rect(header)])

                    # Remove the phrase "Continued on next page" if it exists
                    words = remove_phrase(words, ["–", "Continued", "on", "next", "page."])

                    # Split the words into headers and paragraphs
                    paragraph_words, header_words = get_headers_paragraphs(words)
                    # Remove those headers that are actually just the "x1", "x2", those are loot indicators
                    header_words = [w for w in header_words if not (len(w["text"]) == 2 and w["text"].startswith("x"))]

                    # Divide into sections based on background colors
                    paragraphs_sections = get_sections(paragraph_words, pil_img)
                    draw_sections(paragraphs_sections, im)

                    # The headers are all one-liners
                    header_words = merge_words_same_line(header_words)

                    paragraphs_sections = split_paragraphs_into_columns(
                        paragraphs_sections,
                        column_list,
                        header_words,
                    )

                    headers = []
                    # entry["sections"] = list()

                    # Now go through the headers and match them
                    # with the appropiate paragraph
                    entry["sections"] = marry_headers_to_paragraphs(header_words, paragraphs_sections, debug=False)

                    book.append(entry)
                    im.save("blocker.png")

                case PageType.CONTINUED_SCENARIO:
                    # We need to combine this page with the previous page.
                    # And then consider a 6 column layout.
                    # get the previous page, get the current page, combine the images

                    # New Entry
                    entry = dict()
                    # The type should be set to scenario, since it's a continuation of the scenario
                    entry["type"] = PageType.SCENARIO.name

                    # Book is list, remove the last item, we are replacing it with the combined page analysis
                    book.pop()

                    # And remove

                    page_one = pdf.pages[page_offset + page_number - 1]
                    page_two = pdf.pages[page_offset + page_number]

                    image_page_one = page_one.to_image(72).original
                    image_page_two = page_two.to_image(72).original

                    # use pixel sizes, not PDF units
                    combined_width = image_page_one.width + image_page_two.width
                    combined_height = max(image_page_one.height, image_page_two.height)

                    combined_image = Image.new("RGB", (combined_width, combined_height))

                    combined_image.paste(image_page_one, (0, 0))
                    combined_image.paste(image_page_two, (image_page_one.width, 0))
                    combined_image.save("blocker.png")

                    page_two_images = page_two.images
                    # transpose all the images and words on page two to the right by the width of page one
                    for img in page_two_images:
                        img["x0"] += image_page_one.width
                        img["x1"] += image_page_one.width

                    page_two_words = page_two.extract_words()
                    for w in page_two_words:
                        w["x0"] += image_page_one.width
                        w["x1"] += image_page_one.width

                    combined_words = page_one.extract_words() + page_two_words
                    combined_images = page_one.images + page_two_images

                    im = PageImage(page_one, original=combined_image, resolution=72)

                    # Make pdfplumber's coord mapping be identity:
                    im.scale = 1
                    im.bbox = (0, 0, combined_image.width, combined_image.height)

                    type, header = find_header(combined_images)

                    # Retrieve the title and reference based on the header position and expected offsets
                    title = rect_inside(header, 110, 24, 430, 38)
                    reference = rect_inside(header, 35, 22, 70, 38)
                    second_page_header = rect_inside(header, 705, 16, 445, 26)
                    im.draw_rect(second_page_header, stroke="red", fill=TRANSPARENT, stroke_width=5)

                    entry["title"] = words_to_text(words_in_bbox(combined_words, title))
                    reference = words_in_bbox(combined_words, reference)
                    reference = words_to_text(reference)

                    # reference is two bits of text, a number, a •, and then a code.
                    # need a regex to split that into the number and the code
                    match = re.match(r"(\d+)\s*•\s*(\w+)", reference)
                    if match:
                        entry["number"] = match.group(1)
                        entry["location"] = match.group(2)

                    # We will exclude all words that are in the headers
                    combined_words = filter_words(combined_words, [bbox_to_rect(header)])
                    combined_words = filter_words(combined_words, [bbox_to_rect(second_page_header)])

                    # Remove the phrase "Continued on next page" if it exists
                    combined_words = remove_phrase(combined_words, ["–", "Continued", "on", "next", "page."])

                    # Split the words into headers and paragraphs
                    paragraph_words, header_words = get_headers_paragraphs(combined_words)

                    # Combine the header words that are on the same line
                    header_words = merge_words_same_line(header_words)
                    # Remove the header words that are 2 characters long, starting with 'x'
                    header_words = [w for w in header_words if not (len(w["text"]) == 2 and w["text"].startswith("x"))]

                    # Get sections for the paragraphs
                    paragraphs_sections = get_sections(paragraph_words, combined_image)
                    draw_sections(paragraphs_sections, im)

                    # Determine the columns separators based on the header position and expected offsets
                    col_separators = [header[0] + 215,
                                      header[0] + 220 + 190,
                                      header[0] + (220 * 2) + 190,
                                      header[0] + (220 * 3) + 190,
                                      header[0] + (220 * 4) + 190]

                    # x boundaries from left edge -> separators -> right edge
                    x_edges = [0, *col_separators, combined_width]

                    column_list = [
                        {
                            "x0": x_edges[i],
                            "x1": x_edges[i + 1],
                            "top": 0,
                            "bottom": page_height,
                        }
                        for i in range(len(x_edges) - 1)
                    ]

                    # Divide the paragraphs into columns, account for headers
                    paragraphs_sections = split_paragraphs_into_columns(
                        paragraphs_sections,
                        column_list,
                        header_words,
                    )

                    # Now go through the headers and match them
                    # with the appropriate paragraph
                    entry["sections"] = marry_headers_to_paragraphs(header_words, paragraphs_sections)

                    im.save("blocker.png")
                    book.append(entry)

                case PageType.UNKNOWN:
                    pass

            # save this book to a json file
            filename = "./input/books/scenario-book.json"

            with open(filename, "w") as f:
                json.dump(book, f, indent=4, ensure_ascii=False)
                f.close()

        except Exception as e:
            print("Error processing file:", entry)
            print(e)

input_file = "./input/books/scenario-book.json"
book = json.load(open(input_file))

# remove all items that don't have a number key, those are wrong
book = [b for b in book if "number" in b]

# sort on number key
book.sort(key=lambda x: int(x["number"]))

# save this book to a json file
with open(input_file, "w") as f:
    json.dump(book, f, indent=4, ensure_ascii=False)
    f.close()