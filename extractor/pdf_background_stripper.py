import os

from pypdf import PdfReader, PdfWriter
from pypdf.generic import NameObject, DictionaryObject, DecodedStreamObject

def strip_backgrounds(resources, page_width, page_height, tolerance=0.95):
	if not resources:
		return

	xobjects = resources.get("/XObject")
	if not xobjects:
		return

	for name in list(xobjects.keys()):
		xobj = xobjects[name]
		subtype = xobj.get("/Subtype")

		# --- IMAGE XOBJECT ---
		if subtype == NameObject("/Image"):
			w = xobj.get("/Width", 0)
			h = xobj.get("/Height", 0)

			if w >= page_width * tolerance and h >= page_height * tolerance:
				del xobjects[name]

		# --- FORM XOBJECT ---
		elif subtype == NameObject("/Form"):
			bbox = xobj.get("/BBox")
			if bbox:
				w = float(bbox[2]) - float(bbox[0])
				h = float(bbox[3]) - float(bbox[1])

				if w >= page_width * tolerance and h >= page_height * tolerance:
					del xobjects[name]
					continue

			# Otherwise recurse (icons etc live here)
			form_resources = xobj.get("/Resources")
			if isinstance(form_resources, DictionaryObject):
				strip_backgrounds(form_resources, page_width, page_height, tolerance)

def strip_background_annots(page):
	# Some PDFs use watermark annotations as background
	if "/Annots" in page:
		page["/Annots"] = [
			a for a in page["/Annots"]
			if a.get_object().get("/Subtype") != NameObject("/Watermark")
		]

def add_background_color(writer, page, page_width, page_height, rgb=(0, 1, 0)):
	r, g, b = rgb
	cmds = (
		f"{r} {g} {b} rg\n"
		f"0 0 {page_width} {page_height} re\n"
		"f\n"
	).encode("ascii")

	bg = DecodedStreamObject()
	bg.set_data(cmds)
	bg_ref = writer._add_object(bg)

	contents = page.get("/Contents")

	# Normalize to ArrayObject no matter what we got
	if contents is None:
		page[NameObject("/Contents")] = bg_ref
	else:
		if not isinstance(contents, ArrayObject):
			contents = ArrayObject(list(contents) if isinstance(contents, list) else [contents])
		page[NameObject("/Contents")] = ArrayObject([bg_ref, *contents])


# find all the files in the input/books folder that match the pattern "fh-scenario-book-*.pdf"
input_folder = "../worldhaven/images/books/frosthaven"
output_folder= "../input/books/"

# check if output folder exists, if not create it
if not os.path.exists(output_folder):
	os.makedirs(output_folder)

input_books = [f for f in os.listdir(input_folder) if f.startswith("fh-scenario-book-") and f.endswith(".pdf")]

for book in input_books:

	reader = PdfReader(os.path.join(input_folder, book))

	writer = PdfWriter()
	for i, page in enumerate(reader.pages):
		media_box = page.mediabox
		page_width = float(media_box.width)
		page_height = float(media_box.height)

		resources = page.get("/Resources")
		strip_backgrounds(resources, page_width, page_height)
		strip_background_annots(page)

		writer.add_page(page)

	output_filename = book.replace("fh-scenario-book-", "stripped-scenario-book-")
	with open(os.path.join(output_folder, output_filename), "wb") as f:
		writer.write(f)

input_books = [f for f in os.listdir(input_folder) if f.startswith("fh-section-book-") and f.endswith(".pdf")]

for book in input_books:

	reader = PdfReader(os.path.join(input_folder, book))

	writer = PdfWriter()
	for i, page in enumerate(reader.pages):
		media_box = page.mediabox
		page_width = float(media_box.width)
		page_height = float(media_box.height)

		resources = page.get("/Resources")
		strip_backgrounds(resources, page_width, page_height)
		strip_background_annots(page)

		writer.add_page(page)

	output_filename = book.replace("fh-section-book-", "stripped-section-book-")
	with open(os.path.join(output_folder, output_filename), "wb") as f:
		writer.write(f)
