"""
Build the local image index used by Stage 5.

    python scripts/build_image_index.py                 # generate + index
    python scripts/build_image_index.py --no-generate   # index what is there
    python scripts/build_image_index.py --dir some/dir

The index answers one question: "have we seen this picture before, and
when was it first published?" — which is what catches a real photograph
recaptioned as today's news.

A real deployment would seed `data/seed_images/` with actual images whose
provenance is known. This script does not download anything by default:
fetching arbitrary pictures from the internet is slow, breaks without a
network, and would put files of unclear licence into the repository.
Instead it *generates* placeholder images — flat coloured cards with the
scenario printed on them — one per entry in a small scenario list. They
are visually distinct, so DINOv2 gives each a stable and well-separated
embedding, which is all the index needs to be demonstrable end to end.

Everything it writes is marked `"demo": true`, and matches carry that
flag all the way to the verdict. Point `--dir` at real images with a real
metadata.jsonl to build the genuine article; the format is one JSON
object per line:

    {"file": "chennai_flood_2015.jpg", "context": "...",
     "first_seen": "2015-12-02", "source_url": "https://...",
     "publisher": "...", "domain": "...", "demo": false}
"""

import argparse
import json
import logging
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.images import local_index                         # noqa: E402


# Scenarios chosen to mirror the recycled-media cases the seed fact-check
# corpus covers, so the two demo datasets tell one story.
SCENARIOS = [
    {
        "file": "chennai_flood_2015.png",
        "title": "Chennai floods",
        "context": "Flooding in Chennai during the December 2015 storms.",
        "first_seen": "2015-12-02",
        "source_url": "https://example-demo.invalid/archive/chennai-floods-2015",
        "publisher": "Demo Image Archive",
        "domain": "boomlive.in",
        "colour": (38, 84, 124),
    },
    {
        "file": "army_convoy_2019.png",
        "title": "Army convoy",
        "context": "An army convoy at a Republic Day rehearsal, January 2019.",
        "first_seen": "2019-01-20",
        "source_url": "https://example-demo.invalid/archive/republic-day-rehearsal-2019",
        "publisher": "Demo Image Archive",
        "domain": "factchecker.in",
        "colour": (66, 84, 46),
    },
    {
        "file": "crowd_rally_2018.png",
        "title": "Crowd at a rally",
        "context": "A crowd at a political rally in Hyderabad, November 2018.",
        "first_seen": "2018-11-11",
        "source_url": "https://example-demo.invalid/archive/hyderabad-rally-2018",
        "publisher": "Demo Image Archive",
        "domain": "altnews.in",
        "colour": (124, 62, 38),
    },
    {
        "file": "missing_child_2017.png",
        "title": "Missing child appeal",
        "context": "A missing-child appeal from 2017; the child was traced within days.",
        "first_seen": "2017-06-14",
        "source_url": "https://example-demo.invalid/archive/missing-child-2017",
        "publisher": "Demo Image Archive",
        "domain": "newschecker.in",
        "colour": (96, 64, 112),
    },
    {
        "file": "burning_building_2016.png",
        "title": "Building fire",
        "context": "A warehouse fire in Surat, March 2016.",
        "first_seen": "2016-03-08",
        "source_url": "https://example-demo.invalid/archive/surat-fire-2016",
        "publisher": "Demo Image Archive",
        "domain": "factcrescendo.com",
        "colour": (140, 46, 34),
    },
    {
        "file": "queue_atm_2016.png",
        "title": "ATM queue",
        "context": "A queue outside an ATM during demonetisation, November 2016.",
        "first_seen": "2016-11-12",
        "source_url": "https://example-demo.invalid/archive/atm-queue-2016",
        "publisher": "Demo Image Archive",
        "domain": "boomlive.in",
        "colour": (52, 92, 76),
    },
    {
        "file": "flood_rescue_2018.png",
        "title": "Flood rescue",
        "context": "A rescue boat during the Kerala floods, August 2018.",
        "first_seen": "2018-08-19",
        "source_url": "https://example-demo.invalid/archive/kerala-floods-2018",
        "publisher": "Demo Image Archive",
        "domain": "thequint.com",
        "colour": (30, 96, 110),
    },
    {
        "file": "vaccine_camp_2021.png",
        "title": "Vaccination camp",
        "context": "A COVID-19 vaccination camp, June 2021.",
        "first_seen": "2021-06-22",
        "source_url": "https://example-demo.invalid/archive/vaccination-camp-2021",
        "publisher": "Demo Image Archive",
        "domain": "thip.media",
        "colour": (70, 70, 132),
    },
]

DEMO_NOTE = (
    "DEMO DATA - a generated placeholder image, not a photograph. It stands in "
    "for a catalogued picture so the recycled-media check can be demonstrated "
    "offline. Any match against it is a demo finding."
)

SIZE = (384, 384)


def generate(directory):
    """
    Write one placeholder image per scenario, plus metadata.jsonl.

    Returns the number of images written, or 0 if Pillow is missing.
    Existing files are left alone, so a real image dropped in by hand is
    never overwritten by a placeholder.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("Pillow is not installed; cannot generate placeholder images.")
        print("Install it (pip install Pillow) or point --dir at real images.")
        return 0

    os.makedirs(directory, exist_ok=True)
    written = 0

    for index, scenario in enumerate(SCENARIOS):
        path = os.path.join(directory, scenario["file"])

        if os.path.exists(path):
            continue

        image = Image.new("RGB", SIZE, scenario["colour"])
        draw = ImageDraw.Draw(image)

        # Distinct geometry per entry as well as distinct colour: two flat
        # colour fields can embed closer together than two real
        # photographs would, and the index is meant to demonstrate
        # separation, not to flatter itself.
        step = 24 + (index * 7) % 40

        for offset in range(0, SIZE[0] * 2, step):
            draw.line(
                [(offset, 0), (offset - SIZE[1], SIZE[1])],
                fill=tuple(min(255, channel + 60) for channel in scenario["colour"]),
                width=3 + index % 4,
            )

        draw.rectangle([24, 24, SIZE[0] - 24, 120], fill=(250, 250, 250))
        draw.text((36, 40), scenario["title"], fill=(20, 20, 20))
        draw.text((36, 60), f"first seen {scenario['first_seen']}", fill=(60, 60, 60))
        draw.text((36, 80), "DEMO PLACEHOLDER - not a photograph", fill=(150, 30, 30))

        image.save(path)
        written += 1

    with open(os.path.join(directory, "metadata.jsonl"), "w", encoding="utf-8") as handle:
        for scenario in SCENARIOS:
            record = {
                "file": scenario["file"],
                "context": scenario["context"],
                "first_seen": scenario["first_seen"],
                "source_url": scenario["source_url"],
                "publisher": scenario["publisher"],
                "domain": scenario["domain"],
                "demo": True,
                "demo_note": DEMO_NOTE,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=None, help="the image directory to index")
    parser.add_argument("--out", default=None, help="where to write the index")
    parser.add_argument("--no-generate", action="store_true",
                        help="index what is on disk, generating nothing")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    directory = args.dir or local_index.image_dir()
    output = args.out or local_index.index_path()

    if not args.no_generate:
        written = generate(directory)
        print(f"{written} placeholder images written to {directory}")

    if not os.path.isdir(directory):
        print(f"no such directory: {directory}")
        return 1

    print(f"embedding images in {directory} with DINOv2 (first run downloads the model)...")

    count = local_index.build(directory=directory, output=output)

    if not count:
        print("nothing indexed - is there anything in the directory, and is torch installed?")
        return 1

    print(f"{count} images indexed -> {output}")

    metadata = local_index.load_metadata(directory)
    demo = sum(1 for record in metadata.values() if record.get("demo"))

    if demo:
        print(f"{demo} of them are DEMO DATA: generated placeholders, not photographs.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
