from pathlib import Path

for svg in Path(".").glob("*.svg"):
    text = svg.read_text()

    if "color-scheme: light dark" in text or "color-scheme: dark light" in text:
        print("here")
        name = svg.stem

        light = text.replace("color-scheme: light dark;", "")
        dark = text.replace("color-scheme: light dark;", "")

        Path(f"{name}_light.svg").write_text(light)
        Path(f"{name}_dark.svg").write_text(dark)

        svg.unlink()