"""Unit tests for wikitext cleaner and section extractor."""
from src.ingestion.cleaner import WikiCleaner


def test_strip_templates_and_infoboxes():
    cleaner = WikiCleaner()
    raw = "{{Infobox scientist | name = Isaac Newton | field = Physics}}\nIsaac Newton was an English mathematician."
    cleaned = cleaner.clean_text_block(raw)
    assert "Infobox" not in cleaned
    assert "Isaac Newton was an English mathematician." in cleaned


def test_strip_references_and_html():
    cleaner = WikiCleaner()
    raw = "Gravity attracts masses.<ref name=\"book\">Principia, 1687.</ref> <span style=\"color:red;\">Universal</span> gravitation.<ref/>"
    cleaned = cleaner.clean_text_block(raw)
    assert "Principia" not in cleaned
    assert "<ref" not in cleaned
    assert "style=" not in cleaned
    assert "Gravity attracts masses. Universal gravitation." in cleaned


def test_strip_wikilinks_and_tables():
    cleaner = WikiCleaner()
    raw = "{| class=\"wikitable\"\n|- \n| Row 1\n|}\nSee [[Albert Einstein|Einstein]] and [[General relativity]]."
    cleaned = cleaner.clean_text_block(raw)
    assert "wikitable" not in cleaned
    assert "Row 1" not in cleaned
    assert "See Einstein and General relativity." in cleaned


def test_extract_sections():
    cleaner = WikiCleaner()
    raw = """Albert Einstein was a theoretical physicist.

== Early Life ==
Einstein was born in Ulm, Germany.

=== Education ===
He attended the Federal Polytechnic School in Zurich.

== References ==
<references />
"""
    sections = cleaner.extract_sections(raw, "Albert Einstein")
    assert len(sections) == 3

    # Lead section
    assert sections[0].heading_path == "Albert Einstein > Overview"
    assert "theoretical physicist" in sections[0].content

    # Early Life
    assert sections[1].heading == "Early Life"
    assert sections[1].heading_path == "Albert Einstein > Early Life"
    assert "Ulm, Germany" in sections[1].content

    # Education (sub-heading)
    assert sections[2].heading == "Education"
    assert sections[2].heading_path == "Albert Einstein > Early Life > Education"
    assert "Zurich" in sections[2].content

    # References should be stripped
    headings = [s.heading for s in sections]
    assert "References" not in headings

