"""High-speed Wikipedia text and markup cleaner."""
import re
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class WikiSection:
    heading: str
    level: int
    heading_path: str
    content: str


class WikiCleaner:
    """
    Cleans raw Wikitext / Wikipedia dump markup into clean, human-readable text
    while extracting the hierarchical heading tree.
    """
    def __init__(self):
        # Matches == Heading == or ## Heading
        self._heading_pattern = re.compile(
            r"^(?:(={2,5})\s*(.*?)\s*\1\s*$|(#{2,5})\s*(.*?)$)",
            re.MULTILINE
        )
        
        # Tags and templates
        self._ref_tags = re.compile(r"<ref[^>]*>.*?</ref>", re.DOTALL | re.IGNORECASE)
        self._self_closing_ref = re.compile(r"<ref[^/>]*/>", re.IGNORECASE)
        self._html_tags = re.compile(r"<[^>]+>", re.DOTALL)
        self._wiki_tables = re.compile(r"\{\|.*?\|\}", re.DOTALL)
        self._file_images = re.compile(r"\[\[(File|Image):.*?\]\]", re.IGNORECASE)
        self._external_links = re.compile(r"\[https?://[^\s\]]+(?:\s+([^\]]+))?\]")
        self._category_links = re.compile(r"\[\[Category:[^\]]+\]\]", re.IGNORECASE)
        self._bold_italics = re.compile(r"\'{2,5}")
        self._multiple_newlines = re.compile(r"\n{3,}")
        self._multiple_spaces = re.compile(r"[ \t]{2,}")

    def clean_text_block(self, text: str) -> str:
        """Strips markup from a single block of body text."""
        if not text:
            return ""

        # Remove reference tags
        text = self._ref_tags.sub("", text)
        text = self._self_closing_ref.sub("", text)

        # Remove tables {| ... |}
        text = self._wiki_tables.sub("", text)

        # Strip file and image links
        text = self._file_images.sub("", text)

        # Remove category tags
        text = self._category_links.sub("", text)

        # Remove templates {{ ... }} - handles up to 3 levels of nesting
        for _ in range(3):
            text = re.sub(r"\{\{[^{}]*\}\}", "", text, flags=re.DOTALL)

        # Convert wikilinks [[Target|Anchor]] -> Anchor, [[Article]] -> Article
        def _replace_wikilink(match: re.Match) -> str:
            content = match.group(1)
            if "|" in content:
                parts = content.split("|", 1)
                return parts[1]
            return content

        text = re.sub(r"\[\[([^\]]+)\]\]", _replace_wikilink, text)

        # External links [http://example.com Anchor] -> Anchor
        def _replace_ext_link(match: re.Match) -> str:
            anchor = match.group(1)
            return anchor if anchor else ""

        text = self._external_links.sub(_replace_ext_link, text)

        # HTML tags
        text = self._html_tags.sub(" ", text)

        # Bold and italics formatting
        text = self._bold_italics.sub("", text)

        # Clean spaces and newlines
        text = self._multiple_spaces.sub(" ", text)
        text = self._multiple_newlines.sub("\n\n", text)
        return text.strip()

    def extract_sections(self, raw_text: str, article_title: str) -> List[WikiSection]:
        """
        Splits article wikitext into hierarchical sections.
        Maintains heading path (e.g. "Early life > Education").
        """
        if not raw_text:
            return []

        # Find all headings with their start/end spans
        headings: List[Tuple[int, int, str, int]] = []
        for m in self._heading_pattern.finditer(raw_text):
            if m.group(1):
                level = len(m.group(1))
                title = m.group(2).strip()
            else:
                level = len(m.group(3))
                title = m.group(4).strip()

            # Ignore standard trailing Wikipedia sections
            if title.lower() in ("references", "external links", "see also", "further reading", "notes"):
                headings.append((m.start(), m.end(), title, -1))  # marker to drop content
            else:
                headings.append((m.start(), m.end(), title, level))

        sections: List[WikiSection] = []
        path_stack: List[Tuple[int, str]] = [(1, article_title)]

        if not headings:
            # Entire text is a single top-level section
            cleaned = self.clean_text_block(raw_text)
            if cleaned:
                sections.append(WikiSection(
                    heading=article_title,
                    level=1,
                    heading_path=f"{article_title} > Overview",
                    content=cleaned,
                ))
            return sections

        # Handle lead section before first heading
        first_start = headings[0][0]
        lead_raw = raw_text[:first_start]
        lead_cleaned = self.clean_text_block(lead_raw)
        if lead_cleaned:
            sections.append(WikiSection(
                heading=article_title,
                level=1,
                heading_path=f"{article_title} > Overview",
                content=lead_cleaned,
            ))

        # Handle sections between headings
        for i, (start, end, h_title, h_level) in enumerate(headings):
            if h_level == -1:
                continue  # Skip references / see also

            # Update heading path hierarchy stack
            while path_stack and path_stack[-1][0] >= h_level:
                path_stack.pop()
            path_stack.append((h_level, h_title))
            heading_path = " > ".join(t for _, t in path_stack)

            # Extract text slice until next heading
            next_start = headings[i + 1][0] if i + 1 < len(headings) else len(raw_text)
            sec_raw = raw_text[end:next_start]
            sec_cleaned = self.clean_text_block(sec_raw)

            if sec_cleaned:
                sections.append(WikiSection(
                    heading=h_title,
                    level=h_level,
                    heading_path=heading_path,
                    content=sec_cleaned,
                ))

        return sections

