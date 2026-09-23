try:
    import regex as re
except ImportError:
    import re

from typing import List, Optional, Tuple, Any, NamedTuple
from src.common.schemas import DocumentChunk, RawArticle, make_chunk_id
from src.ingestion.cleaner import WikiCleaner, WikiSection
from src.common.tokenizer import get_tokenizer
from src.common.logging import get_logger

logger = get_logger("ingestion.chunker")

# Sentence boundary regex with abbreviation guards and punctuation preservation
try:
    SENTENCE_SPLIT_REGEX = re.compile(
        r"(?<=(?<!\b(?:Mr|Mrs|Ms|Dr|Prof|Gen|Rep|Sen|Gov|St|Mt|U\.S|U\.K|e\.g|i\.e|vs|approx|vol|dept|univ|etc|al))[.!?])"
        r"(?:\s+|\n+)(?=[A-Z0-9\"'(\[])",
        re.UNICODE
    )
except Exception:
    SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


class _Sentence(NamedTuple):
    """Internal: sentence text with pre-computed token count (F3: eliminates redundant tokenization)."""
    text: str
    tokens: int


class HeadingAwareChunker:
    """
    Sentence-preserving, heading-aware chunker for Wikipedia articles.

    Features:
    1. Preserves hierarchical section paths (e.g. 'Albert Einstein > Early Life > Education').
    2. Does not assume every article has headings (falls back gracefully to Overview).
    3. Respects target token ceiling (~256 tokens) with 15–20% sentence-aware overlap.
    4. Avoids splitting sentences unnecessarily; only splits individual run-on sentences.
    5. Handles short articles (stubs emitted as single chunk) and long sections (sliding window).
    6. Uses the selected embedding model's fast tokenizer for exact token counts.
    7. Generates deterministic 64-bit chunk_ids: (doc_id << 16) | chunk_seq.
    """
    def __init__(
        self,
        target_tokens: int = 256,
        overlap_tokens: int = 40,      # ~15.6% of 256
        min_tokens: int = 25,
        model_name: str = "BAAI/bge-small-en-v1.5",
        tokenizer: Optional[Any] = None,
        prepend_headers: bool = True,
    ):
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.min_tokens = min_tokens
        self.prepend_headers = prepend_headers
        self.cleaner = WikiCleaner()

        # Load embedding model tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            try:
                self.tokenizer = get_tokenizer(model_name)
            except Exception as e:
                logger.warning(f"Could not load HuggingFace tokenizer ({e}). Using calibrated BPE approximation.")
                self.tokenizer = None

    def count_tokens(self, text: str) -> int:
        """Counts exact tokens using the fast model tokenizer with fast fallback."""
        if not text:
            return 0
        if self.tokenizer is not None:
            try:
                return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
            except Exception:
                pass
        # Calibrated fallback: 1 English word ≈ 1.33 subword tokens
        return max(1, int(len(text.split()) * 1.33))

    def _split_into_sentences(self, text: str) -> List[_Sentence]:
        """Splits section content into natural sentence units with pre-computed token counts.

        Each sentence is tokenized exactly once here (F3). Downstream methods
        use the cached .tokens attribute instead of re-tokenizing.
        """
        raw_sentences = SENTENCE_SPLIT_REGEX.split(text)
        sentences: List[_Sentence] = []

        for s in raw_sentences:
            if not s:
                continue
            s_clean = s.strip()
            if not s_clean:
                continue

            # Check if a single sentence exceeds the target chunk ceiling
            s_tok = self.count_tokens(s_clean)
            if s_tok > self.target_tokens:
                # Sub-split oversized run-on sentence on semicolons or commas
                sub_parts = re.split(r"(?<=[;,])\s+", s_clean)
                if len(sub_parts) <= 1:
                    # No clause punctuation either: split by words
                    words = s_clean.split()
                    step = max(1, self.target_tokens - self.overlap_tokens)
                    for i in range(0, len(words), step):
                        slice_words = words[i:i + self.target_tokens]
                        if slice_words:
                            piece = " ".join(slice_words)
                            sentences.append(_Sentence(piece, self.count_tokens(piece)))
                else:
                    cur_sub = ""
                    cur_tok = 0
                    for p in sub_parts:
                        p_tok = self.count_tokens(p)
                        if cur_sub:
                            candidate_tok = cur_tok + p_tok + 1  # +1 for joining space (approximate)
                        else:
                            candidate_tok = p_tok
                        if candidate_tok > self.target_tokens and cur_sub:
                            sentences.append(_Sentence(cur_sub, cur_tok))
                            cur_sub = p
                            cur_tok = p_tok
                        else:
                            cur_sub = f"{cur_sub} {p}".strip() if cur_sub else p
                            cur_tok = candidate_tok
                    if cur_sub:
                        sentences.append(_Sentence(cur_sub, cur_tok))
            else:
                sentences.append(_Sentence(s_clean, s_tok))

        return sentences

    def _sliding_window_chunks(self, sentences: List[_Sentence]) -> List[Tuple[str, int]]:
        """
        Builds overlapping passages from sentences without breaking sentence boundaries.
        Maintains 15–20% token overlap between consecutive chunks.

        Returns list of (passage_text, passage_token_count) tuples.
        Token counts are computed from cached sentence counts (F3: zero re-tokenization).
        """
        if not sentences:
            return []

        passages: List[Tuple[str, int]] = []
        current_sentences: List[_Sentence] = []
        current_tokens = 0

        for sent in sentences:
            # If adding this sentence exceeds target ceiling, flush current chunk
            if current_tokens + sent.tokens > self.target_tokens and current_sentences:
                passages.append((" ".join(s.text for s in current_sentences), current_tokens))

                # Build overlap from the tail of current sentences
                overlap_accum: List[_Sentence] = []
                overlap_tok = 0
                for prev_sent in reversed(current_sentences):
                    if overlap_tok + prev_sent.tokens <= self.overlap_tokens or not overlap_accum:
                        overlap_accum.insert(0, prev_sent)
                        overlap_tok += prev_sent.tokens
                        if overlap_tok >= self.overlap_tokens:
                            break
                    else:
                        break

                current_sentences = overlap_accum
                current_tokens = overlap_tok

            current_sentences.append(sent)
            current_tokens += sent.tokens

        if current_sentences:
            passages.append((" ".join(s.text for s in current_sentences), current_tokens))

        return passages

    def chunk_article(self, article: RawArticle) -> List[DocumentChunk]:
        """
        Chunks an article into contextual, sentence-preserving passages.
        Preserves title and section_path, and generates deterministic chunk IDs.
        """
        if not article.text or not article.text.strip():
            return []

        # Extract hierarchical sections (or fallback to Overview)
        sections = self.cleaner.extract_sections(article.text, article.title)
        if not sections:
            # Entire text has no headings: treat as single Overview section
            cleaned_body = self.cleaner.clean_text_block(article.text)
            if not cleaned_body:
                return []
            sections = [
                WikiSection(
                    heading=article.title,
                    level=1,
                    heading_path=f"{article.title} > Overview",
                    content=cleaned_body,
                )
            ]

        emitted_chunks: List[DocumentChunk] = []
        chunk_seq = 0

        # Process each section
        for sec in sections:
            sentences = self._split_into_sentences(sec.content)
            if not sentences:
                continue

            passages = self._sliding_window_chunks(sentences)

            for passage_text, passage_tokens in passages:
                passage_text = passage_text.strip()

                # Merge small trailing fragments into previous chunk if possible
                if passage_tokens < self.min_tokens and emitted_chunks:
                    prev_chunk = emitted_chunks[-1]
                    combined_text = f"{prev_chunk.text} {passage_text}"
                    combined_tokens = prev_chunk.token_count + passage_tokens
                    if combined_tokens <= int(self.target_tokens * 1.25):
                        # Append to previous chunk
                        emitted_chunks[-1] = DocumentChunk(
                            chunk_id=prev_chunk.chunk_id,
                            doc_id=prev_chunk.doc_id,
                            title=prev_chunk.title,
                            url=prev_chunk.url,
                            section_path=prev_chunk.section_path,
                            text=combined_text,
                            token_count=combined_tokens,
                            faiss_id=prev_chunk.faiss_id,
                        )
                        continue

                # Prepend contextual heading path
                if self.prepend_headers:
                    full_chunk_text = f"{article.title} - {sec.heading_path}\n{passage_text}"
                else:
                    full_chunk_text = passage_text

                # Exact token count of the final chunk text (header is new text, must tokenize)
                total_chunk_tokens = self.count_tokens(full_chunk_text)
                cid = make_chunk_id(int(article.id), chunk_seq)

                chunk = DocumentChunk(
                    chunk_id=cid,
                    doc_id=int(article.id),
                    title=article.title,
                    url=article.url,
                    section_path=sec.heading_path,
                    text=full_chunk_text,
                    token_count=total_chunk_tokens,
                    faiss_id=cid,
                )
                emitted_chunks.append(chunk)
                chunk_seq += 1

        return emitted_chunks
