"""Dataset Loading, Evidence Mapping, and Mapping Limitation Tracking.

Supports:
- NQ-open (Google Natural Questions Open: CC BY-SA 3.0 / Apache 2.0)
- TriviaQA (Mandar Joshi et al.: Apache 2.0)
- Pilot Corpus Evidence Mapping with DPR-standard answer span extraction
- Documented mapping limitations:
  1. Lexical False Positives: Spurious matches for short entities (e.g. '1994', 'May', 'US').
  2. Lexical False Negatives: Paraphrased answers not present in gold alias list.
  3. Corpus Coverage Discrepancy: Queries whose answers lie outside the pilot shard index.
"""

from typing import List, Dict, Any, Optional, Set, Tuple, Sequence, Union
from dataclasses import dataclass, field
import re
import json
import os
from pathlib import Path
import pandas as pd

from src.evaluation.generation_metrics import normalize_answer
from src.common.logging import get_logger

logger = get_logger("evaluation.dataset")


@dataclass
class QASample:
    """Standard representation of an open-domain QA evaluation query."""
    query_id: str
    question: str
    answers: List[str]
    aliases: List[str] = field(default_factory=list)
    entity_titles: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def all_acceptable_answers(self) -> List[str]:
        """Returns deduplicated union of primary answers and alias variations."""
        seen = set()
        res = []
        for a in self.answers + self.aliases:
            cleaned = a.strip()
            if cleaned and cleaned.lower() not in seen:
                seen.add(cleaned.lower())
                res.append(cleaned)
        return res


@dataclass
class GroundTruthEvidence:
    """Grounded evidence mapping linking a QA query to pilot corpus chunks."""
    query_id: str
    question: str
    gold_answers: List[str]
    gold_chunk_ids: Set[int] = field(default_factory=set)
    gold_doc_ids: Set[int] = field(default_factory=set)
    relevance_scores: Dict[int, float] = field(default_factory=dict)
    is_grounded_in_corpus: bool = False
    mapping_notes: List[str] = field(default_factory=list)
    limitations_flagged: List[str] = field(default_factory=list)


@dataclass
class MappingDiagnostics:
    """Statistical summary and diagnostics of evidence mapping."""
    total_queries: int
    grounded_queries: int
    ungrounded_queries: int
    total_positive_chunks: int
    avg_positive_chunks_per_query: float
    short_answer_warnings: int
    title_matches: int
    span_matches: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "grounded_queries": self.grounded_queries,
            "ungrounded_queries": self.ungrounded_queries,
            "grounding_ratio": round(self.grounded_queries / max(1, self.total_queries), 4),
            "total_positive_chunks": self.total_positive_chunks,
            "avg_positive_chunks_per_query": round(self.avg_positive_chunks_per_query, 2),
            "short_answer_warnings": self.short_answer_warnings,
            "title_matches": self.title_matches,
            "span_matches": self.span_matches,
        }


class EvidenceMapper:
    """Maps open QA queries to relevant chunks in the pilot Wikipedia corpus."""

    def __init__(self, min_answer_len_for_span: int = 3):
        self.min_answer_len = min_answer_len_for_span

    @staticmethod
    def prepare_records(chunks_df: pd.DataFrame) -> List[Tuple[int, int, str, str]]:
        """Pre-normalizes chunk records once for high-throughput vectorized/loop matching."""
        if "_norm_title" in chunks_df.columns and "_norm_text" in chunks_df.columns:
            return list(zip(
                chunks_df["chunk_id"].astype(int),
                chunks_df["doc_id"].astype(int),
                chunks_df["_norm_title"].astype(str),
                chunks_df["_norm_text"].astype(str),
            ))
        
        return [
            (
                int(cid),
                int(did),
                normalize_answer(str(title)),
                normalize_answer(str(text)),
            )
            for cid, did, title, text in zip(
                chunks_df["chunk_id"],
                chunks_df["doc_id"],
                chunks_df["title"],
                chunks_df["text"],
            )
        ]

    def map_query_to_chunks(
        self,
        sample: QASample,
        chunks_df_or_records: Union[pd.DataFrame, List[Tuple[int, int, str, str]]],
    ) -> GroundTruthEvidence:
        """Finds all relevant chunks containing evidence for sample.

        Relevance grading criteria:
        - 2.0 (High): Chunk title matches an entity title AND text contains the answer span.
        - 1.0 (Standard): Chunk text contains the exact answer span.
        - 0.5 (Weak): Chunk title matches entity title, but exact answer span is paraphrased.
        """
        evidence = GroundTruthEvidence(
            query_id=sample.query_id,
            question=sample.question,
            gold_answers=sample.all_acceptable_answers,
        )

        all_answers = sample.all_acceptable_answers
        if not all_answers:
            evidence.mapping_notes.append("No gold answers provided for query.")
            return evidence

        # Prepare normalized answers and regex patterns with word boundaries
        answer_pairs = []
        short_warning = False
        for ans in all_answers:
            norm_ans = normalize_answer(ans)
            if not norm_ans:
                continue
            if len(norm_ans) < self.min_answer_len or norm_ans.isdigit():
                short_warning = True
            escaped = re.escape(norm_ans)
            pat = re.compile(r"\b" + escaped + r"\b", re.IGNORECASE)
            answer_pairs.append((norm_ans, pat))

        if short_warning:
            evidence.limitations_flagged.append(
                "Potential lexical false positive: gold answer entity is short (<3 chars or numeric)."
            )

        # Entity title patterns
        norm_titles = {normalize_answer(t) for t in sample.entity_titles if t}
        norm_entity_list = [t for t in norm_titles if len(t) >= 2]

        records = (
            chunks_df_or_records
            if isinstance(chunks_df_or_records, list)
            else self.prepare_records(chunks_df_or_records)
        )

        # Search chunks for evidence
        for cid, doc_id, norm_chunk_title, norm_chunk_text in records:
            # Check title match
            title_matched = (norm_chunk_title in norm_titles) if norm_titles else False
            if not title_matched and norm_entity_list:
                for et_norm in norm_entity_list:
                    if et_norm in norm_chunk_title:
                        title_matched = True
                        break

            # Fast C substring filter first, only run regex if substring present!
            span_matched = False
            for norm_ans, pat in answer_pairs:
                if norm_ans in norm_chunk_text:
                    if pat.search(norm_chunk_text):
                        span_matched = True
                        break

            if span_matched and title_matched:
                evidence.gold_chunk_ids.add(cid)
                evidence.gold_doc_ids.add(doc_id)
                evidence.relevance_scores[cid] = 2.0
            elif span_matched:
                evidence.gold_chunk_ids.add(cid)
                evidence.gold_doc_ids.add(doc_id)
                evidence.relevance_scores[cid] = 1.0
            elif title_matched:
                evidence.gold_chunk_ids.add(cid)
                evidence.gold_doc_ids.add(doc_id)
                evidence.relevance_scores[cid] = 0.5

        evidence.is_grounded_in_corpus = len(evidence.gold_chunk_ids) > 0

        if not evidence.is_grounded_in_corpus:
            evidence.mapping_notes.append(
                "Query answer evidence is not present in this corpus partition (Out-of-Corpus Query)."
            )

        return evidence

    def map_dataset(
        self,
        samples: Sequence[QASample],
        chunks_df: pd.DataFrame,
    ) -> Tuple[List[GroundTruthEvidence], MappingDiagnostics]:
        """Maps a collection of QA samples across the corpus and computes mapping diagnostics."""
        logger.info(f"Pre-normalizing {len(chunks_df)} chunks for high-throughput evidence mapping...")
        records = self.prepare_records(chunks_df)
        logger.info(f"Pre-normalized {len(records)} records. Mapping {len(samples)} QA samples...")

        results: List[GroundTruthEvidence] = []
        grounded_count = 0
        total_positive_chunks = 0
        short_warnings = 0
        title_matches_count = 0
        span_matches_count = 0

        for sample in samples:
            ev = self.map_query_to_chunks(sample, records)
            results.append(ev)

            if ev.is_grounded_in_corpus:
                grounded_count += 1
                total_positive_chunks += len(ev.gold_chunk_ids)

            if ev.limitations_flagged:
                short_warnings += 1

            for cid, score in ev.relevance_scores.items():
                if score >= 2.0:
                    title_matches_count += 1
                    span_matches_count += 1
                elif score == 1.0:
                    span_matches_count += 1
                elif score == 0.5:
                    title_matches_count += 1

        diagnostics = MappingDiagnostics(
            total_queries=len(samples),
            grounded_queries=grounded_count,
            ungrounded_queries=len(samples) - grounded_count,
            total_positive_chunks=total_positive_chunks,
            avg_positive_chunks_per_query=(
                float(total_positive_chunks) / float(grounded_count)
                if grounded_count > 0 else 0.0
            ),
            short_answer_warnings=short_warnings,
            title_matches=title_matches_count,
            span_matches=span_matches_count,
        )

        return results, diagnostics


def get_curated_pilot_benchmark_samples() -> List[QASample]:
    """Provides a curated set of benchmark QA queries grounded in the pilot Wikipedia corpus.

    Carefully constructed from topics present in shards 0000 and 0001
    (e.g., Brad Mehldau, Oscar Peterson, Vassilios Skouris, Summer Olympics,
    Salayea District, Centrolene, Sri Lankan Tamils, Rhodes State Office Tower, etc.)
    plus explicit out-of-domain queries to test context insufficiency detection.
    """
    return [
        QASample(
            query_id="nq_pilot_001",
            question="What instrument does Brad Mehldau play?",
            answers=["piano"],
            aliases=["jazz piano", "grand piano"],
            entity_titles=["Brad Mehldau"],
            metadata={"domain": "music", "expected_in_corpus": True},
        ),
        QASample(
            query_id="nq_pilot_002",
            question="Who recorded the jazz album Night Train?",
            answers=["Oscar Peterson"],
            aliases=["The Oscar Peterson Trio", "Oscar Peterson Trio"],
            entity_titles=["Night Train (Oscar Peterson album)"],
            metadata={"domain": "music", "expected_in_corpus": True},
        ),
        QASample(
            query_id="nq_pilot_003",
            question="What position did Vassilios Skouris hold in the European Union?",
            answers=["President of the Court of Justice of the European Union", "President of the Court of Justice"],
            aliases=["President of the European Court of Justice", "Court of Justice President"],
            entity_titles=["Vassilios Skouris"],
            metadata={"domain": "politics", "expected_in_corpus": True},
        ),
        QASample(
            query_id="nq_pilot_004",
            question="In which county of Liberia is Salayea District located?",
            answers=["Lofa County"],
            aliases=["Lofa"],
            entity_titles=["Salayea District"],
            metadata={"domain": "geography", "expected_in_corpus": True},
        ),
        QASample(
            query_id="nq_pilot_005",
            question="What type of animal is Centrolene?",
            answers=["frog", "glass frog"],
            aliases=["frogs", "glass frogs", "Centrolenidae"],
            entity_titles=["Centrolene"],
            metadata={"domain": "biology", "expected_in_corpus": True},
        ),
        QASample(
            query_id="nq_pilot_006",
            question="Where is the Rhodes State Office Tower situated?",
            answers=["Columbus", "Columbus, Ohio"],
            aliases=["Columbus, OH"],
            entity_titles=["Rhodes State Office Tower"],
            metadata={"domain": "architecture", "expected_in_corpus": True},
        ),
        QASample(
            query_id="nq_pilot_007",
            question="What office did Luis Castiglioni hold in Paraguay?",
            answers=["Vice President", "Vice President of Paraguay"],
            aliases=["Minister of Foreign Affairs"],
            entity_titles=["Luis Castiglioni"],
            metadata={"domain": "politics", "expected_in_corpus": True},
        ),
        QASample(
            query_id="nq_pilot_008",
            question="In which district of Turkey is Çavdarhisar located?",
            answers=["Kütahya Province", "Kütahya"],
            aliases=["Kutahya"],
            entity_titles=["Çavdarhisar"],
            metadata={"domain": "geography", "expected_in_corpus": True},
        ),
        QASample(
            query_id="nq_pilot_009",
            question="What sport club is Karşıyaka S.K. known for in Turkey?",
            answers=["football", "basketball"],
            aliases=["association football", "Karşıyaka Basket"],
            entity_titles=["Karşıyaka S.K."],
            metadata={"domain": "sports", "expected_in_corpus": True},
        ),
        QASample(
            query_id="nq_pilot_010",
            question="Which historical city is Altstadt Spandau located in?",
            answers=["Berlin", "Spandau"],
            aliases=["Berlin, Germany"],
            entity_titles=["Altstadt Spandau"],
            metadata={"domain": "geography", "expected_in_corpus": True},
        ),
        QASample(
            query_id="nq_pilot_011",
            question="What nationality is actress Emília Vášáryová?",
            answers=["Slovak", "Slovakia"],
            aliases=["Czechoslovak"],
            entity_titles=["Emília Vášáryová"],
            metadata={"domain": "entertainment", "expected_in_corpus": True},
        ),
        QASample(
            query_id="nq_pilot_012",
            question="Which team or country did Suriname compete for at the 1996 Summer Olympics?",
            answers=["Suriname"],
            aliases=["Republic of Suriname"],
            entity_titles=["Suriname at the 1996 Summer Olympics"],
            metadata={"domain": "sports", "expected_in_corpus": True},
        ),
        QASample(
            query_id="nq_pilot_013",
            question="What plant genus is Teucrium polium classified under?",
            answers=["Teucrium", "Lamiaceae"],
            aliases=["mint family"],
            entity_titles=["Teucrium polium"],
            metadata={"domain": "biology", "expected_in_corpus": True},
        ),
        QASample(
            query_id="nq_pilot_014",
            question="What ethnicity formed the ancient Jaffna kingdom?",
            answers=["Sri Lankan Tamils", "Tamils"],
            aliases=["Tamil people"],
            entity_titles=["Sri Lankan Tamils"],
            metadata={"domain": "history", "expected_in_corpus": True},
        ),
        QASample(
            query_id="nq_pilot_015",
            question="What role did Nuaym ibn Masud play in Islamic history?",
            answers=["Battle of the Trench", "companion of Muhammad"],
            aliases=["Khandaq", "Sahaba"],
            entity_titles=["Nuaym ibn Masud"],
            metadata={"domain": "history", "expected_in_corpus": True},
        ),
        QASample(
            query_id="nq_pilot_016",
            question="Who played Raj in the TV sitcom What's Happening!!?",
            answers=["Ernest Thomas"],
            aliases=["Ernest Lee Thomas"],
            entity_titles=["Ernest Thomas"],
            metadata={"domain": "entertainment", "expected_in_corpus": True},
        ),
        QASample(
            query_id="nq_pilot_017",
            question="Which district is Zorzor District located within in Liberia?",
            answers=["Lofa County", "Lofa"],
            aliases=["Liberia"],
            entity_titles=["Zorzor District"],
            metadata={"domain": "geography", "expected_in_corpus": True},
        ),
        QASample(
            query_id="nq_pilot_018",
            question="What is the capital of Turkey located near Çarşamba?",
            answers=["Samsun", "Samsun Province"],
            aliases=["Black Sea region"],
            entity_titles=["Çarşamba"],
            metadata={"domain": "geography", "expected_in_corpus": True},
        ),
        QASample(
            query_id="nq_pilot_019",
            question="Which country was actress Gladys Rodríguez born in?",
            answers=["Puerto Rico"],
            aliases=["San Juan, Puerto Rico"],
            entity_titles=["Gladys Rodríguez"],
            metadata={"domain": "entertainment", "expected_in_corpus": True},
        ),
        QASample(
            query_id="nq_pilot_020",
            question="What video game platform was Airforce Delta Strike developed for?",
            answers=["PlayStation 2", "PS2"],
            aliases=["Konami"],
            entity_titles=["Airforce Delta Strike"],
            metadata={"domain": "gaming", "expected_in_corpus": True},
        ),
        # Explicit Out-of-Corpus Queries to test negative retrieval and context insufficiency detection
        QASample(
            query_id="nq_out_001",
            question="Who won the Nobel Prize in Physics in 2024?",
            answers=["John Hopfield", "Geoffrey Hinton"],
            aliases=["Hopfield and Hinton"],
            entity_titles=["2024 Nobel Prize in Physics"],
            metadata={"domain": "physics", "expected_in_corpus": False},
        ),
        QASample(
            query_id="nq_out_002",
            question="What is the deepest known location in the Mariana Trench?",
            answers=["Challenger Deep"],
            aliases=["the Challenger Deep"],
            entity_titles=["Challenger Deep"],
            metadata={"domain": "oceanography", "expected_in_corpus": False},
        ),
        QASample(
            query_id="nq_out_003",
            question="Who directed the 2023 movie Oppenheimer?",
            answers=["Christopher Nolan"],
            aliases=["Nolan"],
            entity_titles=["Oppenheimer (film)"],
            metadata={"domain": "film", "expected_in_corpus": False},
        ),
    ]


def load_qa_dataset(filepath: Optional[str] = None) -> List[QASample]:
    """Loads QA samples from JSONL or returns curated pilot benchmark samples."""
    if filepath and os.path.exists(filepath):
        logger.info(f"Loading evaluation dataset from {filepath}")
        samples = []
        with open(filepath, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                qid = str(data.get("query_id") or data.get("id") or f"q_{idx}")
                q = data.get("question", "")
                ans = data.get("answer") or data.get("answers") or []
                if isinstance(ans, str):
                    ans = [ans]
                aliases = data.get("aliases", [])
                entities = data.get("entity_titles") or data.get("titles") or []
                samples.append(QASample(
                    query_id=qid,
                    question=q,
                    answers=ans,
                    aliases=aliases,
                    entity_titles=entities,
                    metadata=data.get("metadata", {}),
                ))
        return samples

    logger.info("No external QA dataset path provided; using curated pilot benchmark samples.")
    return get_curated_pilot_benchmark_samples()
