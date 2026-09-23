"""Generates synthetic Wikipedia articles in Parquet format for testing and benchmarking."""
import argparse
import random
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

SAMPLE_TOPICS = [
    ("Quantum computing", "Physics", [
        "Quantum computing is a rapidly-emerging technology that harnesses the laws of quantum mechanics to solve problems too complex for classical computers.",
        "Today, IBM Quantum hardware is available to thousands of developers. Our engineers deliver increasingly powerful quantum processing units and software tools.",
        "Qubits can exist in a multidimensional state known as superposition, allowing exponential scaling of computational state spaces.",
        "Fault-tolerant quantum computation requires quantum error correction codes like surface codes to protect quantum information from decoherence."
    ]),
    ("Artificial intelligence", "Computer science", [
        "Artificial intelligence was founded as an academic discipline in 1956, and in the years since has experienced several waves of optimism and disappointment.",
        "Machine learning is a subset of artificial intelligence that focuses on building applications that learn from data and improve their accuracy over time.",
        "Deep learning architectures such as deep neural networks and recurrent neural networks have been applied to computer vision, speech recognition, and natural language processing.",
        "Large language models trained on massive corpora have demonstrated emergent zero-shot and few-shot reasoning capabilities across diverse cognitive benchmarks."
    ]),
    ("Relativity", "Physics", [
        "The theory of relativity usually encompasses two interrelated physics theories by Albert Einstein: special relativity and general relativity.",
        "Special relativity applies to all physical phenomena in the absence of gravity. General relativity explains the law of gravitation and its relation to other forces of nature.",
        "Einstein determined that massive objects cause a distortion in space-time, which is felt as gravity.",
        "Gravitational waves were predicted by Albert Einstein in 1916 on the basis of his theory of general relativity and first detected by LIGO in 2015."
    ]),
    ("Database management system", "Computer science", [
        "A database management system is system software for creating and managing databases. A DBMS makes it possible for end users to create, protect, read, update and delete data in a database.",
        "Relational databases became dominant in the 1980s. These model data as rows and columns in a series of tables, and the vast majority use SQL for writing and querying data.",
        "Vector databases index high-dimensional embeddings using algorithms like Hierarchical Navigable Small World (HNSW) and Inverted File Flat (IVF).",
        "Hybrid search combines dense vector similarity matching with sparse inverted indices to maximize retrieval recall on both semantic and exact lexical terms."
    ]),
]

SECTIONS = [
    ("== Overview ==", 2),
    ("== History ==", 2),
    ("=== Origins and Early Research ===", 3),
    ("=== Modern Developments ===", 3),
    ("== Architecture and Methods ==", 2),
    ("== Applications ==", 2),
    ("== See also ==", 2),
    ("== References ==", 2),
]


def generate_article_wikitext(topic: str, cat: str, paragraphs: list) -> str:
    lines = [
        f"{{{{Infobox scientific discipline | name = {topic} | field = {cat} }}}}",
        f"'''{topic}''' is a prominent area of study in {cat}.<ref name=\"smith2020\">Smith, J. (2020). Fundamentals of Science.</ref>",
        "",
    ]
    for heading, level in SECTIONS:
        lines.append(f"{heading}")
        if "See also" in heading:
            lines.append(f"* [[Computation]]\n* [[Mathematics]]\n* [[Information theory]]\n")
        elif "References" in heading:
            lines.append("<references />\n")
        else:
            chosen_p = random.choice(paragraphs)
            lines.append(f"{chosen_p}<ref>Journal of Advanced Research, 2024.</ref>\n")
            lines.append(f"Further details on [[{topic}|this field]] highlight significant empirical advancements.\n")
    lines.append(f"[[Category:{cat}]]\n")
    return "\n".join(lines)


def generate_dataset(output_path: str, count: int = 1000):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ids = []
    titles = []
    urls = []
    texts = []
    categories = []

    print(f"Generating {count} synthetic Wikipedia articles...")
    for i in range(1, count + 1):
        topic_base, cat, paragraphs = random.choice(SAMPLE_TOPICS)
        title = f"{topic_base} (Variant {i})" if i > len(SAMPLE_TOPICS) else topic_base
        url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        wikitext = generate_article_wikitext(title, cat, paragraphs)

        ids.append(i)
        titles.append(title)
        urls.append(url)
        texts.append(wikitext)
        categories.append([cat])

    table = pa.Table.from_arrays(
        [
            pa.array(ids, type=pa.int64()),
            pa.array(titles, type=pa.string()),
            pa.array(urls, type=pa.string()),
            pa.array(texts, type=pa.string()),
            pa.array(categories, type=pa.list_(pa.string())),
        ],
        names=["id", "title", "url", "text", "categories"],
    )

    pq.write_table(table, str(path), compression="snappy")
    print(f"Successfully wrote {count} articles ({path.stat().st_size / (1024*1024):.2f} MB) to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic Wikipedia Parquet dataset")
    parser.add_argument("--output", type=str, default="/DATA/suraj/m1/search_engine/data/raw/wikipedia_sample.parquet")
    parser.add_argument("--count", type=int, default=1000)
    args = parser.parse_args()
    generate_dataset(args.output, args.count)

