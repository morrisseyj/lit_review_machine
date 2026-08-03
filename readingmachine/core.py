"""
Core analytical pipeline for ReadingMachine.

This module implements the primary ReadingMachine workflow for converting
source documents into an inspectable, insight-level corpus representation
and then into an iterative thematic synthesis.

The core pipeline is organized around four classes:

    Ingestor
    Insights
    Clustering
    Summarize

Together these classes transform a corpus through the following stages:

    documents
        ↓
    full text
        ↓
    chunks
        ↓
    chunk insights
        ↓
    meta-insights
        ↓
    embeddings
        ↓
    semantic clusters
        ↓
    cluster summaries
        ↓
    theme schemas
        ↓
    insight-to-theme mappings
        ↓
    populated themes
        ↓
    orphan detection and reintegration
        ↓
    schema repair / re-theming
        ↓
    redundancy reduction

Corpus Reading
--------------
Implemented by `Ingestor`, `Insights`, and `Clustering`.

The reading portion of the pipeline converts source documents into a
structured insight representation. Documents are ingested, cleaned,
chunked, read against explicit research questions, embedded, and grouped
into provisional semantic clusters.

The resulting insights are the primary analytical unit of the pipeline.
Clusters are used as computational scaffolding for later synthesis, not as
final themes or analytical conclusions.

Thematic Synthesis
------------------
Implemented by `Summarize`.

The synthesis portion of the pipeline converts clustered insights into
theme-level narratives. It generates cluster summaries, builds theme
schemas, maps insights to themes, populates themes, audits coverage through
orphan detection, reintegrates omitted insights, repairs unstable schemas,
and performs a final redundancy pass.

Theme structures are expected to evolve across iterations. Schema revision
is driven by synthesis outcomes, orphan patterns, failed integrations, and
evidence of representational overload.

State Architecture
------------------
The module operates over two persistent state objects:

CorpusState
    Stores the corpus-reading layer: questions, full text, chunks, and
    insights, including embeddings and cluster assignments when generated.

SummaryState
    Stores the thematic-synthesis layer: cluster summaries, theme schemas,
    mappings, populated themes, orphan outputs, and redundancy-reduced
    summaries.

This separation preserves the distinction between reading and synthesis
while maintaining traceability from final summaries back to source
insights and citations.

Persistence and Recovery
------------------------
Long-running stages support persistence and recovery through Parquet
state files, pickle checkpoints, and deterministic fingerprints.

This allows interrupted workflows to resume safely and helps detect state
drift between runs.

Design Principles
-----------------
The core pipeline reflects several ReadingMachine principles:

Coverage preservation
    The workflow audits and reintegrates omitted insights rather than
    allowing early summarization losses to become permanent.

Traceability
    Insights, themes, summaries, and citations remain linked through
    persistent identifiers.

Inspectability
    Intermediate artifacts are stored explicitly so users can examine how
    the corpus representation and thematic structure evolve.

Iterative synthesis
    Theme schemas are revised in response to observed synthesis failures
    rather than treated as fixed outputs.

Separation of reading and synthesis
    Document reading, insight extraction, clustering, and thematic
    organization are represented as distinct stages.

Scalability
    Chunking, batching, clustering, checkpointing, and context-window-aware
    synthesis support large corpora and long-running LLM workflows.

Typical Usage
-------------
The pipeline is typically executed as a sequence of state-transforming
operations:

    corpus = CorpusState.load(...)

    ingestor = Ingestor(...)
    ingestor.ingest_papers()
    ingestor.update_metadata()
    ingestor.gen_unique_citations()
    ingestor.chunk_papers()

    insights = Insights(...)
    insights.get_chunk_insights()
    insights.get_meta_insights()

    clustering = Clustering(...)
    clustering.embed_insights()
    clustering.reduce_dimensions()
    clustering.generate_clusters(...)

    summarizer = Summarize(...)
    summarizer.summarize_clusters()
    summarizer.gen_theme_schema()
    summarizer.map_insights_to_themes()
    summarizer.populate_themes()
    summarizer.address_orphans()
    summarizer.gen_theme_schema()
    summarizer.map_insights_to_themes()
    summarizer.populate_themes()
    summarizer.address_orphans()
    summarizer.address_redundancy()

Each stage updates CorpusState or SummaryState, enabling intermediate
outputs to be inspected, persisted, rewound, or reused across runs.
"""
# import custom libraries

import json

from . import config, utils
from .state import CorpusState, SummaryState
from .prompts import Prompts

# import standard libraries
from typing import List, Any, Optional
import pandas as pd
import numpy as np
from copy import deepcopy
import os
from pathlib import Path
from collections import defaultdict
import pymupdf
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pickle
import tiktoken
import re
import umap
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
import itertools
import networkx as nx
import math


class Ingestor:
    """
    Document ingestion, corpus preparation, and chunk generation for ReadingMachine.

    The Ingestor class transforms raw source documents into the structured
    corpus representation used by the ReadingMachine reading pipeline. It is
    responsible for converting files on disk into validated, metadata-rich,
    deduplicated, and chunked corpus objects suitable for insight extraction.

    The ingestion workflow operates over a CorpusState and performs several
    distinct stages:

        file discovery
            ↓
        document ingestion
            ↓
        metadata extraction
            ↓
        corpus reconciliation
            ↓
        deduplication
            ↓
        citation generation
            ↓
        chunk generation

    Supported document formats
    --------------------------
    - PDF
    - HTML

    PDF files are ingested page by page using PyMuPDF.

    HTML files undergo a hybrid extraction process consisting of:

        HTML cleaning
            ↓
        body-text extraction
            ↓
        character chunking
            ↓
        LLM-assisted content extraction

    This approach combines deterministic preprocessing with model-based
    content identification.

    Metadata extraction
    -------------------
    The class can extract and normalize document metadata using an LLM.

    The following fields are populated or updated:

    - `paper_title`
    - `paper_author`
    - `paper_date`

    Metadata extraction operates over document text and includes type
    normalization, missing-value handling, and resume support for long
    corpus-processing runs.

    Corpus reconciliation
    ---------------------
    During ingestion, filesystem documents are matched against existing
    corpus metadata using:

    - `paper_id`
    - `question_id`

    The ingestion process identifies:

    - successfully ingested papers
    - ingestion failures
    - unmatched files
    - expected papers with no corresponding document

    These diagnostics are exposed through dedicated attributes to support
    user review and recovery.

    Deduplication
    -------------
    The class supports human-in-the-loop duplicate detection.

    Two deduplication strategies are used:

    1. Exact duplicate detection
    - metadata-based
    - content-based

    2. Similarity-based review
    - title similarity
    - full-text shingle similarity

    Potential duplicates are exported for manual review and are only applied
    to the corpus once explicitly confirmed by the user.

    Citation and identifier generation
    ----------------------------------
    Following metadata validation, the class can generate citation-based
    paper identifiers and formatted in-text citation strings.

    These identifiers provide stable, human-readable references that are
    used throughout downstream ReadingMachine outputs while preserving links
    back to the original filename-derived identifiers.

    Chunk generation
    ----------------
    The final ingestion stage converts full-text documents into overlapping
    reading chunks.

    Chunk generation includes:

    - text normalization
    - greedy chunk splitting
    - duplicate chunk removal
    - boilerplate filtering

    The resulting chunk representation forms the input to the insight
    extraction stage described in the ReadingMachine methodology.

    Pipeline outputs
    ----------------
    The ingestion workflow produces and updates:

    - `corpus_state.full_text`
    - `corpus_state.chunks`
    - `corpus_state.insights`

    These structures form the corpus-reading layer that supports:

    - insight extraction
    - clustering
    - thematic synthesis
    - citation tracing

    Design principles
    -----------------
    The ingestion stage prioritizes:

    Traceability
        Documents remain linked to identifiers throughout ingestion,
        metadata extraction, chunking, and downstream synthesis.

    Recoverability
        Long-running metadata extraction tasks support checkpointing and
        resume workflows.

    Human oversight
        Deduplication decisions are surfaced for review rather than applied
        automatically.

    Conservative cleaning
        Cleaning and chunk filtering focus on removing obvious extraction
        artifacts while preserving substantive document content.

    Attributes
    ----------
    corpus_state : CorpusState
        Working corpus state mutated throughout ingestion.

    file_path : str
        Root directory containing source documents.

    llm_client : Any
        Language model client used for HTML extraction, metadata extraction,
        and citation generation.

    ai_model : str
        Model identifier used for ingestion-time LLM calls.

    ingestion_errors : list[str]
        Files that failed ingestion.

    pickle_path : str
        Directory used for metadata extraction checkpoints and resume
        support.

    fuzzy_check_path : str
        Directory used for duplicate-review outputs.
    """

    def __init__(
        self,
        llm_client: Any,
        ai_model: str,
        corpus_state: Optional[CorpusState] = None,
        questions: Optional[List[str]] = None,
        papers: Optional[pd.DataFrame] = None,
        file_path: str = os.path.join(os.getcwd(), config.CORPUS_LOCATION),
        pickle_path: str = config.PICKLE_SAVE_LOCATION, # For storing the pickles of LLM metadata retreival for resume
        fuzzy_check_path: str = config.FUZZY_CHECK_PATH
    ) -> None:
        """
        Initialize an Ingestor for full-text ingestion and chunk preparation.

        Validates or constructs the CorpusState used for ingestion, normalizes
        question text in the insights table, and stores configuration needed for
        document loading, LLM-assisted metadata extraction, resume support, and
        manual fuzzy-check workflows.

        Parameters
        ----------
        llm_client : Any
            LLM client used for ingestion-time model calls, including metadata
            extraction.

        ai_model : str
            Model identifier used for LLM-assisted ingestion tasks.

        corpus_state : CorpusState, optional
            Existing CorpusState to use as the ingestion input. If provided,
            `questions` and `papers` must not also be provided.

        questions : list[str] or pd.DataFrame, optional
            Question data used to construct a new CorpusState when `papers` is
            provided. Passed through `utils.validate_format()`.

        papers : pd.DataFrame, optional
            Paper metadata table used as the initial `insights` table when
            constructing a new CorpusState.

        file_path : str, default=os.path.join(os.getcwd(), config.CORPUS_LOCATION)
            Directory containing source documents to ingest.

        pickle_path : str, default=config.PICKLE_SAVE_LOCATION
            Path used to store pickled intermediate outputs from LLM metadata
            retrieval for resume support.

        fuzzy_check_path : str, default=config.FUZZY_CHECK_PATH
            Path used for fuzzy-check artifacts during ingestion workflows.

        Attributes
        ----------
        RUN : str
            Fixed run label set to `"ingest"`.

        corpus_state : CorpusState
            Deep-copied, validated CorpusState used and modified by ingestion.

        ingestion_errors : list[str]
            Container for ingestion errors encountered during processing.

        Notes
        -----
        Initialization accepts either an existing CorpusState or the components
        needed to construct one from `questions` and `papers`.

        The resulting CorpusState is deep-copied, so subsequent ingestion steps
        operate on the Ingestor's internal state rather than mutating the object
        passed by the caller.

        The initializer requires the ingestion input to contain the corpus
        metadata columns needed by downstream ingestion, including paper
        identifiers, title, author, date, DOI, download status, and messy ID
        fields.
        """

        self.RUN = "ingest"
        self.fuzzy_check_path = fuzzy_check_path
    
        self.corpus_state = deepcopy(
            utils.validate_format(
                corpus_state=corpus_state,
                questions=questions,
                injected_value=papers,
                state_required_cols=[
                    "question_id", "question_text", "search_string_id", "search_string",
                    "paper_id", "paper_title", "paper_author", "paper_date", "doi",
                    "download_status", "messy_question_id", "messy_paper_id"
                ],
                injected_required_cols=["question_id", "question_text"]
            )
        )

        self.corpus_state.enforce_canonical_question_text()
        self.file_path: str = file_path
        self.llm_client: Any = llm_client
        self.ai_model: str = ai_model
        self.ingestion_errors: List[str] = []
        self.pickle_path: str = pickle_path

    @staticmethod
    def _pprint_dict(d: dict):
        """
        Format a dictionary of iterable values as an indented multiline string.

        Creates a human-readable representation in which each dictionary key is
        placed on its own line and each associated value is displayed on
        subsequent indented lines.

        Parameters
        ----------
        d : dict
            Dictionary whose values are iterable collections.

        Returns
        -------
        str
            Formatted multiline string suitable for logging, console output, or
            diagnostic messages.

        Notes
        -----
        This function does not print the dictionary directly. It returns the
        formatted string, allowing the caller to print, log, or otherwise use
        the output.

        The function assumes each dictionary value is iterable and will iterate
        over its contents when constructing the formatted representation.
        """
        lines = []
        # Attach keys to a list of their values with indentation for readability and with newlines
        for key, value in d.items():
            lines.append(f"{key}\n")
            for item in value:
                lines.append(f"    {item}\n")
            lines.append("\n")
        # Convert to a string which when retured will print the dictionary in a readable format with keys and indented values
        line_str = "".join(lines)

        return(line_str)
                     
    def _list_files(self) -> List[str]:
        """
        Discover ingestible documents and validate filename uniqueness.

        Recursively searches `self.file_path` for supported document types and
        returns the discovered files. Currently supported formats are:

        - `.pdf`
        - `.html`

        Before returning the file list, the method verifies that all filenames
        (not full paths) are unique across the directory tree.

        Parameters
        ----------
        None

        Returns
        -------
        list[pathlib.Path]
            Absolute paths to all discovered ingestible documents.

        Raises
        ------
        ValueError
            If duplicate filenames are detected. The error message includes the
            conflicting filenames and their full paths.

        Notes
        -----
        Filename uniqueness is required because ReadingMachine derives
        `paper_id` values from filenames during ingestion. Duplicate filenames
        would therefore produce ambiguous document identifiers and break the
        linkage between documents, chunks, insights, and downstream synthesis
        artifacts.

        The duplicate check is performed on filenames only, not full paths. Two
        files with the same name in different subdirectories are treated as a
        conflict.

        The returned file list is not guaranteed to be sorted and reflects the
        order returned by `os.walk()`.
        """
        list_of_path_obj: List[str] = []
        # First get all the files that are html and pdfs in the directory and subdirectories and store as path objects in a list
        for root, _, files in os.walk(self.file_path):
            for file in files:
                p = Path(root) / file
                if p.is_file() and p.suffix.lower() in [".pdf", ".html"]:
                     list_of_path_obj.append(p)

        
        # Then check for duplicate file names (not paths) in the list of path objects - this is important as the paper_id is derived from the file name, 
        # so if there are duplicate file names this will cause issues with linking papers to insights later in the pipeline. 
        # If there are duplicate file names, print out the conflicting file names and their absolute paths to allow the user to resolve before ingestion.
        list_of_files = [p.name for p in list_of_path_obj]
        # If there are duplicates find them and raise an error with the full path for the user
        if len(list_of_files) != len(set(list_of_files)):
            conflicting_files = defaultdict(list)
            for file_name in list_of_files:
                if list_of_files.count(file_name) > 1:
                    for f, p in zip(list_of_files, list_of_path_obj):
                        if f == file_name:
                            conflicting_files[file_name].append(p.resolve())
            raise ValueError(
                "Duplicate file names detected in ingestion directory. " 
                "Please ensure all files have unique names to avoid ingestion errors." 
                "Conflicting files and their paths are as follows:\n" 
                + self._pprint_dict(conflicting_files)
            )
        
        # Otherwise return the list of files
        return [p.absolute() for p in list_of_path_obj]

    def _ingest_pdf(self, path: str) -> List[str]:
        """
        Extract page-level text from a PDF document.

        Opens a PDF using PyMuPDF and extracts the text content of each page
        individually.

        Parameters
        ----------
        path : str
            Path to the PDF document.

        Returns
        -------
        list[str]
            List of page-level text strings in document order. Each element
            corresponds to a single PDF page.

        Notes
        -----
        Page boundaries are preserved because each page is returned as a
        separate string rather than concatenating the document into a single
        text block.

        The function performs no text cleaning, normalization, OCR, or metadata
        extraction. It returns the raw text produced by PyMuPDF's
        `Page.get_text()` method.
        """
        with pymupdf.open(path) as doc:
            return [doc[i].get_text() for i in range(doc.page_count)]

    @staticmethod
    def _html_cleaner(html_content: str) -> str:
        """
        Extract visible document text from HTML content.

        Parses raw HTML with BeautifulSoup, removes common non-content elements,
        and returns the visible text contained within the document's `<body>`
        element.

        Parameters
        ----------
        html_content : str
            Raw HTML document content.

        Returns
        -------
        str
            Plain-text representation of the document body. Returns an empty
            string if no `<body>` element is present.

        Notes
        -----
        The following elements are removed before text extraction:

        - `script`
        - `style`
        - `nav`
        - `header`
        - `footer`
        - `aside`

        Text is extracted using `body.get_text(separator="\\n", strip=True)`,
        which preserves a newline-separated structure while removing leading and
        trailing whitespace from individual text fragments.

        This function performs lightweight content extraction only. It does not
        attempt article extraction, boilerplate detection, readability scoring,
        or semantic identification of main content regions.
        """
        soup = BeautifulSoup(html_content, "html.parser")
        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            tag.decompose()
        body = soup.find('body')
        if body:
            return body.get_text(separator='\n', strip=True)
        return ""

    @staticmethod
    def _html_chunker(clean_html: str, chunk_size: int = 16000) -> List[str]:
        """
        Split cleaned HTML text into fixed-size character chunks.

        Divides cleaned HTML-derived text into sequential character-based chunks
        for downstream processing. This is primarily used to keep large HTML
        documents within manageable input limits before LLM-assisted content
        extraction and metadata identification.

        Parameters
        ----------
        clean_html : str
            Plain-text content extracted from an HTML document.

        chunk_size : int, default=16000
            Maximum number of characters per chunk.

        Returns
        -------
        list[str]
            List of text chunks in original document order.

            - Empty input returns `[""]`.
            - Input shorter than `chunk_size` returns a single-element list.
            - Larger inputs are split into multiple fixed-size chunks.

        Notes
        -----
        Chunk boundaries are based solely on character count. The function does
        not attempt to preserve paragraphs, sentences, sections, or other
        semantic structure.

        Chunk size is measured in characters rather than tokens. Actual token
        counts depend on the content and tokenizer used by downstream language
        models.

        This helper is intentionally simple because its purpose is only to
        constrain input size prior to LLM processing. Semantic chunking occurs
        later in the ReadingMachine pipeline during corpus chunk generation.
        """
        if len(clean_html) == 0:
            return [""]
        elif len(clean_html) > chunk_size:
            chunks: List[str] = []
            start = 0
            end = chunk_size
            while start < len(clean_html):
                chunks.append(clean_html[start:end])
                start += chunk_size
                end += chunk_size
            return chunks
        else:
            return [clean_html]

    def _llm_parse_html(self, html_list: List[str], prompt: str) -> List[str]:
        """
        Extract cleaned text from HTML-derived segments using an LLM.

        Processes each HTML text segment with the configured chat-completion
        model and returns the model-generated cleaned text. Each segment is sent
        with the supplied system prompt and wrapped in `[START_TEXT]` /
        `[END_TEXT]` markers as the user message.

        Parameters
        ----------
        html_list : list[str]
            List of cleaned or partially cleaned HTML text segments to process.

        prompt : str
            System prompt instructing the model how to extract or clean the
            meaningful document content.

        Returns
        -------
        list[str]
            Model-generated text output for each input segment. If the first
            element of `html_list` is an empty string, returns `[""]`.

        Notes
        -----
        This method assumes `html_list` contains at least one element.

        The method does not use the shared `call_chat_completion()` wrapper and
        does not catch API errors. Exceptions from the LLM client will propagate
        to the caller.

        Returned values are the raw message contents from the model response;
        no additional parsing, validation, or normalization is performed.
        """
        if html_list[0] == "":
            return [""]
        output: List[str] = []
        for chunk in html_list:
            sys_prompt = prompt
            user_prompt = f"[START_TEXT] {chunk} [END_TEXT]"
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ]
            response = self.llm_client.chat.completions.create(
                model=self.ai_model,
                messages=messages
            )
            output.append(response.choices[0].message.content)
        return output

    def _paper_ingestor(self, file_full_path: str) -> List[str]:
        """
        Ingest a single document into ordered text segments.

        Dispatches document parsing based on file extension and returns the
        extracted text as a list of segments.

        Supported formats are:

        - `.pdf`: extracted page by page using `_ingest_pdf()`
        - `.html`: read from disk, cleaned with `_html_cleaner()`, split with
        `_html_chunker()`, and processed with `_llm_parse_html()`

        Parameters
        ----------
        file_full_path : str
            Path to the document file.

        Returns
        -------
        list[str]
            Ordered text segments extracted from the document.

            For PDF files, each element corresponds to one page. For HTML files,
            each element corresponds to an LLM-processed chunk of body text.

        Notes
        -----
        HTML ingestion uses both deterministic preprocessing and LLM-assisted
        content extraction. The deterministic step removes common layout
        elements and extracts body text; the LLM step attempts to isolate the
        substantive page content from each chunk.

        Unsupported file types return `["Unsupported file type"]` rather than
        raising an exception.
        """
        if Path(file_full_path).suffix.lower() == ".pdf":
            return self._ingest_pdf(file_full_path)
        
        elif Path(file_full_path).suffix.lower() == ".html":
            with open(file_full_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            clean_html = self._html_cleaner(html_content)
            html_chunks = self._html_chunker(clean_html)
            print("File is html, sending to LLM for final parsing...")
            return self._llm_parse_html(html_chunks, prompt=Prompts().extract_main_html_content())
        else:
            return ["Unsupported file type"]

    def ingest_papers(self) -> pd.DataFrame:
        """
        Ingest all supported documents and reconcile them with the corpus state.

        Discovers PDF and HTML files under `self.file_path`, extracts their text,
        matches ingested files to existing corpus metadata using `question_id`
        and `paper_id`, and populates `corpus_state.full_text` with the resulting
        document text.

        The ingestion workflow performs the following steps:

        1. Discover ingestible files.
        2. Extract text from each document.
        3. Record ingestion successes and failures.
        4. Match ingested files to existing corpus records.
        5. Identify unmatched files and missing expected documents.
        6. Populate `corpus_state.full_text`.
        7. Filter `corpus_state.insights` to successfully ingested papers.
        8. Record dropped, failed, and mismatched records for review.

        File identity is derived from:

        - filename stem → `paper_id`
        - parent directory name → `question_id`

        These identifiers are used to align ingested documents with existing
        corpus metadata.

        Returns
        -------
        None
            Returned when ingestion completes successfully.

        list[str]
            Returned only when the user elects to abort after ingestion errors.
            Contains the paths of files that failed ingestion.

        pd.DataFrame
            Returned only when the user elects to abort after file/metadata ID
            mismatches. Contains the mismatched records.

        Raises
        ------
        ValueError
            If no supported PDF or HTML files are found in the configured corpus
            directory.

        Side Effects
        ------------
        Mutates `self.corpus_state`:

        - populates `corpus_state.full_text`
        - filters `corpus_state.insights` to successfully ingested papers
        - removes temporary ingestion-only fields from `insights`

        Populates diagnostic attributes:

        - `ingestion_errors`
        - `failed_id_matches`
        - `dropped_papers`

        Notes
        -----
        Documents that cannot be matched to existing corpus metadata are tracked
        through `failed_id_matches`.

        Expected papers that have no successfully ingested file are tracked in
        `dropped_papers` and are removed from the active corpus state used by
        downstream pipeline stages.

        This method does not modify files on disk. All filtering and cleanup
        occur within the in-memory CorpusState.

        Interactive confirmation prompts are used when ingestion failures or
        metadata mismatches are detected, allowing the user to inspect the
        diagnostic attributes before deciding whether to continue.
        """
        working_insights = (
            self.corpus_state.insights.copy()
            .assign(paper_id=lambda x: x["paper_id"].astype(str)) # set as string to handle NaN values so avoid merge issues on str and float (NaN)
        )

        list_of_papers_by_page: List[List[str]] = []
        ingestion_status: List[int] = [] # to track ingestion success (1) or failure (0)
        self.ingestion_errors = []

        list_of_files = self._list_files()
        if len(list_of_files) == 0:
            raise ValueError(f"No PDF or HTML files found in the specified directory: {self.file_path}. Please add files to ingest or check the directory path.")

        for count, file in enumerate(list_of_files, start=1):
            print(f"Ingesting paper {count} of {len(list_of_files)}...")
            try:
                pages = self._paper_ingestor(file)
                list_of_papers_by_page.append(pages)
                ingestion_status.append(1)
            except Exception as e:
                print(f"Error ingesting file {file}: {e}")
                list_of_papers_by_page.append([str(e)])
                self.ingestion_errors.append(file)
                ingestion_status.append(0)

        # Confirm ingestion errors
        if self.ingestion_errors:
            abort_failed_ingest = None
            while abort_failed_ingest not in ["y", "n"]:
                abort_failed_ingest = input(
                    "Ingestion errors occurred. Examine .ingestion_errors and corpus_state.full_text.\n"
                    "Hit 'c' to confirm having read this message:\n\n\n"
                ).lower()
            
            if abort_failed_ingest == "y":
                print("Aborting ingestion. Please review the ingestion errors (returned below and accessible via .ingestion_errors) and the corpus_state.full_text object to see which papers were not ingested successfully.\n\n\n")
                return(self.ingestion_errors)
            else:
                pass # continue with ingestion despite errors

        # Create an ingestion status dataframe tracking ingestion success/failure linked by paper_id and question_id
        ingestion_status_df = pd.DataFrame({
            "paper_id": [os.path.splitext(os.path.basename(path))[0] for path in list_of_files],
            "question_id": [os.path.basename(os.path.dirname(path)) for path in list_of_files],
            "ingestion_status": ingestion_status, 
            "pages": list_of_papers_by_page
        })


        # Clean up working insights to reflect what came in mathcing an id, what was in there for which no file was found, and what came in that did not match an id
        working_insights = (
            ingestion_status_df.merge(working_insights, how="outer", on=["question_id", "paper_id"]) # Outer merge to make sure all the ingestion status records and all the insights records are included so we don't lose anyting
            .assign(ingestion_ids_matched = lambda x: np.where(x["question_text"].isna(), "import file does not match possible id", "match")) # First check if the question_text is na. If so, that means the ingested item did not match to a paper_id and question_id and therefore is unmatched
            .query("paper_id.notna()") # Filter any records for which paper_id is na, as this means there was not paper (usually a hangover from entering just questions to match with a folder of files - i.e. dump files in a folder and query them)
            .assign(ingestion_status=lambda x: np.where(x["ingestion_status"].isna(), 0, x["ingestion_status"])) # Set the ingestion status to 0 if it is na, meaning there was no paper ingested and thus no match
            .assign(ingestion_ids_matched=lambda x: np.where(x["ingestion_status"]==0, "existing id had no corresponding file", x["ingestion_ids_matched"])) # Check if the ingestion status is 0. If so, there was no paper ingested and thus no match - these are essentially docs that were not downloaded
        )

        # Get all the ids for which no file was found
        failed_id_matches = working_insights[working_insights["ingestion_ids_matched"] != "match"] # Get all the non-match ids: both files that came in without an id and those with an id for which no file came in
        self.failed_id_matches= failed_id_matches

        if failed_id_matches.shape[0] > 0:

            unmatched_files = failed_id_matches[
                failed_id_matches["ingestion_ids_matched"] == "import file does not match possible id"
            ]

            missing_files = failed_id_matches[
                failed_id_matches["ingestion_ids_matched"] == "existing id had no corresponding file"
            ]


            abort_failed_match = None
            while abort_failed_match not in ['y', 'n']:
                abort_failed_match = input(
                "\n\nWarning: File / metadata ID mismatches detected.\n\n"
                f"Files ingested that did not match an existing paper_id: {len(unmatched_files)}\n"
                f"Insights rows with no corresponding file: {len(missing_files)}\n\n"
                "If you want files to link to specific questions, place them inside folders.\n"
                "If you invoked the getlit.py module in this library this warning IS likely relevant to you.\n" \
                "If you are reading your own corpus, you can likely ignore this message.\n\n"
                "If you ignore this warning any paper ids that did not have a matching file will be deleted from corpus_state.insights. You can look these up later by exploring the corpus_state.insights object created earlier in the pipeline.\n\n"    
                "Do you wish to abort ingestion to review the failed id matches? (y/n):\n"
            ).lower()
            
            if abort_failed_match == 'y':
                print("Aborting ingestion. Please review the failed id matches (returned below and accessible via .failed_id_matches).\n\n")
                return(failed_id_matches)
            else:
                pass #continue with ingestion despite failed id matches


        # Identify all undownloaded files for the user to see what they are not getting. 
        # Note not tracking these is a design. This package manages reading. It has a module that helps with identifying papers to read, but it is the users responsibility to get the papers. 
        # So we record this error here but it is not persisted to corpus_state
        self.dropped_papers = working_insights[working_insights["ingestion_status"] == 0]

        # Get the all the ingested papers
        working_insights = working_insights[working_insights["ingestion_status"] == 1] # Filter to just the papers that were ingested successfully, as these are the ones we have insights for and can track through the pipeline.
        # Populate the full text corpus_state object
        full_text = (
            working_insights[["paper_id", "pages"]]
            .assign(full_text=lambda x: [" ".join(pages) for pages in x["pages"]])
            .drop(columns=["pages"]) # Drop pages as they take up memory and are not needed, we have the full text now
        )
        # Set the full text as a corpus_state attribute
        self.corpus_state.full_text = full_text

        # Drop pages from insights as well as other fields from the lit retrieve module that are no longer needed:
        working_insights.drop(columns=["pages", "download_status", "messy_question_id", "messy_paper_id"], inplace=True)

        # Set as corpus_state attributes
        self.corpus_state.insights = working_insights

        if self.dropped_papers.shape[0] > 0: # Set as none on init, gets created if there are dropped papers
                print(
                    f"Warning: {self.dropped_papers.shape[0]} paper(s) were not downloaded from your original list. These papers are listed in the .dropped_papers attribute.\n"
                    "You can review these papers to see what was not ingested successfully, update and potentially re-ingest them, but they will not be included in the rest of the pipeline as we have no text to work with for these papers.\n\n"
                    )
                
        print("\nPaper ingestion complete")

    def _get_metadata_from_llm(self, paper_id: str, text: str) -> dict[str, Any]:
        """
        Extract and normalize document metadata using an LLM.

        Sends document text to the configured language model and requests a
        structured metadata record containing:

        - `paper_id`
        - `paper_title`
        - `paper_author`
        - `paper_date`

        The returned metadata is validated and the publication date is
        normalized to either an integer year or a missing value.

        Parameters
        ----------
        paper_id : str
            Identifier of the paper being processed.

        text : str
            Document text used for metadata extraction. In practice this is
            typically the beginning of the document, where title, authorship,
            and publication information are most likely to appear.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:

            - `paper_id`
            - `paper_title`
            - `paper_author`
            - `paper_date`

        Raises
        ------
        KeyError
            If the returned metadata dictionary does not contain one or more
            required fields.

        ValueError
            If a non-empty, non-"NA" `paper_date` value cannot be converted to
            an integer year.

        Notes
        -----
        Metadata extraction is performed using the shared
        `utils.call_chat_completion()` helper.

        If the model call fails or returns invalid JSON, the helper returns a
        fallback metadata record containing:

            {
                "paper_id": paper_id,
                "paper_title": "NA",
                "paper_author": "NA",
                "paper_date": "NA"
            }

        After validation, `paper_date` values of `"NA"` or empty strings are
        converted to `pd.NA`. Other string values are converted to integers.
        """
        
        # Set variables for the call_chat_completion function from utils
        sys_prompt = Prompts().get_metadata()
        user_prompt = f"paper_id: {paper_id}\nTEXT:\n{text}"
        fallback = {
            "paper_id": paper_id,
            "paper_title": "NA",
            "paper_author": "NA",
            "paper_date": "NA"
        }
    
        response_dict = utils.call_chat_completion(llm_client=self.llm_client,
                                                    ai_model=self.ai_model,
                                                    sys_prompt=sys_prompt,
                                                    user_prompt=user_prompt,
                                                    return_json=True,
                                                    fall_back=fallback)

        # Validate keys
        required_keys = ["paper_id", "paper_title", "paper_author", "paper_date"]
        for key in required_keys:
            if key not in response_dict:
                raise KeyError(f"Metadata extraction failed: missing key '{key}'")

        # Clean paper_date
        paper_date = response_dict["paper_date"]
        if isinstance(paper_date, str):
            paper_date = paper_date.strip()
            if paper_date.upper() == "NA" or paper_date == "":
                response_dict["paper_date"] = pd.NA
            else:
                response_dict["paper_date"] = int(paper_date)
        else:
            response_dict["paper_date"] = paper_date

        return response_dict
    
    @staticmethod
    def _metadata_type_check(x, desired_type):
        """
        Normalize a metadata value to an expected Python type.

        Performs lightweight missing-value handling and type coercion for
        metadata fields returned by the LLM during ingestion.

        Parameters
        ----------
        x : Any
            Metadata value to normalize.

        desired_type : type
            Expected output type. Currently intended for `str` and `int`.

        Returns
        -------
        Any
            Normalized value of the requested type, or `pd.NA` when the value is
            missing or unsupported.

        Raises
        ------
        ValueError
            If `desired_type is int` and a non-empty string cannot be converted
            with `int()`.

        Notes
        -----
        The function treats `pd.NA`, empty strings, and the string `"NA"` as
        missing values.

        For string values:

        - surrounding whitespace is stripped
        - strings are converted to `int` when `desired_type is int`
        - stripped strings are returned when `desired_type is str`

        Values already matching the requested type are returned unchanged.
        Unsupported values are returned as `pd.NA`.

        This helper performs minimal validation and is intended to standardize
        LLM-extracted metadata before updating corpus metadata tables.
        """
        # Missing or explicit NA
        if pd.isna(x):
            return pd.NA
        if isinstance(x, str):
            s = x.strip()
            if s.upper() == "NA" or s == "":
                return pd.NA
            # Convert year-like strings when int requested
            if desired_type is int:
                return int(s)  # assumes s is YYYY
            if desired_type is str:
                return s

        # Non-string already correct type
        if desired_type is int and isinstance(x, int):
            return x
        if desired_type is str and isinstance(x, str):
            return x

        # If something unexpected comes back (e.g., list/dict), treat as missing
        return pd.NA
    
    def _populate_metadata(self, 
                           metadata_check_df: pd.DataFrame, # The dataframe containing the columns neccesary for the metadata check which is paper_id, paper_title, paper_author, paper_date, and full_text.
                           recovered_metadata_check: Optional[List[pd.DataFrame]] = None # The recoevered metadata from a previous run if the metadata check was interrupted and needs to be resumed. 
                           ) -> List[pd.DataFrame]:
        
        """
        Extract, normalize, and persist metadata records for ingested documents.

        Iterates over `metadata_check_df`, extracts publication metadata for each
        paper using `_get_metadata_from_llm()`, normalizes the returned values,
        and stores each result as a single-row DataFrame. After each paper is
        processed, the accumulated metadata list is saved to disk so interrupted
        runs can be resumed.

        Parameters
        ----------
        metadata_check_df : pd.DataFrame
            DataFrame containing the documents to check. Expected columns are:

            - `paper_id`
            - `full_text`

        recovered_metadata_check : list[pd.DataFrame], optional
            Previously saved metadata records from an interrupted run. If
            provided, these records are used as the starting output list and the
            same number of rows are skipped from `metadata_check_df`.

        Returns
        -------
        list[pd.DataFrame]
            List of single-row metadata DataFrames. Each element contains:

            - `paper_id`
            - `paper_title`
            - `paper_author`
            - `paper_date`

        Side Effects
        ------------
        Creates `self.pickle_path` if needed and writes the accumulated metadata
        records to:

            {self.pickle_path}/metadata_check.pkl

        after each processed paper.

        Notes
        -----
        Only the first 5000 characters of `full_text` are passed to the LLM for
        metadata extraction.

        Metadata values are normalized with `_metadata_type_check()` before
        being stored. Titles and authors are normalized as strings; publication
        dates are normalized as integer years or `pd.NA`.

        Resume behavior is position-based: when recovered metadata is supplied,
        the method skips the first `len(recovered_metadata_check)` rows of
        `metadata_check_df`. It does not verify that recovered `paper_id` values
        match the skipped rows.
        """
        # If there is no recovered metadata passed to the function then start with an empty list, otherwise start with the recovered metadata
        if recovered_metadata_check is None:
            output = []
        else:
            output = recovered_metadata_check
            # Get the length of this list to see how many entries have been handled and drop them
            start = len(recovered_metadata_check) 
            metadata_check_df = metadata_check_df[start:] 

        # Now iterate either over the full df populating the empty list or over the partial dataframe populating the partially completed list
        for idx, row in metadata_check_df.iterrows():
            print(f"Checking metadata for paper {idx + 1} of {metadata_check_df.shape[0]}...")
            paper_id = row["paper_id"]
            text = row["full_text"][:5000] if row["full_text"] else ""
            metadata = self._get_metadata_from_llm(paper_id, text) # Get the metadat from the llm
            # Call the metadata type check function to ensure the metadata is in the correct format and handle any unexpected formats that may come back from the llm, such as lists or dicts, which we want to treat as missing values. This is important for ensuring the integrity of the metadata and avoiding issues later in the pipeline when we rely on this metadata for linking insights to papers and for any analyses that involve metadata.
            author = self._metadata_type_check(metadata.get("paper_author"), str)
            title  = self._metadata_type_check(metadata.get("paper_title"), str)
            year   = self._metadata_type_check(metadata.get("paper_date"), int)
            
            # create a new dataframe with the metadata pinned to paper id
            paper_meta_df = pd.DataFrame({
                "paper_id": [paper_id],
                "paper_title": [title],
                "paper_author": [author],
                "paper_date": [year]
            })

            output.append(paper_meta_df)
            
            # Save to pickle to allow for resume, make sure path exists - use safe pickle to ensure atomic save
            os.makedirs(self.pickle_path, exist_ok=True)
            utils.safe_pickle(output, os.path.join(self.pickle_path, "metadata_check.pkl"))

        return(output)

    def update_metadata(self) -> pd.DataFrame:
        """
        Update paper metadata in the corpus state using LLM extraction.

        Builds a metadata-check table by joining the current `insights` metadata
        with `full_text`, extracts metadata for each paper, and replaces the
        existing metadata columns in `corpus_state.insights` with the extracted
        values.

        The following fields are updated:

        - `paper_title`
        - `paper_author`
        - `paper_date`

        Returns
        -------
        pd.DataFrame
            Updated `corpus_state.insights` DataFrame containing the extracted
            metadata fields.

        Raises
        ------
        pandas.errors.MergeError
            If merging extracted metadata back into `corpus_state.insights`
            violates the `validate="one_to_one"` constraint on `paper_id`.

        Side Effects
        ------------
        Mutates `self.corpus_state.insights` by replacing the existing metadata
        columns with LLM-extracted metadata.

        Reads from or writes to:

            {self.pickle_path}/metadata_check.pkl

        through `_populate_metadata()`.

        Notes
        -----
        If a metadata-check pickle already exists, the user is prompted to either
        resume from the saved metadata records or rerun metadata extraction from
        the beginning.

        Metadata extraction and per-paper persistence are delegated to
        `_populate_metadata()`.

        The merge back into `corpus_state.insights` is performed on `paper_id`.
        The method assumes one row per `paper_id` in both the existing insights
        table and the extracted metadata table.
        """
        #Create the metadata check which is the dataframe containing the columns i need for the check
        metadata_check_df = (
            self.corpus_state.insights.copy()[["paper_id", "paper_title", "paper_author", "paper_date"]]
            .merge(self.corpus_state.full_text[["paper_id", "full_text"]],
                how="left",
                on=["paper_id"])
        )

        # Check whether a file exists which shows previoulsy completed meta data check results and ask the user what they want to do: rerun or resume

        if os.path.isfile(os.path.join(self.pickle_path, "metadata_check.pkl")):
            reload_meta_data = None
            while reload_meta_data not in ["1", "2"]:
                reload_meta_data = input(
                    "A metadata check pickle file was found. This indicates that a metadata check was previously run but may not have completed. \n"
                    "Do you want to: (pick the corresponding number)\n"
                    "  1 - Reload/resume the previous metadata check\n"
                    "  2 - Re-run the metadata check from the start\n"
                ).lower()

            # If they want to resume, get the data and run with recovered_metadata 
            if reload_meta_data == "1":
                with open(os.path.join(self.pickle_path, "metadata_check.pkl"), "rb") as f:
                    recovered_metadata = pickle.load(f)
                metadata_list = self._populate_metadata(metadata_check_df, recovered_metadata_check=recovered_metadata)
            # If they want to rerun, run the check from the start with an empty list for the output
            else:
                metadata_list = self._populate_metadata(metadata_check_df)
        
        # If the file does not exist, run the metadata check from the start with an empty list for the output
        else:
            metadata_list = self._populate_metadata(metadata_check_df)
         
        # Now process the list: concat to full metadata
        full_meta_data_df = pd.concat(metadata_list, ignore_index=True)

        # Merge back to corpus_state.insights. Drop the old metadata columns and replace with the new metadata from the llm. This ensures that any metadata that was missing or incorrect is updated with the llm response, while any existing correct metadata is retained. We merge on paper_id to ensure we are updating the correct records, and we validate one_to_one to ensure there are no duplicate paper_ids which would indicate an issue with the data integrity.
        updated_insights = (
            self.corpus_state.insights
            .drop(columns=["paper_title", "paper_author", "paper_date"])
            .merge(full_meta_data_df, how="left", on="paper_id", validate="one_to_one")
        )

        # Update the corpus_state.insights to now have the correct metadata. 
        # Note we don't save here as its not the end of the object and we have the pickle to handle recovery if we need to re run.
        self.corpus_state.insights = updated_insights

        return self.corpus_state.insights
    
    def drop_duplicates(self, threshold: float = 0.9) -> None:
        """
        Generate a manual review file for potential duplicate documents.

        Identifies candidate duplicate papers using full-text shingle similarity
        and writes the resulting review table to disk. The method does not apply
        deduplication directly; users are expected to inspect the review file,
        delete duplicate rows, and then call `update_state()` to apply the
        reviewed corpus.

        Parameters
        ----------
        threshold : float, default=0.9
            Jaccard similarity threshold used to group documents as candidate
            duplicates. Higher values require greater full-text overlap.

        Returns
        -------
        None

        Side Effects
        ------------
        Creates the review directory if needed and writes:

            {self.fuzzy_check_path}/{self.RUN}/duplicate_check.csv

        Notes
        -----
        Duplicate detection is delegated to `utils.prepare_dedup_review()` with
        `engine="shingles"`. This means exact full-text duplicates are first
        removed, then remaining documents are grouped using shingle-based
        Jaccard similarity.

        This method supports a human-in-the-loop deduplication workflow. It
        surfaces likely duplicate groups but leaves final deletion decisions to
        the user.

        The active `corpus_state` is not modified by this method.
        """
        # Set the save path for use to insepct
        save_dir = os.path.join(self.fuzzy_check_path, self.RUN)
        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, "duplicate_check.csv")

        df_for_review = utils.prepare_dedup_review(state = self.corpus_state, threshold=threshold, engine="shingles")
        df_for_review.to_csv(output_path, index=False)

        print(
            f"Potential duplicate review file saved to {output_path}.\n"
            "Delete duplicate rows and keep one per paper.\n"
            "Then run update_state()."
        )

        return None

    def update_state(self, 
                     filename: str, 
                     encoding: str = "utf-8",
                     output_cols = [
                        "question_id",
                        "question_text",
                        "search_string_id",
                        "search_string",
                        "paper_id",
                        "paper_title",
                        "paper_author",
                        "paper_date",
                        "doi"
                        ]
                    ) -> pd.DataFrame:
        """
        Apply a manually reviewed deduplication file to the corpus state.

        Loads a reviewed CSV or Excel file produced by the duplicate-review
        workflow, uses it as the updated `insights` table, and filters
        `corpus_state.full_text` so that it remains aligned with the retained
        paper IDs.

        Parameters
        ----------
        filename : str
            Name of the reviewed CSV or Excel file located under:

                {self.fuzzy_check_path}/{self.RUN}/

        encoding : str, default="utf-8"
            File encoding used when reading CSV files. Ignored for Excel files.

        output_cols : list, optional
            Columns to retain from the reviewed file. By default, the method
            keeps the paper- and question-level metadata columns needed for
            downstream ingestion and chunking.

        Returns
        -------
        pd.DataFrame
            Updated `corpus_state.insights` DataFrame.

        Raises
        ------
        ValueError
            If the reviewed file is not found at the expected path.

        ValueError
            If the reviewed file contains no retained `paper_id` values.

        Side Effects
        ------------
        Mutates `self.corpus_state`:

        - replaces `corpus_state.insights` with the reviewed records
        - filters `corpus_state.full_text` to retained `paper_id` values

        Notes
        -----
        This method is designed for the human-in-the-loop deduplication workflow
        started by `drop_duplicates()`. The user reviews the duplicate-check
        file, removes duplicate rows manually, and then calls this method to
        apply the reviewed corpus.

        The method does not save the updated CorpusState to disk. It assumes
        subsequent ingestion steps, such as chunking, will continue in the same
        workflow.

        If present, `paper_author` values such as empty strings, `"No author
        found"`, `"NA"`, `"null"`, `pd.NA`, and `np.nan` are normalized to
        `None`.
        """
        # Set the filepath to the reviewed file based on the provided filename and the expected location of the fuzzy check outputs. 
        filepath = os.path.join(self.fuzzy_check_path, self.RUN, filename)

        # Check file exists
        if not os.path.isfile(filepath):
            raise ValueError(f"Reviewed file not found at expected location: {filepath}. Please ensure the file is in the correct directory and the filename is correct.")
        
        #Get the updated insights df with the correct schema
        insights_df = CorpusState.load_insights_from_csv_xslx(filepath=filepath, encoding=encoding, output_cols=output_cols)

        # Clean the author field:
        if "paper_author" in insights_df.columns:
            insights_df["paper_author"] = insights_df["paper_author"].replace(
                ["", "No author found", "NA", "null", pd.NA, np.nan],
                None
            )

        # Align with full text:
        # Get unique valid paper ids from the deduped insights
        valid_ids = set(insights_df["paper_id"])
        # Sanity check to make sure some papers came in
        if len(valid_ids) == 0:
            raise ValueError(f"No records found in {filepath}. Please ensure you have kept at least one record per paper and that the paper_id column is intact.")
        
        # Make temp copy of full text to operate over
        temp_full_text = self.corpus_state.full_text.copy()

        # Filter full text
        temp_full_text = (
           temp_full_text[
                temp_full_text["paper_id"].isin(valid_ids)
            ].copy()
        )

        # Filter insights - to make sure only unique ids
        insights_df = (
            insights_df[
                insights_df["paper_id"].isin(valid_ids)
            ].copy()
        )

        # reassign to corpus state
        self.corpus_state.full_text = temp_full_text
        self.corpus_state.insights = insights_df

        # Note we don't save as we are now going to chunk this in the same class
        # Return for inspection
        return self.corpus_state.insights

    @staticmethod
    def _make_unique_list(x: list) -> list:
        """
        Make duplicate values unique by appending occurrence numbers.

        Processes a list of values and ensures that each element in the returned
        list is unique. Values that appear only once are left unchanged. Values
        that appear multiple times are suffixed with an occurrence counter:

            item
            → item_1
            → item_2
            → item_3
            ...

        Parameters
        ----------
        x : list
            Input list of values.

        Returns
        -------
        list
            List with duplicate values disambiguated using numeric suffixes.

        Examples
        --------
        >>> _make_unique_list(["Smith 2020", "Jones 2021", "Smith 2020"])
        ["Smith 2020_1", "Jones 2021", "Smith 2020_2"]

        >>> _make_unique_list(["A", "A", "A"])
        ["A_1", "A_2", "A_3"]

        >>> _make_unique_list(["A", "B", "C"])
        ["A", "B", "C"]

        Notes
        -----
        The function preserves input order.

        This helper is primarily used when generating citation identifiers from
        author-year combinations. Multiple documents may share the same
        author-year reference (for example, several papers by the same author in
        the same year), so suffixes are added to create unique identifiers while
        preserving the original citation label.
        """
        # First iterate a dict to count items
        totals = {}

        for item in x:
            totals[item] = totals.get(item, 0) + 1

        # Then iterate again to create the unique list, using the totals to append integers to duplicates
        seen = {}
        result = []

        for item in x:
            if totals[item] == 1:
                result.append(item)
            else:
                seen[item] = seen.get(item, 0) + 1
                result.append(f"{item}_{seen[item]}")

        return result   

    def gen_unique_citations(self) -> None:
        """
        Generate unique paper identifiers and formatted in-text citations.

        Creates citation-based `paper_id` values from paper author and date
        metadata, updates the corpus state to use those identifiers, and uses an
        LLM to generate formatted in-text citation strings.

        The method performs the following steps:

        1. Rename the existing `paper_id` column in `insights` to
        `filename_stub`.
        2. Generate provisional citation keys from first author surname and
        publication year.
        3. Disambiguate duplicate citation keys by appending numeric suffixes.
        4. Update `full_text.paper_id` to use the new citation-based identifiers.
        5. Send citation strings to the LLM in batches for formatting.
        6. Disambiguate duplicate formatted citations.
        7. Add the resulting `in_text_citation` column to `corpus_state.insights`.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If any `paper_author` or `paper_date` values are missing or empty.

        Side Effects
        ------------
        Mutates `self.corpus_state`:

        - renames `insights.paper_id` to `insights.filename_stub`
        - creates a new citation-based `insights.paper_id`
        - updates `full_text.paper_id` to match the new citation-based IDs
        - adds `insights.in_text_citation`

        Notes
        -----
        Citation-based IDs are generated from the first author name and
        publication year. Author strings are cleaned by replacing semicolons
        with commas, taking the first comma-separated author, removing spaces,
        and appending the publication year.

        Duplicate citation-based IDs are made unique with `_make_unique_list()`,
        producing suffixes such as `_1`, `_2`, and `_3`.

        Formatted in-text citations are generated by the LLM in batches of 10
        using `Prompts().gen_in_text_citation()` and
        `utils.call_chat_completion()`.

        If an LLM batch returns an error, the method prints a warning and
        continues. Citations from failed batches may be missing from the final
        state.

        The method prints the first few generated `paper_id` and
        `in_text_citation` values for inspection and does not save the updated
        CorpusState to disk.
        """
        # Make a copy of working insights so that we only mutate at the end and don't change things in stages if code crashes
        working_insights = self.corpus_state.insights.copy()

        # First we change the paper_id field to filename_stub to reflect that it is no longer the paper id but the original filename stub that we have for each paper, which is what we will use to link the unique citations to the papers. We then create a new paper_id field which is the unique citation. We generate the unique citation by combining the author and date fields, and then we check for duplicates and append an integer if there are duplicates. Finally we save this as a new dataframe in the state called unique_citations.
        working_insights = working_insights.rename(columns={"paper_id": "filename_stub"})
        # Now we create the paper_id field by combining the author and dat fields and then checking for uniqueness
        # First check whether there are any empty author or date values, if so throw an error:
        mask = working_insights["paper_author"].isna() | working_insights["paper_date"].isna() | (working_insights["paper_author"] == "") | (working_insights["paper_date"] == "")
        
        if mask.any():
            raise ValueError(
                "Missing author or date values found in insights. Please ensure all papers have author and date metadata before generating unique citations.\n\n" \
                "To do this update the file at data/potential_duplicates/ and re run the update_state function to update the insights with the correct metadata, then run this function again.\n\n"
                )

        # if mask succeeds convert paper_date to numeric to keep the data model clean
        working_insights["paper_date"] = pd.to_numeric(working_insights["paper_date"], errors="coerce")
    
        # Now generate the unique citation by combining the author and date fields
        # Clean the citations
        paper_date = (
            working_insights["paper_date"]
            .fillna(-1)
            .astype(int)
            .astype(str)
            .replace("-1", "nd")
        )
        
        paper_author = (
            working_insights["paper_author"]
            .str.replace(";", ",", regex=False)
            .str.split(",")
            .str[0]
            .str.replace(" ", "", regex=False)
            .str.strip()
        )

        working_insights["paper_id"] = (
            paper_author + paper_date
        )

        working_insights["paper_id"] = self._make_unique_list(
            working_insights["paper_id"].tolist()
        )

        # Before we get unique citations we also update the paper id in corpus_state.full_text
        working_full_text = self.corpus_state.full_text.copy() #Make a copy for safety so that i can mutate state at the end and avoid any distortions from code crashing
        working_full_text = (
            working_full_text
            .rename(columns={"paper_id": "filename_stub"})
            .merge(working_insights[["filename_stub", "paper_id"]], 
                   how="left", on="filename_stub")
            .drop(columns=["filename_stub"])
        )

        # Now we get the unique citations back from the LLM
        # First create a dictionary of paper_id keys and citation values
        citations = working_insights["paper_author"] + " " + paper_date
        citations_dict = {}
        for paper_id, citation in zip(working_insights["paper_id"], citations):
            citations_dict[paper_id] = citation

        out_clean_citations = []
        step = 10
        total_batches = math.ceil(len(citations_dict) / step)
        citation_keys = list(citations_dict.keys())
        count = 0

        for i in range(0, len(citations_dict), step):
            count += 1
            print(f"Processing batch {count} of {total_batches}...")
            batch = {
                key: citations_dict[key] for key in citation_keys[i:i+step]
            }
            citations_json = json.dumps(batch, indent=2, ensure_ascii=False)
            user_prompt = citations_json
            sys_prompt = Prompts().gen_in_text_citation()
            json_schema = {
                "name": "formatted_citations",
                "schema": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "string"
                    }
                }
            }

            fall_back = {}

            response, error = utils.call_chat_completion(
                llm_client=self.llm_client,
                ai_model=self.ai_model,
                sys_prompt=sys_prompt,
                user_prompt=user_prompt,
                return_json=True,
                json_schema=json_schema,
                fall_back=fall_back, 
                return_with_error = True
            )

            if error:
                print(f"Error processing batch {count}: {error}. Some citations will be missing.")

            clean_citations = pd.DataFrame(
                response.items(), 
                columns=["paper_id", "in_text_citation"]
            )

            out_clean_citations.append(clean_citations)


        # Now we concat all the results
        out_clean_citations_df = pd.concat(out_clean_citations, ignore_index=True)
        # Now we get unique values for the citations
        out_clean_citations_df["unique_in_text_citation"] = self._make_unique_list(out_clean_citations_df["in_text_citation"].tolist())

        # Now merge on paper_id to make sure everything matches
        working_insights = (
            working_insights
            .merge(
                out_clean_citations_df[["paper_id", "unique_in_text_citation"]],
                how="left",
                on="paper_id"
            )
            .rename(columns={"unique_in_text_citation": "in_text_citation"})
        )

        # Mutate the state at the end of this
        self.corpus_state.insights = working_insights
        self.corpus_state.full_text = working_full_text
        print(self.corpus_state.insights[["paper_id", "in_text_citation"]].head())

    @staticmethod
    def _drop_duplicate_chunks(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove exact duplicate chunks within individual documents.

        Drops rows that contain identical combinations of `paper_id` and
        `chunk_text`, ensuring that repeated text segments within the same
        document are processed only once.

        Parameters
        ----------
        df : pd.DataFrame
            Chunk-level DataFrame containing:

            - `paper_id`
            - `chunk_text`

        Returns
        -------
        pd.DataFrame
            DataFrame with duplicate `(paper_id, chunk_text)` rows removed and
            the index reset.

        Raises
        ------
        ValueError
            If either `paper_id` or `chunk_text` is missing from the input
            DataFrame.

        Notes
        -----
        Deduplication is performed within documents rather than across the
        corpus. Identical chunk text appearing in different papers is retained.

        Only exact text matches are removed. No fuzzy matching, similarity
        comparison, or semantic deduplication is performed.

        This step helps reduce redundant LLM processing that can arise from PDF
        parsing artifacts, duplicated page elements, repeated headers/footers,
        or document layout issues.
        """
        if not set(["paper_id", "chunk_text"]).issubset(df.columns):
            raise ValueError("Input DataFrame must contain 'paper_id' and 'chunk_text' columns.")
        
        df = df.drop_duplicates(subset=["paper_id", "chunk_text"])
        return df.reset_index(drop=True)
    
    @staticmethod
    def _drop_boilerplate(df) -> pd.DataFrame:
        """
        Remove high-frequency repeated chunks within individual documents.

        Identifies chunk text that appears repeatedly within the same paper and
        removes occurrences whose frequency exceeds a fixed threshold. This is
        intended to reduce the impact of recurring document artifacts such as
        headers, footers, navigation text, and page labels.

        Parameters
        ----------
        df : pd.DataFrame
            Chunk-level DataFrame containing:

            - `paper_id`
            - `chunk_text`

        Returns
        -------
        pd.DataFrame
            DataFrame containing only chunk occurrences whose within-paper
            frequency is less than or equal to 10. The index is reset.

        Notes
        -----
        Chunk frequency is calculated separately for each `(paper_id,
        chunk_text)` combination.

        The current implementation removes chunks that appear more than 10 times
        within a single paper.

        This method does not attempt to identify boilerplate semantically.
        Filtering is based solely on repeated exact text matches within a
        document.

        Entire papers are never removed; only high-frequency repeated chunk
        instances are filtered.

        The threshold is intentionally conservative to reduce the risk of
        removing legitimate repeated content.
        """
        counts = df.groupby(["paper_id", "chunk_text"])["chunk_text"].transform("size")
        boilerplate = df[counts <= 10].reset_index(drop=True)
        return boilerplate

    def chunk_papers(
        self,
        chunk_size: int = 3500,
        chunk_overlap: int = 350
    ) -> None:
        """
        Chunk ingested full-text documents and update corpus state.

        Normalizes each document in `corpus_state.full_text`, splits the text
        into overlapping character-based chunks, removes repeated chunk
        artifacts, and updates the CorpusState with the resulting chunk-level
        representation.

        The method performs the following steps:

        1. Normalize full text to reduce PDF line-break and whitespace artifacts.
        2. Split each document into greedy overlapping chunks.
        3. Explode document-level chunk lists into `corpus_state.chunks`.
        4. Assign sequential `chunk_id` values.
        5. Remove exact duplicate chunks.
        6. Remove high-frequency repeated chunks within papers.
        7. Rebuild `corpus_state.insights` so each paper-level record is
        associated with its chunk IDs.
        8. Save the updated CorpusState to the `07_full_text_and_chunks` state
        directory.

        Parameters
        ----------
        chunk_size : int, default=3500
            Maximum chunk size, measured using character length.

        chunk_overlap : int, default=350
            Number of characters of overlap between consecutive chunks.

        length_function : callable, default=len
            Currently unused. Retained for compatibility with earlier splitter
            interfaces.

        separators : list[str], optional
            Currently unused. Retained for compatibility with earlier splitter
            interfaces.

        is_separator_regex : bool, default=False
            Currently unused. Retained for compatibility with earlier splitter
            interfaces.

        Returns
        -------
        None

        Side Effects
        ------------
        Mutates `self.corpus_state`:

        - populates `corpus_state.chunks`
        - temporarily adds and then removes `full_text.chunk_text`
        - rebuilds `corpus_state.insights` by merging chunk IDs onto paper-level
        metadata
        - saves the updated CorpusState to disk

        Notes
        -----
        The greedy chunker prioritizes splitting at paragraph breaks, then
        sentence-ending periods, then spaces. If no acceptable split point is
        found, it performs a hard character cut.

        Chunking is character-based rather than token-based.

        The splitter avoids very early split points by ignoring candidate split
        positions that occur before 30 percent of the current window.

        Chunk cleaning removes exact duplicate chunks and chunks that appear
        more than 10 times within the same paper. It does not currently remove
        table-like chunks, although a table-cleaning call is present but
        commented out.

        `full_text` remains the source document text. The generated chunks are
        stored separately in `corpus_state.chunks`.
        """

        def _normalize_text(text: str) -> str:

            if not isinstance(text, str):
                return text

            # remove soft hyphens
            text = text.replace('\xad', '')

            # fix broken words
            text = re.sub(r'(?<=\w)\n(?=\w)', '', text)

            # preserve paragraph breaks
            text = re.sub(r'\n{2,}', '<<PARA>>', text)

            # flatten remaining newlines
            text = re.sub(r'\n', ' ', text)

            # restore paragraphs
            text = text.replace('<<PARA>>', '\n\n')
            
            # normalize whitespace
            text = re.sub(r'\s+', ' ', text)

            return text.strip()

        
        def greedy_chunk_text(
            text: str,
            chunk_size: int = 3500,
            chunk_overlap: int = 350
        ) -> list[str]:
            """
            Greedy text chunker with prioritized backward splitting.

            Strategy:
            - Take up to `chunk_size`
            - Walk backward to nearest:
                1. "\n\n"
                2. "."
                3. " "
            - If none found, hard cut

            Ensures:
            - No chunk exceeds chunk_size
            - Chunks are as large as possible
            - Overlap is preserved

            Parameters
            ----------
            text : str
            chunk_size : int
            chunk_overlap : int

            Returns
            -------
            List[str]
            """

            chunks = []
            start = 0
            text_length = len(text)

            while start < text_length:
                # Tentative end
                end = min(start + chunk_size, text_length)

                if end == text_length:
                    chunks.append(text[start:].strip())
                    break

                window = text[start:end]

                # Priority 1: double newline
                split_idx = window.rfind("\n\n")

                # Priority 2: sentence end
                if split_idx == -1:
                    split_idx = window.rfind(".")

                # Priority 3: space
                if split_idx == -1:
                    split_idx = window.rfind(" ")

                # If nothing found, force cut
                if split_idx == -1 or split_idx < int(0.3 * len(window)):
                    split_idx = len(window)

                # Adjust end
                end = start + split_idx

                chunk = text[start:end].strip()
                if chunk:
                    chunks.append(chunk)

                # Move start with overlap
                start = max(end - chunk_overlap, 0)

            return chunks


        full_text_list = (
            self.corpus_state.full_text["full_text"]
            .fillna("")
            .apply(_normalize_text)
            .to_list()
        )

        chunks_list = [
            greedy_chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            for text in full_text_list
            ]
        # Create the chunks corpus_state from the full_text corpus_state
        self.corpus_state.full_text["chunk_text"] = chunks_list
        self.corpus_state.chunks = self.corpus_state.full_text[["paper_id", "chunk_text"]].explode("chunk_text").reset_index(drop=True).copy()
        self.corpus_state.chunks["chunk_id"] = [f"chunk_{i+1}" for i in range(self.corpus_state.chunks.shape[0])]

        # Chunks from full_text as its now joined by paper and question id
        self.corpus_state.full_text.drop(columns=["chunk_text"], inplace=True)

        # Now clean up the chunks by dropping duplicates, boilerplayte and tables
        temp_chunks = self.corpus_state.chunks.copy()
        # Remove NA to avoid crashing on split in the functions
        temp_chunks = temp_chunks[temp_chunks["chunk_text"].notna()]
        # Now call the cleaning functions
        temp_chunks = self._drop_duplicate_chunks(temp_chunks)
        temp_chunks = self._drop_boilerplate(temp_chunks)
        # temp_chunks = self._drop_extreme_table_chunks(temp_chunks)

        self.corpus_state.chunks = temp_chunks.reset_index(drop=True).copy()

        # Get chunk_id into insights
        temp_insights = self.corpus_state.insights.copy()
        self.corpus_state.insights = (
            self.corpus_state.chunks
            .drop(columns=["chunk_text"])
            .merge(temp_insights, how="left", on="paper_id")
        )

        # Save the updated corpus_state
        self.corpus_state.save(os.path.join(config.STATE_SAVE_LOCATION, "07_full_text_and_chunks"))

    def chunk_sanity_check(self):
        """
        Summarize chunking statistics for each paper.

        Computes basic descriptive statistics over the current
        `corpus_state.chunks` table to help validate the chunking process and
        identify unusually large, small, or fragmented documents.

        Returns
        -------
        pd.DataFrame
            One row per paper with the following columns:

            - `paper_id`
            - `num_chunks`
            - `avg_chunk_length`
            - `min_chunk_length`
            - `max_chunk_length`
            - `total_words`

            Results are sorted by `num_chunks` in descending order.

        Notes
        -----
        Chunk length is estimated as the number of whitespace-separated words in
        `chunk_text`.

        Average, minimum, and maximum chunk lengths are reported in words rather
        than characters.

        This method performs a diagnostic check only and does not modify the
        CorpusState.

        The output is intended for inspection of chunking behavior, helping to
        identify documents that may require investigation due to unexpectedly
        high chunk counts, unusually short chunks, or other extraction issues.
        """
        # Get the first few records of the chunks corpus_state to inspect

        temp_chunks = self.corpus_state.chunks.copy()

        temp_chunks["chunk_len"] = temp_chunks["chunk_text"].str.split().str.len()

        summary = (
            temp_chunks
            .groupby("paper_id")
            .agg(
                num_chunks=("chunk_id", "nunique"),
                avg_chunk_length=("chunk_len", "mean"),
                min_chunk_length=("chunk_len", "min"),
                max_chunk_length=("chunk_len", "max"),
                total_words=("chunk_len", "sum")
            )
            .reset_index()
        )

        summary.sort_values("num_chunks", ascending=False, inplace=True)

        return summary

class Insights:
    """
    Insight extraction stage of the ReadingMachine pipeline.

    The Insights class converts a prepared corpus representation into
    research-question-relevant insight records. It operates over the chunked
    corpus produced by the Ingestor and generates the insight-level
    representation that forms the foundation of all downstream analytical
    stages.

    The insight-generation workflow proceeds in two passes:

        chunk-level reading
            ↓
        meta-insight generation
            ↓
        complete insight corpus

    Chunk-level reading
    -------------------
    The first pass processes each chunk independently.

    Every chunk is evaluated against the complete set of research questions,
    and the language model extracts any claims, findings, arguments,
    observations, or evidence relevant to those questions.

    The resulting chunk-level insights:

    - retain links to `paper_id`
    - retain links to `chunk_id`
    - retain links to `question_id`
    - receive unique `insight_id` values

    Chunks that contain no relevant insights are still recorded to preserve
    traceability and support resumable execution.

    Meta-insight generation
    -----------------------
    The second pass identifies insights that emerge only when larger portions
    of a paper are considered together.

    For each paper and research question, the model receives:

    - document-level text
    - previously generated chunk insights
    - the focal research question
    - contextual information about the remaining questions

    This allows the system to identify arguments, findings, and conceptual
    relationships that span multiple chunks and may not be visible during
    isolated chunk-level reading.

    Generated meta-insights are appended to the existing chunk-level
    insights and receive `meta_insight_*` identifiers.

    Resume and recovery
    -------------------
    Both chunk-level insight generation and meta-insight generation support
    checkpoint-based recovery.

    Intermediate outputs are written to pickle files throughout execution,
    allowing long-running extraction jobs to resume after:

    - API interruptions
    - process termination
    - user cancellation
    - system failures

    Recovery is performed using chunk-level and content-level identifiers so
    that already processed units are not re-read unnecessarily.

    Pipeline role
    -------------
    The Insights stage transforms:

        chunks
            ↓
        chunk insights
            ↓
        meta insights
            ↓
        insight corpus

    The resulting insight corpus becomes the primary analytical
    representation used by subsequent ReadingMachine stages, including:

    - embedding generation
    - clustering
    - theme construction
    - orphan detection
    - thematic synthesis

    Design principles
    -----------------
    The class reflects several core ReadingMachine principles:

    Question-guided reading
        All extraction is performed relative to explicit research questions
        rather than generic summarization.

    Traceability
        Every generated insight remains linked to its originating paper,
        chunk, and research question.

    Coverage preservation
        Meta-insights provide a mechanism for identifying information that
        may span multiple chunks and would otherwise be missed by strictly
        local reading.

    Recoverability
        Long-running extraction workflows support checkpointing and resume
        operations.

    Inspectable intermediate representations
        Chunk-level insights and meta-insights remain explicit analytical
        objects rather than being immediately collapsed into higher-level
        summaries.

    Attributes
    ----------
    corpus_state : CorpusState
        Working corpus state containing chunks, metadata, and generated
        insights.

    llm_client : Any
        Language model client used for insight extraction.

    ai_model : str
        Model identifier used for insight-generation calls.

    paper_context : str
        Review-specific context supplied to insight-generation prompts.

    pickle_path : str
        Directory used to store checkpoint files for recovery.

    chunk_insights_pickle_file : str
        Checkpoint file for chunk-level insight generation.

    meta_insights_pickle_file : str
        Checkpoint file for meta-insight generation.

    max_token_length : int
        Maximum document length threshold used when preparing document-level
        inputs for meta-insight generation.
    """
    def __init__(
        self,
        corpus_state: "CorpusState",
        llm_client: Any,
        ai_model: str,
        paper_context: str, 
        max_token_length: int = 100000,
        pickle_path: str = config.PICKLE_SAVE_LOCATION, 
        chunk_insights_pickle_file: str="chunk_insights.pkl", 
        meta_insights_pickle_file: str="meta_insights.pkl"
    ) -> None:
        """
        Initialize the insight-extraction stage.

        Creates an Insights object for generating chunk-level and document-level
        meta-insights from a prepared CorpusState. The initializer validates that
        the corpus state contains the metadata required for insight extraction,
        stores LLM configuration, creates the pickle checkpoint directory, and
        canonicalizes question text in the working state.

        Parameters
        ----------
        corpus_state : CorpusState
            Prepared CorpusState containing questions, paper metadata, citations,
            full text, and chunk records. The state is deep-copied before use.

        llm_client : Any
            LLM client used for insight-extraction calls.

        ai_model : str
            Model identifier used for LLM calls.

        paper_context : str
            Corpus- or project-level context supplied to the insight-extraction
            prompts.

        max_token_length : int, default=100000
            Maximum text length used when preparing document-level meta-insight
            prompts.

        pickle_path : str, default=config.PICKLE_SAVE_LOCATION
            Directory used to store intermediate insight-extraction checkpoints.

        chunk_insights_pickle_file : str, default="chunk_insights.pkl"
            Filename used for chunk-level insight checkpoint data.

        meta_insights_pickle_file : str, default="meta_insights.pkl"
            Filename used for document-level meta-insight checkpoint data.

        Notes
        -----
        The supplied CorpusState is validated through `utils.validate_format()`
        and then deep-copied, so extraction mutates the Insights instance's
        internal state rather than the caller's original object.

        The required state columns include question metadata, paper metadata,
        DOI, and `in_text_citation`, which is used to preserve citation
        provenance in extracted insights.

        This class corresponds to the ReadingMachine stage that converts the
        prepared corpus representation into insight-level units for downstream
        embedding, clustering, theme construction, and synthesis.
        """
        self.llm_client: Any = llm_client
        self.ai_model: str = ai_model
        self.pickle_path: str = pickle_path
        os.makedirs(self.pickle_path, exist_ok=True)
        
        self.paper_context: str = paper_context
        self.chunk_insights_pickle_file = chunk_insights_pickle_file
        self.meta_insights_pickle_file = meta_insights_pickle_file
        self.max_token_length: int = max_token_length


        # Ensure corpus_state has all required columns before processing
        self.corpus_state = deepcopy(
            utils.validate_format(
            corpus_state=corpus_state,
            questions=None,
            injected_value=None,
            state_required_cols=[
                "question_id", "question_text", "search_string_id", "search_string",
                "paper_id", "paper_title", "paper_author", "paper_date", "doi", "in_text_citation"
            ],
            injected_required_cols=None
            )
        )

        self.corpus_state.enforce_canonical_question_text()

    def _generate_chunk_insights(self, insights: List = None) -> pd.DataFrame:
        """
        Generate chunk-level insights for all unprocessed chunks.

        Processes each chunk in `corpus_state.chunks` with the configured LLM and
        extracts research-question-relevant insights. Each chunk is evaluated
        against the full set of research questions, and extracted insights are
        linked back to their source `paper_id` and `chunk_id`.

        Parameters
        ----------
        insights : list[pd.DataFrame], optional
            Previously generated chunk-insight outputs, usually recovered from a
            checkpoint pickle. If provided, chunks whose `chunk_id` values already
            appear in the recovered outputs are skipped.

        Returns
        -------
        pd.DataFrame
            Chunk-level insight table containing extracted insights, source
            identifiers, paper metadata, and generated `insight_id` values.

        Notes
        -----
        The method supports both fresh and resumed execution:

        - `insights is None`: start a new run
        - `insights == []`: treat as a valid empty recovered state
        - `insights` contains DataFrames: resume and skip processed chunks

        If the model returns no insights for a chunk, the chunk is still recorded
        with missing `question_id` and `insight` values. This preserves evidence
        that the chunk was processed.

        Intermediate results are checkpointed every 10 processed chunks to:

            {self.pickle_path}/{self.chunk_insights_pickle_file}

        The returned DataFrame is not automatically assigned to
        `self.corpus_state.insights`. State mutation is handled by the public
        orchestration method `get_chunk_insights()`.

        Chunk-level insight IDs are assigned only to rows with non-missing
        insight text and take the form:

            chunk_insight_1
            chunk_insight_2
            ...

        This method implements the chunk-level reading pass in ReadingMachine:
        it converts bounded text chunks into atomic, question-relevant insight
        records while preserving links back to the source paper and chunk.
        """

        if insights is None:
            insights = []
            processed_chunks = []
        else:
            if len(insights) > 0:
                insights_df = pd.concat(insights).reset_index(drop=True)
                processed_chunks = insights_df["chunk_id"].unique().tolist()
            else:
                # Explicitly handle injected empty list
                processed_chunks = []

        remaining_chunks = self.corpus_state.chunks[
            ~self.corpus_state.chunks["chunk_id"].isin(processed_chunks)
        ]

        # Generate a list of all the research questions with ids in the form <rq_id>: <rq_text> for the llm to consdier against each chunk
        rqs_ids = [f"{row['question_id']}: {row['question_text']}" for _, row in self.corpus_state.questions.iterrows()]
        rqs_ids_str = "\n".join(rqs_ids)

        # Create a temp copy of the remaining chunks to iterate over, this is just to be safe and not mutate the original dataframe as we go through the loop
        temp_state_insights = remaining_chunks.copy()

        # Iterate over each chunk
        for idx, (df_index, row) in enumerate(temp_state_insights.iterrows()):
            print(f"Processing chunk {len(processed_chunks) + idx + 1} of {temp_state_insights.shape[0] + len(processed_chunks)}...")

            # Extract fields from row
            paper_id: str = row["paper_id"]
            chunk_text = row["chunk_text"] if pd.notna(row["chunk_text"]) else ""
            chunk_id: str = str(row["chunk_id"])

            # Build prompts
            sys_prompt: str = Prompts().gen_chunk_insights(paper_context=self.paper_context)
            user_prompt: str = (
                f"RESEARCH QUESTIONS:\n{rqs_ids_str}\n\n"
                f"TEXT CHUNK:\n{chunk_text}\n"
            )

            fall_back = {
                "results": {}
            }

            response_dict = utils.call_chat_completion(ai_model = self.ai_model,
                                 llm_client = self.llm_client,
                                 sys_prompt = sys_prompt,
                                 user_prompt = user_prompt,
                                 return_json = True, 
                                 fall_back=fall_back)
            
            # Turn the response into a df with columns question_id and insight, where each row is a different insight, and the insights are the insights for that question id that were extracted from the chunk. This will make it easier to merge into the corpus_state later.
            results = response_dict.get("results", {})
            
            response_df = (
                pd.DataFrame(results.items(), columns=["question_id", "insight"])
                .explode("insight") #explode on insights to make tidy data from the lists
                .reset_index(drop=True)
                           )
            # If the response is empty (no insights were found) add a row with the chunk id and paper id
            if response_df.empty:
                response_df = pd.DataFrame([{
                    "question_id": pd.NA,
                    "insight": pd.NA, 
                    "chunk_id": chunk_id,
                    "paper_id": paper_id
                }])
            # Otherwise just add the chunk id and paper id to the response df
            else: 
                # Add paper_id and chunk_id to the response_df 
                response_df["chunk_id"] = chunk_id
                response_df["paper_id"] = paper_id

            # Append to insights list
            insights.append(response_df)
            # Batch writes every 10 chunks
            if (idx + 1) % 10 == 0:
                utils.safe_pickle(insights, os.path.join(self.pickle_path, self.chunk_insights_pickle_file))

        # Convert insights list to DataFrame
        print("Converting insights to DataFrame and merging into corpus_state...")
        insights_complete_df: pd.DataFrame = pd.concat(insights).reset_index(drop=True) if insights else pd.DataFrame(columns=["question_id", "insight", "chunk_id", "paper_id"])
        
        # Merge into global insights table - first drop existing insight columns if present (these can be created by previous runs of recover_chunk_insights_generation)
        base_insights = (
            self.corpus_state.insights
            .drop(columns=["insight"], errors="ignore") # drop the metadata columns if they exist as we will merge them back in from the corpus_state later, but ignore if they don't exist as this function can be run multiple times and they will only be there after the first run
        )
        
        # Now we merge this chunk insights with all the insights data and metadata. 
        # Notably we drop question_id from the corpus_state.insights as previously if papers wwere not associated with a question when importing them, the question_id was NA
        # Now we have all the chunks and thier insights associated with a question_id thus this becomes the primary df
        working_insights_df = (
            insights_complete_df
            .merge(
                base_insights.drop(columns=["question_id"], errors="ignore"), # So that we don't duplicate question_id columns
                how="left",
                on=["paper_id", "chunk_id"]
                ))
        
        # Add insight_ids
        mask = working_insights_df["insight"].notna()

        # Create numeric ids only where valid
        working_insights_df["insight_id"] = pd.NA
        working_insights_df.loc[mask, "insight_id"] = range(1, mask.sum() + 1)

        # Explicitly transform only valid rows to final string form
        working_insights_df.loc[mask, "insight_id"] = (
            "chunk_insight_" +
            working_insights_df.loc[mask, "insight_id"].astype(int).astype(str)
        )

        return working_insights_df
    
    
    def _recover_chunk_insights_generation(self):
        """
        Resume chunk-level insight extraction from a checkpoint.

        Loads previously generated chunk-insight DataFrames from
        `self.chunk_insights_pickle_file` and passes them to
        `_generate_chunk_insights()`, which skips chunks that have already been
        processed and continues extraction from the remaining chunks.

        Returns
        -------
        pd.DataFrame
            Chunk-level insight DataFrame generated by resuming from the
            recovered checkpoint.

        Side Effects
        ------------
        Reads from:

            {self.pickle_path}/{self.chunk_insights_pickle_file}


        Notes
        -----
        This method assumes the checkpoint file exists and contains a list of
        partial chunk-insight DataFrames. Missing, corrupted, or incompatible
        pickle files will raise errors from `open()`, `pickle.load()`, or the
        downstream recovery logic.

        Recovery is based on previously processed `chunk_id` values.

        This method does not mutate `self.corpus_state`. It loads the checkpoint,
        continues processing remaining chunks, and returns the completed
        insight table to the caller.
        """
        print("Opening pickle file to recover chunk insights generation...")
        with open(os.path.join(self.pickle_path, self.chunk_insights_pickle_file), "rb") as f:
            recover_chunk_insights = pickle.load(f)
        
        chunk_insights = self._generate_chunk_insights(insights=recover_chunk_insights)
        
        return(chunk_insights)


    def get_chunk_insights(self) -> pd.DataFrame:
        """
        Generate or recover chunk-level insights.

        This method is the public entry point for the chunk-level insight
        extraction stage. It checks whether a pickle file containing
        previously generated insights exists and prompts the user to either:

        - recover/resume the previous run, or
        - regenerate insights from scratch.

        The underlying extraction logic is implemented in
        `_generate_chunk_insights`.

        Returns
        -------
        pd.DataFrame
            Chunk-level insight table assigned to
            `self.corpus_state.insights`.

        Side Effects
        ------------
        Mutates `self.corpus_state.insights` with the generated or recovered
        chunk-level insights.

        Notes
        -----
        Resume functionality allows long-running LLM extraction processes
        to continue from partial results stored in a pickle file.
        """

        if os.path.exists(os.path.join(self.pickle_path, self.chunk_insights_pickle_file)):
            recover = None
            while recover not in ['r', 'n']:
                recover = input(
                    f"Insights already exist on file at {os.path.join(self.pickle_path, self.chunk_insights_pickle_file)}. "
                    "Do you wish to recover/restore or regenerate insights?\n"
                    "Hit 'r' to recover/restore from file, or 'n' to generate new insights (this will overwrite existing file):\n"
                    ).lower()
            if recover == 'r':
                chunk_insights = self._recover_chunk_insights_generation()
            else:
                print("Overwriting existing chunk insights pickle file...")
                chunk_insights = self._generate_chunk_insights()
        else:
            chunk_insights = self._generate_chunk_insights()

        # Update the state with the chunk insights that we have generated or recovered
        # Note we don't save here as we can recover from the pickle and we will save after getting the meta insights.
        self.corpus_state.insights = chunk_insights
        return(self.corpus_state.insights)


    def _prepare_meta_insights_df(self) -> pd.DataFrame:
        """
        Prepare the document-level inputs used for meta-insight extraction.

        Builds a tidy DataFrame containing the full-document content, associated
        chunk-level insights, and research-question identifiers needed to
        generate document-level meta-insights. Each row represents a
        paper/question/content-chunk combination.

        Returns
        -------
        pd.DataFrame
            DataFrame containing:

            - `paper_id`
            - `question_id`
            - `paper_content`
            - `paper_chunk_insights`
            - `content_chunk_id`

        Raises
        ------
        ValueError
            If chunk-level insights have not yet been generated.

        Side Effects
        ------------
        Mutates `self.corpus_state.chunks` by adding meta-content chunks whose
        `chunk_id` values begin with `meta_chunk_`.

        Notes
        -----
        Each paper is checked against `self.max_token_length`. Papers exceeding
        the limit are split with `string_breaker()` before meta-insight
        generation; shorter papers are passed as a single content chunk.

        For each paper and research question, previously generated chunk-level
        insights are collected and joined into `paper_chunk_insights`. These are
        provided to the LLM so it can identify higher-level insights that span
        multiple chunks while avoiding simple repetition of existing chunk
        insights.

        Before adding new meta chunks to `corpus_state.chunks`, any existing
        chunks whose IDs start with `meta_chunk_` are removed to avoid duplicate
        meta chunks across repeated runs.
        """
        
        # Must run chunk insights first
        if "insight" not in self.corpus_state.insights.columns:
            raise ValueError(
                "Meta-insights cannot be created prior to generating chunk insights. "
                "Please run .get_chunk_insights before .get_meta_insights."
            )
        
        rqs = self.corpus_state.questions
        # Create the final list of paper that we will populate as we develop all the data for checking for meta insights for each paper
        list_of_papers = []
        # Get the full text and paper id
        for _, row in self.corpus_state.full_text.iterrows():
            paper_id = row["paper_id"]
            paper_content = row["full_text"]
            # Check that the whole paper fits in the model context window, if not break into chunks and process separately 
            # (this is a bit of a hack but it allows us to at least get some meta insights from papers that exceed the context window, 
            # which is likely to be the case for many academic papers with the full text included)
            token_count = self.estimate_tokens(paper_content, self.ai_model)
            if token_count > self.max_token_length:
                paper_content_list = self.string_breaker(paper_content, max_token_length=self.max_token_length)
            else:
                paper_content_list = [paper_content]

            # Get the chunk insights for the paper id and the question_id
            for rqid in rqs["question_id"].to_list():
                paper_question_chunk_insights = self.corpus_state.insights[
                    (self.corpus_state.insights["paper_id"] == paper_id) & (self.corpus_state.insights["question_id"] == rqid)
                ]["insight"].dropna().tolist()
                paper_question_chunk_insights = "\n".join(paper_question_chunk_insights) if paper_question_chunk_insights else ""
            
                # Now build the dataframe that we can call against the LLM and that we can use to determine resume points
                meta_insight_check_df = pd.DataFrame({
                    "paper_id": [paper_id] * len(paper_content_list),
                    "question_id": [rqid] * len(paper_content_list),
                    "paper_content": paper_content_list,
                    "paper_chunk_insights": [paper_question_chunk_insights] * len(paper_content_list),
                })

                list_of_papers.append(meta_insight_check_df)

        # Concat all the papers in the list
        meta_insight_check_df = pd.concat(list_of_papers).reset_index(drop=True)    

        # Create an id for full_content chunks both for resuming and traceability
        meta_insight_check_df["content_chunk_id"] = [f"meta_chunk_{i+1}" for i in range(meta_insight_check_df.shape[0])]

        # Update the corpus_state.chunks with the meta chunks and their ids
        temp_chunks = self.corpus_state.chunks.copy()
        # Check if the meta_chunks were already created in a previous run. If so remove them from the corpus_state object before we concat so that they don't get duplicated. 
        if temp_chunks["chunk_id"].str.startswith("meta_chunk_").any():
            temp_chunks = temp_chunks[~temp_chunks["chunk_id"].str.startswith("meta_chunk_")]
        
        self.corpus_state.chunks = pd.concat([
            meta_insight_check_df[["paper_id", "paper_content", "content_chunk_id"]].rename(columns={"paper_content": "chunk_text", "content_chunk_id": "chunk_id"}),
            temp_chunks
            ])

        return(meta_insight_check_df)
        

    def get_meta_insights(self) -> pd.DataFrame:
        """
        Generate, recover, and append document-level meta-insights.

        Creates meta-insights that may span multiple chunks within the same
        paper. Unlike chunk-level insights, which are extracted from bounded text
        chunks, meta-insights are generated from larger paper-level content
        segments together with the existing chunk insights for the same paper and
        research question.

        The method supports checkpoint recovery. If a meta-insight pickle exists,
        the user is prompted to either resume from the saved outputs or
        regenerate meta-insights from scratch.

        Returns
        -------
        pd.DataFrame
            Updated `corpus_state.insights` DataFrame containing both chunk-level
            insights and appended meta-insight rows.

        Raises
        ------
        ValueError
            If chunk-level insights have not been generated before calling this
            method.

        Side Effects
        ------------
        Mutates `self.corpus_state`:

        - may remove previously generated `meta_insight_` rows when regenerating
        - adds meta-content chunks to `corpus_state.chunks` through
        `_prepare_meta_insights_df()`
        - appends generated meta-insights to `corpus_state.insights`
        - saves the updated CorpusState to the `08_insights` state directory

        Reads from and writes to:

            {self.pickle_path}/{self.meta_insights_pickle_file}

        Notes
        -----
        Meta-insight generation operates over rows produced by
        `_prepare_meta_insights_df()`, where each row represents a
        paper/question/content-chunk combination.

        For each row, the LLM receives:

        - the specific research question under consideration
        - the relevant paper text segment
        - existing chunk-level insights for that paper and question
        - the remaining research questions as context

        Generated meta-insights are assigned IDs only when insight text is
        present. IDs take the form:

            meta_insight_1
            meta_insight_2
            ...

        Recovered runs skip content chunks whose `content_chunk_id` values are
        already present in the recovered pickle data.

        The final save occurs here because this method completes the Insights
        stage by combining chunk-level insights and meta-insights into the
        corpus-level insight representation used downstream.
        """
        # Must run chunk insights first
        if "insight" not in self.corpus_state.insights.columns:
            raise ValueError(
                "Meta-insights cannot be created prior to generating chunk insights. "
                "Please run .get_chunk_insights before .get_meta_insights."
            )
        
        # Check if there is a pickle file with previously generated meta_insights and ask the user if they want to recover or regenerate
        meta_insights = None
        if os.path.exists(os.path.join(self.pickle_path, self.meta_insights_pickle_file)):
            recover = None
            while recover not in ['r', 'n']:
                recover = input(
                    f"Meta-insights already exist on file at {os.path.join(self.pickle_path, self.meta_insights_pickle_file)}. "
                    "Do you wish to recover/restore or regenerate meta-insights?\n"
                    "Hit 'r' to recover/restore from file, or 'n' to generate new meta-insights (this will overwrite existing file):\n"
                    ).lower()
            if recover == 'r':
                print("Recovering meta-insights from file...")
                with open(os.path.join(self.pickle_path, self.meta_insights_pickle_file), "rb") as f:
                    meta_insights = pickle.load(f)
            else:
                # Clean up any meta insights that have been generated in self.corpus_state.insights to avoid duplication when we generate new ones.
                self.corpus_state.insights = self.corpus_state.insights[
                    ~self.corpus_state.insights["insight_id"].astype(str).str.startswith("meta_insight_")
                    ]
                meta_insights = None

        # If meta insights exists (its a list of dicts), so we concat them to get the ids of all the processed content chunks so we can drop them from the meta_insight_check_df and only process the remaining content chunks. 
        if meta_insights is not None:
            meta_insights_run_df = pd.concat(meta_insights).reset_index(drop=True)
            processed_meta_chunks = meta_insights_run_df["content_chunk_id"].unique().tolist()
        else: 
            processed_meta_chunks = [] #if it doesn't exist its empty so we will exclude nothing

        # Drop the processed content chunks
        meta_insight_check_df = self._prepare_meta_insights_df()
        meta_insight_check_df = meta_insight_check_df[~meta_insight_check_df["content_chunk_id"].isin(processed_meta_chunks)]

        # Now prepare to pass to the LLM for checking
        meta_insights_df_lst = [] if meta_insights is None else meta_insights
        for idx, row in meta_insight_check_df.iterrows():
            print(f"Processing meta-insights for content piece {idx + len(processed_meta_chunks) + 1} of {meta_insight_check_df.shape[0] + len(processed_meta_chunks)}...")

            # Extract fields from row
            paper_id: str = row["paper_id"]
            rq_id = row["question_id"]
            rq_text = self.corpus_state.questions[self.corpus_state.questions["question_id"] == rq_id]["question_text"].iloc[0] if rq_id in self.corpus_state.questions["question_id"].tolist() else ""
            rq = f"{rq_id}: {rq_text}"
            other_rqs = self.corpus_state.questions[self.corpus_state.questions["question_id"] != rq_id]
            other_rqs_str = "\n".join([f"{row['question_id']}: {row['question_text']}" for _, row in other_rqs.iterrows()])
            paper_content: str = row["paper_content"] if pd.notna(row["paper_content"]) else ""
            paper_chunk_insights: str = row["paper_chunk_insights"] if pd.notna(row["paper_chunk_insights"]) else ""
            content_chunk_id: str = row["content_chunk_id"]


            # Build prompts
            sys_prompt: str = Prompts().gen_meta_insights(paper_context=self.paper_context)
            user_prompt: str = (
                f"SPECIFIC RESEARCH QUESTION FOR CONSIDERATION:\n{rq}\n"
                f"PAPER TEXT:\n{paper_content}\n"
                f"EXISTING CHUNK INSIGHTS:\n{paper_chunk_insights}\n\n"
                f"OTHER RESEARCH QUESTIONS IN THE REVIEW (context only):\n{other_rqs_str}\n\n"
            )

            fall_back = {
                "results": {}
            }

            response_dict = utils.call_chat_completion(ai_model = self.ai_model,
                                llm_client = self.llm_client,
                                sys_prompt = sys_prompt,
                                user_prompt = user_prompt,
                                return_json = True, 
                                fall_back=fall_back)
            
            results = pd.DataFrame(response_dict["results"])

            # If the response is empty use an empty list otherwise get the list of results from results
            meta_list = results["meta_insight"].tolist() if not results.empty else []

            meta_insight_df = pd.DataFrame({
                "paper_id": [paper_id],
                "question_id": [rq_id],
                "content_chunk_id": [content_chunk_id],
                "meta_insight": [meta_list]
            })

            meta_insights_df_lst.append(meta_insight_df)

            # Save using safe_pickle to assure atomic save
            utils.safe_pickle(meta_insights_df_lst, os.path.join(self.pickle_path, self.meta_insights_pickle_file))


        # Now join up the list of dataframes to get a single value
        meta_insights_complete_df: pd.DataFrame = pd.concat(meta_insights_df_lst).reset_index(drop=True) if meta_insights_df_lst else pd.DataFrame(columns=["paper_id", "question_id", "content_chunk_id", "meta_insight"])
        # Explode on meta_insight (currently a list for each content chunk)
        meta_insights_complete_df = meta_insights_complete_df.explode("meta_insight").reset_index(drop=True) 

        # Now we want to append to self.corpus_state.insights (which already has the chunk level insights)
        # First rename meta_insight to insight to match .corpus_state.insights, i also rename content_chunk_id to chunk_id to match the corpus_state and ensure the meta_chunks have an ID of thier own. 
        meta_insights_complete_df.rename(columns={"meta_insight": "insight", 
                                                  "content_chunk_id": "chunk_id"}, 
                                                  inplace=True)
        #Then add insight ids to meta insights to distinguish them from chunk insights and to have a unique id for each insight which we can use for traceability 
        # First create mask so that we only iterate insight_id count for insights, not for NA insights
        mask = meta_insights_complete_df["insight"].notna()

        # Create numeric ids only where valid
        meta_insights_complete_df["insight_id"] = pd.NA
        meta_insights_complete_df.loc[mask, "insight_id"] = range(1, mask.sum() + 1)

        # Explicitly transform only valid rows to final string form
        meta_insights_complete_df.loc[mask, "insight_id"] = (
            "meta_insight_" +
            meta_insights_complete_df.loc[mask, "insight_id"].astype(int).astype(str)
        )

        # Then manipulate corpus_state.insights so that we can merge with meta_insights to get a complete df that we can later append to insights
        temp_insights = self.corpus_state.insights.copy()
        temp_insights = (
            temp_insights
            .drop(columns=["question_id", "question_text", "insight_id", "insight", "chunk_id"]) #Drop these to avoid conflicts
            .drop_duplicates(subset=["paper_id"])
            ) 
        meta_insights_complete = meta_insights_complete_df.merge(
            temp_insights, 
            on="paper_id", 
            how="left"
            )
        # Now we have the meta insights with all the metadata and we can append to the corpus_state insights
        self.corpus_state.insights = pd.concat([self.corpus_state.insights, meta_insights_complete], ignore_index=True)
        # Save to parquet and return
        self.corpus_state.save(os.path.join(config.STATE_SAVE_LOCATION, "08_insights"))
        return self.corpus_state.insights

    @staticmethod
    def ensure_list(x):
        """
        Normalize a value to list form.

        Converts common scalar, missing, and array-like values into a consistent
        list representation.

        Conversion rules
        ----------------
        - `list` → returned unchanged
        - `numpy.ndarray` → converted using `tolist()`
        - missing value (`pd.NA`, `np.nan`, etc.) → `[]`
        - any other value → wrapped in a single-element list

        Parameters
        ----------
        x : Any
            Value to normalize.

        Returns
        -------
        list
            List representation of the input.

        Examples
        --------
        >>> ensure_list(["a", "b"])
        ["a", "b"]

        >>> ensure_list(np.array([1, 2]))
        [1, 2]

        >>> ensure_list(pd.NA)
        []

        >>> ensure_list("text")
        ["text"]

        Notes
        -----
        This helper is commonly used when processing LLM outputs and DataFrame
        columns whose values may inconsistently appear as scalars, arrays, or
        missing values.

        The function assumes `pd.isna(x)` is valid for the supplied value.
        Unexpected container types may raise errors depending on pandas'
        missing-value handling behavior.
        """
        if isinstance(x, list):
            return x
        if isinstance(x, np.ndarray):
            return x.tolist()
        if pd.isna(x):
            return []
        return [x]  # fallback for any other type
    
    @staticmethod
    def estimate_tokens(text, model):
        """
        Estimate the number of tokens in a text string.

        Uses the tokenizer associated with the specified model to estimate
        how many tokens a piece of text will consume when sent to an LLM.

        Parameters
        ----------
        text : str
            Text to tokenize.

        model : str
            Model name used to select the appropriate tokenizer.

        Returns
        -------
        int
            Estimated token count.

        Notes
        -----
        Token counts are calculated using `tiktoken.encoding_for_model()`
        and therefore reflect the tokenizer associated with the specified
        model rather than simple character or word counts.

        This helper is primarily used to determine whether a document exceeds
        the configured context window and must be split before LLM
        processing.
        """
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    
    @staticmethod
    def string_breaker(text, max_token_length):
        """
        Split long text into smaller segments for LLM processing.

        Breaks a document into sequential text segments when the document is
        too large to fit comfortably within a model's context window.

        Parameters
        ----------
        text : str
            Text to split.

        max_token_length : int
            Maximum context length assumed for the target model.

        Returns
        -------
        list[str]
            Sequential text segments covering the full input text.

        Notes
        -----
        The function uses a conservative target segment size equal to
        approximately 75 percent of `max_token_length`.

        Splitting is performed on whitespace-separated words and is based on
        character length rather than true token counts. The resulting
        segments therefore approximate, but do not guarantee, compliance
        with the specified token limit.

        Word order is preserved and no overlap is introduced between
        segments.

        This helper is used when preparing document-level inputs for
        meta-insight generation, allowing very large papers to be processed
        in multiple passes.
        """
        max_length = max_token_length * 0.75
        words = text.split()
        current_chunk = ""
        chunks = []
        for word in words:
            if len(current_chunk) + len(word) + 1 <= max_length:
                current_chunk += (" " if current_chunk else "") + word
            else:
                chunks.append(current_chunk)
                current_chunk = word
        if current_chunk:
            chunks.append(current_chunk)
        return chunks
    
class Clustering:
    """
    Embedding and clustering stage of the ReadingMachine pipeline.

    The Clustering class converts extracted insights into a structured
    semantic representation that can be used to organize the corpus for
    downstream synthesis.

    The workflow consists of three stages:

        insight embeddings
            ↓
        dimensionality reduction
            ↓
        density-based clustering

    These operations transform the insight corpus into a set of provisional
    semantic groups that serve as computational scaffolding for later
    thematic synthesis.

    Importantly, clusters are not treated as analytical conclusions.

    ReadingMachine does not assume that embedding neighborhoods correspond
    directly to themes, concepts, arguments, or findings. Instead, clusters
    are used to:

    - organize large insight sets
    - provide manageable synthesis units
    - improve computational efficiency
    - supply an initial semantic structure for theme generation

    The final thematic structure emerges later through iterative synthesis,
    orphan detection, reinsertion, and theme revision rather than from the
    clustering process itself.

    Pipeline role
    -------------
    The clustering stage transforms:

        insights
            ↓
        embeddings
            ↓
        reduced embeddings
            ↓
        semantic clusters

    The resulting clusters provide the organizational layer used by
    downstream summarization and theme-generation stages.

    To support large-scale synthesis, the clustering workflow also includes
    cluster-size management. Oversized clusters are partitioned into bounded
    groups so that subsequent summarization steps remain compatible with LLM
    context-window constraints.

    Parameter tuning
    ----------------
    The class provides diagnostics for both dimensionality reduction and
    clustering.

    UMAP parameter tuning evaluates whether reduced embedding spaces retain
    useful structure using research-question labels as a proxy signal.

    HDBSCAN parameter tuning evaluates:

    - cluster compactness
    - outlier rates
    - downstream summarization feasibility

    These diagnostics are intended to support human parameter selection
    rather than fully automated optimization.

    Design principles
    -----------------
    The clustering stage is designed around several principles:

    Semantic organization, not interpretation
        Clusters organize insights but do not determine analytical meaning.

    Coverage preservation
        Outlier handling and bounded partitioning ensure that all insights
        remain available for downstream synthesis.

    Scalability
        Embeddings, dimensionality reduction, and clustering provide a
        computationally tractable representation of large insight corpora.

    Traceability
        Cluster assignments remain linked to the underlying insight records
        and associated metadata.

    Context-window awareness
        Cluster-size constraints are explicitly incorporated to support
        subsequent LLM-based summarization workflows.

    Attributes
    ----------
    corpus_state : CorpusState
        Working corpus state containing insight records and clustering
        outputs.

    llm_client : Any
        Client used for embedding generation.

    embedding_model : str
        Embedding model identifier.

    embedding_dims : int
        Number of embedding dimensions requested from the embedding model.

    valid_embeddings_df : pd.DataFrame
        Subset of insights eligible for embedding and clustering.

    insight_embeddings_array : np.ndarray
        Full embedding vectors for valid insights.

    reduced_insight_embeddings_array : np.ndarray
        UMAP-reduced embedding vectors used for clustering.

    rq_valid_embeddings_dfs : dict[str, pd.DataFrame]
        Research-question-specific embedding subsets used during clustering
        and parameter tuning.

    cum_prop_cluster : pd.DataFrame
        Cluster-distribution diagnostics generated during analysis.

    umap_param_tuning_results : pd.DataFrame
        Results of UMAP parameter tuning.

    hdbscan_tuning_results : pd.DataFrame
        Results of HDBSCAN parameter tuning.
    """
    def __init__(
        self,
        corpus_state: CorpusState,
        llm_client: Any,
        embedding_model: str,
        embedding_dims: int = 1024,
        embeddings_pickle_path: str = os.path.join(os.getcwd(), "data", "pickles", "insight_embeddings.pkl")
    ):
        """
        Initialize the clustering stage.

        Validates and deep-copies the supplied CorpusState, stores embedding
        configuration, and prepares the subset of insight rows that are eligible
        for embedding and clustering.

        Parameters
        ----------
        corpus_state : CorpusState
            CorpusState containing generated insights and associated paper,
            question, citation, and chunk metadata.

        llm_client : Any
            Client used to request embeddings.

        embedding_model : str
            Embedding model identifier.

        embedding_dims : int, default=1024
            Number of embedding dimensions to request from the embedding model.

        embeddings_pickle_path : str, default=os.path.join(os.getcwd(), "data", "pickles", "insight_embeddings.pkl")
            Path used to persist or recover generated insight embeddings.

        Attributes
        ----------
        valid_embeddings_df : pd.DataFrame
            Working DataFrame of valid insight rows prepared by
            `_gen_valid_embeddings_df()`.

        insight_embeddings_array : np.ndarray
            Placeholder for full insight embedding vectors.

        reduced_insight_embeddings_array : np.ndarray
            Placeholder for reduced-dimensionality embeddings.

        cum_prop_cluster : pd.DataFrame
            Placeholder for clustering diagnostics or cumulative cluster
            proportion outputs.

        Notes
        -----
        The supplied CorpusState is validated through `utils.validate_format()`
        and deep-copied before use, so clustering operations work on this class's
        internal state.

        The initializer requires extracted insight text, chunk identifiers,
        citation metadata, paper metadata, and question metadata to be present.

        The actual filtering and preparation of embeddable insights is delegated
        to `_gen_valid_embeddings_df()`.
        """
        self.corpus_state = deepcopy(
            utils.validate_format(
            corpus_state=corpus_state,
            questions = None,
            injected_value=None,
            state_required_cols=[
                "question_id", "question_text", "search_string_id", "search_string",
                "paper_id", "paper_title", "paper_author", "paper_date", "doi", "in_text_citation", "chunk_id", "insight"
            ],
            injected_required_cols=None
            )
        )

        self.llm_client = llm_client
        self.embedding_model = embedding_model
        self.embedding_dims = embedding_dims
        self.embeddings_pickle_path = embeddings_pickle_path

        self.valid_embeddings_df: pd.DataFrame = self._gen_valid_embeddings_df()
        self.insight_embeddings_array: np.ndarray = np.array([])
        self.reduced_insight_embeddings_array: np.ndarray = np.array([])
        self.cum_prop_cluster: pd.DataFrame = pd.DataFrame()

    def _gen_valid_embeddings_df(self):
        """
        Prepare the subset of insights eligible for embedding.

        Filters the current insight table to retain only rows containing both
        valid insight text and a non-empty in-text citation. The resulting
        DataFrame is used as the input set for embedding generation.

        Returns
        -------
        pd.DataFrame
            DataFrame containing only embeddable insight records.

        Notes
        -----
        Rows are retained only when:

        - `insight` is not missing
        - `in_text_citation` is not empty after string conversion and trimming

        The returned DataFrame has its index reset to provide a stable
        positional alignment between DataFrame rows and embedding vectors
        generated later in the clustering workflow.

        This method does not mutate `self.corpus_state.insights`.
        """

        valid = self.corpus_state.insights[
            self.corpus_state.insights["insight"].notna()
        ].copy()

        valid = valid[valid["in_text_citation"].astype(str).str.strip() != ""]

        # Ensure clean positional index for embedding alignment
        valid = valid.reset_index(drop=True)

        return valid
    
    def embed_insights(self) -> np.ndarray:
        """
        Generate or recover vector embeddings for valid insights.

        Creates embeddings for each row in `valid_embeddings_df` using the
        configured embedding model. Embeddings are stored by `insight_id`, saved
        incrementally for recovery, merged back into `valid_embeddings_df`, and
        converted into a NumPy array for downstream clustering.

        Returns
        -------
        pd.DataFrame
            DataFrame containing:

            - `insight_id`
            - `full_insight_embedding`

        Side Effects
        ------------
        Creates the parent directory for `self.embeddings_pickle_path` if needed.

        Reads from or writes to:

            self.embeddings_pickle_path

        Mutates clustering attributes:

        - adds `full_insight_embedding` to `self.valid_embeddings_df`
        - populates `self.insight_embeddings_array`

        Notes
        -----
        If an embeddings pickle already exists, the user is prompted to either
        resume from the saved embeddings or regenerate embeddings from scratch.

        Resume behavior is key-based. Previously processed rows are identified
        by `insight_id`, so recovery is robust to row order changes.

        Embeddings are checkpointed every 10 generated embeddings using
        `utils.safe_pickle()`. A final save is performed after all remaining
        embeddings are generated.

        Duplicate embedding records are removed using `insight_id`.

        Embedding text is taken from the `insight` column of
        `valid_embeddings_df`.

        The method assumes each valid insight has a stable, unique `insight_id`
        and that the embedding model returns vectors with consistent dimensions.
        """

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.embeddings_pickle_path), exist_ok=True)
       
       # Create empty df to populate with embeddings and insight_id
        processed_embeddings_df = pd.DataFrame(columns = ["insight_id", "full_insight_embedding"])
        
        # Initialize column for alignment (only once)
        if "full_insight_embedding" not in self.valid_embeddings_df.columns:
            self.valid_embeddings_df["full_insight_embedding"] = None

        # --- Resume / recovery handling ---
        if os.path.exists(self.embeddings_pickle_path):
            recover = None
            while recover not in ['r', 'n']:
                recover = input(
                    f"Embeddings pickle found at '{self.embeddings_pickle_path}'. "
                    "Do you wish to resume or regenerate embeddings?\n"
                    "Hit 'r' to resume, or 'n' to regenerate (overwrite existing pickle):\n"
                ).lower()

            if recover == 'r':
                print("Loading existing embeddings for resume...")
                # If recovering load the embeddings and overwrite the initially created procesed_embeddings_df
                with open(self.embeddings_pickle_path, "rb") as f:
                    processed_embeddings_df = pickle.load(f)

                # Defensive handling
                if processed_embeddings_df is None:
                    processed_embeddings_df = pd.DataFrame(columns = ["insight_id", "full_insight_embedding"])

                # Calculate the embeddings to process  by dropping the already processed ones
                embeddings_to_process_df = self.valid_embeddings_df[~self.valid_embeddings_df["insight_id"].isin(processed_embeddings_df["insight_id"])]

            else:
                print("Overwriting existing embeddings pickle...")
                # If not resuming processed_embeddings_df is empty to embeddings to process is just the valid_embeddings_df
                embeddings_to_process_df = self.valid_embeddings_df[~self.valid_embeddings_df["insight_id"].isin(processed_embeddings_df["insight_id"])]

        else:
                # If no pickle exists then we process all the valid embeddings and the processed_embeddings_df is empty
                embeddings_to_process_df = self.valid_embeddings_df.copy()

        # --- Generate embeddings ---
        total = self.valid_embeddings_df.shape[0]
        count = processed_embeddings_df.shape[0]

        new_rows = []
        for idx, row in embeddings_to_process_df.iterrows():
            count += 1

            print(f"Embedding insight {count} of {total}")

            response = self.llm_client.embeddings.create(
                input=row["insight"],
                model=self.embedding_model,
                dimensions=self.embedding_dims
            )

            emb = response.data[0].embedding
            current_embedding_dict = {
                "insight_id": row["insight_id"],
                "full_insight_embedding": emb
            }
            new_rows.append(current_embedding_dict)

            # Incremental save every 10 embeddings
            if (count) % 10 == 0:
                processed_embeddings_df = (
                    pd.concat([processed_embeddings_df,
                               pd.DataFrame(new_rows)],
                               ignore_index=True)
                    .drop_duplicates(subset=["insight_id"])
                )
                utils.safe_pickle(processed_embeddings_df, self.embeddings_pickle_path)
                new_rows = []


        # Final save to ensure completeness - last rows get covered
        if new_rows:
            processed_embeddings_df = (
                pd.concat([processed_embeddings_df,
                           pd.DataFrame(new_rows)],
                           ignore_index=True)
                .drop_duplicates(subset=["insight_id"])
            )
        utils.safe_pickle(processed_embeddings_df, self.embeddings_pickle_path)
        
        # Set attributes for downstream use---
        # First update valid_embedding df
        # If this is a recompute of embeddings we want to drop the old embeddings from the valid_embeddings_df so that we don't get _x and _y columns 
        if "full_insight_embedding" in self.valid_embeddings_df.columns:
            self.valid_embeddings_df = self.valid_embeddings_df.drop(columns=["full_insight_embedding"])
        # Then we merge 
        self.valid_embeddings_df = (
            self.valid_embeddings_df
            .merge(processed_embeddings_df, on="insight_id", how="left")
        )
        # Then valid_embeddings_array 
        self.insight_embeddings_array = np.vstack(self.valid_embeddings_df["full_insight_embedding"])
        # -----

        return processed_embeddings_df

    def reduce_dimensions(
        self, 
        full_embeddings: np.array = None,
        n_neighbors: int = 15,
        min_dist: float = 0.25,
        n_components: int = 10,
        metric: str = "cosine",
        random_state: int = config.seed,
        update_attributes: bool = True
    ) -> np.ndarray:
        """
        Reduce insight embedding dimensionality with UMAP.

        Projects high-dimensional insight embeddings into a lower-dimensional
        space for clustering and diagnostic exploration. If no embedding matrix
        is supplied, the method uses `self.insight_embeddings_array`.

        Parameters
        ----------
        full_embeddings : np.ndarray, optional
            Embedding matrix to reduce. If omitted, uses
            `self.insight_embeddings_array`.

        n_neighbors : int, default=15
            Size of the local neighborhood used by UMAP for manifold estimation.

        min_dist : float, default=0.25
            Minimum distance between points in the reduced embedding space.

        n_components : int, default=10
            Number of dimensions in the reduced embedding space.

        metric : str, default="cosine"
            Distance metric used by UMAP.

        random_state : int, default=config.seed
            Random seed used for reproducible dimensionality reduction.

        update_attributes : bool, default=True
            If True, store the reduced embeddings on the instance and prepare
            downstream clustering data structures. If False, return the reduced
            matrix without mutating class attributes.

        Returns
        -------
        np.ndarray
            Reduced embedding matrix.

        Side Effects
        ------------
        When `update_attributes=True`, mutates:

        - `self.reduced_insight_embeddings_array`
        - `self.valid_embeddings_df["reduced_insight_embedding"]`
        - `self.rq_valid_embeddings_dfs`

        Notes
        -----
        `self.valid_embeddings_df["reduced_insight_embedding"]` stores each
        reduced vector as a Python list.

        `self.rq_valid_embeddings_dfs` is a dictionary mapping each
        `question_id` to the subset of `valid_embeddings_df` for that research
        question.

        Set `update_attributes=False` when tuning UMAP parameters or running
        diagnostic projections without changing downstream clustering state.
        """
        # See if full_embeddings were provided, otherwise use the class's insight embeddings (this is done to make other methods in the class more explicit)
        if full_embeddings is None:
            full_embeddings = self.insight_embeddings_array

        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric,
            random_state=random_state
        )

        reduced_embeddings = reducer.fit_transform(full_embeddings)

        if update_attributes:
            # create some attributes for downstream use - skip this when tuning params
            self.reduced_insight_embeddings_array = reduced_embeddings
            self.valid_embeddings_df["reduced_insight_embedding"] = [row.tolist() for row in self.reduced_insight_embeddings_array]
            self.rq_valid_embeddings_dfs = {
                rq: self.valid_embeddings_df[self.valid_embeddings_df["question_id"] == rq].copy() for rq in self.valid_embeddings_df["question_id"].unique()
            }

        return reduced_embeddings

    def calc_silhouette(self, reduced_embeddings: np.array = None, rq_exclude: list[str] = None) -> float:
        """
        Calculate a silhouette diagnostic for a reduced embedding space.

        Computes a silhouette score on the reduced embeddings using
        `question_id` values as proxy labels. This diagnostic is used to assess
        whether the dimensionality-reduction parameters preserve enough
        structure to support distinct clustering under reduced information.

        Parameters
        ----------
        reduced_embeddings : np.ndarray, optional
            Reduced embedding matrix to evaluate. If omitted, uses
            `self.reduced_insight_embeddings_array`.

        rq_exclude : list[str], optional
            Research question IDs to exclude from the calculation.

        Returns
        -------
        float
            Silhouette score computed using Euclidean distance. Higher values
            indicate stronger separation under the proxy labeling scheme.

        Notes
        -----
        This method does not evaluate the final clustering solution directly.
        Instead, it uses research-question assignments as a practical proxy for
        testing whether the reduced embedding space retains meaningful
        separation after dimensionality reduction.

        The diagnostic is primarily useful when tuning UMAP parameters before
        density-based clustering.

        The score is printed before being returned.
        """

        if reduced_embeddings is None:
            reduced_embeddings = self.reduced_insight_embeddings_array

        sil_df = self.valid_embeddings_df.copy()
        sil_df["reduced_insight_embedding"] = [row.tolist() for row in reduced_embeddings]

        if rq_exclude:
            # Exclude any rows where question_id is in the rq_exclude list
            sil_df = sil_df[~sil_df["question_id"].isin(rq_exclude)]

        score = silhouette_score(
            X=np.vstack(sil_df["reduced_insight_embedding"].to_list()),
            labels=sil_df["question_id"].to_numpy(),
            metric="euclidean"
        )
        print(f"Silhouette score: {score}")
        return score

    def tune_umap_params(
        self,
        n_neighbors_list: list[int] = [5, 15, 30, 50, 75, 100],
        min_dist_list: list[float] = [0.0, 0.1, 0.2, 0.5],
        n_components_list: list[int] = [5, 10, 20],
        metric_list: list[str] = ["cosine", "euclidean"],
        rq_exclude: list[str] = None
    ) -> None:
        """
        Tune UMAP dimensionality-reduction parameters using a silhouette proxy.

        Runs a grid search over UMAP parameter combinations and evaluates each
        reduced embedding space with `calc_silhouette()`. Research-question IDs
        are used as proxy labels to assess whether the reduced space preserves
        enough separation to support downstream clustering.

        Parameters
        ----------
        n_neighbors_list : list[int], default=[5, 15, 30, 50, 75, 100]
            UMAP `n_neighbors` values to evaluate.

        min_dist_list : list[float], default=[0.0, 0.1, 0.2, 0.5]
            UMAP `min_dist` values to evaluate.

        n_components_list : list[int], default=[5, 10, 20]
            UMAP output dimensionalities to evaluate.

        metric_list : list[str], default=["cosine", "euclidean"]
            UMAP distance metrics to evaluate.

        rq_exclude : list[str], optional
            Research question IDs to exclude from silhouette scoring.

        Returns
        -------
        None

        Side Effects
        ------------
        Populates `self.umap_param_tuning_results` with a DataFrame sorted by
        descending silhouette score.

        Notes
        -----
        This method does not tune clusters directly. It evaluates whether UMAP
        parameter settings preserve useful structure in the reduced embedding
        space, using research-question labels as a practical proxy.

        Each parameter combination is reduced with `update_attributes=False`, so
        temporary projections do not overwrite the class's stored reduced
        embeddings.

        The search is run across the full insight set, not separately within
        each research question.
        """
        results = []
        param_grid = list(itertools.product(n_neighbors_list, min_dist_list, n_components_list, metric_list))
        total_runs = len(param_grid)

        for run_num, (n_neighbors, min_dist, n_components, metric) in enumerate(param_grid, start=1):
            print(f"Run {run_num} of {total_runs}: n_neighbors={n_neighbors}, min_dist={min_dist}, n_components={n_components}, metric={metric}")
            reduced_embeddings = self.reduce_dimensions(
                full_embeddings=self.insight_embeddings_array,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                n_components=n_components,
                metric=metric, 
                update_attributes=False  # do not store these temp reductions
            )
            # Calculate silhouette score, optionally excluding specified questions
            score = self.calc_silhouette(reduced_embeddings=reduced_embeddings, rq_exclude=rq_exclude)
            results.append({
                "n_neighbors": n_neighbors,
                "min_dist": min_dist,
                "n_components": n_components,
                "metric": metric,
                "silhouette_score": score
            })

        # Convert results to DataFrame for easy analysis
        results_df = pd.DataFrame(results)
        self.umap_param_tuning_results = results_df.sort_values("silhouette_score", ascending=False)
        print(self.umap_param_tuning_results)

    @staticmethod
    def cluster(embedding_matrix, min_cluster_size: int = 5, metric: str = "euclidean", cluster_selection_method: str = "eom", min_samples = None):
        """
        Cluster embeddings using HDBSCAN.

        Applies HDBSCAN to an embedding matrix and returns both hard cluster
        assignments and soft membership probabilities.

        Parameters
        ----------
        embedding_matrix : np.ndarray
            Matrix of embeddings to cluster. Each row represents a single
            observation.

        min_cluster_size : int, default=5
            Minimum number of observations required to form a cluster.

        metric : str, default="euclidean"
            Distance metric used by HDBSCAN.

        cluster_selection_method : str, default="eom"
            Method used to extract flat clusters from the HDBSCAN hierarchy.

        min_samples : int or None, default=None
            Controls how conservatively points are assigned to clusters. Higher
            values generally produce more noise points (`-1`). If None, HDBSCAN
            uses its default behavior.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing:

            - `cluster_labels`: cluster assignment for each observation.
            Noise points are labeled `-1`.
            - `cluster_probs`: membership probabilities returned by HDBSCAN.

        Notes
        -----
        This method performs clustering only. It does not update class
        attributes or modify the CorpusState.

        Cluster labels are generated independently for the supplied embedding
        matrix and are not guaranteed to be stable across different clustering
        runs or parameter settings.

        Membership probabilities indicate the strength of assignment of each
        point to its cluster and can be useful when evaluating cluster quality
        or identifying ambiguous assignments.
        """
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric=metric,
            cluster_selection_method=cluster_selection_method, 
            min_samples=min_samples
        )

        cluster_labels = clusterer.fit_predict(embedding_matrix)
        cluster_probs = clusterer.probabilities_

        return cluster_labels, cluster_probs
    
    @staticmethod
    def calc_davies_bouldain_score(embeddings_matrix, cluster_labels):
        """
        Compute the Davies–Bouldin score for a clustering result.

        Evaluates cluster compactness and separation using the Davies–Bouldin
        index. Lower scores indicate tighter, better-separated clusters.

        Parameters
        ----------
        embeddings_matrix : np.ndarray
            Embedding matrix used for clustering. Each row represents a single
            observation.

        cluster_labels : np.ndarray
            Cluster assignments for each observation. Noise points should be
            labeled `-1`.

        Returns
        -------
        tuple[float, int]
            A tuple containing:

            - `db_score`: Davies–Bouldin score computed on clustered points
            only. Returns `np.nan` when fewer than two clusters remain after
            removing noise points.
            - `num_outliers`: Number of observations labeled as noise (`-1`).

        Notes
        -----
        Noise points are excluded before calculating the score because they do
        not belong to any cluster.

        If fewer than two clusters remain after removing noise points, the score
        is undefined and `np.nan` is returned.

        This metric is primarily used as a clustering diagnostic and parameter
        tuning signal.

        Because HDBSCAN can generate non-convex and variable-density clusters,
        the Davies–Bouldin score should be interpreted as a heuristic rather
        than a definitive measure of clustering quality.
        """
        mask = cluster_labels != -1
        num_outliers = np.sum(~mask)

        filtered_embeddings = embeddings_matrix[mask]
        filtered_labels = cluster_labels[mask]

        if len(set(filtered_labels)) < 2:
            return np.nan, num_outliers
        db_score = davies_bouldin_score(filtered_embeddings, filtered_labels)
        return float(db_score), num_outliers

    def _estimate_max_cluster_sizes(self, context_window_constraint, hdbscan_cluster_size_cap):
        """
        Estimate per-question cluster size limits for downstream processing.

        Calculates a maximum cluster size for each research question based on
        the average length of its insights and a downstream context-window
        constraint. The resulting limits are intended to keep clusters small
        enough to fit within later LLM-based synthesis stages.

        Parameters
        ----------
        context_window_constraint : int
            Approximate maximum context size available for downstream processing.

        hdbscan_cluster_size_cap : int
            Hard upper bound on cluster size. Estimated limits will never exceed
            this value.

        Returns
        -------
        dict[str, int]
            Mapping from `question_id` to estimated maximum cluster size.

        Notes
        -----
        The estimate is based on the average insight length for each research
        question.

        For each question:

        1. Average insight length is calculated in words.
        2. The number of insights that could fit within the context window is
        estimated.
        3. A conservative scaling factor of 1.5 is applied to reduce the risk of
        exceeding the downstream context limit.
        4. The estimate is capped by `hdbscan_cluster_size_cap`.

        The resulting values are heuristics rather than strict guarantees
        because downstream token counts depend on prompt structure, formatting,
        citation metadata, and model tokenization behavior.

        Only insights with non-missing `insight` text are included in the
        calculation.
        """
        # Get a clean set of insights
        valid_insights = self.valid_embeddings_df[self.valid_embeddings_df["insight"].notna()]

        # Calculate the average insight lengh in words for each research question
        avg_insight_len_by_rq = {}
        for rq in valid_insights["question_id"].unique():
            rq_insights_list = valid_insights[valid_insights["question_id"] == rq]["insight"].tolist()
            rq_avg_insight_len = np.mean([len(insight.split()) for insight in rq_insights_list])
            avg_insight_len_by_rq[rq] = rq_avg_insight_len

        
        # Calculate the max cluser size for each research question based on the average insight length and the context window constraint
        max_cluster_size_by_rq = {}
        for rq, avg_len in avg_insight_len_by_rq.items():
                # Estimate the number of insights that would fit in the context window based on the average insight length
                estimated_insights_in_window = context_window_constraint / (avg_len * 1.5)  # Use a multiplier to be conservative
                max_cluster_size_by_rq[rq] = int(min(estimated_insights_in_window, hdbscan_cluster_size_cap))

        return max_cluster_size_by_rq

    def tune_hdbscan_params(
        self,
        min_cluster_sizes: list[int] = [5, 10, 15, 20],
        metrics: list[str] = ["euclidean", "manhattan"],
        min_sample_ratios: list[float] = [0.5, 0.25, 0.1, 0.05],
        cluster_selection_method: list[str] = ["eom", "leaf"],
        context_window_constraint: int = 90000, 
        hdbscan_cluster_size_cap: int = 1000,
        outlier_cluster_size_cap: int = 300
        ) -> None:

        """
        Tune HDBSCAN parameters separately for each research question.

        Runs a grid search over HDBSCAN parameter combinations for each
        research-question-specific embedding subset. Each configuration is
        evaluated using clustering diagnostics and downstream summarization-size
        constraints.

        Parameters
        ----------
        min_cluster_sizes : list[int], default=[5, 10, 15, 20]
            Candidate HDBSCAN `min_cluster_size` values.

        metrics : list[str], default=["euclidean", "manhattan"]
            Distance metrics to evaluate.

        min_sample_ratios : list[float], default=[0.5, 0.25, 0.1, 0.05]
            Ratios used to derive `min_samples` from `min_cluster_size`.

        cluster_selection_method : list[str], default=["eom", "leaf"]
            HDBSCAN cluster selection methods to evaluate.

        context_window_constraint : int, default=90000
            Approximate downstream context limit used to estimate maximum
            feasible cluster sizes per research question.

        hdbscan_cluster_size_cap : int, default=1000
            Hard cap applied when estimating maximum HDBSCAN cluster sizes.

        outlier_cluster_size_cap : int, default=300
            Maximum number of outlier insights to include in a single downstream
            outlier-summary group.

        Returns
        -------
        None

        Side Effects
        ------------
        Populates `self.hdbscan_tuning_results` with a DataFrame of valid tuning
        results sorted by:

        1. `question_id`
        2. lower `outlier_fraction`
        3. lower `db_score`

        Notes
        -----
        Clustering is performed independently within each research question using
        the reduced embeddings stored in `self.rq_valid_embeddings_dfs`.

        For each parameter combination, the method records:

        - Davies–Bouldin score
        - number of outliers
        - outlier fraction
        - estimated maximum HDBSCAN cluster size
        - estimated number of HDBSCAN clusters
        - estimated number of outlier groups
        - total number of downstream clusters to summarize

        `min_samples` is derived from `min_cluster_size * min_sample_ratio`,
        rounded to an integer, and only configurations with
        `2 <= min_samples <= min_cluster_size` are retained.

        Rows with undefined Davies–Bouldin scores are dropped before results are
        stored.

        This method is intended for human-in-the-loop parameter selection rather
        than automatic optimization. The diagnostics balance cluster compactness,
        outlier coverage, and downstream summarization feasibility.
        """
        
        #Calculate the max cluster sizes
        max_cluster_size_by_rq = self._estimate_max_cluster_sizes(context_window_constraint=context_window_constraint, 
                                                                  hdbscan_cluster_size_cap=hdbscan_cluster_size_cap)

        # Pass sample size values to get the full grid of search options
        param_grid = (
            pd.DataFrame(itertools.product(min_cluster_sizes, metrics, min_sample_ratios, cluster_selection_method), 
                         columns=["min_cluster_size", "metric", "min_sample_ratio", "cluster_selection_method"])
            .assign(min_samples=lambda df: (df["min_cluster_size"] * df["min_sample_ratio"]).round().astype(int))
            .query("min_samples >= 2 & min_samples <= min_cluster_size")  # Ensure min_samples is less than min_cluster_size and greater than 2 (required)
            .drop_duplicates(subset=["min_cluster_size", "metric", "min_samples"])
        )

        rqs = self.valid_embeddings_df["question_id"].unique()
        total_runs = len(param_grid) * len(rqs)
        data = [self.rq_valid_embeddings_dfs[rq] for rq in rqs]
        results = []
        for idx, (d, rq) in enumerate(zip(data, rqs)):
            print(f"Tuning HDBSCAN for {rq}...(run {idx + 1} of {total_runs})")
            embeddings_matrix = np.vstack(d["reduced_insight_embedding"].to_list())
            for row in param_grid.itertuples(index = False):
                cluster_labels, _ = self.cluster(
                    embeddings_matrix,
                    min_cluster_size=row.min_cluster_size,
                    metric=row.metric,
                    min_samples=row.min_samples,
                    cluster_selection_method=row.cluster_selection_method
                    )
                
                db_score, num_outliers = self.calc_davies_bouldain_score(embeddings_matrix, cluster_labels)

                outlier_cluster_count = math.ceil(num_outliers / outlier_cluster_size_cap) if num_outliers > 0 else 0
                hdbscan_cluster_count = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0) 

                results.append({
                    "question_id": rq,
                    "min_cluster_size": row.min_cluster_size,
                    "metric": row.metric,
                    "cluster_selection_method": row.cluster_selection_method,
                    "min_sample_ratio": row.min_sample_ratio,
                    "min_samples": row.min_samples,
                    "db_score": db_score,
                    "num_outliers": num_outliers,
                    "outlier_fraction": num_outliers / len(cluster_labels) if len(cluster_labels) > 0 else 0,
                    "max_hdbscan_cluster_size": max_cluster_size_by_rq[rq],
                    "max_outlier_cluster_size": outlier_cluster_size_cap,
                    "final_hdbscan_cluster_count": hdbscan_cluster_count,
                    "final_outlier_cluster_count": outlier_cluster_count,
                    "total_final_clusters_to_summarize": hdbscan_cluster_count + outlier_cluster_count
                    })

        results_df = pd.DataFrame(results)

        valid_results_df = (
            results_df
            .dropna(subset=["db_score"])
            .sort_values(["question_id", "outlier_fraction", "db_score"], ascending=[True, True, True])
            .assign(outlier_fraction = lambda x: x["outlier_fraction"].round(3))
            .assign(db_score = lambda x: x["db_score"].round(3))
        )

        self.hdbscan_tuning_results = valid_results_df
        print(self.hdbscan_tuning_results)

    @staticmethod
    def density_seeded_partition(X: np.ndarray, K: int):
        """
        Partition embeddings into size-bounded, spatially coherent groups.

        Creates a constrained partition of an embedding matrix so that every
        point is assigned to exactly one group and no group exceeds size `K`.
        The method uses density-guided seed selection followed by balanced
        nearest-neighbor assignment.

        Parameters
        ----------
        X : np.ndarray
            Embedding matrix of shape `(n_points, n_features)`.

        K : int
            Maximum number of points allowed in each output group.

        Returns
        -------
        tuple[np.ndarray, list[int]]
            A tuple containing:

            - `labels`: array assigning each input point to a partition ID.
            - `seeds`: original row indices selected as partition seeds.

        Notes
        -----
        This is a constrained partitioning method, not a standard clustering
        algorithm.

        HDBSCAN is used only during seed selection to identify dense regions
        of the embedding space. Final assignments are produced by a
        capacity-constrained nearest-neighbor procedure.

        The method is designed for downstream summarization constraints where
        large clusters or outlier groups must be broken into bounded groups
        that can fit within LLM context limits.

        The intended guarantees are:

        - every point is assigned
        - each point is assigned once
        - no group exceeds size `K`

        The resulting groups are approximately spatially coherent but do not
        optimize a global clustering objective.
        """

        # ----- COMPONENT FUNCTIONS START -----

        def select_seeds_with_density(X: np.ndarray, n_seeds: int):
            """
            Select spatially distributed seed points using density estimates.

            Repeatedly runs HDBSCAN on the remaining candidate points,
            selects the point with the highest membership probability as a
            seed, and removes that seed's `K` nearest neighbors from
            subsequent seed selection.

            Parameters
            ----------
            X : np.ndarray
                Embedding matrix of shape `(n_points, n_features)`.

            n_seeds : int
                Number of seed points to select.

            Returns
            -------
            list[int]
                Original row indices of selected seed points.

            Notes
            -----
            Seed selection is density-guided and destructive. After each seed
            is chosen, nearby points are removed from the candidate pool. This
            reduces seed concentration in dense regions and encourages
            coverage across the embedding space.

            HDBSCAN probabilities are used only as density proxies for seed
            placement. They are not used to assign the final partitions.
            """
            remaining_idx = np.arange(len(X))
            seeds = []

            for _ in range(n_seeds):
                if len(remaining_idx) == 0:
                    break

                X_sub = X[remaining_idx]

                # Run HDBSCAN
                clusterer = HDBSCAN(min_cluster_size=5)
                clusterer.fit(X_sub)

                # Pick highest density point
                probs = clusterer.probabilities_
                seed_local_idx = np.argmax(probs)
                seed_global_idx = remaining_idx[seed_local_idx]

                seeds.append(seed_global_idx)

                # Remove K nearest points
                seed = X_sub[seed_local_idx]
                dists = np.linalg.norm(X_sub - seed, axis=1)
                idx = np.argsort(dists)[:K]

                remaining_idx = np.delete(remaining_idx, idx)

            return seeds


        def assign_points_balanced(X, seed_indices, K):
            """
            Assign points to seeds with balanced, capacity-limited growth.

            Precomputes distances from every point to every seed, then grows
            seed groups in round-robin order. Each seed repeatedly claims its
            nearest unassigned point until all points are assigned or the seed
            reaches capacity.

            Parameters
            ----------
            X : np.ndarray
                Embedding matrix of shape `(n_points, n_features)`.

            seed_indices : list[int]
                Original row indices of the selected seed points.

            K : int
                Maximum number of points allowed in each assigned group.

            Returns
            -------
            np.ndarray
                Array of shape `(n_points,)` containing the assigned group
                label for each input point.

            Notes
            -----
            Each point is assigned exactly once.

            No assigned group exceeds size `K`.

            Assignment is local and iterative rather than globally optimized.
            Unlike KMeans, this method does not recompute centroids or
            minimize a global within-cluster objective.

            Because groups grow in round-robin order with a shared capacity
            limit, the resulting partitions are approximately balanced while
            remaining tied to local distance from seed points.
            """
            n = len(X)
            assigned = np.full(n, -1, dtype=int)

            seeds = X[seed_indices]

            # distance matrix (n_points x n_seeds)
            dist_matrix = np.linalg.norm(X[:, None, :] - seeds[None, :, :], axis=2)

            sorted_points = [np.argsort(dist_matrix[:, i]) for i in range(len(seeds))]
            pointers = [0] * len(seeds)
            cluster_sizes = [0] * len(seeds)

            remaining = set(range(n))

            while remaining:
                for i in range(len(seeds)):
                    if cluster_sizes[i] >= K:
                        continue

                    while pointers[i] < n:
                        candidate = sorted_points[i][pointers[i]]
                        pointers[i] += 1

                        if candidate in remaining:
                            assigned[candidate] = i
                            remaining.remove(candidate)
                            cluster_sizes[i] += 1
                            break

                    if not remaining:
                        break

            return assigned

        # ----- COMPONENT FUNCTIONS END -----

        n_points = len(X)
        n_seeds = math.ceil(n_points / K)

        seeds = select_seeds_with_density(X, n_seeds)
        labels = assign_points_balanced(X, seeds, K)

        return labels, seeds

    def generate_clusters(self,
                          clustering_param_dict: dict,
                          context_window_constraint: int = 90000,
                          hdbscan_cluster_size_cap: int = 1000, 
                          outlier_cluster_size_cap: int = 300) -> pd.DataFrame:
        """
        Generate final cluster assignments for insights.

        Clusters reduced insight embeddings independently within each research
        question, post-processes oversized groups into bounded partitions, merges
        the resulting cluster labels back into `corpus_state.insights`, and saves
        the clustered state.

        Parameters
        ----------
        clustering_param_dict : dict
            Mapping from `question_id` to HDBSCAN parameter dictionaries. Each
            parameter dictionary should contain:

            - `min_cluster_size`
            - `metric`
            - `cluster_selection_method`
            - `min_samples`

        context_window_constraint : int, default=90000
            Approximate downstream context limit used to estimate maximum
            allowable core-cluster sizes.

        hdbscan_cluster_size_cap : int, default=1000
            Hard upper bound applied when estimating maximum core-cluster size.

        outlier_cluster_size_cap : int, default=300
            Maximum allowed size for outlier-derived groups.

        Returns
        -------
        pd.DataFrame
            Updated `corpus_state.insights` DataFrame containing cluster labels,
            cluster probabilities, full embeddings, and reduced embeddings.

        Side Effects
        ------------
        Mutates `self.corpus_state.insights` by replacing any existing cluster
        and embedding columns with newly generated clustering results.

        Saves the updated CorpusState to:

            {config.STATE_SAVE_LOCATION}/09_clusters

        Notes
        -----
        HDBSCAN is run separately for each research question using the reduced
        embeddings in `self.rq_valid_embeddings_dfs`.

        If parameters are not supplied for every research question, the user is
        prompted to either use default HDBSCAN parameters for missing questions
        or abort.

        Initial HDBSCAN labels are used for structure discovery. Oversized
        clusters are then split using `density_seeded_partition()` so downstream
        summarization groups remain within practical size limits.

        Core HDBSCAN clusters use per-question size caps estimated from
        `context_window_constraint` and `hdbscan_cluster_size_cap`.

        Outlier groups use `outlier_cluster_size_cap`.

        Cluster labels are globally unique across research questions:

        - positive labels are used for core clusters
        - negative labels are used for outlier-derived groups

        Partitioned groups are assigned `cluster_prob = 1.0`. Unpartitioned
        HDBSCAN groups retain their original HDBSCAN membership probabilities.

        The final merge back into `corpus_state.insights` is performed on:

        - `question_id`
        - `paper_id`
        - `chunk_id`
        - `insight_id`
        """
        # Get research questions
        rqs = self.valid_embeddings_df["question_id"].unique()

        # Estimate max cluster sizes (for core clusters only)
        max_cluster_size_by_rq = self._estimate_max_cluster_sizes(
            context_window_constraint=context_window_constraint,
            hdbscan_cluster_size_cap=hdbscan_cluster_size_cap
        )

        # Resolve parameters
        if len(clustering_param_dict) != len(rqs):
            use_default = None
            while use_default not in ['y', 'n']:
                use_default = input(
                    "You did not enter specific clustering parameters for each research question. "
                    "Use default parameters? (y/n): "
                ).lower()
                if use_default == 'n':
                    raise KeyboardInterrupt("Provide parameters for each research question.")
            params = [
                clustering_param_dict.get(
                    rq,
                    {"min_cluster_size": 5, "metric": "euclidean", "cluster_selection_method": "eom", "min_samples": 5}
                )
                for rq in rqs
            ]
        else:
            params = [clustering_param_dict[rq] for rq in rqs]

        data = [self.rq_valid_embeddings_dfs[rq] for rq in rqs]

        # 🔑 GLOBAL label counters (ensure uniqueness across all RQs)
        next_positive_label = 1
        next_negative_label = -1

        # --- CLUSTERING LOOP ---
        for i, (d, rq, param) in enumerate(zip(data, rqs, params)):

            embeddings_matrix = np.vstack(d["reduced_insight_embedding"].to_list())

            cluster_labels, cluster_probs = self.cluster(
                embedding_matrix=embeddings_matrix,
                min_cluster_size=param["min_cluster_size"],
                metric=param["metric"],
                cluster_selection_method=param["cluster_selection_method"],
                min_samples=param["min_samples"],
            )

            d = d.copy()
            d["cluster"] = cluster_labels
            d["cluster_prob"] = cluster_probs

            new_rows = []

            # Process each cluster independently
            for label in sorted(d["cluster"].unique()):

                cluster_df = d[d["cluster"] == label].copy()
                cluster_size = len(cluster_df)

                # Decide cap
                if label >= 0:
                    cap = max_cluster_size_by_rq[rq]
                else:
                    cap = outlier_cluster_size_cap

                # Split if needed
                if cluster_size > cap:
                    X_cluster = np.vstack(cluster_df["reduced_insight_embedding"].to_list())

                    sub_labels, _ = self.density_seeded_partition(X_cluster, K=cap)

                    unique_sub = np.unique(sub_labels)

                    if label >= 0:
                        # Assign new positive labels
                        label_map = {
                            old: next_positive_label + idx
                            for idx, old in enumerate(unique_sub)
                        }
                        cluster_df["cluster"] = [label_map[l] for l in sub_labels]
                        next_positive_label += len(unique_sub)

                    else:
                        # Assign new negative labels
                        label_map = {
                            old: next_negative_label - idx
                            for idx, old in enumerate(unique_sub)
                        }
                        cluster_df["cluster"] = [label_map[l] for l in sub_labels]
                        next_negative_label -= len(unique_sub)

                    cluster_df["cluster_prob"] = 1.0

                new_rows.append(cluster_df)

            # Rebuild dataframe for this RQ
            d = pd.concat(new_rows, ignore_index=True)
            data[i] = d

        # Combine all RQs
        clustered_df = pd.concat(data, ignore_index=True)

        # Clean existing cluster columns
        cluster_cols = ["cluster", "cluster_prob", "full_insight_embedding", "reduced_insight_embedding"]
        self.corpus_state.insights = self.corpus_state.insights.drop(
            columns=[col for col in cluster_cols if col in self.corpus_state.insights.columns]
        )

        # Merge results back
        self.corpus_state.insights = self.corpus_state.insights.merge(
            clustered_df[[
                "question_id", "paper_id", "chunk_id", "insight_id",
                "cluster", "cluster_prob",
                "full_insight_embedding", "reduced_insight_embedding"
            ]],
            on=["question_id", "paper_id", "chunk_id", "insight_id"],
            how="left"
        )

        # Save
        self.corpus_state.save(os.path.join(config.STATE_SAVE_LOCATION, "09_clusters"))

        return self.corpus_state.insights

class Summarize:
    """
    Thematic synthesis stage of the ReadingMachine pipeline.

    The Summarize class converts the clustered insight representation
    generated during corpus reading into a structured thematic synthesis.
    It implements the iterative synthesis workflow described in the
    ReadingMachine methodology, transforming insight-level representations
    into theme-level narratives while preserving coverage, traceability,
    and opportunities for revision.

    Unlike earlier pipeline stages, which focus on extracting and organizing
    information, the Summarize stage is responsible for constructing the
    thematic representation of the corpus.

    The synthesis workflow proceeds through a sequence of stages:

        cluster summaries
            ↓
        theme schema generation
            ↓
        insight-to-theme mapping
            ↓
        theme population
            ↓
        orphan detection
            ↓
        orphan reintegration
            ↓
        schema repair / re-theming
            ↓
        schema optimization
            ↓
        redundancy reduction

    These stages may be repeated multiple times. Theme schemas are not
    treated as fixed structures; instead they evolve in response to
    representational failures, orphan patterns, and evidence that certain
    themes are carrying too much conceptual load.

    Cluster Summaries
    -----------------
    The synthesis process begins by summarizing semantically related
    clusters of insights.

    Clusters are processed in semantic-proximity order using centroid-based
    path estimation. The resulting cluster summaries serve as a compressed
    representation of the clustered insight space and provide the initial
    scaffolding for theme generation.

    Importantly, clusters are not treated as themes. They function as
    organizational structures that help the model reason over large insight
    sets prior to thematic synthesis.

    Theme Schema Generation
    -----------------------
    Theme schemas define the conceptual categories used to organize
    insights.

    Each theme contains:

    - a theme label
    - a theme description
    - mapping instructions

    Theme schemas are generated independently for each research question
    and may be revised repeatedly as synthesis progresses.

    Mapping and Population
    ----------------------
    Once a schema exists, insights are mapped into one or more themes.

    Themes are then populated by synthesizing their assigned insights into
    narrative summaries. Theme lengths are allocated proportionally based
    on mapped insight volume while respecting practical model-output
    constraints.

    The resulting populated themes become the primary analytical
    representation of the corpus.

    Coverage Preservation
    ---------------------
    ReadingMachine prioritizes coverage over compression.

    To support this objective, populated themes undergo a completeness
    audit in which mapped insights are checked against the synthesized
    narrative.

    Insights that are not represented in the summary are identified as
    orphans and are reintroduced through an iterative integration process.

    Orphan handling serves as the primary mechanism for preventing
    information loss during synthesis.

    Schema Repair and Stabilization
    -------------------------------
    When orphan integration fails, when themes become overloaded, or when
    summaries exhibit signs of excessive compression, the schema may be
    revised.

    The repair workflow operates in two stages:

        repair planning
            ↓
        repair implementation

    This separation helps reduce performative repair and maintains an
    inspectable record of why schema modifications were proposed.

    Once representational adequacy is achieved, schemas may undergo a
    separate optimization stage focused on improving thematic coherence.

    Stable schemas are preserved across iterations and excluded from
    unnecessary reprocessing.

    Redundancy Reduction
    --------------------
    After thematic synthesis has stabilized, a final redundancy pass is
    performed.

    Theme summaries are reviewed sequentially within each research
    question and revised to reduce repeated content while preserving unique
    information.

    This step improves readability without modifying the underlying
    thematic structure.

    Persistence and Recovery
    ------------------------
    The class operates over two complementary state objects:

        CorpusState
        SummaryState

    CorpusState contains the insight-level representation generated by the
    reading stages of the pipeline.

    SummaryState stores the evolving synthesis artifacts generated during
    thematic analysis, including:

    - cluster summaries
    - theme schemas
    - insight mappings
    - populated themes
    - orphan audits
    - redundancy outputs

    Long-running operations support checkpoint-based recovery through
    pickle serialization and state fingerprinting.

    Resume operations verify both CorpusState and SummaryState before
    continuing execution, helping prevent corruption caused by state drift
    between synthesis sessions.

    Design Principles
    -----------------
    The Summarize stage reflects several core ReadingMachine principles:

    Coverage Preservation
        Information omitted during synthesis is identified and reintroduced
        through orphan detection and reintegration.

    Iterative Synthesis
        Theme schemas are expected to evolve in response to observed
        representational failures.

    Traceability
        Theme summaries remain linked to mapped insights, citations, and
        research questions.

    Inspectability
        Intermediate synthesis artifacts are preserved rather than
        discarded, allowing users to examine how thematic structures
        develop over time.

    Separation of Reading and Synthesis
        Corpus reading produces the insight representation; thematic
        synthesis operates over that representation without re-reading the
        source documents.

    Stability Through Evidence
        Schema revisions are driven by synthesis outcomes, completeness
        audits, and failed integrations rather than arbitrary regeneration.

    Attributes
    ----------
    corpus_state : CorpusState
        Insight-level corpus representation used as the foundation for
        thematic synthesis.

    summary_state : SummaryState
        Persistent synthesis-state object storing cluster summaries,
        schemas, mappings, populated themes, orphan outputs, and redundancy
        passes.

    llm_client : Any
        Language model client used throughout the synthesis workflow.

    ai_model : str
        Model identifier used for synthesis operations.

    paper_output_length : int
        Approximate target length for synthesized outputs.

    insight_embeddings_array : np.ndarray
        Embedding representation loaded from the clustering stage and used
        for cluster ordering operations.

    summary_save_location : str
        Directory used to persist SummaryState artifacts.
    """
    def __init__(self,
                 corpus_state: Any,
                 llm_client: Any,
                 ai_model: str,
                 paper_output_length: int,  # Approximate total paper length in words
                 summary_save_location: str = config.SUMMARY_SAVE_LOCATION, 
                 pickle_save_location: str = config.PICKLE_SAVE_LOCATION,
                 insight_embedding_path = os.path.join(os.getcwd(), "data", "pickles", "insight_embeddings.pkl"), 
                 use_organizing_proposition = False):
        """
        Initialize the summarization stage.

        Sets up the Summarize object used to convert clustered insights into
        iterative thematic synthesis artifacts. The initializer verifies that
        insight embeddings from the clustering stage are available, deep-copies
        the supplied CorpusState, loads or initializes SummaryState, and stores
        LLM and output-length configuration for downstream summarization methods.

        Parameters

        corpus_state : CorpusState
        Corpus state containing clustered insight records and associated
        metadata from earlier pipeline stages.

        llm_client : Any
        Client used for LLM calls during cluster summarization, theme
        generation, theme mapping, theme population, orphan handling, citation
        repair, and redundancy reduction.

        ai_model : str
        Model identifier used for summarization-stage LLM calls.

        paper_output_length : int
        Approximate target length, in words, for the final synthesized output.
        Downstream methods use this value to allocate summary lengths across
        research questions and themes.

        summary_save_location : str, default=config.SUMMARY_SAVE_LOCATION
        Directory used by SummaryState to persist summarization artifacts as
        Parquet files.

        pickle_save_location : str, default=config.PICKLE_SAVE_LOCATION
        Directory used by downstream summarization methods for intermediate
        pickle-based recovery artifacts.

        insight_embedding_path : str, default=os.path.join(os.getcwd(), "data", "pickles", "insight_embeddings.pkl")
        Path to the serialized insight embeddings generated during the
        clustering stage.

        Raises

        FileNotFoundError
        If insight_embedding_path does not exist.

        Side Effects

        Creates summary_save_location if it does not already exist.

        If existing summary Parquet files are found, prompts the user to either
        reload the existing SummaryState or delete those files and start with a
        new empty SummaryState.

        Attributes

        insight_embeddings_array : np.ndarray or pd.DataFrame
        Embedding artifact loaded from insight_embedding_path.

        corpus_state : CorpusState
        Deep-copied corpus state used by the summarization stage.

        summary_state : SummaryState
        Summary-state object used to store cluster summaries, theme schemas,
        theme mappings, populated themes, orphan outputs, and redundancy outputs.

        llm_client : Any
        LLM client stored for downstream calls.

        ai_model : str
        Model identifier stored for downstream calls.

        paper_output_length : int
        Target final output length used by downstream synthesis methods.

        summary_save_location : str
        Directory used for persisted SummaryState artifacts.

        Notes

        The summarization stage operates on both CorpusState and SummaryState.
        CorpusState provides the clustered insight-level representation generated
        by earlier ReadingMachine stages. SummaryState records the evolving
        synthesis artifacts produced by the summarization workflow.

        Existing summary artifacts are not silently overwritten. When Parquet
        files are detected in summary_save_location, the user must choose
        whether to resume from them or regenerate the summary state from scratch.
        """ 
        # Check that the embeddings have been created from the clustering step. If so, load. If not send the user back to run clustering
        if not os.path.exists(insight_embedding_path):
            raise FileNotFoundError(f"Insight embeddings pickle not found at {insight_embedding_path}. Please run clustering first or amend the path to where you pickled your insight embeddings.")
        else:
            with open(insight_embedding_path, "rb") as f:
                self.insight_embeddings_array: np.ndarray = pickle.load(f)
        
        # Load the two states that this class operates on. Note this class has a SummaryState which will manage all the objects the class generates
        self.corpus_state: CorpusState = deepcopy(corpus_state)
        # Check whether SummaryState already has some summaries on disk - if so offer to reload or regenerate (regen will overwrite everything)
        os.makedirs(summary_save_location, exist_ok=True)
        path = Path(summary_save_location)
        possible_summary_files = list(path.glob("*.parquet"))
        if possible_summary_files:
            load = None
            while load not in ["1", "2"]:
                load = input(
                    f"Previous summarization files found in '{summary_save_location}'. Do you want to:\n"
                    "(1) reload existing summaries\n"
                    "(2) regenerate summaries (NOTE: this will overwrite all existing summary attributes)\n"
                    "Enter 1 or 2:\n"
                ).strip()

            if load == "1":
                self.summary_state = SummaryState.load(summary_save_location=summary_save_location)
                print("Summaries loaded. To see where you are in the summarization process run var.summary_state.status())")

            else:
                # Clear out existing summaries if we are regenerating
                for file in Path(summary_save_location).glob("*.parquet"):
                    file.unlink()  
                # And load the empty summary_state
                self.summary_state = SummaryState(summary_save_location=summary_save_location)
        
        # If no summary files found, just load the empty summary state
        else:
            self.summary_state = SummaryState(summary_save_location=summary_save_location)

        self.llm_client: Any = llm_client
        self.ai_model: str = ai_model
        self.paper_output_length: int = paper_output_length
        self.summary_save_location = summary_save_location
        self.pickle_save_location = pickle_save_location
        self.use_organizing_proposition = use_organizing_proposition

    def _calculate_centroid(self, col="full_insight_embedding"):
        """
        Compute cluster centroids for downstream cluster ordering.

        Generates a centroid vector for each `(question_id, cluster)` pair by
        averaging the embedding vectors assigned to that cluster. These
        centroids provide a compact representation of cluster location within
        the embedding space and are subsequently used to estimate semantic
        proximity between clusters.

        Within ReadingMachine, cluster ordering is used during cluster
        summarization to arrange semantically related clusters adjacent to one
        another. This supports the generation of a structured cluster-summary
        narrative that later serves as the input to theme-schema generation.

        Parameters
        ----------
        col : str, default="full_insight_embedding"
            Column in `corpus_state.insights` containing the embedding vectors
            used to compute cluster centroids.

        Returns
        -------
        pd.DataFrame
            DataFrame containing one row per `(question_id, cluster)` pair
            with the columns:

            - `question_id`
            - `cluster`
            - `centroid`

            The `centroid` column contains the mean embedding vector for the
            corresponding cluster.

        Notes
        -----
        Centroids are computed independently within each research question.

        Several validation and cleaning steps are applied before centroid
        calculation:

        - rows without valid embedding vectors are excluded
        - ragged embeddings are filtered using the modal vector length
        - embeddings containing NaN values are removed

        Clusters that contain no valid embeddings after cleaning are skipped.

        Centroids are used only as a geometric representation of cluster
        position. They do not determine thematic structure and should be
        understood as part of the semantic scaffolding used to organize
        insights prior to theme generation.
        """
        rows = []
        for rq, d in self.corpus_state.insights.groupby("question_id", sort=False):
            # get clusters with at least one non-null embedding
            for cl, g in d.groupby("cluster", sort=False):
                vecs = [v for v in g[col].tolist() if isinstance(v, (list, tuple, np.ndarray))]
                if not vecs:
                    continue  # skip empty cluster
                A = np.asarray(vecs, dtype=np.float32)
                # guard ragged
                if not np.all([len(v) == A.shape[1] for v in A]):
                    # filter by the modal length
                    L = pd.Series([len(v) for v in vecs]).mode().iloc[0]
                    A = np.asarray([v for v in vecs if len(v) == L], dtype=np.float32)
                # drop rows with NaN
                mask = ~np.isnan(A).any(axis=1)
                A = A[mask]
                if A.size == 0:
                    continue
                centroid = A.mean(axis=0, dtype=np.float32)
                rows.append({"question_id": rq, "cluster": cl, "centroid": centroid})
        
        return pd.DataFrame(rows, columns=["question_id", "cluster", "centroid"])

    def _estimate_shortest_path(self):
        """
        Estimate semantic ordering paths through non-outlier clusters.

        Computes cluster centroids and uses pairwise cosine distance between
        centroid embeddings to derive an ordered path through the clusters for
        each research question. The resulting order places semantically nearby
        clusters adjacent to one another, providing a coherent sequence for
        downstream cluster summarization.

        In ReadingMachine, this ordering is used to construct the cluster-summary
        narrative that scaffolds initial theme-schema generation. It affects the
        order in which cluster summaries are produced and presented, but does not
        itself determine the final thematic structure.

        Procedure
        ---------
        1. Compute cluster centroids from insight embeddings.
        2. Remove the HDBSCAN outlier cluster (`-1`) from path estimation.
        3. Compute pairwise cosine distances between remaining centroids.
        4. Estimate an ordered path through the clusters:

        - for zero or one non-outlier cluster, use the available cluster order
        - for fewer than 10 clusters, evaluate all permutations exactly
        - for 10 or more clusters, use NetworkX's approximate traveling
            salesman path algorithm

        5. Append `-1` to the end of each returned order so outlier-derived
        insights are summarized after core clusters.

        Returns
        -------
        dict
            Dictionary mapping each research question with at least one valid
            non-outlier centroid to an ordered cluster list:

            {
                question_id: {
                    "order": [cluster_1, cluster_2, ..., -1]
                }
            }

        Notes
        -----
        Outlier cluster `-1` is excluded from centroid-path estimation and appended
        at the end of the returned ordering. The value is appended even when no
        outlier insights are present for a given research question.

        Research questions with no valid non-outlier centroids are omitted from the
        returned dictionary.

        The path is based on centroid geometry and should be understood as semantic
        scaffolding for summary generation, not as a conceptual or thematic
        ordering of the corpus.
        """
        print("Calculating centroids for each cluster...")
        centroids = self._calculate_centroid(col="full_insight_embedding")

        # handle outliers later
        centroids = centroids[centroids["cluster"] != -1].copy()
        print("Estimating shortest path through clusters for each research question...")
        shortest_paths = {}

        for rq, df in centroids.groupby("question_id", sort=False):
            clusters = df["cluster"].tolist()
            C = np.stack(df["centroid"].to_list()).astype(np.float32)

            # prefer cosine distance on embeddings
            # normalize to unit vectors
            norms = np.linalg.norm(C, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            U = C / norms
            # cosine distance matrix
            D = 1.0 - (U @ U.T)
            np.fill_diagonal(D, 0.0)

            n = len(clusters)
            if n <= 1:
                order = clusters
                final_order = order + [-1]
            elif n < 10:
                best_len = np.inf
                best_perm = None
                for perm in itertools.permutations(range(n)):
                    length = sum(D[perm[i], perm[i+1]] for i in range(n-1))
                    if length < best_len:
                        best_len = length
                        best_perm = perm
                order = [clusters[i] for i in best_perm]
                final_order = order + [-1]  # add outlier cluster at the end
            else:
                G = nx.Graph()
                for i in range(n):
                    for j in range(i+1, n):
                        G.add_edge(clusters[i], clusters[j], weight=float(D[i, j]))
                order = nx.approximation.traveling_salesman_problem(G, weight="weight", cycle=False)
                final_order = order + [-1]  # add outlier cluster at the end
            shortest_paths[rq] = {"order": final_order}
    
        return shortest_paths
    

    def summarize_clusters(self, frozen_summary_window = 5):
        """
        Generate cluster summaries for all research questions.

        Produces the cluster-summary narrative that forms the first synthesis
        layer of the ReadingMachine thematic workflow. Each cluster is
        summarized independently and the resulting summaries are stored as a
        structured representation of the corpus that is subsequently used for
        theme-schema generation.

        Clusters are processed in the semantic order estimated by
        `_estimate_shortest_path()`. This ordering places nearby clusters in the
        embedding space adjacent to one another, creating a locally coherent
        narrative of the corpus while avoiding the need to expose the model to
        the full cluster set simultaneously.

        To maintain local continuity, previously generated cluster summaries are
        provided as frozen context using a bounded sliding window. Context is:

            - restricted to the current research question
            - limited to the most recent `frozen_summary_window` summaries

        This approach allows neighboring clusters to influence one another while
        reducing long-range path dependence and limiting the impact of early
        summaries on later synthesis.

        The workflow proceeds as follows:

            shortest-path cluster ordering
                ↓
            cluster summarization
                ↓
            cluster-summary narrative
                ↓
            theme-schema generation

        If cluster summaries already exist, the user may either:

            (1) reload the existing summaries
            (2) regenerate summaries

        Regeneration resets all downstream synthesis artifacts because they are
        derived from the cluster-summary representation.

        Parameters
        ----------
        frozen_summary_window : int, default=5
            Number of previously generated cluster summaries to provide as
            frozen context when generating each new summary.

        Returns
        -------
        list[pd.DataFrame] or None
            Returns `self.summary_state.cluster_summary_list`, which contains a
            single DataFrame with the columns:

            - `question_id`
            - `question_text`
            - `cluster`
            - `summary`

            Returns `None` when existing summaries are reloaded and no new
            summarization is performed.

        Side Effects
        ------------
        Mutates:

        - `self.summary_state.cluster_summary_list`

        May reset:

        - `theme_schema_list`
        - `mapped_theme_list`
        - `populated_theme_list`
        - `orphan_list`
        - `redundancy_list`

        Persists the resulting summaries through `SummaryState.save()`.

        Notes
        -----
        Cluster summaries are not treated as analytical conclusions or final
        themes. They function as a compressed structural representation of the
        clustered insight space and provide the scaffolding used to construct
        the initial theme schema.

        Outlier cluster `-1`, when present, is summarized after all core
        clusters for a research question.

        The generated summaries remain linked to the underlying clustered
        insights through the retained cluster identifiers, preserving
        traceability between thematic synthesis and the source corpus.
        """
        
        if self.summary_state.cluster_summary_list is not None and len(self.summary_state.cluster_summary_list) > 0:
            new = None
            while new not in ["1", "2"]:
                new = input(
                    "Cluster summaries already exist on disk. Do you want to:\n"
                    "(1) reload existing summaries\n"
                    "(2) regenerate summaries (NOTE:this will overwrite existing summaries and delete all existing theme mapping, populated themes, orphans and redundancy passes; as these all derive from summaries)? \n"
                    "Enter 1 or 2:\n"
                ).lower()
            if new == "1":
                print(
                    "Summaries loaded.\n" \
                    "Summaries can be accessed via the variable.cluster_summary_list attribute of this class. There is only one item in the list thus: `variable.cluster_summary_list[0]`."
                    )
                return(None) # Return to exit the function and avoid re-running summarization
            else:
                # If we are regenerating summeries we need to delete all existing outputs and reset the attributes to none as they are loaed on init if they exist
                print("Re-running summarization of clusters...")
                # Delete all the existing outputs and set thier values to empty lists
                self.summary_state.restart(confirm="yes")
    
        # We are going to send the insights to the LLM in the order of the shortest path, so that the most similar clusters are summarized close together
        # This will add coherence to the final paper when the summaries are stitched together
        # It will also aid in the applicaion of the sliding window for summary clean up
        shortest_paths = self._estimate_shortest_path()
        
        # Create list to populate with summaries from the LLM
        summaries_dict_lst: List[dict] = []

        # Get the numbers to show progress 
        total_clusters = sum(len(shortest_paths[rq_id]["order"]) for rq_id in shortest_paths)
        count = 1

        # Loop over unique research questions
        for _, row in self.corpus_state.questions.iterrows():
            rq_id = row["question_id"]
            rq_text = row["question_text"]
            rq_df: pd.DataFrame = self.corpus_state.insights[self.corpus_state.insights["question_id"] == rq_id].copy()

            # Loop over clusters for this research question - in shortest path order
            for cluster in shortest_paths[rq_id]["order"]:
                print(f"Summarizing cluster {cluster} for research question {rq_id} (total progress: {count} of {total_clusters})...")
                count += 1
                # Skip any cases where chunks might have had no insights (and therefore no cluster)
                if pd.isna(cluster) or cluster == "NA":
                    continue

                cluster_df: pd.DataFrame = rq_df[rq_df["cluster"] == cluster]
                # get the insights, they are list of single strings. So make sure they are valid string to send to the LLM
                insights: List[str] = (
                    cluster_df
                    .apply(
                        lambda row: f'{row["insight"]} ({row["in_text_citation"]})',
                        axis=1,
                    )
                    .tolist()
                )

                # # --- Sampling step for large clusters ---
                # MAX_WORDS = 70000

                # if sum(len(i.split()) for i in insights) > MAX_WORDS:
                #     insights = utils.sample_to_word_limit(
                #         insights,
                #         max_words=MAX_WORDS,
                #         seed=config.seed
                #     )

                #     print(
                #         f"Cluster {cluster} (RQ {rq_id}): sampled {len(insights)} insights to fit word budget"
                #     )
                # #------
                
                if any(i is None for i in insights):
                    raise ValueError("Insight format error: each insight must be a string or a single-item list containing a string.")

                # Build system prompt from predefined method
                sys_prompt: str = Prompts().summarize_clusters(frozen_summary_window=frozen_summary_window)

                # Get the summaries frozen so far if there are any
                # Get only the last N summaries (local window) for frozen context
                frozen_summaries = (
                    [d["summary"] for d in summaries_dict_lst 
                    if d["question_id"] == rq_id][-frozen_summary_window:]
                    if summaries_dict_lst else []
                )

                frozen_summaries_str = "\n".join(frozen_summaries) if frozen_summaries else ""
                insights_str = "\n".join(insights)

                # Build user prompt for LLM
                user_prompt: str = (
                    f"Research question id: {rq_id}\n"
                    f"Research question text: {rq_text}\n"
                    "PRECEDING CLUSTER SUMMARIES (for context only; may be empty):\n"
                    f"{frozen_summaries_str}\n"
                    f"Cluster: {cluster}\n"
                    "INSIGHTS:\n" 
                    f"{insights_str}\n"
                )

                json_schema = {
                    "name": "cluster_summary",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "summary": {"type": "string"}
                        },
                        "required": ["summary"],
                        "additionalProperties": False
                    }
                }
                fall_back = {"question_id": rq_id, "question_text": rq_text, "cluster": cluster, "summary": ""}

                response = utils.call_chat_completion(
                    sys_prompt=sys_prompt,
                    user_prompt=user_prompt,
                    llm_client=self.llm_client,
                    ai_model=self.ai_model,
                    return_json=True,
                    json_schema=json_schema,
                    fall_back=fall_back,
                    max_tokens=4000
                )

                summary = response.get("summary")
                if not summary: 
                    print(f"Warning: No summary generated for cluster {cluster} of research question {rq_id}. Storing empty summary.")
                
                response_dict = {
                    "question_id": rq_id,
                    "question_text": rq_text,
                    "cluster": cluster,
                    "summary": summary
                }
                
                summaries_dict_lst.append(response_dict)

        # Now convert the list of dict responses to a dataframe for easier handling and saving
        summaries_df: pd.DataFrame = pd.DataFrame(summaries_dict_lst)

        print(
            f"Summaries saved here: {self.summary_save_location} and accesible via `variable.cluster_summary_list[0]`.\n"
        )

        self.summary_state.cluster_summary_list = [summaries_df]

        # Save summaries as this is LLM output we may want to reuse later - save as parquet
        os.makedirs(self.summary_save_location, exist_ok=True)
        self.summary_state.save()

        return self.summary_state.cluster_summary_list
    
    def _llm_gen_initial_schema(self, user_prompt, sys_prompt):
        """
        Generate an initial theme schema from cluster-summary narratives.

        Calls the configured language model to construct a thematic schema from
        the cluster-summary representation of a research question. The schema
        defines the conceptual categories that will be used during subsequent
        insight-to-theme mapping and theme population stages.

        The model is required to return a structured schema consisting of one
        or more themes. Each theme contains:

        - a theme label
        - a theme description
        - mapping instructions describing which insights belong in the theme

        The method enforces a strict JSON response schema and returns only the
        extracted theme definitions.

        Parameters
        ----------
        user_prompt : str
            Prompt containing the cluster-summary narrative and any additional
            context required for schema generation.

        sys_prompt : str
            System prompt defining the rules, objectives, and constraints for
            theme-schema generation.

        Returns
        -------
        list[dict]
            List of theme definitions. Each theme contains:

            - `theme_label`
            - `theme_description`
            - `instructions`

            Returns an empty list if schema generation fails or no themes are
            produced.

        Notes
        -----
        This method performs only the LLM schema-generation call. It does not
        validate, persist, or apply the resulting schema.

        The underlying JSON response also includes a `no_change` flag used by
        later iterative schema-refinement workflows. This flag is intentionally
        discarded here because the initial schema-generation stage requires only
        the theme definitions themselves.

        Errors returned by the LLM wrapper are logged and result in an empty
        theme list rather than an exception.
        """
        fall_back = {
            "themes": [],
            "no_change": False
        }
        
        if self.use_organizing_proposition:
            json_schema = {
                "name": "theme_schema_generator",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "themes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "theme_label": {
                                        "type": "string"
                                    },
                                    "theme_description": {
                                        "type": "string"
                                    },
                                    "organizing_proposition": {
                                        "type": ["string", "null"]
                                    },
                                    "instructions": {
                                        "type": "string"
                                    }
                                },
                                "required": [
                                    "theme_label",
                                    "theme_description",
                                    "organizing_proposition",
                                    "instructions"
                                ],
                                "additionalProperties": False
                            }
                        },
                        "no_change": {
                            "type": "boolean"
                        }
                    },
                    "required": [
                        "themes",
                        "no_change"
                    ],
                    "additionalProperties": False
                }
            }

        else:    
            json_schema = {
                "name": "theme_schema_generator",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "themes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "theme_label": {
                                        "type": "string"
                                    },
                                    "theme_description": {
                                        "type": "string"
                                    },
                                    "instructions": {
                                        "type": "string"
                                    }
                                },
                                "required": [
                                    "theme_label",
                                    "theme_description",
                                    "instructions"
                                ],
                                "additionalProperties": False
                            }
                        },
                        "no_change": {
                            "type": "boolean"
                        }
                    },
                    "required": [
                        "themes",
                        "no_change"
                    ],
                    "additionalProperties": False
                }
            }

        response, error = utils.call_chat_completion(
            sys_prompt=sys_prompt,
            user_prompt=user_prompt,
            llm_client=self.llm_client,
            ai_model=self.ai_model,
            fall_back=fall_back,
            return_json=True,
            json_schema=json_schema, 
            return_with_error=True
        )

        if error:
            print(f"Error during LLM schema generation: {error}. Returning empty theme list.")

        themes = response.get("themes", [])
        return themes
    
    def _llm_gen_schema_repair_plan(self, user_prompt, sys_prompt):
        """
        Generate a structured repair plan for an overloaded or incomplete theme schema.

        Calls the configured language model to produce a schema-repair plan when
        theme population, orphan reinsertion, or completeness checks indicate
        that the current thematic structure is failing to adequately represent
        the underlying insight set.

        The generated repair plan identifies:

        - themes that failed completeness checks
        - concepts contributing most to representational overload
        - concepts that should be extracted into new themes
        - concepts that should be moved to existing themes
        - proposed scope changes for affected themes
        - schema-level adjustments required to improve coverage

        The repair plan functions as an intermediate planning artifact rather
        than a direct schema modification. Downstream methods use the plan to
        construct revised theme schemas while maintaining an inspectable record
        of why changes were proposed.

        Parameters
        ----------
        user_prompt : str
            Prompt containing the current schema, failed themes, orphan
            information, completeness diagnostics, and any other context
            required for schema repair.

        sys_prompt : str
            System prompt defining the objectives and constraints of the repair
            process.

        Returns
        -------
        dict
            Structured repair plan containing:

            - `theme_repairs`
            - `schema_repairs`

            If repair generation fails, returns an empty repair plan of the form:

            {
                "theme_repairs": [],
                "schema_repairs": []
            }

        Notes
        -----
        This method corresponds to the schema-repair stage described in the
        ReadingMachine methodology, where overloaded themes are decomposed into
        smaller conceptual units before later optimization and recombination.

        The repair plan is intentionally separated from schema implementation.
        This separation supports inspectability and reduces the risk of
        performative repair, where a model reports changes without actually
        modifying the thematic structure.

        The returned object is a planning artifact only. It does not directly
        modify SummaryState or any stored theme schema.
        """
        fall_back = {
            "repair_plan": {
                "theme_repairs": [],
                "schema_repairs": []
            },
        } 

        if self.use_organizing_proposition:
            json_schema = {
                "name": "theme_schema_repair_plan",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "repair_plan": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "theme_repairs": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "properties": {
                                            "source_theme_id": {
                                                "type": "integer"
                                            },
                                            "source_theme_label": {
                                                "type": "string"
                                            },
                                            "completeness_check": {
                                                "type": "string",
                                                "enum": ["fail"]
                                            },
                                            "concepts_ranked_by_representational_load": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "additionalProperties": False,
                                                    "properties": {
                                                        "concept": {
                                                            "type": "string"
                                                        },
                                                        "estimated_load": {
                                                            "type": "string",
                                                            "enum": [
                                                                "high",
                                                                "medium",
                                                                "low"
                                                            ]
                                                        },
                                                        "evidence_from_summary_or_failed_batches": {
                                                            "type": "string"
                                                        },
                                                        "independently_synthesizable": {
                                                            "type": "boolean"
                                                        }
                                                    },
                                                    "required": [
                                                        "concept",
                                                        "estimated_load",
                                                        "evidence_from_summary_or_failed_batches",
                                                        "independently_synthesizable"
                                                    ]
                                                }
                                            },
                                            "extractions": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "additionalProperties": False,
                                                    "properties": {
                                                        "concept": {
                                                            "type": "string"
                                                        },
                                                        "action": {
                                                            "type": "string",
                                                            "enum": [
                                                                "new_theme",
                                                                "move_to_existing_theme"
                                                            ]
                                                        },
                                                        "target_theme_id": {
                                                            "type": [
                                                                "integer",
                                                                "null"
                                                            ]
                                                        },
                                                        "new_theme_label": {
                                                            "type": [
                                                                "string",
                                                                "null"
                                                            ]
                                                        },
                                                        "new_theme_core_scope": {
                                                            "type": [
                                                                "string",
                                                                "null"
                                                            ]
                                                        },
                                                        "new_theme_organizing_proposition": {
                                                            "type": [
                                                                "string",
                                                                "null"
                                                            ]
                                                        },
                                                        "new_theme_inclusions": {
                                                            "type": "array",
                                                            "items": {
                                                                "type": "string"
                                                            }
                                                        },
                                                        "new_theme_exclusions": {
                                                            "type": "array",
                                                            "items": {
                                                                "type": "string"
                                                            }
                                                        },
                                                        "receiving_theme_scope_update": {
                                                            "type": [
                                                                "string",
                                                                "null"
                                                            ]
                                                        },
                                                        "receiving_theme_organizing_proposition_update": {
                                                            "type": [
                                                                "string",
                                                                "null"
                                                            ]
                                                        },
                                                        "reason": {
                                                            "type": "string"
                                                        }
                                                    },
                                                    "required": [
                                                        "concept",
                                                        "action",
                                                        "target_theme_id",
                                                        "new_theme_label",
                                                        "new_theme_core_scope",
                                                        "new_theme_organizing_proposition",
                                                        "new_theme_inclusions",
                                                        "new_theme_exclusions",
                                                        "receiving_theme_scope_update",
                                                        "receiving_theme_organizing_proposition_update",
                                                        "reason"
                                                    ]
                                                }
                                            },
                                            "source_theme_resolution": {
                                                "type": "object",
                                                "additionalProperties": False,
                                                "properties": {
                                                    "outcome": {
                                                        "type": "string",
                                                        "enum": [
                                                            "rename_and_narrow",
                                                            "dissolve_and_reallocate"
                                                        ]
                                                    },
                                                    "residual_label": {
                                                        "type": [
                                                            "string",
                                                            "null"
                                                        ]
                                                    },
                                                    "residual_core_scope": {
                                                        "type": [
                                                            "string",
                                                            "null"
                                                        ]
                                                    },
                                                    "residual_organizing_proposition": {
                                                        "type": [
                                                            "string",
                                                            "null"
                                                        ]
                                                    },
                                                    "residual_inclusions": {
                                                        "type": "array",
                                                        "items": {
                                                            "type": "string"
                                                        }
                                                    },
                                                    "residual_exclusions": {
                                                        "type": "array",
                                                        "items": {
                                                            "type": "string"
                                                        }
                                                    },
                                                    "residual_expected_to_pass": {
                                                        "type": "boolean"
                                                    },
                                                    "dissolution_reason": {
                                                        "type": [
                                                            "string",
                                                            "null"
                                                        ]
                                                    }
                                                },
                                                "required": [
                                                    "outcome",
                                                    "residual_label",
                                                    "residual_core_scope",
                                                    "residual_organizing_proposition",
                                                    "residual_inclusions",
                                                    "residual_exclusions",
                                                    "residual_expected_to_pass",
                                                    "dissolution_reason"
                                                ]
                                            },
                                            "repair_narrative": {
                                                "type": "string"
                                            }
                                        },
                                        "required": [
                                            "source_theme_id",
                                            "source_theme_label",
                                            "completeness_check",
                                            "concepts_ranked_by_representational_load",
                                            "extractions",
                                            "source_theme_resolution",
                                            "repair_narrative"
                                        ]
                                    }
                                },
                                "schema_repairs": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "properties": {
                                            "affected_theme_ids": {
                                                "type": "array",
                                                "items": {
                                                    "type": "integer"
                                                }
                                            },
                                            "repair_narrative": {
                                                "type": "string"
                                            }
                                        },
                                        "required": [
                                            "affected_theme_ids",
                                            "repair_narrative"
                                        ]
                                    }
                                }
                            },
                            "required": [
                                "theme_repairs",
                                "schema_repairs"
                            ]
                        }
                    },
                    "required": [
                        "repair_plan"
                    ]
                }
            }

        else:
            json_schema = {
                "name": "theme_schema_repair_plan",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "repair_plan": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "theme_repairs": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "properties": {
                                            "source_theme_id": {"type": "integer"},
                                            "source_theme_label": {"type": "string"},
                                            "completeness_check": {
                                                "type": "string",
                                                "enum": ["fail"]
                                            },
                                            "concepts_ranked_by_representational_load": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "additionalProperties": False,
                                                    "properties": {
                                                        "concept": {"type": "string"},
                                                        "estimated_load": {
                                                            "type": "string",
                                                            "enum": ["high", "medium", "low"]
                                                        },
                                                        "evidence_from_summary_or_failed_batches": {"type": "string"},
                                                        "independently_synthesizable": {"type": "boolean"}
                                                    },
                                                    "required": [
                                                        "concept",
                                                        "estimated_load",
                                                        "evidence_from_summary_or_failed_batches",
                                                        "independently_synthesizable"
                                                    ]
                                                }
                                            },
                                            "extractions": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "additionalProperties": False,
                                                    "properties": {
                                                        "concept": {"type": "string"},
                                                        "action": {
                                                            "type": "string",
                                                            "enum": ["new_theme", "move_to_existing_theme"]
                                                        },
                                                        "target_theme_id": {
                                                            "type": ["integer", "null"]
                                                        },
                                                        "new_theme_label": {
                                                            "type": ["string", "null"]
                                                        },
                                                        "new_theme_core_scope": {
                                                            "type": ["string", "null"]
                                                        },
                                                        "new_theme_inclusions": {
                                                            "type": "array",
                                                            "items": {"type": "string"}
                                                        },
                                                        "new_theme_exclusions": {
                                                            "type": "array",
                                                            "items": {"type": "string"}
                                                        },
                                                        "receiving_theme_scope_update": {
                                                            "type": ["string", "null"]
                                                        },
                                                        "reason": {"type": "string"}
                                                    },
                                                    "required": [
                                                        "concept",
                                                        "action",
                                                        "target_theme_id",
                                                        "new_theme_label",
                                                        "new_theme_core_scope",
                                                        "new_theme_inclusions",
                                                        "new_theme_exclusions",
                                                        "receiving_theme_scope_update",
                                                        "reason"
                                                    ]
                                                }
                                            },
                                            "source_theme_resolution": {
                                                "type": "object",
                                                "additionalProperties": False,
                                                "properties": {
                                                    "outcome": {
                                                        "type": "string",
                                                        "enum": ["rename_and_narrow", "dissolve_and_reallocate"]
                                                    },
                                                    "residual_label": {
                                                        "type": ["string", "null"]
                                                    },
                                                    "residual_core_scope": {
                                                        "type": ["string", "null"]
                                                    },
                                                    "residual_inclusions": {
                                                        "type": "array",
                                                        "items": {"type": "string"}
                                                    },
                                                    "residual_exclusions": {
                                                        "type": "array",
                                                        "items": {"type": "string"}
                                                    },
                                                    "residual_expected_to_pass": {"type": "boolean"},
                                                    "dissolution_reason": {
                                                        "type": ["string", "null"]
                                                    }
                                                },
                                                "required": [
                                                    "outcome",
                                                    "residual_label",
                                                    "residual_core_scope",
                                                    "residual_inclusions",
                                                    "residual_exclusions",
                                                    "residual_expected_to_pass",
                                                    "dissolution_reason"
                                                ]
                                            },
                                            "repair_narrative": {"type": "string"}
                                        },
                                        "required": [
                                            "source_theme_id",
                                            "source_theme_label",
                                            "completeness_check",
                                            "concepts_ranked_by_representational_load",
                                            "extractions",
                                            "source_theme_resolution",
                                            "repair_narrative"
                                        ]
                                    }
                                },
                                "schema_repairs": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "properties": {
                                            "affected_theme_ids": {
                                                "type": "array",
                                                "items": {"type": "integer"}
                                            },
                                            "repair_narrative": {"type": "string"}
                                        },
                                        "required": [
                                            "affected_theme_ids",
                                            "repair_narrative"
                                        ]
                                    }
                                }
                            },
                            "required": [
                                "theme_repairs",
                                "schema_repairs"
                            ]
                        }
                    },
                    "required": ["repair_plan"]
                }
            }

        # Generate the repair instructions for this schema
        response = utils.call_chat_completion(
            sys_prompt=sys_prompt,
            user_prompt=user_prompt,
            llm_client=self.llm_client,
            ai_model=self.ai_model,
            fall_back=fall_back,
            return_json=True,
            json_schema=json_schema,
        )


        repair_plan = response.get("repair_plan", {
            "theme_repairs": [],
            "schema_repairs": []
        })

        return repair_plan

    def _llm_apply_schema_repair_plan(self, unstable_schema_rq, sys_prompt, user_prompt):
        """
        Apply a schema-repair plan to an unstable question-specific schema.

        Calls the configured language model to implement a previously generated
        schema-repair plan against the current unstable schema for a single
        research question. The method returns a revised list of theme definitions
        using the standard theme-schema structure required by downstream mapping
        and population stages.

        Parameters
        ----------
        unstable_schema_rq : pd.DataFrame
            Current unstable theme schema for one research question. Used to build
            the fallback schema returned if the LLM call fails.

        sys_prompt : str
            System prompt defining the constraints for applying the repair plan.

        user_prompt : str
            Prompt containing the unstable schema and the repair plan to apply.

        Returns
        -------
        list[dict]
            Revised theme schema containing dictionaries with:

            - `theme_label`
            - `theme_description`
            - `instructions`

            If the LLM call fails or returns invalid JSON, returns the existing
            unstable schema converted to this dictionary format.

        Notes
        -----
        This method implements the repair plan only. It does not optimize the
        resulting schema, assign theme IDs, persist the schema, or mutate
        SummaryState.

        Separating repair-plan generation from repair-plan implementation helps
        reduce performative repair by requiring the model first to diagnose
        structural problems and then to apply those repairs in a separate call.
        """
        
        if self.use_organizing_proposition:
            fall_back = {
                "themes": unstable_schema_rq[
                    [
                        "theme_label",
                        "theme_description",
                        "organizing_proposition",
                        "instructions",
                    ]
                ].to_dict(orient="records")
            }

        else:
            fall_back = {
                "themes": unstable_schema_rq[
                    ["theme_label", "theme_description", "instructions"]
                ].to_dict(orient="records")
            }
        
        if self.use_organizing_proposition:
            json_schema = {
                "name": "theme_schema_repair_implementer",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "themes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "theme_label": {
                                        "type": "string"
                                    },
                                    "theme_description": {
                                        "type": "string"
                                    },
                                    "organizing_proposition": {
                                        "type": ["string", "null"]
                                    },
                                    "instructions": {
                                        "type": "string"
                                    }
                                },
                                "required": [
                                    "theme_label",
                                    "theme_description",
                                    "organizing_proposition",
                                    "instructions"
                                ],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": [
                        "themes"
                    ],
                    "additionalProperties": False
                }
            }

        else:
            json_schema = {
                "name": "theme_schema_repair_implementer",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "themes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "theme_label": {"type": "string"},
                                    "theme_description": {"type": "string"},
                                    "instructions": {"type": "string"}
                                },
                                "required": [
                                    "theme_label",
                                    "theme_description",
                                    "instructions"
                                ],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["themes"],
                    "additionalProperties": False
                }
            }

        response = utils.call_chat_completion(
            sys_prompt=sys_prompt,
            user_prompt=user_prompt,
            llm_client=self.llm_client,
            ai_model=self.ai_model,
            fall_back=fall_back,
            return_json=True,
            json_schema=json_schema
        )

        themes = response.get("themes", [])
        
        return themes

    def _llm_apply_schema_optimization(self, sys_prompt, user_prompt):
        """
        Optimize a repaired theme schema without reintroducing overload.

        Calls the configured language model to review a repaired schema and make
        conservative improvements to theme coherence, boundaries, and organization.
        The optimizer may return a revised schema or indicate that no improvement is
        needed.

        Parameters
        ----------
        sys_prompt : str
            System prompt defining the optimization rules and constraints.

        user_prompt : str
            Prompt containing the repaired schema and any context needed to assess
            whether optimization is safe.

        Returns
        -------
        list[dict] or str
            Returns a list of optimized theme definitions when changes are proposed.
            Each theme dictionary contains:

            - `theme_label`
            - `theme_description`
            - `instructions`

            Returns the string `"no change"` when the optimizer indicates that the
            existing schema should be retained.

        Notes
        -----
        Optimization is intentionally separate from decomposition and repair. The
        repair step prioritizes resolving overloaded or incomplete themes; the
        optimization step may then improve conceptual coherence only when doing so
        does not recreate the representational overload that caused earlier
        failures.

        This method performs only the LLM optimization call. It does not assign
        theme IDs, persist outputs, or mutate SummaryState.
        """
        fallback_optimizer_response = {
            "no_change": True,
            "themes": []
        }

        json_schema = {
            "name": "theme_schema_optimizer",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "no_change": {
                        "type": "boolean"
                    },
                    "themes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "theme_label": {"type": "string"},
                                "theme_description": {"type": "string"},
                                "instructions": {"type": "string"}
                            },
                            "required": [
                                "theme_label",
                                "theme_description",
                                "instructions"
                            ],
                            "additionalProperties": False
                        }
                    }
                },
                "required": [
                    "no_change",
                    "themes"
                ],
                "additionalProperties": False
            }
        }

        response = utils.call_chat_completion(
            sys_prompt=sys_prompt,
            user_prompt=user_prompt,
            llm_client=self.llm_client,
            ai_model=self.ai_model,
            fall_back=fallback_optimizer_response,
            return_json=True,
            json_schema=json_schema
        )

        optimized_schema = response.get("themes", [])

        if response.get("no_change", False):
            print("Optimizer indicated no change needed. Retaining existing schema.")
            return("no change")

        else:
            optimized_schema = response.get("themes", [])
        return optimized_schema


    def _run_llm_schema_gen(self, source: str) -> pd.DataFrame:
        """
        Generate, repair, optimize, and stabilize a thematic schema.

        This method is the central schema-generation and schema-refinement
        orchestrator within the ReadingMachine synthesis workflow. It is used both
        for initial theme-schema generation and for subsequent schema revisions
        triggered by theme-population diagnostics.

        Two operating modes are supported:

            "cluster summaries"
                Generates an initial theme schema from the cluster-summary
                narrative produced during cluster summarization.

            "populated themes"
                Revises an existing schema using evidence generated during
                theme population, completeness checking, orphan handling,
                and schema evaluation.

        Initial Schema Generation
        -------------------------
        When operating on cluster summaries, the method generates an initial
        theme schema independently for each research question.

        For each question:

            cluster summaries
                ↓
            LLM schema generation
                ↓
            initial theme schema

        The generated schema consists of:

            - theme labels
            - theme descriptions
            - mapping instructions

        All generated themes are marked as:

            stable = False
            optimized = False
            needs_repair = True

        because they have not yet undergone empirical testing through the
        mapping and population stages.

        Schema Refinement
        -----------------
        When operating on populated themes, the method performs iterative
        schema stabilization.

        Questions are divided into:

            stable schemas
            unstable schemas

        Stable schemas are carried forward unchanged.

        Only unstable schemas are processed further.

        For each unstable question, one of two pathways is followed:

        1. Schema Repair
        ----------------
        If the schema has been flagged as requiring repair
        (`needs_repair == True`):

            schema history
                ↓
            repair-plan generation
                ↓
            repair-plan implementation
                ↓
            revised schema

        The repair workflow uses:

            - previous schema iterations
            - completeness-check failures
            - failed summary batches
            - representational overload signals

        to identify concepts that should be:

            - extracted into new themes
            - reassigned to existing themes
            - used to redefine theme boundaries

        2. Schema Optimization
        ----------------------
        If the schema does not require repair:

            schema history
                ↓
            optimization review
                ↓
            optimized schema

        The optimizer may either:

            - propose schema improvements
            - indicate that no further changes are required

        Schemas receiving a "no change" decision are marked as stable and are
        excluded from future refinement iterations.

        Convergence Logic
        -----------------
        Schema stabilization is tracked at the research-question level.

        A question is considered converged when the optimizer indicates that no
        further modifications are necessary.

        When all research questions are stable, the method terminates and returns
        `None`, signalling that schema development has converged and that
        downstream synthesis can proceed directly to redundancy handling and
        rendering.

        Parameters
        ----------
        source : str
            Source representation used to generate or refine the schema.

            Must be one of:

            - `"cluster summaries"`
            - `"populated themes"`

        Returns
        -------
        pd.DataFrame or None
            Updated theme schema containing:

            - `theme_id`
            - `theme_label`
            - `theme_description`
            - `instructions`
            - `question_id`
            - `question_text`
            - `needs_repair`
            - `optimized`
            - `stable`
            - `schema_produced_by`

            Returns `None` when all research questions have reached schema
            stability.

        Side Effects
        ------------
        Mutates:

        - `self.last_schema_repair_theme_repairs`
        - `self.last_schema_repair_schema_repairs`

        Reads from:

        - `self.summary_state.cluster_summary_list`
        - `self.summary_state.theme_schema_list`
        - `self.summary_state.populated_theme_list`

        Uses and updates schema-state metadata including:

        - `stable`
        - `optimized`
        - `needs_repair`

        Notes
        -----
        Theme IDs are regenerated on every schema iteration to maintain a
        contiguous global numbering scheme.

        Schema repair and schema optimization are intentionally separated into
        distinct LLM operations. Repair addresses representational failures and
        coverage problems, while optimization focuses on improving thematic
        coherence once representational adequacy has been established.

        This method does not persist the generated schema directly. Persistence
        is handled by the higher-level schema-generation workflow that appends
        the resulting DataFrame to `SummaryState.theme_schema_list`.

        The method assumes that stability is evaluated at the research-question
        level and that all themes belonging to a question share the same
        stability status during a given iteration.
        """
        if source not in ["cluster summaries", "populated themes"]:
            raise ValueError("Invalid source for theme schema generation. Source must be either 'cluster summaries' or 'populated themes'.")
        
        #update the theme_schema_list with values from populated themes so that they are avialble here for conditional flow
        if source == "populated themes" and self.summary_state.populated_theme_list:
            status_cols = ["needs_repair", "optimized", "stable"]

            self.summary_state.theme_schema_list[-1] = (
                self.summary_state.theme_schema_list[-1]
                .drop(columns=status_cols, errors="ignore") # Drop these in case they were created in a partial pass previously and will result in name dediup x_, y_ upon merge
                .merge(
                    self.summary_state.populated_theme_list[-1][
                        ["theme_label", "question_id"] + status_cols
                    ],
                    how="left",
                    on=["question_id", "theme_label"]
                )
            )

        out_df_list = []
        self.last_schema_repair_theme_repairs = []
        self.last_schema_repair_schema_repairs = [] 
        no_change_count = self.summary_state.theme_schema_list[-1][self.summary_state.theme_schema_list[-1]["stable"] == True]["question_id"].nunique() if self.summary_state.theme_schema_list else 0
        
        # initialize the primary dataframes for the two input branches:
        if source == "cluster summaries":
            # Grab data from summaries
            source_df = self.summary_state.cluster_summary_list[0].copy()
            stable_schema = None
            for idx, row in self.corpus_state.questions.iterrows():
                print(f"Generating theme schema for question {row['question_id']} (total: {idx + 1} of {len(self.corpus_state.questions)})...")
                question_id = row["question_id"]
                question_text = row["question_text"]

                # Use source df and get the clusters for this rq
                rq_df = source_df[source_df["question_id"] == question_id].copy()
                summary = "\n\n".join(rq_df["summary"].tolist())

                # Generate user and sys prompt
                user_prompt = (
                    f"Research Question: {question_text}\n"
                    "TEXT TO ANALYZE:\n"
                    f"{summary}\n"
                )
                sys_prompt = Prompts().gen_theme_schema_cluster_source(provide_organizing_proposition=self.use_organizing_proposition)
                # Get the initial schema for this question from the LLM

                theme_list = self._llm_gen_initial_schema(user_prompt, sys_prompt)
                themes_df = pd.DataFrame(theme_list)
            
                # Add the metadata back to the results
                themes_df["needs_repair"] = True
                themes_df[["optimized", "stable"]] = False # These are generated from the cluster summaries so that need to be checked for repairs and are not optimized or stable yet           
                themes_df["question_id"] = question_id
                themes_df["question_text"] = question_text
                themes_df["schema_produced_by"] = "initial_cluster_schema"

                out_df_list.append(themes_df)

        else:
            # If its not cluster summaries then its from orphans so first we prepare for schema repair
            # First identify any viable and unviable schema
            stable_schema = self.summary_state.theme_schema_list[-1][self.summary_state.theme_schema_list[-1]["stable"]]
            unstable_schema = self.summary_state.theme_schema_list[-1][~self.summary_state.theme_schema_list[-1]["stable"]]
            
            for idx, row in self.corpus_state.questions.iterrows():
                question_id = row["question_id"]
                question_text = row["question_text"]
           
                # First check whether the question is unstable - if not we can skip and add to the output df as is, if it is unstable we need to send to the LLM for revision
                unstable_schema_rq = unstable_schema[unstable_schema["question_id"] == question_id].copy()
                if unstable_schema_rq.empty:
                    stable_schema_rq = stable_schema[stable_schema["question_id"] == question_id].copy()
                    stable_output_columns = [
                        "theme_label",
                        "theme_description",
                    ]
                    # Add the organizing proposition if its selected
                    if self.use_organizing_proposition:
                        stable_output_columns.append("organizing_proposition")

                    stable_output_columns.extend(
                        [
                            "instructions",
                            "question_id",
                            "question_text",
                            "stable",
                            "needs_repair",
                            "optimized",
                        ]
                    )

                    out_df_list.append(stable_schema_rq[stable_output_columns])
                    continue

                else:
                    # If its not stable we generate the full history of schema iterations for this question as this will be used in the repair plan and optimization prompts, and filtered for the last iteration in implement repair plan prompt
                    print(f"Generating theme schema for question {row['question_id']} (total: {idx + 1} of {len(self.corpus_state.questions)})...")
                    
                    # Generate the full history of the theme summaries and schema rules for this question so that i can pass it to the model
                    full_history = []

                    for i, (s, p) in enumerate(zip(self.summary_state.theme_schema_list, self.summary_state.populated_theme_list)):
                        merged_schema_pop_df = (
                            s[s["question_id"] == question_id]
                            .merge(p[["thematic_summary", "theme_id", "question_id"]], 
                                    how ="left", 
                                    on=["question_id", "theme_id"])
                            .assign(completeness_check=lambda x: x["thematic_summary"].apply(lambda y: "fail" if pd.notna(y) and "--- FAILED BATCH SUMMARIES ---" in y else "pass"))
                            .assign(iteration=i)
                            .assign(word_count=lambda x: x["thematic_summary"].str.split("--- FAILED BATCH SUMMARIES ---").str[0].str.split().str.len().fillna(0).astype(int))
                            .assign(word_count=lambda x: np.where(x["thematic_summary"].str.contains("--- FAILED BATCH SUMMARIES ---", na=False), None, x["word_count"]))
                            .assign(schema_has_failures=lambda x: (x["completeness_check"] == "fail").any())
                            .assign(is_current_iteration=lambda x: x["iteration"] == len(self.summary_state.theme_schema_list) - 1)
                        )
                        full_history.append(merged_schema_pop_df)

                    full_history_df = pd.concat(full_history, ignore_index=True)

                    # Avoid invalid JSON NaN values
                    full_history_df = full_history_df.where(pd.notna(full_history_df), None)

                    # Conditonally make the json based on whether we are passing the organizing proposition or not
                    history_theme_columns = [
                        "theme_id",
                        "theme_label",
                        "theme_description",
                    ]

                    if self.use_organizing_proposition:
                        history_theme_columns.append("organizing_proposition")

                    history_theme_columns.extend(
                        [
                            "instructions",
                            "completeness_check",
                            "word_count",
                            "thematic_summary",
                        ]
                    )

                    full_history_by_iteration_dict = {
                        str(iteration): {
                            "iteration": int(iteration),
                            "is_current_iteration": bool(group["is_current_iteration"].iloc[0]),
                            "schema_has_failures": bool(group["schema_has_failures"].iloc[0]),
                            "themes": group[history_theme_columns].to_dict(orient="records"),
                        }
                        for iteration, group in full_history_df.sort_values(
                            ["iteration", "theme_id"]
                        ).groupby("iteration", sort=True)
                    }

                    # Now we route
                    # 1. if the question needs repair it goes to the schema repair process
                    # 2. if the question does not need repairs it goes to optimization

                    # First check is the schema is stable for this question, if so move to next question
                    needs_repair = unstable_schema_rq["needs_repair"].fillna(False).astype(bool).any()
                    if needs_repair:
                        print("Schema for this question has been marked as needing repairs based on the completeness check and word count signals. Running repair process...")

                        # Generate the content for the repair plan prompt
                        # Turn full history into json for the prompt
                        full_history_json = json.dumps(
                                full_history_by_iteration_dict,
                                ensure_ascii=False,
                                indent=2,
                                allow_nan=False,
                            )
                            
                        # generate the repair instructions for this schema
                        user_prompt_gen_repair = (
                            f"RESEARCH QUESTION: {question_text}\n\n"
                            "-------------------------------------------------------------\n\n"
                            "HISTORIC EFFORTS AT SCHEMA DEVELOPMENT:\n"
                            f"{full_history_json}\n\n"
                            "-------------------------------------------------------------\n\n"
                        )

                        sys_prompt_gen_repair = Prompts().gen_theme_schema_repair_instructions(provide_organizing_proposition=self.use_organizing_proposition)

                        # Get the repair plan
                        repair_plan = self._llm_gen_schema_repair_plan(user_prompt_gen_repair, sys_prompt_gen_repair)

                        if repair_plan.get("theme_repairs") == [] and repair_plan.get("schema_repairs") == []:
                            print("LLM did not propose any repairs for this question. Reusing old schema and marking as unstable, not optimized and needs repair.")
                            themes_df = unstable_schema_rq.copy()
                            themes_df["stable"] = False # Set the stable flag to true for all themes in this question as the model has indicated that there is no need to change the schema and therefore they are stable now
                            themes_df["needs_repair"] = True # Assuming error so repair did not happen therefor needs repair stays True
                            themes_df["optimized"] = False # If there are no repairs then they have not been optimized 
                            themes_df["schema_produced_by"] = "repair"
                            themes_df["question_id"] = question_id
                            themes_df["question_text"] = question_text
                            # Append to the out_df_list
                            out_df_list.append(themes_df)
                            continue

                        # Assign the repair plans as attributes so that i can see what they are proposing for debugging
                        theme_repairs = repair_plan.get("theme_repairs", [])
                        schema_repairs = repair_plan.get("schema_repairs", [])
                        theme_repairs_df = pd.DataFrame(theme_repairs)
                        schema_repairs_df = pd.DataFrame(schema_repairs)
                        theme_repairs_df["question_id"] = question_id
                        schema_repairs_df["question_id"] = question_id
                        self.last_schema_repair_theme_repairs.append(theme_repairs_df)
                        self.last_schema_repair_schema_repairs.append(schema_repairs_df)

                        # Now implement the plan
                        print("Implementing repair plan...")
                        # get the repair plan as json for the LLM
                        repair_plan_json = json.dumps(repair_plan, indent=2, ensure_ascii=False)

                        # We send the implement repair prompt the latest schema iteraton with all the information so get the last iteration
                        last_iteration = full_history_df["iteration"].max()
                        full_history_last_iteration = full_history_by_iteration_dict[str(last_iteration)]
                        # Convert full_history_last_iteration to json for the prompt
                        full_history_last_iteration_json = json.dumps(
                            full_history_last_iteration,
                                ensure_ascii=False,
                                indent=2,
                                allow_nan=False,
                        )

                        # Create all the prompts
                        user_prompt = (
                            f"RESEARCH QUESTION: {question_text}\n\n"
                            "-------------------------------------------------------------\n\n"
                            "CURRENT UNSTABLE SCHEMA:\n"
                            f"{full_history_last_iteration_json}\n\n"
                            "-------------------------------------------------------------\n\n"
                            "REPAIR PLAN:\n"
                            f"{repair_plan_json}\n\n"
                        )
                        # Then the sys prompt
                        sys_prompt = Prompts().implement_schema_repairs(provide_organizing_proposition=self.use_organizing_proposition)

                        # Get the repaired themes from the LLM
                        theme_list = self._llm_apply_schema_repair_plan(
                            unstable_schema_rq=unstable_schema_rq, 
                            sys_prompt=sys_prompt, 
                            user_prompt=user_prompt
                            )
                        # Convert the repaired theme list to a dataframe
                        themes_df = pd.DataFrame(theme_list)
                        # Set the needs_repair flag to false for all themes in this question as the model has undertaken repairs and therefore they are not viable yet - vability will be set for this update after orphan insertion
                        themes_df["needs_repair"] = pd.NA # We dont know whether they are viable or not until we test them so set to NA for now
                        themes_df["optimized"] = False # If there are repairs then they have not been optimized 
                        themes_df["stable"] = False # If they are not optimized they are not stable

                        # Now add the metadata back in for the themes for this question - covering both now stable and unstable themes
                        themes_df["question_id"] = question_id
                        themes_df["question_text"] = question_text
                        themes_df["schema_produced_by"] = "repair"
                        
                        # Append to the final list
                        out_df_list.append(themes_df)

                    else: # now checking if it does not need repairs, and it was not stable it must need optimizing
                        print("Schema for this question does not need repairs, sending for optimization...")
                        # generate the user prompt for schema optimization
                        # First get the full history
                        full_history_json = json.dumps(
                                full_history_by_iteration_dict,
                                ensure_ascii=False,
                                indent=2,
                                allow_nan=False,
                            )
                        
                        user_prompt = (
                            f"RESEARCH QUESTION: {question_text}\n\n"
                            f"SCHEMA HISTORY: {full_history_json}\n\n"
                        )

                        sys_prompt = Prompts().gen_theme_schema_optimize(provide_organizing_proposition=self.use_organizing_proposition)

                        optimized_schema = self._llm_apply_schema_optimization(sys_prompt=sys_prompt, user_prompt=user_prompt)

                        if optimized_schema == "no change": # If we get back no change we need to set to stable, and use the existing schema for this question
                            print("No changes proposed by optimizer. Marking schema as stable for this question.")
                            themes_df = unstable_schema_rq.copy()
                            themes_df["stable"] = True # Set the stable flag to true for all themes in this question as the model has indicated that there is no need to change the schema and therefore they are stable now
                            themes_df["optimized"] = True # If there is no change then they are optimized and therefore stable
                            themes_df["schema_produced_by"] = "optimizer_no_change"
                            themes_df["question_id"] = question_id
                            themes_df["question_text"] = question_text
                            # Append to the out_df_list
                            out_df_list.append(themes_df)
                            # Increment the no change count
                            no_change_count += 1
                            # Check if no change count equals the total number of research questions, if so end the stabilization iterations
                            if no_change_count == self.corpus_state.questions.shape[0]:
                                print(
                                    "This iteration has not made any changes to the schema for any of the research questions.\n"
                                    "This means there are no errors in your populated themes and no obvious optimiaztion options for the schema to improve the mapping of insights to themes.\n"
                                    "You should consider iterations done.\n"
                                    "The final populated theme list is available in `self.summary_state.populated_theme_list[-1]`.\n"
                                    "You should move to redundancy handling/rendering."
                                )
                                return(None)
                        else:
                            themes = optimized_schema
                            themes_df = pd.DataFrame(themes)
                            themes_df["optimized"] = False # Set the optimized flag to false for all themes in this question as the model has undertaken optimizations and these need to be tested
                            themes_df["stable"] = False # Set the stable flag to false for all themes in this question as the model has undertaken optimizations and therefore we need to test wh
                            themes_df["needs_repair"] = pd.NA # We dont know whether they need repairs or not until we test them so set to NA for now
                            themes_df["schema_produced_by"] = "optimizer_change"
                            themes_df["question_id"] = question_id
                            themes_df["question_text"] = question_text
                            # Append to the out_df_list
                            out_df_list.append(themes_df)

        # Now we have to concat the repaired/optimized/stable themes 
        # Concat all the questions
        output = pd.concat(out_df_list, ignore_index=True, sort=False)
        # We want to sort the themes by questions and the order they came in. So first we add a theme id
        output["theme_id"] = [i + 1 for i in range(len(output))]
        # Sort the output
        output = output.sort_values(by=["question_id", "theme_id"], ignore_index=True)
        # Then we re-sort so that they run from theme 1 up, and are global
        # NOTE THIS IS A CENTRAL CONDITION. FOR THIS REASON THERE IS A FLAGGING FUNCTION AT LOAD AND SAVE WHICH COMPLAINS TO THE USER IF SOME CHANGE TO THE CODE HAS RESULTED IN theme_id NOT BEING AN INT.       
        output["theme_id"] = [i + 1 for i in range(len(output))]
        #Drop the columns that i added to schema for the LLM to use but don't want otherwise subsequent processing will generate column name conflicts _y, and _x
        output = output.drop(columns=["thematic_summary", "completeness_check", "word_count"], errors="ignore")
        return(output)

    def gen_theme_schema(self, force: bool = False) -> pd.DataFrame:
        """
        Generate, reload, append, or regenerate a theme-schema pass.

        Public entry point for theme-schema generation in the summarization
        workflow. This method manages sequencing around schema creation and
        refinement, while delegating the actual LLM-based schema generation,
        repair, optimization, and stabilization logic to `_run_llm_schema_gen()`.

        The theme schema defines the conceptual categories and mapping
        instructions used to assign insights to themes and later synthesize
        theme-level summaries.

        Operating Modes
        ---------------
        Initial schema generation
            If no schema exists, generate the first schema from the cluster
            summaries stored in `summary_state.cluster_summary_list`.

        Iterative refinement
            If a schema already exists, optionally generate a new schema pass
            from populated-theme outputs after the current mapping, population,
            and orphan-handling cycle has been completed.

        Latest-pass regeneration
            Replace the most recent schema pass and clear downstream artifacts
            that depend on it.

        Forced generation
            If `force=True`, bypass normal sequencing checks and append a new
            schema pass generated from cluster summaries. This mode is intended
            for development and can leave SummaryState internally inconsistent.

        Parameters
        ----------
        force : bool, default=False
            If True, bypass sequencing checks and append a schema generated from
            cluster summaries. Must be a boolean.

        Returns
        -------
        pd.DataFrame or None
            Newly generated schema DataFrame, or None when:

            - the user chooses to load the latest existing schema
            - schema refinement reaches convergence and no new schema is produced

            Generated schemas typically contain:

            - `theme_id`
            - `theme_label`
            - `theme_description`
            - `instructions`
            - `question_id`
            - `question_text`
            - `needs_repair`
            - `optimized`
            - `stable`
            - `schema_produced_by`

        Raises
        ------
        ValueError
            If `force` is not a boolean.

        ValueError
            If cluster summaries have not yet been generated.

        ValueError
            If the user attempts to generate a new refinement pass before
            populated themes exist.

        ValueError
            If the user attempts to generate a new refinement pass before orphan
            handling has been completed for the current population pass.

        Side Effects
        ------------
        May mutate:

        - `self.summary_state.theme_schema_list`
        - `self.summary_state.mapped_theme_list`
        - `self.summary_state.populated_theme_list`
        - `self.summary_state.orphan_list`
        - `self.summary_state.redundancy_list`

        Persists SummaryState after successful schema generation or regeneration.

        Notes
        -----
        Schema passes are stored sequentially in
        `self.summary_state.theme_schema_list`.

        A new refinement pass may only be appended after a complete synthesis
        cycle has been run through orphan handling. This guardrail helps ensure
        that schema revisions are based on evidence from actual mapping and
        population behavior rather than on untested schema definitions.

        When regenerating the latest schema pass, downstream artifacts are cleared
        through `SummaryState.rewind_to()` before the latest schema is replaced.
        This preserves the invariant that mapping, population, orphan, and
        redundancy artifacts do not remain ahead of the active schema.

        If `_run_llm_schema_gen()` returns None, schema convergence has been
        reached and no new schema is appended or overwritten.
        """
        if force not in [True, False]:
            raise ValueError("Invalid value for 'force' parameter. Must be a boolean (True or False).")

         # --- Root validation ---
        if not self.summary_state.cluster_summary_list:
            raise ValueError(
                "No cluster summaries found. Please run cluster summarization first."
            )
        ##### 
        # Experimental mode where force is True and guardrails are skipped
        if force:
            print("WARNING: Force flag is True: Skipping validation and sequencing checks and generating new schema pass from cluster summaries. This may cause the state to become unstable. This mode should be used for testing purposes only.")
            new_schema = self._run_llm_schema_gen(source="cluster summaries")
            self.summary_state.theme_schema_list.append(new_schema)
            self.summary_state.save()
            return new_schema
        #####

        # Else run all the regular validation and sequencing checks and proceed as the user specifies/system alllows            
        schema_len = len(self.summary_state.theme_schema_list)
        populate_len = len(self.summary_state.populated_theme_list)
        orphan_len = len(self.summary_state.orphan_list)

        # Determine the input source (from cluster summaries or from orphan handling):
        # --- CASE 1: First schema pass, build from clusters
        if schema_len == 0:
            source_name = "cluster summaries"
            new_schema = self._run_llm_schema_gen(source=source_name)
            self.summary_state.theme_schema_list.append(new_schema)
            self.summary_state.save()
            return new_schema  

        # --- Existing schema passes, check what user wants---
        choice = None
        while choice not in ["1", "2", "3"]:
            choice = input(
                f"You currently have {schema_len} theme schema pass(es).\n"
                "Choose an option:\n"
                "(1) Load the latest schema pass\n"
                "(2) Generate a NEW schema pass (requires orphan incorporation)\n"
                "(3) Regenerate the LATEST schema pass (overwrite last pass)\n"
                "Enter 1, 2, or 3:\n"
            ).strip()

        # --- Option 1: Reload ---
        if choice == "1":
            print("Latest schema loaded.")
            return None

        # --- Option 2: Append new iteration ---
        if choice == "2":

            # Enforce full structural cycle completion
            if populate_len == 0:
                raise ValueError(
                    "No populated themes found. Please run populate_themes() first."
                )

            if populate_len != orphan_len:
                raise ValueError(
                    "You must run handle_orphans() before generating a new schema pass."
                )

            source_name = "populated themes"
            new_schema = self._run_llm_schema_gen(source=source_name)
            if new_schema is None:
                # This means the LLM has indicated that there are no changes to the schema worth making, which likely means we have reached the optimal schema for the current state of populated themes and orphans. In this case we should not add a new schema pass as it is identical to the last one, so we return None to indicate no new schema was generated.
                return None
            self.summary_state.theme_schema_list.append(new_schema)

            self.summary_state.save()
            return new_schema

        # --- Option 3: Regenerate last pass ---
        if choice == "3":

            # Rewind to just before the last schema pass
            last_index = schema_len - 1
            self.summary_state.rewind_to("schema", last_index)

            # Re-evaluate state AFTER rewind
            populate_len_after = len(self.summary_state.populated_theme_list)

            if last_index == 0 and populate_len_after == 0:
                source_name = "cluster summaries"
            else:
                source_name = "populated themes"

            new_schema = self._run_llm_schema_gen(source=source_name)
            if new_schema is None:
                # This means the LLM has indicated that there are no changes to the schema worth making,
                # which likely means we have reached the optimal schema for the current state of
                # populated themes and orphans. In this case we should not overwrite the last schema
                # pass as the new one is identical to it, so we return None to indicate no new schema
                # was generated.
                return None

            # Replace last schema pass
            self.summary_state.theme_schema_list[-1] = new_schema

            self.summary_state.save()
            return new_schema

    def _validate_and_cast_theme_ids(self, df, allowed_ids, prompt, output):
        """
        Validate and normalize LLM-generated theme assignments.

        Verifies that all theme identifiers returned during the insight-to-theme
        mapping stage correspond to themes present in the active schema. After
        validation, theme identifiers are converted to integer type to maintain
        consistency with the schema representation used throughout SummaryState.

        This validation acts as a guardrail against schema drift, hallucinated
        theme identifiers, and formatting inconsistencies in LLM outputs.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing theme assignments returned by the LLM.

            Must contain:

            - `theme_id`

        allowed_ids : Iterable
            Collection of valid theme identifiers defined by the current theme
            schema.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with:

            - validated theme assignments
            - `theme_id` converted to integer dtype

        Raises
        ------
        ValueError
            If one or more returned theme identifiers are not present in the
            current schema.

        Notes
        -----
        Validation is performed using string representations of theme IDs before
        integer conversion. This allows the method to accept identifiers that may
        be returned by the LLM as either strings or integers while still enforcing
        membership in the active schema.

        The method performs only identifier validation and type normalization. It
        does not verify that individual insight assignments are conceptually
        correct, only that they reference valid themes.

        Maintaining integer `theme_id` values is important because downstream
        SummaryState operations assume integer identifiers when ordering themes,
        joining synthesis artifacts, validating state integrity, and generating
        stable theme references across schema iterations.
        """
        allowed_set = set(str(i) for i in allowed_ids)
        returned_set = set(df["theme_id"].astype(str))

        invalid_ids = returned_set - allowed_set
        if invalid_ids:
            raise ValueError(
                f"Invalid theme_id(s) returned by LLM: {invalid_ids}. "
                f"Prompt:\n\n{prompt}\n\n"
                f"Outout:\n\n{output}\n\n"
                f"Allowed IDs: {sorted(allowed_set)}"
            )

        df["theme_id"] = df["theme_id"].astype(int)
        return df
    

    def _map_insights_via_llm(
        self, 
        batch_size,
        already_mapped_insight_ids,
        mapped_insights_df_list, 
        in_progress_path, 
        mode
        ):
        """
        Map insights to themes using the active theme schema.

        Performs the core insight-to-theme classification stage of the
        ReadingMachine synthesis workflow. Insights are processed in batches
        and assigned to one or more themes defined in the current theme schema.

        Mapping is performed independently for each research question using
        the most recent schema stored in
        `self.summary_state.theme_schema_list[-1]`.

        The workflow proceeds as follows:

            active theme schema
                ↓
            batched insight classification
                ↓
            insight-to-theme mappings
                ↓
            complete mapping snapshot

        Stable Schema Reuse
        -------------------
        Questions whose schemas have already been marked as stable are not
        remapped.

        Instead, mappings from the previous mapping pass are reused and seeded
        into the current result set. Only insights belonging to unstable
        questions are submitted to the LLM.

        This reduces unnecessary remapping and helps preserve stability across
        schema iterations.

        Resume and Recovery
        -------------------
        The method supports resumable execution.

        Previously mapped insights can be supplied through
        `already_mapped_insight_ids`, allowing interrupted mapping runs to
        continue from the last completed batch.

        A state fingerprint containing:

        - CorpusState fingerprint
        - SummaryState fingerprint

        is stored alongside intermediate outputs. These fingerprints can be
        used to detect state drift between mapping sessions.

        Parameters
        ----------
        batch_size : int
            Number of insights to include in each LLM mapping request.

        already_mapped_insight_ids : list
            Insight identifiers that have already been mapped. These insights
            are excluded from processing.

        mapped_insights_df_list : list[pd.DataFrame]
            Existing mapping outputs accumulated from previous batches or
            recovery operations.

        in_progress_path : str
            Path where intermediate mapping state is serialized during
            execution.

        mode : str
            Mapping mode.

            Must be one of:

            - `"normal"`: standard execution with state validation
            - `"force"`: bypass resume-state validation safeguards

        Returns
        -------
        pd.DataFrame
            Complete insight-to-theme mapping table containing:

            - `insight_id`
            - `theme_id`
            - `question_id`

            Each row represents a single insight-theme assignment.

            Because insights may map to multiple themes, a single
            `insight_id` may appear in multiple rows.

        Raises
        ------
        ValueError
            If `mode` is not `"normal"` or `"force"`.

        Side Effects
        ------------
        Writes incremental progress to `in_progress_path` after every
        successfully processed batch.

        Stores:

        - mapping outputs
        - state fingerprints
        - execution mode

        in the serialized checkpoint object.

        Notes
        -----
        Mapping is constrained by the active theme schema. Theme identifiers
        returned by the LLM are validated against the schema using
        `_validate_and_cast_theme_ids()` before being accepted.

        The method uses a strict JSON schema to constrain LLM outputs to
        valid `(insight_id, theme_id)` assignments.

        Mapping outputs are treated as a complete state snapshot rather than
        a delta. Even when only unstable questions are processed, the final
        output contains mappings for all insights through a combination of:

        - reused stable mappings
        - newly generated unstable mappings

        Duplicate mappings are removed using:

            (insight_id, theme_id)

        ensuring idempotent behavior across retries, resumes, and repeated
        executions.

        This method performs only classification. It does not populate
        themes, generate summaries, evaluate completeness, or perform orphan
        handling.
        """
        
        if mode not in ["force", "normal"]:
            raise ValueError("Invalid mode. Mode must be either 'force' or 'normal'.")

        # Check if there are any stable questions in the current schema and exclude those from remapping
        if any(self.summary_state.theme_schema_list[-1]["stable"].to_list()):
            # Get the queston ids of the stable questions
            stable_questions = self.summary_state.theme_schema_list[-1][self.summary_state.theme_schema_list[-1]["stable"] == True]["question_id"].unique().tolist()
            # Use the stable questions to get the mappings from the last run which was stable
            stable_mapping = self.summary_state.mapped_theme_list[-1][self.summary_state.mapped_theme_list[-1]["question_id"].isin(stable_questions)].copy()
            if not stable_mapping.empty:
                # use these "stable mappings" to seed the mapped insights and mapped insight ids 
                # mapped_insights_df_list is a list of dfs so we need to concat, drop duplicates and then convert back to a list to add to the mapped insights list so that progress, resume etc all still work
                mapped_insights_df_list.append(stable_mapping)
                mapped_insighs_df = (
                    pd.concat(mapped_insights_df_list, ignore_index=True, sort=False)
                    .drop_duplicates(subset=["insight_id", "theme_id"])
                )
                # Back to list
                mapped_insights_df_list = [mapped_insighs_df]
                # Already mapped insight ids is a list so i just extend and set to drop duplicates
                already_mapped_insight_ids.extend(stable_mapping["insight_id"].tolist())
                # Drop duplicates to avoid multiple seeding if i resume
                already_mapped_insight_ids = list(set(already_mapped_insight_ids))
        
        # Create a meta object of the state against which any resume can be checked to ensure state has not been changed between resume calls
        # This gets pickled along with the progess in mapped insights
        state_meta = {
            "corpus_hash": self.corpus_state.fingerprint(),
            "summary_hash": self.summary_state.fingerprint()
        }

        # Work on a temporary copy
        temp_state_insights = self.corpus_state.insights.copy()

        # Remove already mapped insights if resuming
        if already_mapped_insight_ids:
            temp_state_insights = temp_state_insights[
                ~temp_state_insights["insight_id"].isin(
                    already_mapped_insight_ids
                )
            ]

        # Iterate through each research question
        for _, q_row in self.corpus_state.questions.iterrows():

            question_id = q_row["question_id"]
            question_text = q_row["question_text"]

            # Filter insights for this question
            q_insights_df = temp_state_insights[
                (temp_state_insights["question_id"] == question_id) &
                (temp_state_insights["insight"].notna()) &
                (temp_state_insights["insight"] != "")
            ].copy()

            # Get schema for this question
            q_schema_df = self.summary_state.theme_schema_list[-1][
                self.summary_state.theme_schema_list[-1]["question_id"] == question_id
            ].copy()

            q_schema_json = q_schema_df[
                ["theme_id", "theme_label", "theme_description", "instructions"]
            ].to_json(orient="records", indent=2)


            # Get valid theme IDs for this question.
            # The prompt-validation code uses integers, while the structured output
            # schema expects theme IDs to be returned as strings.
            # Get the inputs to the sys prompt call: the list of allowd theme_ids to tag to, the other theme id and the conflict theme id
            allowed_theme_ids = q_schema_df["theme_id"].astype(int).tolist()
            # Convert the allowed theme IDs to strings for use in the JSON schema
            allowed_theme_id_strings = [
                str(theme_id)
                for theme_id in allowed_theme_ids
            ]
            # Get the rest of the inputs to the sys prompt call: the other theme id and the conflict theme id
            other_theme_rows = q_schema_df[q_schema_df["theme_label"].str.lower() == "other"]["theme_id"].astype(str)
            other_theme_id = other_theme_rows.iloc[0] if not other_theme_rows.empty else None
            conflicts_theme_rows = q_schema_df[q_schema_df["theme_label"].str.lower() == "conflict"]["theme_id"].astype(str)
            conflicts_theme_id = conflicts_theme_rows.iloc[0] if not conflicts_theme_rows.empty else None

            # Pre-build JSON schema (explicit)
            json_schema = {
                "name": "insight_to_theme_mapper",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "mapped_data": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "insight_id": {
                                        "type": "string"
                                    },
                                    "theme_id": {
                                        "type": "array",
                                        "items": {
                                            "type": "string",
                                            "enum": allowed_theme_id_strings
                                        },
                                        "minItems": 1
                                    }
                                },
                                "required": [
                                    "insight_id",
                                    "theme_id"
                                ],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["mapped_data"],
                    "additionalProperties": False
                }
            }

            # Calculate total batches for this question
            total_batches_for_question = math.ceil(
                len(q_insights_df) / batch_size
            )

            # Batch loop
            for batch_index, start_index in enumerate(
                range(0, len(q_insights_df), batch_size),
                start=1
            ):

                print(
                    f"Mapping insights to themes for question "
                    f"{question_id} - batch "
                    f"{batch_index} of "
                    f"{total_batches_for_question}..."
                )

                batch_df = q_insights_df.iloc[
                    start_index : start_index + batch_size
                ]

                if batch_df.empty:
                    continue

                current_batch_str = "\n".join(
                    f"{row.insight_id}: {row.insight}"
                    for row in batch_df.itertuples()
                )

                

                sys_prompt = Prompts().theme_map_to_schema(
                    allowed_ids=allowed_theme_ids,
                    other_theme_id=other_theme_id,
                    conflicts_theme_id=conflicts_theme_id
                    )
                
                user_prompt = (
                    f"RESEARCH QUESTION: {question_text}\n"
                    "THEMATIC CODEBOOK:\n"
                    f"{q_schema_json}\n\n"
                    "INSIGHTS TO MAP:\n"
                    f"{current_batch_str}\n\n"
                )

                response = utils.call_chat_completion(
                    sys_prompt=sys_prompt,
                    user_prompt=user_prompt,
                    llm_client=self.llm_client,
                    ai_model=self.ai_model,
                    fall_back={"mapped_data": []},
                    return_json=True,
                    json_schema=json_schema
                )

                batch_results_df = pd.DataFrame(
                    response.get("mapped_data", [])
                )

                if batch_results_df.empty:
                    continue

                # Expand theme_id
                batch_results_df = batch_results_df.explode("theme_id")
                batch_results_df["question_id"] = question_id
                # Validate that all returned theme_ids are valid
                batch_results_df = self._validate_and_cast_theme_ids(
                    batch_results_df,
                    allowed_theme_ids, 
                    prompt=user_prompt,
                    output=response
                )

                mapped_insights_df_list.append(batch_results_df)
                mapped_insights_df_with_meta = {"mapped_insights_df_list": mapped_insights_df_list, 
                                                "state_meta": state_meta, 
                                                "mode": mode}

                # Save progress safely after each batch = use safe pickle to avoid corruption if something goes wrong during the write
                utils.safe_pickle(mapped_insights_df_with_meta, in_progress_path)
            
        # Check insights existed and were mapped, then concat and return the mapped insights
        if not mapped_insights_df_list:
            mapped_insights_df = pd.DataFrame()
        else:
            mapped_insights_df = pd.concat(
                mapped_insights_df_list,
                ignore_index=True
            )

        return(mapped_insights_df)
    
    def map_insights_to_themes(
        self,
        batch_size: int = 75,
        force: bool = False
    ) -> pd.DataFrame:
        """
        Generate, resume, reload, or regenerate an insight-to-theme mapping pass.

        Public entry point for the insight-to-theme mapping stage of the
        ReadingMachine synthesis workflow. This method coordinates mapping
        execution, resume handling, state validation, state realignment, and
        persistence of mapping outputs.

        Mapping assigns individual insights to one or more themes defined by
        the most recent theme schema and produces the thematic membership layer
        used by downstream theme-population and orphan-handling stages.

        Workflow
        --------
        The mapping stage transforms:

            theme schema
                ↓
            insight classification
                ↓
            insight-to-theme mappings
                ↓
            populated themes

        Mapping is performed independently for each research question and is
        constrained by the active theme schema.

        Resume and Recovery
        -------------------
        The method supports resumable execution through an intermediate
        checkpoint file.

        When a partial mapping process is detected, the user may:

            (1) resume the previous mapping run
            (2) discard the partial run and start again

        Resume operations verify that both CorpusState and SummaryState remain
        unchanged by comparing stored fingerprints against the current state.

        If state drift is detected, resume is aborted to prevent corruption of
        the synthesis history.

        Existing Mapping Passes
        -----------------------
        When mapping passes already exist, the user may:

            (1) reload the most recent mapping pass
            (2) regenerate mappings for the latest schema

        Regeneration realigns schema and mapping history before constructing a
        replacement mapping pass.

        Force Mode
        ----------
        If `force=True`, normal sequencing validation and history-alignment
        safeguards are bypassed.

        The resulting mapping pass is appended directly to
        `SummaryState.mapped_theme_list` regardless of the existing state.

        This mode is intended for development and debugging and may leave the
        synthesis state internally inconsistent.

        Parameters
        ----------
        batch_size : int, default=75
            Number of insights included in each LLM mapping request.

        force : bool, default=False
            If True, bypass sequencing validation, resume-state validation, and
            history-alignment safeguards.

        Returns
        -------
        pd.DataFrame or None
            Mapping DataFrame for the active mapping pass containing at least:

            - `insight_id`
            - `theme_id`
            - `question_id`

            Returns None when the user chooses to reload an existing mapping
            pass rather than generating a new one.

        Raises
        ------
        ValueError
            If no theme schema exists.

        ValueError
            If a resume attempt detects that CorpusState or SummaryState has
            changed since the checkpoint was created.

        Side Effects
        ------------
        Mutates:

        - `self.summary_state.mapped_theme_list`

        May realign:

        - `self.summary_state.theme_schema_list`
        - `self.summary_state.mapped_theme_list`

        Persists SummaryState after successful mapping generation.

        Creates and removes:

        - `mapped_theme_in_progress.pickle`

        used for checkpoint-based recovery.

        Notes
        -----
        Actual classification is delegated to `_map_insights_via_llm()`.

        Mapping outputs are stored as iterative passes in
        `self.summary_state.mapped_theme_list`, allowing theme schemas and
        mappings to evolve together through successive synthesis iterations.

        Stable schemas are not remapped. Their existing mappings are reused by
        `_map_insights_via_llm()`, allowing refinement efforts to focus only on
        unstable portions of the thematic structure.

        Mapping is many-to-many. Individual insights may be assigned to
        multiple themes, including special themes such as "Other" and
        "Conflict" when present in the schema.
        """

        # Check that it is possible to run this step: i.e. there is a theme schema to map to
        if not self.summary_state.theme_schema_list:
            raise ValueError(
                "No theme schema found. Please run gen_theme_schema() first. No force option possible."
            )
        
        # Set up the variabales i will need in the logic below
        # Get the lengths of the theme schema and mapped themes to handle state integirity
        schema_len = len(self.summary_state.theme_schema_list)
        mapped_len = len(self.summary_state.mapped_theme_list)

        # First check for resume 
        #### --------------------------------------------------
        #### RESUMPTION / RECOVERY LOGIC
        #### --------------------------------------------------

        # Get the progress path for the mapping process as i will need this immediately in the logic
        in_progress_path = os.path.join(
            self.summary_save_location, "mapped_theme_in_progress.pickle"
        )

        # Set this as empty for no resume and update the value in the resume branch if triggered
        already_mapped_insight_ids = []

        # Check for existing in-progress file
        if os.path.exists(in_progress_path):

            resume_choice = None
            while resume_choice not in ["1", "2"]:
                resume_choice = input(
                    "A partial mapping process was detected. Do you want to:\n"
                    "1) resume from the last saved point? \n"
                    "2) start a new mapping process? \n"
                    "Enter 1 or 2:\n"
                ).strip()

            if resume_choice == "1":
                with open(in_progress_path, "rb") as f:
                    mapped_insights_df_list_meta = pickle.load(f)
                    mapped_insights_df_list = mapped_insights_df_list_meta["mapped_insights_df_list"]
                    pickled_meta = mapped_insights_df_list_meta["state_meta"]
                    mode = mapped_insights_df_list_meta["mode"]
                
                # Check that the state has not changed between resume calls
                current_meta = {
                    "corpus_hash": self.corpus_state.fingerprint(),
                    "summary_hash": self.summary_state.fingerprint()
                }

                if current_meta != pickled_meta:
                    raise ValueError(
                        "State change detected between resume call. Resume invalid to prevent state corruption. "
                        "Force override if you know what you are doing or run without resume."
                    )

                if mapped_insights_df_list:
                    recovered_df = pd.concat(
                        mapped_insights_df_list,
                        ignore_index=True
                    )
                    already_mapped_insight_ids = (
                        recovered_df["insight_id"].unique().tolist()
                    )

                print("Resuming mapping process from last saved point...")
                mapped_insights_df = self._map_insights_via_llm(
                    batch_size=batch_size,
                    already_mapped_insight_ids=already_mapped_insight_ids,
                    mapped_insights_df_list=mapped_insights_df_list,
                    in_progress_path=in_progress_path, 
                    mode = mode
                )

                # Now check the mode to know how to handle.
                # If the mode is force we append as that is the defualt for force
                if mode == "force":
                    self.summary_state.mapped_theme_list.append(mapped_insights_df)
                    # Save state
                    os.makedirs(self.summary_save_location, exist_ok=True)
                    self.summary_state.save()
                    # Remove in-progress file now that mapping and save is complete
                    if os.path.exists(in_progress_path):
                        os.remove(in_progress_path)
                    # Return to exit the function
                    return self.summary_state.mapped_theme_list[-1]
                # mode is normal, so we follow pattern for ensuring state integrity
                else: 
                    # Make sure the state is integreal by realigning the schema and mapping history to the last coherent point 
                    min_len = min(schema_len, mapped_len)
                    self.summary_state.theme_schema_list = self.summary_state.theme_schema_list[:min_len + 1]
                    self.summary_state.mapped_theme_list = self.summary_state.mapped_theme_list[:min_len]
                    # Append the new mapped insights to the state
                    self.summary_state.mapped_theme_list.append(mapped_insights_df)
                    # Save state
                    os.makedirs(self.summary_save_location, exist_ok=True)
                    self.summary_state.save()
                    # Remove in-progress file now that mapping and save is complete
                    if os.path.exists(in_progress_path):
                        os.remove(in_progress_path)
                    # Return to exit the function
                    return self.summary_state.mapped_theme_list[-1]
            else:
                os.remove(in_progress_path)
                print("Deleted in-progress file. Starting new mapping process.")


        ####--------------------------------------------------
        #  Force flag enabled - skip all validation and resume checks. Run a mapping of the insights to the latest themes and append to the state.
        if force:
            print("WARNING: Force flag is True: Skipping validation and resumption checks and mapping insights to themes. This may cause the state to become unstable. This mode should be used for testing purposes only.")
            
            # Get the mapped df from the llm with resume elements set to empty as we want fresh run
            mapped_insights_df = self._map_insights_via_llm(
                batch_size=batch_size,
                already_mapped_insight_ids=[],
                mapped_insights_df_list=[],
                in_progress_path=in_progress_path, 
                mode = "force"
            )
            
            # Append to state (because force defaults to append), save, clean-up and return
            self.summary_state.mapped_theme_list.append(mapped_insights_df)
            self.summary_state.save()
            # Remove in-progress file now that mapping and save is complete
            if os.path.exists(in_progress_path):
                os.remove(in_progress_path)
            return mapped_insights_df
        ######-----------------------------

        # If mappings exist, offer choice
        if mapped_len > 0:
            new_choice = None
            while new_choice not in ["1", "2"]:
                new_choice = input(
                    "Mapped themes already exist on disk. Do you want to:\n"
                    "(1) reload existing mapped themes\n"
                    "(2) remap insights to the latest theme schema. "
                    "This will realign schema and mapping history.\n"
                    "Enter 1 or 2:\n"
                ).strip()

            if new_choice == "1":
                print("Mapped themes loaded.")
                return None

            # Realign to last coherent pre-mapping corpus_state
            min_len = min(schema_len, mapped_len)
            self.summary_state.theme_schema_list = self.summary_state.theme_schema_list[:min_len + 1]
            self.summary_state.mapped_theme_list = self.summary_state.mapped_theme_list[:min_len]

            # Get the mappings from the llm - passsing empty values for the resume elements as this is a fresh run
            mapped_insights_df = self._map_insights_via_llm(
                batch_size=batch_size,
                already_mapped_insight_ids=[],
                mapped_insights_df_list=[],
                in_progress_path=in_progress_path, 
                mode = "normal"
            )

        else:
            # No mappings exist — ensure we are in valid pre-mapping corpus_state
            # This is duplicative because we tested earlier, but this makes the logic more explicit
            if schema_len < 1:
                raise ValueError("No schema available to map.")
            
            # If no mappings exist and there is no resume and as we are not forcing, then we should run under normal mode (i.e. a fresh run)
            mapped_insights_df = self._map_insights_via_llm(
                batch_size=batch_size,
                already_mapped_insight_ids=[],
                mapped_insights_df_list=[],
                in_progress_path=in_progress_path,
                mode="normal"
            )
        
        #### --------------------------------------------------
        #### FINALIZATION ON NORMAL MODE (non-force)
        #### --------------------------------------------------

        # Update state
        self.summary_state.mapped_theme_list.append(mapped_insights_df)
        # Save state
        os.makedirs(self.summary_save_location, exist_ok=True)
        self.summary_state.save()
        # Remove in-progress file now that mapping and save is complete
        if os.path.exists(in_progress_path):
            os.remove(in_progress_path)

        return self.summary_state.mapped_theme_list[-1]


    def _estimate_theme_lengths(self, paper_len: int, max_model_output_words: int = 2500) -> pd.DataFrame:
        """
        Estimate target word budgets for theme-level summaries.

        Allocates an approximate summary length to each theme based on the
        number of mapped insights assigned to that theme. Themes with more mapped
        insights receive larger target lengths, while themes with fewer mapped
        insights receive shorter target lengths subject to fixed lower and upper
        bounds.

        This allocation is used during theme population to guide the amount of
        narrative space given to each theme within the overall target output
        length.

        Parameters
        ----------
        paper_len : int
            Approximate target length, in words, for the synthesized output.

        max_model_output_words : int, default=2500
            Maximum target word length assigned to any single theme summary.

        Returns
        -------
        pd.DataFrame
            DataFrame containing:

            - `theme_id`
            - `allocated_length`

        Notes
        -----
        The allocation is based on the latest mapping pass stored in
        `self.summary_state.mapped_theme_list[-1]` and the latest schema stored in
        `self.summary_state.theme_schema_list[-1]`.

        All themes in the active schema are preserved, including themes with no
        mapped insights. Zero-hit themes are assigned the minimum target length so
        that schema structure remains stable across synthesis passes.

        The current hard bounds are:

        - minimum theme length: 375 words
        - maximum theme length: `max_model_output_words`

        Word budgets are heuristic controls for LLM generation rather than strict
        output guarantees. Actual generated summary lengths may differ depending
        on model behavior, prompt structure, and the density of mapped insight
        content.

        Theme identifiers are cast to integer type before return to preserve
        downstream ordering and join consistency.
        """

        MIN_THEME_WORDS = 375
        MAX_THEME_WORDS = max_model_output_words

        # --- Full list of expected themes (anchor) ---
        full_schema = (
            self.summary_state.theme_schema_list[-1][["theme_id"]]
            .copy()
            .astype({"theme_id": int})
            .drop_duplicates()
        )

        # --- Count mapped insights per theme ---
        theme_counts = (
            self.summary_state.mapped_theme_list[-1]
            .copy()
            .astype({"theme_id": int})
            .groupby("theme_id")
            .size()
            .reset_index(name="count")
        )

        # --- Merge to preserve zero-hit themes ---
        theme_counts = full_schema.merge(
            theme_counts,
            on="theme_id",
            how="left"
        )

        theme_counts["count"] = theme_counts["count"].fillna(0)

        # --- Compute global proportions ---
        total_count = theme_counts["count"].sum()

        if total_count > 0:
            theme_counts["proportion"] = (
                theme_counts["count"] / total_count
            )
        else:
            theme_counts["proportion"] = 0

        # --- Allocate proportionally ---
        theme_counts["allocated_length"] = np.ceil(
            theme_counts["proportion"] * paper_len
        ).astype(int)

        # --- Apply hard floor and hard ceiling ---
        theme_counts["allocated_length"] = (
            theme_counts["allocated_length"]
            .clip(lower=MIN_THEME_WORDS, upper=MAX_THEME_WORDS)
        )

        # Defensively ensure that theme_id remains an int to allow for sorting
        theme_counts["theme_id"] = theme_counts["theme_id"].astype(int)

        return theme_counts[["theme_id", "allocated_length"]]


    def _check_length_and_flag(self, df: pd.DataFrame, max_prop: float, max_model_output_words: int = 2800) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split populated themes by length-pressure status.

        Flags themes whose generated summaries are approaching or exceeding a
        specified proportion of their allocated word budget. This diagnostic is
        used during theme population to identify summaries that may require
        expansion, repartitioning, or repair because the theme may contain more
        mapped insight content than can be represented within its current target
        length.

        Parameters
        ----------
        df : pd.DataFrame
            Populated-theme DataFrame containing at least:

            - `allocated_length`
            - `current_length`
            - `stable`

        max_prop : float
            Proportion of `allocated_length` above which a theme is flagged.

        max_model_output_words : int, default=2800
            Hard maximum summary length. Themes already allocated this maximum are
            not flagged.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            Tuple containing:

            - `df_len_ok`: themes not flagged for length pressure
            - `df_len_flagged`: themes flagged for length pressure

        Notes
        -----
        A theme is not flagged when:

        - `current_length` is missing or zero
        - `allocated_length` equals `max_model_output_words`
        - `stable` is True
        - `current_length <= allocated_length * max_prop`

        The method adds a `length_flag` column to the input DataFrame before
        splitting it into flagged and unflagged subsets.

        Length pressure is a heuristic signal. It does not prove that a summary is
        incomplete, but it can indicate that the theme is carrying too much
        representational load or that assigned insights may need to be redistributed
        during schema repair.
        """

        MAX_THEME_WORDS = max_model_output_words

        def flag_row(row):
            # Treat missing summaries as not flagged
            if pd.isna(row["current_length"]) or row["current_length"] == 0:
                return 0

            # Do not flag if theme is already at hard ceiling
            if row["allocated_length"] == MAX_THEME_WORDS:
                return 0

            # If the theme is marked as stable we should not flag - we don't want to re-write.
            if row["stable"] == True:
                return 0
            
            # Standard proportional check
            if row["current_length"] <= row["allocated_length"] * max_prop:
                return 0
            else:
                return 1

        df["length_flag"] = df.apply(flag_row, axis=1)

        df_len_ok = df[df["length_flag"] == 0]
        df_len_flagged = df[df["length_flag"] == 1]

        return df_len_ok, df_len_flagged
    
    def _run_theme_pop(
        self,
        schema_df: pd.DataFrame,
        mapped_themes_df: pd.DataFrame,
        paper_len: int = 8000
    ) -> pd.DataFrame:
        """
        Populate themes by synthesizing mapped insights into thematic narratives.

        Generates theme-level summaries from the insight-to-theme mappings
        produced during the mapping stage. Each theme is synthesized
        independently and becomes the primary analytical representation used for
        completeness auditing, orphan detection, schema repair, and subsequent
        thematic refinement.

        The theme-population workflow proceeds as follows:

            mapped insights
                ↓
            length allocation
                ↓
            theme-level synthesis
                ↓
            populated themes
                ↓
            completeness auditing

        Stable Theme Reuse
        ------------------
        Themes belonging to research questions whose schemas have already been
        marked as stable are not regenerated.

        Instead, populated themes from the most recent population pass are
        reused and carried forward unchanged. Only themes associated with
        unstable schemas are submitted to the LLM for regeneration.

        Length Allocation
        -----------------
        Before synthesis, each theme receives an allocated word budget derived
        from the relative number of mapped insights assigned to that theme.

        These allocations are used to guide summary generation and to later
        evaluate whether themes may be experiencing representational overload.

        Theme Types
        -----------
        Different prompting strategies are used depending on theme type:

        - general themes
        - conflict themes
        - other themes

        This allows conflict-oriented and residual categories to be synthesized
        differently from standard thematic categories.

        Fallback and Sampling
        ---------------------
        If theme population fails or produces an empty summary, the method
        retries synthesis using a sampled subset of insights constrained by a
        maximum word budget.

        Sampling is treated as a temporary context-management strategy rather
        than a coverage decision. Any omitted information is expected to be
        recovered later through completeness checking, orphan detection,
        reintegration, and schema refinement.

        Parameters
        ----------
        schema_df : pd.DataFrame
            Active theme schema containing theme definitions and synthesis-state
            metadata.

            Expected columns include:

            - `theme_id`
            - `theme_label`
            - `theme_description`
            - `question_id`
            - `question_text`
            - `stable`

        mapped_themes_df : pd.DataFrame
            Insight-to-theme mapping table containing at least:

            - `insight_id`
            - `theme_id`
            - `question_id`

        paper_len : int, default=8000
            Approximate target length of the synthesized output. Used when
            allocating theme-level word budgets.

        Returns
        -------
        pd.DataFrame
            Populated theme table containing:

            - `thematic_summary`
            - `question_id`
            - `theme_id`
            - `theme_label`
            - `theme_description`
            - `allocated_length`
            - `current_length`
            - `perc_of_max_length`
            - `needs_repair`
            - `optimized`
            - `stable`

        Notes
        -----
        Themes with no mapped insights are retained and returned with empty
        summaries. This preserves schema stability and allows structural
        categories such as "Other" and "Conflict" to remain present even when
        they are temporarily unused.

        Summary length diagnostics are calculated for every generated theme.
        These diagnostics are later used to identify themes that may require
        schema repair, decomposition, or redistribution of representational
        load.

        This method performs synthesis only. It does not evaluate completeness,
        detect orphaned insights, modify the schema, or persist outputs to
        SummaryState.
        """
        # Calculate the estimated lengths for each theme based on the number of insights mapped to them and merge this info back to the theme schema for use in the prompt when populating themes
        # This is only done if the columsn do not already exist, because later we will iterate on this and in those subsequent cases we just amend the allocated length manually
        # Normalise the id columsn as they come back from the LLM so could be str
        # Copy the df because this could be called on a corpus_state object
        
        # Get the allocated lenghts
        schema_df = schema_df.copy()
        schema_df["theme_id"] = schema_df["theme_id"].astype(int)
        if "allocated_length" not in schema_df.columns:
            schema_df = schema_df.merge(
            self._estimate_theme_lengths(paper_len),
            on="theme_id",
            how="left"
        )
        
        # Get the populated themese for the stable schema 
        stable_schema = schema_df[schema_df["stable"] == True].copy()
        # Check that there is a populated theme list (i.e. its not the first iteeration)
        if self.summary_state.populated_theme_list:
            # Then get the populated themes from the schema
            stable_populated_themes = (
                self.summary_state.populated_theme_list[-1][
                    self.summary_state.populated_theme_list[-1]["question_id"]
                    .isin(stable_schema["question_id"])
                ].copy()
            )
            stable_populated_themes["stable"] = True
        # If it doesn't exist set it as empty
        else:
            stable_populated_themes = pd.DataFrame()

        if stable_populated_themes is not None and not stable_populated_themes.empty:
            stable_populated_themes["stable"] = True

        # Get the unstable schema
        unstable_schema = schema_df[schema_df["stable"] == False].copy()

        # Iterate over the themes from the unstable schema to get the data for the LLM call
        populated_themes = []
        for idx, row in unstable_schema.iterrows():
            print(f"Populating theme {idx + 1} of {unstable_schema.shape[0]}...")
            rq_id = row["question_id"]
            rq_text = row["question_text"]
            theme_id = row["theme_id"]
            theme_label = row["theme_label"]
            theme_description = row["theme_description"]
            organizing_proposition = (
                row.get("organizing_proposition") if self.use_organizing_proposition else None
                )
            allocated_length = row["allocated_length"]
            needs_repair = row.get("needs_repair", pd.NA)
            optimized = row.get("optimized", False)
            stable = row.get("stable", False)
            # Get the insight ids for the specific question and theme
            insight_ids = mapped_themes_df[
                (mapped_themes_df["question_id"] == rq_id) & 
                (mapped_themes_df["theme_id"] == theme_id)
            ]["insight_id"].tolist()
            # Get the insight text from those insight ids

            insights_df = self.corpus_state.insights[
                self.corpus_state.insights["insight_id"].isin(insight_ids)
            ].copy()
            
            # Add in the citations
            insights = (insights_df["insight"] + " (" + insights_df["in_text_citation"] + ")").tolist()
            
            # Check if insights are zero (i.e. an empty conflicts or other catergory got returned by the LLM). If so populate with an empty row
            if len(insights) == 0:
                no_insight_row = {
                    "thematic_summary": "",
                    "question_id": rq_id,
                    "theme_id": theme_id,
                    "theme_label": theme_label,
                    "theme_description": theme_description,
                    "allocated_length": allocated_length,
                    "needs_repair": needs_repair,
                    "optimized": optimized,
                    "stable": stable,
                }

                if self.use_organizing_proposition:
                    no_insight_row["organizing_proposition"] = organizing_proposition

                no_insight_df = pd.DataFrame([no_insight_row])
                continue
            
            insights_str = "\n".join(insights)

            # Get the theme type to pass to the sys prompt to tailor the prompt to the theme type: general, conflict or other.
            if theme_label.strip().lower() == "conflict":
                theme_type = "conflicts"
            elif theme_label.strip().lower() == "other":
                theme_type = "other"
            else:
                theme_type = "general"
                
            # Build the prompt
            sys_prompt = Prompts().populate_themes(theme_len=allocated_length, theme_type=theme_type, provide_organizing_proposition=self.use_organizing_proposition)

            # COnditionally add to the user prompt
            # Set it to empty and populate if the attrbute is set to true
            organizing_proposition_input = ""

            if self.use_organizing_proposition and theme_type == "general":
                organizing_proposition_input = (
                    f"ORGANIZING PROPOSITION: {organizing_proposition}\n"
                )

            user_prompt = (
                f"RESEARCH QUESTION: {rq_text}\n"
                f"THEME LABEL: {theme_label}\n"
                f"THEME DESCRIPTION: {theme_description}\n"
                f"{organizing_proposition_input}"
                f"INSIGHTS TO SYNTHESIZE:\n"
                f"{insights_str}\n\n"
            )
            fall_back = {"thematic_summary": ""}

            json_schema = {
                "name": "theme_populator",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "thematic_summary": {"type": "string"}
                    },
                    "required": ["thematic_summary"],
                    "additionalProperties": False
                }
            }
            # Call the LLM
            response, error = utils.call_chat_completion(
                sys_prompt=sys_prompt,
                user_prompt=user_prompt,
                llm_client=self.llm_client,
                ai_model=self.ai_model,
                fall_back=fall_back,
                return_json=True,
                json_schema=json_schema, 
                return_with_error=True
            )
            
            # Get the summary from the response and tag with metadata in a dataframe
            summary_text = response.get("thematic_summary", "")

            if not summary_text.strip():

                print(f"Empty summary generated for theme_id {theme_id}. Error: {error}. Resubmitting for summary on sampled insights")
                # --- sample when words in insights exceeds 70000 step ---
                MAX_WORDS = 70000           

                # if total number of words in the insights exceeds the max, then sample
                
                insights = utils.sample_to_word_limit(
                    insights,
                    max_words=MAX_WORDS,
                    seed=config.seed
                )

                print(
                    f"Theme {theme_id}: sampled {len(insights)} insights to fit word budget"
                )
                insights_str = "\n".join(insights)

                user_prompt = (
                f"RESEARCH QUESTION: {rq_text}\n"
                f"THEME LABEL: {theme_label}\n"
                f"THEME DESCRIPTION: {theme_description}\n"
                f"{organizing_proposition_input}"
                f"INSIGHTS TO SYNTHESIZE:\n"
                f"{insights_str}\n\n"
                )

                response = utils.call_chat_completion(
                    sys_prompt=sys_prompt,
                    user_prompt=user_prompt,
                    llm_client=self.llm_client,
                    ai_model=self.ai_model,
                    fall_back=fall_back,
                    return_json=True,
                    json_schema=json_schema
                )
                    
            thematic_summary = pd.DataFrame([response.get("thematic_summary", "")], columns=["thematic_summary"])
            thematic_summary["question_id"] = rq_id
            thematic_summary["theme_id"] = int(theme_id)
            thematic_summary["theme_label"] = theme_label
            thematic_summary["theme_description"] = theme_description

            # Conditionally add the organizing proposition to the thematic summary if the attribute is set to true
            if self.use_organizing_proposition:
                thematic_summary["organizing_proposition"] = organizing_proposition

            thematic_summary["allocated_length"] = allocated_length
            thematic_summary["needs_repair"] = needs_repair
            thematic_summary["optimized"] = optimized
            thematic_summary["stable"] = stable
            

            # Get the length of the summary in words and calculate the percentage of the allocated length that this summary represents
            thematic_summary["current_length"] = len(thematic_summary["thematic_summary"].iloc[0].split())
            thematic_summary["perc_of_max_length"] = thematic_summary["current_length"] / allocated_length if allocated_length > 0 else None
            
            # Append the result to the list of dfs the loop is producing, which will be concatenated at the end
            populated_themes.append(thematic_summary)

        # Concat the final list of dfs and return
        populated_themes_df = pd.concat(populated_themes, ignore_index=True)
        # Add back the stable questions and themes with thier populated summaries
        if stable_populated_themes is not None and not stable_populated_themes.empty:
            populated_themes_df = pd.concat([populated_themes_df, stable_populated_themes], ignore_index=True)

        # Sort the values by question id and theme id so that they are in the same order as the schema and therefore able to be exported to the narrative
        populated_themes_df["theme_id"] = populated_themes_df["theme_id"].astype(int) # Before sorting defensively make sure this is int
        populated_themes_df = populated_themes_df.sort_values(by=["question_id", "theme_id"]).reset_index(drop=True)
        
        return(populated_themes_df)

    def _iterative_length_check_and_expand_loop(
        self,
        populated_themes_df: pd.DataFrame,
        max_prop: float,
        paper_len: int
    ) -> pd.DataFrame:
        """
        Iteratively rerun length-pressured theme summaries with larger budgets.

        Reviews populated theme summaries for length pressure and optionally
        regenerates flagged themes with expanded word allocations. This provides a
        human-in-the-loop mechanism for reducing over-compression before moving to
        downstream orphan handling and schema refinement.

        The loop proceeds as follows:

            populated themes
                ↓
            length-pressure check
                ↓
            optional budget expansion
                ↓
            targeted theme repopulation
                ↓
            updated populated themes

        Parameters
        ----------
        populated_themes_df : pd.DataFrame
            Current populated-theme table containing generated summaries, theme
            metadata, allocated lengths, current lengths, and stability flags.

        max_prop : float
            Proportion of allocated length above which a theme is flagged for
            possible expansion.

        paper_len : int
            Approximate target output length used when rerunning theme population.

        Returns
        -------
        pd.DataFrame
            Updated populated-theme table with any regenerated summaries replacing
            their previous versions.

        Raises
        ------
        ValueError
            If expanded `allocated_length` values fail to merge onto the rerun
            schema for flagged themes.

        Notes
        -----
        Flagged themes are rerun with a 20 percent increase in allocated length.

        Only flagged themes are regenerated. Unflagged themes are carried forward
        unchanged.

        Stable themes are not flagged by `_check_length_and_flag()` and therefore
        are not regenerated by this loop.

        The loop continues until either no themes remain flagged or the user chooses
        to accept the current populated themes.

        Length pressure is treated as a heuristic signal of possible excessive
        abstraction or representational overload. It does not prove that a theme is
        incomplete, but it gives the user an opportunity to expand summaries before
        orphan detection and schema repair evaluate coverage more directly.
        """
        # Check whether any of the populated themes exceed a certain proportion of th total allocated length
        df_len_ok, df_len_flagged = self._check_length_and_flag(populated_themes_df, max_prop)
        df_len_flagged = df_len_flagged.copy() # Copy to avoid modifying the original df with the flags, as this will be used in the loop and we want to preserve the original for reference.

        #While any themes exceed the length threshold its a sign that the LLM is undertaking more compression of the granulatiry of the theme than might be optimal (to fit length requirements). 
        # Offer the user the options to re-run these themes with a 20% expansion of the word count allocated to them.
        while df_len_flagged is not None and not df_len_flagged.empty:
            expand_count = None
            while expand_count not in ["1", "2"]:
                expand_count = input(
                    f"{df_len_flagged.shape[0]} themes exceed {max_prop*100}% of their allocated length. This suggests the model is compressing these themes through aggressive abstraction and granularity is being lost. \n"
                    "Do you want to:\n"
                    "1) re-run these themes with a 20% expansion of the word count allocated to them? \n"
                    "2) keep the current populated themes and move on to the next step? \n"
                    "Enter 1 or 2:\n"
                ).lower()

            
            if expand_count == "2":
                # If the user is happy, end the loop. Populated themes is in tact and we keep it. 
                break

            else:
                # If the user wants to expand, we re run the theme populator on the flagged themes with increased allocation and update populated themes with the longer theme summaries
                # First get the schema map for the flagged themes and expand the allocated length by 20%
                df_len_flagged = df_len_flagged.copy()
                df_len_flagged.loc[:, "allocated_length"] = (df_len_flagged["allocated_length"] * 1.2).astype(int)
                # Then get the ids and the corresponding insights
                length_check_theme_ids = df_len_flagged["theme_id"].tolist()
                rerun_mapped_df = self.summary_state.mapped_theme_list[-1][self.summary_state.mapped_theme_list[-1]["theme_id"].isin(length_check_theme_ids)].copy()
                # Rerun on the flagged themes with the expanded length and get new summaries for those themes
                rerun_theme_ids = df_len_flagged["theme_id"].tolist()
                rerun_schema_df = (
                    self.summary_state.theme_schema_list[-1][self.summary_state.theme_schema_list[-1]["theme_id"].isin(rerun_theme_ids)] # First filter the schema to just the themes we want to rerun
                    .drop(columns=["allocated_length"], errors="ignore") # Drop allocated length as we will get the updated length from df_len_flagged 
                    .copy() # Copy so that we are not modifying corpus_state
                )
                # Now get the updated allocated length via a merge
                rerun_schema_df = ( 
                    rerun_schema_df
                    .merge(df_len_flagged[["theme_id", "allocated_length"]], 
                           on="theme_id", 
                           how="left")
                )
                # Add a sanity check to make sure the allocated length came through in the merge, as this is critical for the rerun. Fail loudly if there is a problem
                if rerun_schema_df["allocated_length"].isna().any():
                    raise ValueError(
                        "Expanded allocated_length missing for some rerun themes."
                    )
                
                # Rerun the theme populater on the theme_schema_df
                rerun_populated_themes_df = self._run_theme_pop(rerun_schema_df, rerun_mapped_df, paper_len)
                rerun_len_ok, df_len_flagged = self._check_length_and_flag(rerun_populated_themes_df, max_prop)
                # If this produced any themes for which the length is now ok we replace the odl summaries in populated_theme_list with these updates
                # Always replace all rerun themes (both ok + still flagged)
                rerun_all = pd.concat([rerun_len_ok, df_len_flagged], ignore_index=True)

                rerun_theme_ids = rerun_all["theme_id"].tolist()

                # Drop old versions
                populated_themes_df = populated_themes_df[
                    ~populated_themes_df["theme_id"].isin(rerun_theme_ids)
                ]

                # Add updated versions
                populated_themes_df = pd.concat(
                    [populated_themes_df, rerun_all],
                    ignore_index=True
                )
                    
        # Make sure the final df of populated themes is in the same order as the theme schema for easier comparison and so that it can be exported to the narrative in the correct order
        populated_themes_df["theme_id"] = populated_themes_df["theme_id"].astype(int) # Before sorting defensively make sure this is int
        populated_themes_df = populated_themes_df.sort_values(by=["question_id", "theme_id"]).reset_index(drop=True)
        return(populated_themes_df)
   
    def populate_themes(
        self,
        paper_len: int = 8000,
        max_prop: float = 0.9,
        force: bool = False
    ) -> pd.DataFrame:
        """
        Generate, reload, or regenerate populated theme summaries.

        Public entry point for the theme-population stage of the ReadingMachine
        synthesis workflow. This method synthesizes mapped insights into
        theme-level narrative summaries, manages state sequencing, supports
        regeneration of existing population passes, and persists populated-theme
        outputs to SummaryState.

        The population stage transforms:

            theme schema
                ↓
            insight-to-theme mappings
                ↓
            theme-level narrative summaries
                ↓
            orphan detection and schema refinement

        Parameters
        ----------
        paper_len : int, default=8000
            Approximate target length, in words, for the synthesized output. Used
            to allocate theme-level word budgets.

        max_prop : float, default=0.9
            Proportion of allocated theme length above which a populated theme is
            flagged for possible expansion.

        force : bool, default=False
            If True, bypass normal sequencing and state-alignment checks, populate
            themes from the latest schema and mapping pass, and append the result
            directly to `summary_state.populated_theme_list`.

        Returns
        -------
        pd.DataFrame or None
            Populated-theme DataFrame for the active population pass.

            Returns None when the user chooses to reload the latest existing
            populated themes rather than regenerate them.

        Raises
        ------
        ValueError
            If no mapped themes exist.

        ValueError
            If no mapped themes are available after state-alignment checks.

        Side Effects
        ------------
        May mutate:

        - `self.summary_state.mapped_theme_list`
        - `self.summary_state.populated_theme_list`

        Persists SummaryState after successful population.

        Notes
        -----
        Actual theme synthesis is delegated to `_run_theme_pop()`.

        After population, `_iterative_length_check_and_expand_loop()` gives the
        user the option to regenerate length-pressured themes with expanded word
        budgets before outputs are saved.

        If populated themes already exist, regeneration realigns mapping and
        population history so that downstream artifacts do not remain ahead of the
        active mapping pass.

        Force mode is intended for development and debugging. It may leave the
        synthesis history internally inconsistent because it bypasses normal
        sequencing safeguards.

        Populated themes are stored as sequential passes in
        `self.summary_state.populated_theme_list`, preserving the history of
        synthesis attempts across schema iterations.
        """
        
        # Check that themes have been mapped before we try to populate
        if not self.summary_state.mapped_theme_list:
            raise ValueError("No mapped themes found. Please run map_insights_to_themes() first.")
        
        #######
        # FORCE FLAG LOGIC
        #######
        # Check whether force flag is enabled. If so popylate the themes and append without running any state checks.
        if force:
            print("WARNING: Force flag is True: Skipping validation and resumption checks and populating themes. This may cause the state to become unstable. This mode should be used for testing purposes only.")
            # populate the themes with the insights mapped to them
            populated_themes_df = self._run_theme_pop(
                schema_df=self.summary_state.theme_schema_list[-1].copy(),
                mapped_themes_df=self.summary_state.mapped_theme_list[-1].copy(),
                paper_len=paper_len
            )
            # Iteratively expand the theme lenght until the user is satisfied
            populated_themes_df = self._iterative_length_check_and_expand_loop(
                populated_themes_df=populated_themes_df,
                max_prop=max_prop,
                paper_len=paper_len
            )

            # Append, save and return
            self.summary_state.populated_theme_list.append(populated_themes_df)
            self.summary_state.save()
            return self.summary_state.populated_theme_list[-1]
        
        #######
        # FORCE FLAG LOGIC/ENDS
        #######

        
        mapped_len = len(self.summary_state.mapped_theme_list)
        populate_len = len(self.summary_state.populated_theme_list)

        # If populated themes exist, offer choice
        if populate_len > 0:
            new = None
            while new not in ["1", "2"]:
                new = input(
                    "Populated themes already exist on disk. Do you want to:\n"
                    "(1) reload existing populated themes\n"
                    "(2) repopulate themes based on the current theme schema and mapped insights. This will realign mapping and population history.\n"
                    "Enter 1 or 2:\n"
                ).strip()
            if new == "1":
                print(
                    "Populated themes loaded. Inspect them via variable.populated_theme_list[-1]\n"
                )
                return(None) # Return to exit the function and avoid re-running the population
            
            # Realign to last coherent pre-population corpus_state
            min_len = min(mapped_len - 1, populate_len)
            self.summary_state.mapped_theme_list = self.summary_state.mapped_theme_list[:min_len + 1]
            self.summary_state.populated_theme_list = self.summary_state.populated_theme_list[:min_len]
            
        else:
            # No populated themes exist — ensure we are in valid pre-population corpus_state
            if mapped_len < 1:
                raise ValueError("No mapped themes available to populate from. Please run map_insights_to_themes() first.")
        
        
        # Populate the themes with the insights mapped to them 
        populated_themes_df = self._run_theme_pop(
            schema_df=self.summary_state.theme_schema_list[-1].copy(),
            mapped_themes_df=self.summary_state.mapped_theme_list[-1].copy(),
            paper_len=paper_len
        )
        
        # Iterate on the length of each theme and expand until the user is happy
        populated_themes_df = self._iterative_length_check_and_expand_loop(
            populated_themes_df=populated_themes_df,
            max_prop=max_prop,
            paper_len=paper_len
        )

        # Now add populated_themes_df to the populated theme list and save to disk. Return the updated themes
        self.summary_state.populated_theme_list.append(populated_themes_df)
        self.summary_state.save()
        return self.summary_state.populated_theme_list[-1]

    def _identify_orphans(
        self,
        checked_insights_df,
        mode,
        batch_size
    ) -> pd.DataFrame:
        """
        Audit populated theme summaries for omitted mapped insights.

        Identifies orphan insights: mapped insights that are not reflected in the
        populated thematic summaries to which they were assigned. This method
        performs the coverage-audit stage that checks whether theme population
        preserved the underlying insight content.

        The audit proceeds theme by theme:

            populated theme summary
                ↓
            mapped source insights
                ↓
            LLM mention audit
                ↓
            orphan insight set

        Resume and Recovery
        -------------------
        The method supports checkpoint-based recovery. Previously checked insights
        can be supplied through `checked_insights_df`, and intermediate progress is
        written to `self.orphan_pickle_resume_path` after each batch.

        Each checkpoint stores:

        - checked insight results
        - CorpusState fingerprint
        - SummaryState fingerprint
        - orphan-handling mode

        Stable Question Reuse
        ---------------------
        Research questions whose schemas are marked stable are not re-audited.
        Their orphan results are reused from the previous orphan pass and seeded
        into the current audit output.

        Parameters
        ----------
        checked_insights_df : pd.DataFrame or None
            Previously checked insight-audit results, usually recovered from a
            checkpoint. If None, a new empty audit table is initialized.

        mode : str
            Orphan-handling mode associated with the current run.

            Must be one of:

            - `"replace"`: overwrite the most recent orphan audit
            - `"append"`: append a new orphan audit to the history

        batch_size : int
            Number of mapped insights included in each LLM audit request.

        Returns
        -------
        pd.DataFrame
            DataFrame containing only orphaned insights, with columns:

            - `question_id`
            - `theme_id`
            - `insight_id`
            - `found`

            Returned rows have `found == False`.

        Raises
        ------
        ValueError
            If `mode` is not `"replace"` or `"append"`.

        Side Effects
        ------------
        Writes checkpoint data to `self.orphan_pickle_resume_path` after each
        processed batch.

        Notes
        -----
        The method uses the latest populated themes and latest insight-to-theme
        mappings stored in SummaryState.

        For each populated theme, the method retrieves all mapped source insights
        and asks the LLM which insight IDs are reflected in the theme summary.
        Insights not returned as reflected are treated as orphans.

        The method checks coverage, not conceptual quality. It does not determine
        whether the populated summary is well written, only whether mapped insight
        content appears to be represented.

        Citations are appended to insight text during the audit so the LLM can
        assess whether cited source material is reflected in the thematic summary.

        `theme_id` is cast to integer type before return to preserve downstream
        merge and ordering consistency.
        """

        if mode not in ["replace", "append"]:
            raise ValueError("Mode must be either 'replace' or 'append'.")

        # Pickle path attribute gets set in self.address_orphans()
        self.orphan_pickle_resume_path
        # Get the state meta for the resume checks (so we don't recompute this for every batch in the loop))
        state_meta = {
            "corpus_hash": self.corpus_state.fingerprint(),
            "summary_hash": self.summary_state.fingerprint()
        }

        # Set the output of the loop as either the recovered df if it exists or as an empty df to populate if it does not
        checked_insights_df = pd.DataFrame(columns=["question_id", "theme_id", "insight_id", "found"]) if checked_insights_df is None else checked_insights_df

        # Get the stable questions so that i can avoid re running the orphan check on stable questions
        stable_questions = self.summary_state.theme_schema_list[-1][self.summary_state.theme_schema_list[-1]["stable"] == True]["question_id"].tolist()
        if len(stable_questions) > 0:
            # Get the orphans for the stable questions
            stable_orphans_df = self.summary_state.orphan_list[-1][self.summary_state.orphan_list[-1]["question_id"].isin(stable_questions)].copy() 
            # Add to the checked orphans 
            if not checked_insights_df.empty:
                checked_insights_df = pd.concat([checked_insights_df, stable_orphans_df], ignore_index=True)
                checked_insights_df = checked_insights_df.drop_duplicates(subset = ["question_id", "theme_id", "insight_id"])
            else:
                checked_insights_df = stable_orphans_df.copy()

        checked_insight_id_list = checked_insights_df["insight_id"].tolist() if not checked_insights_df.empty else []
            
        total_batches_to_check = math.ceil(len(self.corpus_state.insights) / batch_size)
        count = (checked_insights_df.shape[0] // batch_size) + 1 if checked_insights_df is not None else 0

        # Now call the loop on these temp dataframes which will allow me to skip the insights that have already been checked and saved in the checked_insights_df which is being updated and saved to pickle as we go to allow for resumption if the process is interupted
        theme_map = self.summary_state.mapped_theme_list[-1].copy() # we need this in the loop, but to prevent copying each loop, i put it here
        for _, row in self.summary_state.populated_theme_list[-1].iterrows():
            # This runs question by question so first we iterate over those
            t_id, q_id, thematic_summary = row["theme_id"], row["question_id"], row["thematic_summary"]
            # Then we get the insights that were initially allocated to that question - as we are checking whether these got allocated
            relevant_ids = theme_map[(theme_map["theme_id"] == t_id) & (theme_map["question_id"] == q_id)]["insight_id"].tolist()
            relevant_insights = self.corpus_state.insights[
                (self.corpus_state.insights["insight_id"].isin(relevant_ids)) & # Get the insight text for the insight_ids that were mapped
                (~self.corpus_state.insights["insight_id"].isin(checked_insight_id_list)) # SKIP checked insights
            ].dropna(subset=["insight_id"]).copy()

            # Add the in text citation to the insight text to make sure its included in the LLM check
            relevant_insights["insight"] = relevant_insights["insight"] + " (" + relevant_insights["in_text_citation"] + ")"

            for i in range(0, len(relevant_insights), batch_size):
                print(f"Checking insights for theme {t_id} and question {q_id}, batch {count} of {total_batches_to_check}...")
                count += 1
                batch = relevant_insights.iloc[i : i + batch_size]
                insight_str = "\n".join(f"{rid}: {itxt}" for rid, itxt in zip(batch["insight_id"], batch["insight"]))

                sys_prompt = Prompts().identify_orphans()
                user_prompt = (
                    "THEMATIC SUMMARY:\n"
                    f"{thematic_summary}\n\n"
                    "SOURCE INSIGHTS:\n"
                    f"{insight_str}\n\n"
                )
                fall_back = {"mentioned_insight_ids": []}
                json_schema = {
                    "name": "mention_audit",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                        "mentioned_insight_ids": {
                            "type": "array",
                            "description": "The list of unique IDs for the insights that are reflected in the thematic summary.",
                            "items": {
                            "type": "string"
                            }
                        }
                        },
                        "required": ["mentioned_insight_ids"],
                        "additionalProperties": False
                        }
                    }

                response = utils.call_chat_completion(
                    sys_prompt=sys_prompt,
                    user_prompt=user_prompt,
                    llm_client=self.llm_client,
                    ai_model=self.ai_model,
                    fall_back=fall_back,
                    return_json=True,
                    max_tokens=4096,
                    json_schema=json_schema
                )
                # get all the found insights as sets for subtraction to get missed insights (i.e. orphans)
                insights_found = response.get("mentioned_insight_ids", [])
                insights_found_set = set(insights_found)
                insights_found_set = {str(i) for i in insights_found}
                # Get the batch to a set of strings to match the found insights set
                batch_set = set(batch["insight_id"].tolist())
                batch_set = {str(i) for i in batch["insight_id"]}
                # Get the missed insights - the orphans
                insights_missed_set = batch_set - insights_found_set
                insights_missed = list(insights_missed_set)
                # create separate dfs with diff bool conditions for found
                batch_found_df = pd.DataFrame({
                    "question_id": q_id,
                    "theme_id": t_id,
                    "insight_id": list(insights_missed),
                    "found": False
                })   
                batch_missed_df = pd.DataFrame({
                    "question_id": q_id,
                    "theme_id": t_id,
                    "insight_id": insights_found,
                    "found": True
                })
                # Get all the results together for the batch
                batch_results_df = pd.concat([batch_found_df, batch_missed_df], ignore_index=True)
                # Append the batch to the list
                checked_insights_df = pd.concat([checked_insights_df, batch_results_df], ignore_index=True)

                # Then bundle into a dict that can be pickled. Mode keeps track of whether the resume path is intended to resume or append
                checked_insights_df_meta_state_mode = {
                    "checked_insights_df": checked_insights_df,
                    "state_meta": state_meta,
                    "mode": mode
                }
                os.makedirs(config.PICKLE_SAVE_LOCATION, exist_ok=True)
                # Pickle the results of the batch so that if the process is interrupted we can resume from the last batch completed without losing all progress. 
                utils.safe_pickle(checked_insights_df_meta_state_mode, self.orphan_pickle_resume_path)

        orphans_df = checked_insights_df[checked_insights_df["found"] == False].copy()
        orphans_df["theme_id"] = orphans_df["theme_id"].astype(int) # make sure this is int for merging and comparison with the theme schema
        return(orphans_df)
    

    def _summarize_failed_orphan_batch(
        self,
        orphans: str,
        question_text: str,
        theme_label: str
    ) -> str:
        """
        Summarize orphan insights that could not be integrated directly.

        Compresses a failed orphan-integration batch into a shorter diagnostic
        summary so that its content is not lost when direct reintegration fails.
        This method is used as a fallback pathway when orphan insertion cannot
        produce a complete revised theme summary, often because the orphan batch is
        too large or the generated output is truncated.

        Parameters
        ----------
        orphans : str
            Formatted orphan-insight text to summarize.

        question_text : str
            Research question used to contextualize the orphan batch.

        theme_label : str
            Theme label associated with the failed orphan batch.

        Returns
        -------
        str
            Concise summary of the orphan batch. Returns `"No summary available."`
            if the LLM call fails or cannot be parsed.

        Notes
        -----
        This method intentionally prioritizes completion over full fidelity.
        Unlike direct orphan reintegration, it may compress, abstract, or merge
        details in order to preserve the main representational content of the
        failed batch.

        The returned summary is a diagnostic and schema-repair artifact, not a
        final thematic synthesis. It is intended to keep failed orphan content
        available for downstream repair, restructuring, or re-theming.

        A strict JSON schema is used so the response can be parsed reliably.
        """
        sys_prompt = Prompts().summarize_failed_orphan_batch()
        user_prompt = (
            f"QUESTION:\n{question_text}\n\n"
            f"THEME:\n{theme_label}\n\n"
            f"ORPHANS:\n{orphans}\n\n"
        )

        json_schema = {
            "name": "failed_orphan_batch_summarizer",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string"
                    }
                },
                "required": ["summary"],
                "additionalProperties": False
            }
        }

        fall_back = {"summary": "No summary available."}

        response = utils.call_chat_completion(
            sys_prompt=sys_prompt,
            user_prompt=user_prompt,
            llm_client=self.llm_client,
            ai_model=self.ai_model,
            fall_back=fall_back,
            return_json=True,
            json_schema=json_schema, 
            max_tokens = 4096
        )

        response_summary = response["summary"]

        return response_summary

    def _identify_missing_citations(self, summary: str, required_citations: list) -> list:
        """
        Identify citations missing from a synthesized summary.

        Compares a set of required citation strings against a generated summary
        and returns any citations that do not appear in the summary text. This
        helper is used during citation-repair and orphan-handling workflows to
        detect source references that may have been omitted during synthesis.

        Parameters
        ----------
        summary : str
            Generated summary text to inspect.

        required_citations : list
            Citation strings that should be represented in the summary.

        Returns
        -------
        list[str]
            Citation strings from `required_citations` that do not appear in the
            summary.

            Returns an empty list if:

            - no required citations are supplied
            - all required citations are present

        Notes
        -----
        Citation matching is performed using simple string containment checks.
        Citations are normalized by converting to strings and stripping leading
        and trailing whitespace before comparison.

        Duplicate citation requirements are removed before checking.

        This method verifies citation presence only. It does not determine
        whether a citation is attached to the correct claim or whether the cited
        content is adequately represented in the summary.
        """
        if not required_citations:
            return []

        required_citations = [str(i).strip() for i in required_citations]
        required_citations = list(dict.fromkeys(required_citations))  # deduplicate defensively

        missing_citations = [
            i for i in required_citations
            if i not in summary
        ]

        return missing_citations

    
    def _address_missing_citations(
        self,
        summary: str,
        missing_citations_df: pd.DataFrame
    ) -> str:
        """
        Repair citation omissions in a synthesized thematic summary.

        Uses the insights associated with missing citations to generate targeted
        sentence-level patches for a thematic summary. The method asks the LLM to
        propose either revisions to existing sentences or new sentences anchored
        after existing text, then applies those patches deterministically.

        Parameters
        ----------
        summary : str
            Thematic summary text to repair.

        missing_citations_df : pd.DataFrame
            DataFrame containing omitted citation evidence. Expected columns are:

            - `in_text_citation`
            - `insight`

        Returns
        -------
        str
            Summary text after applying any valid citation-repair patches. If no
            missing citations are supplied, returns the original summary unchanged.

        Notes
        -----
        Missing citations are grouped by `in_text_citation` before being passed to
        the LLM, along with the associated insight texts. The LLM returns structured
        patch instructions rather than a fully rewritten summary.

        Patches are applied only when their target sentence is found exactly in the
        current summary. This prevents uncontrolled rewrites and keeps the repair
        process deterministic.

        Two patch types are supported:

        - revise an existing sentence
        - insert a new sentence after an anchor sentence

        This method repairs citation presence, not full evidentiary accuracy. It
        does not verify that citations are attached to the most appropriate claims
        beyond the LLM-generated patch instructions.

        The method may print missing-citation inputs, prompts, and proposed patches
        for debugging.
        """

        if missing_citations_df.empty:
            return summary

        missing_citations_grouped = (
            missing_citations_df
            .groupby("in_text_citation")["insight"]
            .apply(list)
            .reset_index()
            .to_dict(orient="records")
        )

        missing_citations_json = json.dumps(missing_citations_grouped, indent=2, ensure_ascii=False)

        user_prompt = (
            f"THEMATIC SUMMARY:\n{summary}\n\n"
            f"MISSING CITATIONS:\n{missing_citations_json}\n\n"
        )

        sys_prompt = Prompts().repair_citation_provenance()

        json_schema = {
            "name": "repair_citation_provenance",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "patches": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "missing_citations": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "revise": {
                                    "type": "boolean"
                                },
                                "original_sentence": {
                                    "type": "string"
                                },
                                "revised_sentence": {
                                    "type": "string"
                                },
                                "anchor_sentence": {
                                    "type": "string"
                                },
                                "new_sentence": {
                                    "type": "string"
                                }
                            },
                            "required": [
                                "missing_citations",
                                "revise",
                                "original_sentence",
                                "revised_sentence",
                                "anchor_sentence",
                                "new_sentence"
                            ],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["patches"],
                "additionalProperties": False
            }
        }

        fall_back = {
            "patches": []
        }

        response = utils.call_chat_completion(
            sys_prompt=sys_prompt,
            user_prompt=user_prompt,
            llm_client=self.llm_client,
            ai_model=self.ai_model,
            fall_back=fall_back,
            return_json=True,
            json_schema=json_schema,
            max_tokens=4096
        )

        for patch in response["patches"]:
            # Check whether the sentences got returned correctly
            if patch["revise"]:

                if patch["original_sentence"] not in summary:
                    print("WARNING: original sentence not found")
                    continue

            else:

                if patch["anchor_sentence"] not in summary:
                    print("WARNING: anchor sentence not found")
                    continue

            # Then deterministically insert the revised sentences
            if patch["revise"]:

                summary = summary.replace(
                    patch["original_sentence"],
                    patch["revised_sentence"],
                    1
                )

            else:

                summary = summary.replace(
                    patch["anchor_sentence"],
                    patch["anchor_sentence"] + " " + patch["new_sentence"],
                    1
                )

        return summary


    def _load_failed_themes(self, remove_latest_iteration: bool = False) -> defaultdict:
        """
        Load and reconcile persisted failed-theme records.

        Retrieves the stored history of theme-population failures and aligns it
        with the current synthesis state. The method removes records associated
        with impossible future iterations and optionally excludes failures from
        the most recent visible synthesis pass.

        Failed-theme records are used during schema repair and re-theming to
        preserve information that could not be fully integrated during theme
        population. These records provide evidence of representational overload,
        truncation, or other synthesis failures that may indicate a need for
        schema restructuring.

        Parameters
        ----------
        remove_latest_iteration : bool, default=False
            If True, remove failure records associated with the latest visible
            synthesis iteration.

            This is primarily used during schema-generation workflows where the
            most recent failed content is already represented in the current
            populated-theme outputs through embedded failure summaries.

        Returns
        -------
        collections.defaultdict[list]
            Dictionary keyed by research-question identifier containing lists of
            failed-theme records.

            Returns an empty `defaultdict(list)` if no failed-theme file exists.

        Notes
        -----
        Failed-theme records are loaded from:

            {config.FAILED_THEMES_PATH}/failed_themes.json

        The method reconciles failure history against the current synthesis state
        using:

            summarize_iteration = len(self.summary_state.orphan_list)

        Records whose iteration exceeds the current visible synthesis state are
        discarded. These can arise from interrupted runs, partial crashes, or
        state rewinds that leave persisted failure records ahead of the active
        SummaryState.

        When `remove_latest_iteration=True`, failures from the latest visible
        iteration are also removed because they are already represented in the
        current populated themes through embedded `"FAILED BATCH SUMMARIES"`
        content and should not be double-counted during schema repair.

        The returned object is always a `defaultdict(list)` to simplify
        downstream accumulation and lookup operations.
        """
        os.makedirs(config.FAILED_THEMES_PATH, exist_ok=True)

        failed_themes_path = Path(config.FAILED_THEMES_PATH) / "failed_themes.json"

        if not failed_themes_path.is_file():
            return defaultdict(list)

        with open(failed_themes_path, "r", encoding="utf-8") as f:
            failed_themes_dict = json.load(f)

        summarize_iteration = len(self.summary_state.orphan_list)

        for k in failed_themes_dict.keys():
            # Drop records from impossible/future iterations caused by partial crashes.
            failed_themes_dict[k] = [
                ft for ft in failed_themes_dict[k]
                if ft["iteration"] <= summarize_iteration + 1
            ]

            # For schema generation, omit latest visible failures because they are
            # already represented in populated themes with FAILED BATCH SUMMARIES.
            if remove_latest_iteration:
                failed_themes_dict[k] = [
                    ft for ft in failed_themes_dict[k]
                    if ft["iteration"] < summarize_iteration
                ]

        return defaultdict(list, failed_themes_dict)        

    def _integrate_orphans(
        self,
        orphans_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Reintegrate orphan insights and audit thematic coverage.

        Performs the coverage-recovery stage of the ReadingMachine synthesis
        workflow. Orphan insights identified during completeness auditing are
        reintegrated into their assigned theme summaries, ensuring that mapped
        insight content is represented before schema refinement and re-theming.

        The integration workflow proceeds as follows:

            populated themes
                ↓
            orphan identification
                ↓
            capacity-bounded reintegration
                ↓
            citation-provenance repair
                ↓
            coverage-corrected themes

        Stable Theme Reuse
        ------------------
        Themes belonging to research questions whose schemas are marked stable
        are not modified.

        Stable themes are carried forward unchanged and are excluded from orphan
        integration.

        Capacity-Bounded Integration
        ----------------------------
        Orphan insights are integrated in large batches sized according to a
        practical model-input budget.

        For each theme:

            current summary
                ↓
            orphan batch
                ↓
            summary revision
                ↓
            next orphan batch
                ↓
            final integrated summary

        Batching is deterministic and capacity-driven. Orphans are never
        discarded simply because they exceed a context limit.

        Failure Handling
        ----------------
        If a batch cannot be successfully integrated, the batch is summarized
        using `_summarize_failed_orphan_batch()`.

        These summaries are:

        - appended to the theme output
        - recorded as failed-theme diagnostics
        - preserved for future schema-repair operations

        This allows representational failures to become explicit repair signals
        rather than silent information loss.

        Citation Provenance Repair
        --------------------------
        After successful integration, the method verifies that all citations
        associated with mapped insights remain represented in the summary.

        Missing citations are identified and repaired using:

            _identify_missing_citations()
                ↓
            _address_missing_citations()

        This preserves traceability between synthesized narratives and the
        underlying source corpus.

        Parameters
        ----------
        orphans_df : pd.DataFrame
            Orphan-insight table containing at least:

            - `question_id`
            - `theme_id`
            - `insight_id`

        Returns
        -------
        pd.DataFrame
            Updated populated-theme table containing:

            - `thematic_summary`
            - `question_id`
            - `theme_id`
            - `theme_label`
            - `theme_description`
            - `question_text`
            - `stable`
            - `needs_repair`
            - `optimized`

        Side Effects
        ------------
        Updates and persists failed-theme history in:

            {config.FAILED_THEMES_PATH}/failed_themes.json

        Uses:

        - `self.summary_state.populated_theme_list[-1]`
        - `self.summary_state.mapped_theme_list[-1]`
        - `self.summary_state.theme_schema_list[-1]`

        to determine integration targets and repair metadata.

        Notes
        -----
        This method enforces coverage rather than thematic optimization. Its
        purpose is to ensure that mapped insights are represented in the
        synthesis output before subsequent schema evaluation.

        Themes that experience integration failures are marked:

            needs_repair = True

        These flags become important inputs to later schema-repair and
        re-theming workflows.

        Failed orphan batches are intentionally preserved rather than discarded.
        In ReadingMachine, inability to integrate information is treated as
        evidence that the current thematic structure may be overloaded or
        conceptually unstable.

        Citation repair is performed only when orphan integration succeeds
        without failed batches. Themes containing failed batch summaries skip
        citation-provenance repair because their content is already known to be
        incomplete.

        The returned DataFrame is sorted by:

            question_id
            theme_id

        to preserve alignment with the active theme schema and support
        deterministic rendering.
        """

        # Load failed themes so that we can update as we go - keep all iterations to preserve history
        failed_themes = self._load_failed_themes(remove_latest_iteration=False)        

        # Prepare output holder for updated theme summaries
        updated_summary_df_lst = []

        #Get the stable themes so that I can add them back in at the end without modification, as these should not be changed and we want to preserve the summaries for these themes as they are.  
        stable_populated_themes = self.summary_state.populated_theme_list[-1][self.summary_state.populated_theme_list[-1]["stable"] == True].copy() 
        unstable_populated_themes = self.summary_state.populated_theme_list[-1][self.summary_state.populated_theme_list[-1]["stable"] == False].copy()

        total_themes = len(unstable_populated_themes)
        count = 1

        total_batches_all_themes = 0

        # ---- Capacity parameters (tune once) ----
        MODEL_INPUT_WORD_LIMIT = 22000   # practical integration limit (well below 128k hard cap)
        # ----------------------------------------

        # Packs as many insights as possible into a single batch without exceeding capacity
        # This maximizes efficiency (few passes) while avoiding model overload
        def pack_batch(insights_df, available_words):
            # Calculate word counts for all rows at once
            word_counts = insights_df["insight"].str.split().str.len()
            
            # Calculate running total and find where it exceeds the limit
            running_total = word_counts.cumsum()
            
            # Filter for rows that stay under the limit
            return insights_df[running_total <= available_words]

        # Iterate over each theme (one row per theme summary)
        for _, row in unstable_populated_themes.iterrows():
            theme_id = int(row["theme_id"])
            theme_label = row["theme_label"]
            question_id = row["question_id"]
            optimized = row.get("optimized", False)
            stable = row.get("stable", False)
            
            # Retrieve research question text for prompt context
            question_text = self.corpus_state.questions[
                self.corpus_state.questions["question_id"] == question_id
            ]["question_text"].iloc[0]

            theme_description = row["theme_description"]
            thematic_summary = row["thematic_summary"]
            
            # Conditionally include organizing proposition if the flag is set
            organizing_proposition = (
            row.get("organizing_proposition")
            if self.use_organizing_proposition
            else None
        )
            
            # Get orphan insight IDs for this theme + question
            theme_orphans = orphans_df[
                (orphans_df["theme_id"] == theme_id) &
                (orphans_df["question_id"] == question_id)
            ]

            if not theme_orphans.empty:

                print(f"Integrating orphans for theme {count} of {total_themes} (Theme ID: {theme_id})...")

                # Resolve orphan IDs → actual insight text
                orphan_data = self.corpus_state.insights[
                    self.corpus_state.insights["insight_id"].isin(theme_orphans["insight_id"])
                ].copy()
                # Add the in_text_citations to the orphan data insights
                orphan_data["insight"] = orphan_data["insight"] + " (" + orphan_data["in_text_citation"] + ")"

                # ---- iterative large-batch integration ----
                # We process all orphans, but in a few large capacity-safe batches
                remaining = orphan_data.copy()
                updated_summary = thematic_summary

                # Setup per theme batch counter
                batch_count = 0
                total_orphans = remaining.shape[0]

                failed_batch_summaries = []

                # cumulative list of orphan citations/authors
                required_citations_seen = []

                while not remaining.empty:

                    batch = pack_batch(remaining, MODEL_INPUT_WORD_LIMIT)

                    # Edge case: single very long insight → force it through
                    if batch.empty:
                        batch = remaining.iloc[[0]]

                    # increment batch counter
                    batch_count += 1
                    total_batches_all_themes += 1

                    # Format batch for prompt (bullet list preserves separability)
                    batch = batch.copy() # Avoid SettingWithCopyWarning

                    
                    # Format citations for the batch, ensuring uniqueness and cleanliness
                    batch_citations_unique = batch["in_text_citation"].unique().tolist()
                    
                    # Accumulate citations/authors across batches for this theme
                    required_citations_seen = list(dict.fromkeys(
                        required_citations_seen + batch_citations_unique
                    ))

                    # Convert cumulative citation list to string as bulleted list
                    batch_citations_str = "\n".join([f"- {c}" for c in required_citations_seen])
                    # get the insights text as bulleted list
                    batch_insights_str = "\n".join([f"- {i}" for i in batch["insight"].tolist()])

                    # Build LLM prompt
                    sys_prompt = Prompts().integrate_orphans(provide_organizing_proposition=self.use_organizing_proposition)

                    # Conditionally include organizing proposition in the user prompt if the flag is set and the proposition is not null
                    organizing_proposition_input = ""

                    if (self.use_organizing_proposition and pd.notna(organizing_proposition)):
                        organizing_proposition_input = (
                            "ORGANIZING PROPOSITION:\n"
                            f"{organizing_proposition}\n\n"
                        )

                    user_prompt = (
                        f"RESEARCH QUESTION: {question_text}\n"
                        f"THEME LABEL: {theme_label}\n\n"
                        "THEME DESCRIPTION:\n"
                        f"{theme_description}\n\n"
                        f"{organizing_proposition_input}"
                        "ORIGINAL SUMMARY:\n"
                        f"{updated_summary}\n\n"
                        "REQUIRED ORPHAN CITATIONS/AUTHORS:\n"
                        f"{batch_citations_str}\n\n"
                        "ORPHAN INSIGHTS:\n"
                        f"{batch_insights_str}\n\n"
                    )

                    # Fallback ensures no regression if model fails
                    fall_back = {"updated_summary": updated_summary}

                    # Enforce strict structured output
                    json_schema = {
                        "name": "orphan_integrator",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "updated_summary": {"type": "string"}
                            },
                            "required": ["updated_summary"],
                            "additionalProperties": False
                        }
                    }

                    # Call LLM to integrate current batch into summary
                    response, error = utils.call_chat_completion(
                        sys_prompt=sys_prompt,
                        user_prompt=user_prompt,
                        llm_client=self.llm_client,
                        ai_model=self.ai_model,
                        fall_back=fall_back,
                        return_json=True,
                        json_schema=json_schema, 
                        max_tokens=4096, 
                        return_with_error=True
                    )

                    # if there is an error
                    if error is not None or response["updated_summary"] == "":
                        print(f"WARNING: Orphan integration failed for theme {theme_id} batch {batch_count}. Capturing summary of failed batch for diagnostics and proceeding with original summary.\nError details: {error if error else 'No error message returned.'}")
                        failed_batch_summary = self._summarize_failed_orphan_batch(
                            orphans=batch_insights_str,
                            question_text=question_text,
                            theme_label=theme_label
                        )

                        failed_batch_details = (
                            f"failed batch summary {batch_count}\n"
                            f"Reason for failure: {error if error else 'Returned empty summary'}\n\n"
                            f"Summary of failed batch contents:\n{failed_batch_summary}\n\n"
                        )

                        failed_batch_summaries.append(failed_batch_details)
                    
                    else:
                                         
                        # Update working summary with integrated content
                        updated_summary = response.get("updated_summary", updated_summary)

                    # Remove processed insights from remaining obligation set
                    remaining = remaining.iloc[len(batch):].copy()

                avg_batch_size = total_orphans / batch_count if batch_count > 0 else 0
                print(
                    f"Theme {theme_id}: {total_orphans} orphans integrated in "
                    f"{batch_count} batches (avg {avg_batch_size:.1f} insights/batch)"
                )

                # Check if there were failed batch summaries
                if failed_batch_summaries:
                    # Skip citation provenance
                    print("Skipping citation provenance check because there were failed batch summaries.")
                    # Update the summary with the failures so they can go back to the model
                    updated_summary += "\n\n--- FAILED BATCH SUMMARIES ---\n\n" + "\n\n".join(failed_batch_summaries) + "\n\n" 
                    # Add the failed themes to the failed themes dict to be saved and used in the next schema gen pass
                    # Get the theme instructions
                    instructions = self.summary_state.theme_schema_list[-1][
                        (self.summary_state.theme_schema_list[-1]["theme_id"] == theme_id) &
                        (self.summary_state.theme_schema_list[-1]["question_id"] == question_id)
                    ]["instructions"].iloc[0]
                    
                    failed_theme_record = {
                        "theme_id": theme_id,
                        "theme_label": theme_label,
                        "theme_description": theme_description,
                        "instructions": instructions,
                        "iteration": len(self.summary_state.orphan_list) + 1,
                    }

                    if self.use_organizing_proposition:
                        failed_theme_record["organizing_proposition"] = organizing_proposition

                    failed_themes[question_id].append(failed_theme_record)

                else: 
                    # If there were no failed batches we proceed with citation provenance repair
                    # First get all the insights that should be in this theme, from there get paper_ids
                    print("Checking for missing citations in the updated summary...")
                    required_insights = self.summary_state.mapped_theme_list[-1][
                        (self.summary_state.mapped_theme_list[-1]["theme_id"] == theme_id) &
                        (self.summary_state.mapped_theme_list[-1]["question_id"] == question_id)
                        ]["insight_id"].to_list()

                    required_citations = (
                        self.corpus_state.insights[self.corpus_state.insights["insight_id"].isin(required_insights)]["in_text_citation"]
                        .dropna()
                        .drop_duplicates()
                        .astype(str)
                        .to_list()
                    )
                    
                    # Have the LLM find the missing citations
                    theme_missing_citations = self._identify_missing_citations(updated_summary, required_citations)

                    # If there are missing citations, attempt to repair them
                    if theme_missing_citations:
                        print(f"Missing citations identified for theme {theme_id}. Attempting to repair citation provenance in the summary...")

                        #Convert to strings 
                        theme_missing_citations = [str(x) for x in theme_missing_citations]
                        # I am going to get insights for paper_ids from corpus_state.insights and then filter for those insights mapped to this theme and question
                        # First get all insights
                        insights = self.corpus_state.insights.copy()
                        insights["in_text_citation"] = insights["in_text_citation"].astype(str)
                        # then prepare mapping
                        mapped = self.summary_state.mapped_theme_list[-1].copy()

                        mapped_theme_insight_ids = mapped[
                            (mapped["question_id"] == question_id) &
                            (mapped["theme_id"] == theme_id)
                        ]["insight_id"].dropna().tolist()

                        # take the intersection
                        theme_missing_citations_df = (
                            # First we take the intersection
                            insights[
                            (insights["in_text_citation"].isin(theme_missing_citations)) &
                            (insights["insight_id"].isin(mapped_theme_insight_ids))
                        ]   # Then we sample
                            #.groupby("in_text_citation", group_keys=False)
                            #.apply(lambda x: x.sample(n=min(len(x), 20), random_state=config.RANDOM_SEED))
                            .reset_index(drop=True)
                            [["in_text_citation", "insight_id", "insight"]]
                            .copy()
                        )

                        # Have the LLM repair the summary to add in missing citations
                        updated_summary = self._address_missing_citations(updated_summary, theme_missing_citations_df)

                # Store final fully integrated summary for this theme
                updated_row_data = {
                    "thematic_summary": updated_summary,
                    "question_id": question_id,
                    "theme_id": int(theme_id),
                    "theme_label": theme_label,
                    "theme_description": theme_description,
                    "question_text": question_text,
                    "stable": stable,
                    "needs_repair": True if failed_batch_summaries else False,
                    "optimized": optimized,
                }

                if self.use_organizing_proposition:
                    updated_row_data["organizing_proposition"] = organizing_proposition

                updated_row = pd.DataFrame([updated_row_data])

            else:
                # No orphans → summary already complete
                print(f"No orphans found for theme {count} of {total_themes}. Skipping integration.")
                updated_row_data = {
                    "thematic_summary": thematic_summary,
                    "question_id": question_id,
                    "theme_id": int(theme_id),
                    "theme_label": theme_label,
                    "theme_description": theme_description,
                    "question_text": question_text,
                    "stable": stable,
                    "needs_repair": False,
                    "optimized": optimized,
                }

                if self.use_organizing_proposition:
                    updated_row_data["organizing_proposition"] = organizing_proposition

                updated_row = pd.DataFrame([updated_row_data])

            updated_summary_df_lst.append(updated_row)
            count += 1

        # Reassemble full dataframe and restore canonical ordering
        theme_no_orphans = pd.concat(updated_summary_df_lst, ignore_index=True)

        # Add back the stable themes
        if not stable_populated_themes.empty:
            stable_columns = [
                "thematic_summary",
                "question_id",
                "theme_id",
                "theme_label",
                "theme_description",
            ]

            if self.use_organizing_proposition:
                stable_columns.append("organizing_proposition")

            stable_columns.extend(
                [
                    "question_text",
                    "stable",
                    "needs_repair",
                    "optimized",
                ]
            )

            stable_populated_themes = stable_populated_themes[stable_columns].copy()
            theme_no_orphans = pd.concat([theme_no_orphans, stable_populated_themes], ignore_index=True)

        # Sort for outputing in the correct order - before this make sure the theme_id is int to prevent sorting issues
        theme_no_orphans["theme_id"] = theme_no_orphans["theme_id"].astype(int)
        theme_no_orphans = theme_no_orphans.sort_values(
            by=["question_id", "theme_id"]
        ).reset_index(drop=True)

        # Save the failed themes so that they can be used in the next schema gen
        os.makedirs(config.FAILED_THEMES_PATH, exist_ok=True)
        failed_themes_path = Path(config.FAILED_THEMES_PATH) / "failed_themes.json"
        with open(failed_themes_path, "w", encoding="utf-8") as f:
            json.dump(dict(failed_themes), f, indent=2, ensure_ascii=False) # We want to work with a defaultdict, but serialize a regular dict for stabiliy

        return theme_no_orphans

    def _get_orphans_and_updated_summary(
        self,
        checked_insights_df,
        mode,
        batch_size
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run orphan detection and reintegration.

        Coordinates the two internal stages of orphan handling: first auditing
        populated theme summaries for omitted mapped insights, then reintegrating
        any identified orphan insights into the relevant summaries.

        Parameters
        ----------
        checked_insights_df : pd.DataFrame or None
            Previously checked orphan-audit results, usually recovered from a
            checkpoint. Passed through to `_identify_orphans()`.

        mode : str
            Orphan-handling mode passed to `_identify_orphans()`.

            Must be one of:

            - `"replace"`
            - `"append"`

        batch_size : int
            Number of mapped insights included in each orphan-audit LLM request.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            Tuple containing:

            - `orphans_df`: orphan insights identified by the audit
            - `updated_summary_df`: populated themes after orphan integration

        Notes
        -----
        If no orphans are identified, the latest populated-theme pass is returned
        unchanged as `updated_summary_df`.

        This method is a coordination helper only. Orphan auditing is delegated to
        `_identify_orphans()`, and reintegration is delegated to
        `_integrate_orphans()`.
        """

        orphans_df = self._identify_orphans(checked_insights_df=checked_insights_df, mode = mode, batch_size=batch_size)
        if orphans_df.shape[0] == 0:
            print("No orphans identified. All insights mapped to themes are reflected in the thematic summaries.")
            # return the last populated theme as nothing is being updated
            updated_summary_df = self.summary_state.populated_theme_list[-1]
        else:
            print(f"{orphans_df.shape[0]} orphan insights identified. Running integration process to integrate them back into the thematic summaries...")
            updated_summary_df = self._integrate_orphans(orphans_df)
        return(orphans_df, updated_summary_df)


    def address_orphans(
        self,
        force: bool = False,
        batch_size: int = 75
    ) -> pd.DataFrame:
        """
        Audit thematic coverage and reintegrate orphan insights.

        Public entry point for the orphan-handling stage of the ReadingMachine
        synthesis workflow. This method identifies mapped insights that are not
        represented in populated theme summaries and, when necessary,
        reintegrates those insights into the affected summaries.

        Orphan handling functions as the primary coverage-correction mechanism
        within ReadingMachine:

            populated themes
                ↓
            orphan audit
                ↓
            orphan reintegration
                ↓
            coverage-corrected themes
                ↓
            schema refinement

        Resume and Recovery
        -------------------
        The method supports checkpoint-based recovery.

        When an orphan-identification process is interrupted, progress is stored
        in:

            orphan_check_in_progress.pickle

        On subsequent runs the user may:

            (1) resume the previous audit
            (2) discard the checkpoint and start again

        Resume operations verify both CorpusState and SummaryState fingerprints
        before continuing.

        Existing Orphan Passes
        ----------------------
        The method maintains alignment between populated-theme passes and orphan
        passes.

        When orphan outputs already exist for the latest populated themes, the
        user may:

            (1) reload the existing orphan audit
            (2) rerun orphan identification and reintegration

        Reruns replace the latest orphan pass rather than appending a new one.

        Force Mode
        ----------
        If `force=True`, sequencing checks and history-alignment safeguards are
        bypassed.

        The resulting orphan audit is appended directly to
        `SummaryState.orphan_list` regardless of the current synthesis state.

        Parameters
        ----------
        force : bool, default=False
            If True, bypass sequencing validation and append a new orphan audit.

        batch_size : int, default=75
            Number of mapped insights included in each orphan-audit request.

        Returns
        -------
        pd.DataFrame or None
            Orphan-insight DataFrame for the active orphan pass.

            Returns None when the user chooses to reload an existing orphan audit
            rather than rerun orphan handling.

        Raises
        ------
        ValueError
            If no populated themes exist.

        ValueError
            If a resume attempt detects that CorpusState or SummaryState has
            changed since the checkpoint was created.

        Side Effects
        ------------
        May mutate:

        - `self.summary_state.populated_theme_list`
        - `self.summary_state.orphan_list`

        Updates the latest populated-theme pass with coverage-corrected
        summaries.

        Persists SummaryState after successful completion.

        Creates, reads, and removes:

            orphan_check_in_progress.pickle

        used for checkpoint-based recovery.

        Notes
        -----
        Orphan detection is delegated to `_identify_orphans()`.

        Orphan reintegration is delegated to `_integrate_orphans()`.

        Themes that cannot successfully integrate all orphan content are marked
        as requiring repair. These failures become important inputs to the next
        schema-generation cycle.

        The orphan stage is not primarily a quality-improvement step. Its
        purpose is to enforce coverage by ensuring that mapped insights are
        represented before schema optimization and re-theming are attempted.

        Orphan audits are stored as iterative passes in
        `self.summary_state.orphan_list`, preserving the history of coverage
        corrections across schema iterations.
        """
        # The logic for this function is as follows:
        # First check if there is a resume file, if so resume according to the mode passed originally
        # Then run a force flag, which passes the mode append and skips the sequencing validation checks
        # Finally, if no force and no reusme, check the state to determine the sequencing needs and offer the user to replace or extend, and then pass that mode to the call - so that it can resume correctly if it crashes

        # Make sure the required state elements exist to check for orphans
        if len(self.summary_state.populated_theme_list) == 0:
            raise ValueError("No populated themes found. Please run populate_themes() first.")

        # ####
        # RESUME LOGIC
        # ### 
        # First check resume logic - this effectively sets the mode and the checked_insights_df
        checked_insights_df = None
        self.orphan_pickle_resume_path = os.path.join(config.PICKLE_SAVE_LOCATION, "orphan_check_in_progress.pickle")
        if os.path.exists(self.orphan_pickle_resume_path):
            resume = None
            while resume not in ["1", "2"]:
                resume = input("A partial orphan identification process was detected. Do you want to:\n"
                               "1) resume from the last saved point? \n"
                               "2) start a new orphan identification process? \n"
                               "Enter 1 or 2:\n").lower()
            if resume == "1":
                with open(self.orphan_pickle_resume_path, "rb") as f:
                    checked_insights_df_meta_state = pickle.load(f)
                    # Check that the pickled state matches the current state
                    pickled_state = checked_insights_df_meta_state["state_meta"]
                    current_state = {
                        "corpus_hash": self.corpus_state.fingerprint(),
                        "summary_hash": self.summary_state.fingerprint()
                    }
                    if pickled_state != current_state:
                        raise ValueError(
                            "The corpus state or summary state has changed between resume. Resume invalid"
                            "Either choose a fresh start or if you know what you are doing use force = True" 
                            )
                    # Load the checked insights data frame from the pickle to resume the process
                    checked_insights_df = checked_insights_df_meta_state["checked_insights_df"]
                    mode = checked_insights_df_meta_state["mode"]

                    print("Resuming orphan identification process from last saved point...")
                    orphans_df, updated_summary_df = self._get_orphans_and_updated_summary(checked_insights_df=checked_insights_df, mode=mode, batch_size=batch_size)
                    self.summary_state.populated_theme_list[-1] = updated_summary_df
                    if mode == "replace":
                        self.summary_state.orphan_list[-1] = orphans_df
                    elif mode == "append":
                        self.summary_state.orphan_list.append(orphans_df)
                    
                    # Clean up the resume pickle as the process is complete and we want to void a resume trigger if we run again
                    # Have to wrap in this try - except as windows keeps hold of the file for reasons that are not clear.
                    try:
                        os.remove(self.orphan_pickle_resume_path)
                    except PermissionError:
                        while True:
                            confirm = input(
                                "\nCould not automatically delete resume file.\n"
                                "This is a known Windows file system issue.\n\n"
                                f"Please manually delete:\n{self.orphan_pickle_resume_path}\n\n"
                                "This file will trigger resume mode on the next run if not removed.\n\n"
                                "Enter 'c' to continue:\n"
                            ).lower().strip()

                            if confirm == "c":
                                break

                    self.summary_state.save()
                    return(self.summary_state.orphan_list[-1])
 
            else:
                print("Starting new orphan identification process and deleting the in progress pickle...")
                # Again have to wrap in try except because of windows file system issues
                try:
                    os.remove(self.orphan_pickle_resume_path)
                except PermissionError:
                    while True:
                        confirm = input(
                            "\nCould not automatically delete resume file.\n"
                            "This is a known Windows file system issue.\n\n"
                            f"Please manually delete:\n{self.orphan_pickle_resume_path}\n\n"
                            "This file will trigger resume mode on the next run if not removed.\n\n"
                            "Enter 'c' to continue:\n"
                        ).lower().strip()

                        if confirm == "c":
                            break

        # #####
        # FORCE FLAG LOGIC
        # #####
        if force:
            print("WARNING: Force flag is True: Skipping validation and resumption checks and addressing orphans. This may cause the state to become unstable. This mode should be used for testing purposes only.")
            mode = "append" # This is a bit meaningless because with force we are always appending, but this makes it clear that we are not replacing any existing orphan audits and just appending a new one on top of the existing state regardless of the current state of the orphan list.       
            orphans_df, updated_summary_df = self._get_orphans_and_updated_summary(checked_insights_df=None, mode="append", batch_size=batch_size) # Mode is append because with force we are always appending. We set it here in case the system crashes, in which case the recover will propagate the append mode. We actually don't need to access the mode before this return as we just append in force
            # We always repair the populated themes with the integrated version. No change under force
            self.summary_state.populated_theme_list[-1] = updated_summary_df
            # We append orphans because this is force and we always append
            self.summary_state.orphan_list.append(orphans_df)

            # Clean up           
            if os.path.exists(self.orphan_pickle_resume_path):
                os.remove(self.orphan_pickle_resume_path)
            self.summary_state.save()
            return(self.summary_state.orphan_list[-1])
        
        # #####
        # SEQUENCING LOGIC
        # #####

        # --- Realign orphan list to coherent corpus_state ---
        min_len = min(len(self.summary_state.populated_theme_list), len(self.summary_state.orphan_list))
        self.summary_state.orphan_list = self.summary_state.orphan_list[:min_len]

        # --- Determine behavior ---
        # Essentially if the lenght of orphan and theme lists are the same the orphan has already been created for the latest theme list. If the user want to run again, we need to replace
        # the last orphan rather than extending it. If orphans is behind, then we just extend the list
        if len(self.summary_state.populated_theme_list) == len(self.summary_state.orphan_list):
            # Orphan already exists for latest populate
            choice = None
            while choice not in ["1", "2"]:
                choice = input(
                    "Orphan handling already exists for the latest populated themes.\n"
                    "(1) Reload existing orphan audit\n"
                    "(2) Re-run orphan identification and incorporation\n"
                    "Enter 1 or 2:\n"
                ).strip()

            if choice == "1":
                print("Latest orphan audit loaded.")
                return None
            # If we are not reloading, and populated themes and orphans are aligned, then we replace the most recent orphan audit with a new one - we are essentially updating the audit, so mode is replace
            mode = "replace" 
            # Otherwise if orphans is behind populated themes we are adding the new audit to align the state
        else:
            mode = "append"

        # Get the orphans and the updated summaries based on whether we are replacing or appending
        orphans_df, updated_summary_df = self._get_orphans_and_updated_summary(checked_insights_df=checked_insights_df, mode=mode, batch_size=batch_size) #Checked insights is None, based on false eval for resume. Mode propagates in case there is a crash
        self.summary_state.populated_theme_list[-1] = updated_summary_df
        if mode == "replace":
            self.summary_state.orphan_list[-1] = orphans_df
        elif mode == "append":
            self.summary_state.orphan_list.append(orphans_df)

        # Now clean up and save
        # Delete the resume pickle as the process is complete and we want to void a resume trigger if we run again
        if os.path.exists(self.orphan_pickle_resume_path):
            try:
                os.remove(self.orphan_pickle_resume_path)
            except PermissionError:
                while True:
                    confirm = input(
                        "\nCould not automatically delete resume file.\n"
                        "This is a known Windows file system issue.\n\n"
                        f"Please manually delete:\n{self.orphan_pickle_resume_path}\n\n"
                        "This file will trigger resume mode on the next run if not removed.\n\n"
                        "Enter 'c' to confirm you have read this message and continue:\n"
                    ).lower().strip()

                    if confirm == "c":
                        break

        self.summary_state.save()
        return self.summary_state.orphan_list[-1]

    
    def _llm_redundancy_check(self) -> pd.DataFrame:
        """
        Reduce redundancy across populated theme summaries.

        Performs a final sequential redundancy-reduction pass over the latest
        populated themes. Each theme summary is reviewed against the already
        refined summaries that precede it within the same research question, and
        the LLM rewrites the current theme to reduce repeated content while
        preserving information unique to that theme.

        The redundancy workflow proceeds as follows:

            populated themes
                ↓
            sequential within-question review
                ↓
            theme-level redundancy reduction
                ↓
            refined theme summaries

        Processing resets at each research question. Summaries from one research
        question are not used as context for another.

        Returns
        -------
        pd.DataFrame
            DataFrame containing refined theme summaries with the same columns as
            the latest populated-theme pass. The `thematic_summary` column is
            replaced with redundancy-reduced text.

        Notes
        -----
        The method assumes that thematic synthesis has already stabilized through
        mapping, population, orphan handling, and schema refinement.

        Redundancy reduction is applied only to the rendered summary text. It does
        not modify:

        - the theme schema
        - insight-to-theme mappings
        - orphan records
        - underlying CorpusState insights

        Because the pass is sequential, earlier refined themes shape the context
        used to revise later themes within the same research question. The ordering
        is determined by `question_id` and `theme_id`.

        A strict JSON schema is used for the LLM response. If the call fails, the
        fallback refined theme is an empty string.
        """

        ordered_themes = self.summary_state.populated_theme_list[-1].sort_values(by=["question_id", "theme_id"]).reset_index(drop=True).copy()
        refined_rows = []

        total_themes = ordered_themes.shape[0]
        count = 1

        for q_id, q_group in ordered_themes.groupby("question_id"):
            previous_theme_text = "" # Reset for each Research Question
            
            for _, row in q_group.iterrows():
                print(f"Addressing redundancy for theme {count} of {total_themes}")
                # Prepare the Payload
                question_text = row["question_text"]
                theme_label = row["theme_label"]
                current_theme_text = row["thematic_summary"]

                # System Prompt (Your structural version)
                sys_prompt = Prompts().address_redundancy()
                
                # User Prompt (Matching your INPUT FORMAT)
                user_prompt = (
                    f"RESEARCH QUESTION: {question_text}\n"
                    f"PREVIOUSLY CLEANED THEMES:\n{previous_theme_text if previous_theme_text else 'None.'}\n\n"
                    f"CURRENT THEME LABEL: {theme_label}\n"
                    f"CURRENT THEME TEXT TO REFINE:\n{current_theme_text}"
                )
                
                fall_back = {"refined_theme": current_theme_text}

                # Execute the reduction
                response = utils.call_chat_completion(
                    sys_prompt=sys_prompt,
                    user_prompt=user_prompt,
                    llm_client=self.llm_client,
                    ai_model=self.ai_model,
                    return_json=True,
                    fall_back=fall_back,
                    json_schema={
                        "name": "redundancy_reduction",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {"refined_theme": {"type": "string"}},
                            "required": ["refined_theme"],
                            "additionalProperties": False
                        }
                    }
                )

                refined_theme = response["refined_theme"]
                
                # Update context for the next theme in this RQ
                previous_theme_text += f"\n\n{refined_theme}"

                # Preserve the row data
                refined_row = row.copy()
                refined_row["thematic_summary"] = refined_theme
                refined_rows.append(refined_row)
                count += 1

        # Push to State
        refined_df = pd.DataFrame(refined_rows)
        refined_df["theme_id"] = refined_df["theme_id"].astype(int)
        return(refined_df)

    def address_redundancy(
        self,
        force: bool = False
    ) -> pd.DataFrame:
        """
        Generate, reload, or regenerate the final redundancy-reduction pass.

        Public entry point for the final text-refinement stage of the
        ReadingMachine synthesis workflow. This method reduces repeated content
        across populated theme summaries after orphan handling has completed.

        Redundancy reduction transforms:

            coverage-corrected populated themes
                ↓
            sequential redundancy review
                ↓
            refined final theme summaries
                ↓
            rendering

        Parameters
        ----------
        force : bool, default=False
            If True, bypass sequencing checks and run redundancy reduction on the
            latest populated themes regardless of orphan-state alignment.

        Returns
        -------
        pd.DataFrame
            Redundancy-reduced theme summaries.

        Raises
        ------
        ValueError
            If no populated themes exist.

        ValueError
            If orphan handling has not been completed for the current populated
            theme pass.

        Side Effects
        ------------
        Mutates:

        - `self.summary_state.redundancy_list`

        Persists SummaryState after successful redundancy reduction.

        Notes
        -----
        Redundancy reduction is intended to run after coverage has been corrected
        through orphan handling. This sequencing ensures repeated material is
        removed only after mapped insights have been represented in the populated
        summaries.

        Only one redundancy pass is retained. Re-running this method overwrites
        `self.summary_state.redundancy_list` with a single new refined DataFrame.

        Force mode is intended for development and debugging and may leave the
        synthesis state internally inconsistent.

        The underlying LLM refinement is delegated to `_llm_redundancy_check()`.
        This method handles sequencing, reload/regeneration behavior, persistence,
        and state updates.
        """
        # Make sure this can run
        if not self.summary_state.populated_theme_list:
            raise ValueError("No populated themes available for redundancy pass.")
        # Force logic: run regardless and populate redundancy attribute and return
        if force:
            print("WARNING: Force flag is True: Skipping validation and resumption checks and addressing redundancy. This may cause the state to become unstable. This mode should be used for testing purposes only.")
            # We just run the redundancy reduction on the latest populated themes and append to the redundancy list without any checks
            refined_df = self._llm_redundancy_check()
            self.summary_state.redundancy_list = [refined_df]
            self.summary_state.save()
            return self.summary_state.redundancy_list[-1]
        
        # If not force then make sure the orphans are not smaller than populated themes - i.e. you have to have run orphans before you can run redundancy. But orphans can be larger - if you for some reason iterate orphans and redundancy, orphans will grow while everythig else stays as is
        if len(self.summary_state.orphan_list) < len(self.summary_state.populated_theme_list):
            raise ValueError("The number of orphan audits and populated themes are not aligned. Please run address_orphans() to align the state before running redundancy reduction.")

        # Now check whether we want to reload or rebuild redundancy
        if len(self.summary_state.redundancy_list) == 1:
            reload = None
            while reload not in ["1", "2"]:
                reload = input(
                    "Redundancy reduction has already been run on the latest populated themes.\n"
                    "(1) Reload existing redundancy reduction\n"
                    "(2) Re-run redundancy reduction (this will overwrite the previous redundancy pass)\n"
                    "Enter 1 or 2:\n"
                ).strip()
            if reload == "1":
                print("Latest redundancy reduction loaded.")
                return self.summary_state.redundancy_list[0] # Return to exit the function and avoid re-running the reduction    
            else:
                print("Re-running redundancy reduction on the latest populated themes...")

        # Do the redundancy check, update redundancy_list, save and return
        refined_df = self._llm_redundancy_check()
        self.summary_state.redundancy_list = [refined_df]
        self.summary_state.save()
        return self.summary_state.redundancy_list[-1]

