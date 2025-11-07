# Import the two projevct modules
from lit_review_machine.outputs import QuestionState, Summaries, SUMMARY_SAVE_LOCATION
from lit_review_machine.prompts import Prompts
from lit_review_machine.utils import ensure_list_of_strings, populate_dict_recursively, call_chat_completion, call_reasoning_model

# Import other libraries
import json
import os
import pickle
from scholarly import scholarly, ProxyGenerator
from collections import defaultdict
import time
import random
import requests
from typing import Optional, Dict, Any, List, Union, Tuple
from unpywall import Unpywall
import pandas as pd
from rapidfuzz import process, fuzz
import numpy as np
import networkx as nx
from semanticscholar import SemanticScholar, SemanticScholarException
import datetime
import pymupdf
from kneed import KneeLocator
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
import umap
from sklearn.metrics import silhouette_score, davies_bouldin_score
import hdbscan
from docx import Document
import datetime
import pyarrow as pa
from copy import deepcopy
import ast
from bs4 import BeautifulSoup
from pathlib import Path
#from crossref.restful import Works
from requests.exceptions import HTTPError
import tiktoken
import itertools
import networkx as nx

 
STATE_SAVE_LOCATION = os.path.join(os.getcwd(), "data", "runs")
  

def validate_format(
    state: Optional["QuestionState"], 
    injected_value: Optional[pd.DataFrame],
    state_required_cols: List[str], 
    injected_required_cols: List[str]
) -> "QuestionState":
    """
    Validates input state or injected DataFrame for required columns.
    Returns a properly initialized QuestionState.

    Args:
        state: An existing QuestionState object (if available).
        injected_value: A DataFrame to inject into a new QuestionState if state is None.
        state_required_cols: List of columns required in state.insights.
        injected_required_cols: List of columns required in the injected DataFrame.

    Returns:
        QuestionState: A valid state object with all required columns.

    Raises:
        ValueError: If neither state nor injected_value is provided,
                    or if required columns are missing,
                    or if 'paper_id' contains any NA values.
    """
    
    if state is None and injected_value is None:
        raise ValueError(
            "Both state and injected_value cannot be None. "
            "You must initialize the class with either a valid 'state' created by prior classes "
            "or with a valid dataframe in tidy format."
        )

    elif state is not None:
        # Verify that the state already has the required columns
        if not set(state_required_cols).issubset(state.insights.columns):
            raise ValueError(
                "State does not contain all the required columns. "
                f"Expected at least {state_required_cols}."
            )
        # Check for paper_id NAs
        if "paper_id" in state.insights.columns and state.insights["paper_id"].isna().any():
            raise ValueError(
                "State contains NA values in 'paper_id'. "
                "All papers must have unique IDs. If you have injected data or modified the state, "
                "'paper_id' must be populated for each paper."
            )
        return state

    else:
        # Create new state from injected DataFrame
        state = QuestionState(insights=injected_value)
        if not set(injected_required_cols).issubset(state.insights.columns):
            raise ValueError(
                "Injected dataframe does not contain all required fields. "
                f"Expected at least {injected_required_cols}."
            )
        # Add missing columns (from state_required but not in injected)
        for field in state_required_cols:
            if field not in injected_required_cols:
                state.insights[field] = np.nan

        # Check for paper_id NAs in injected data
        if "paper_id" in state.insights.columns and state.insights["paper_id"].isna().any():
            raise ValueError(
                "Injected dataframe contains NA values in 'paper_id'. "
                "All papers must have unique IDs. You must populate 'paper_id' for each paper."
            )

        return state


   
class ScholarSearchString:
    """
    Generate search prompts and search strings for research questions using an LLM.

    Workflow:
      1. Initialize with a list of questions and an LLM client.
      2. Build a state object (QuestionState) that tracks questions and insights.
      3. Generate structured messages for each question (`message_maker`).
      4. Query the LLM for search strings and update the state (`searchstring_maker`).
    """

    def __init__(
        self,
        questions: List[str],
        llm_client: Any,
        num_prompts: int = 10,
        search_engine: str = "Semantic Scholar",
        llm_model: str = "gpt-4.1",
        state: Optional[QuestionState] = None,
        messages: Optional[List[List[Dict[str, str]]]] = None,
    ) -> None:
        """
        Initialize the ScholarSearchString object.

        Args:
            questions (List[str]): List of research questions.
            llm_client (Any): LLM API client instance (e.g., OpenAI client).
            num_prompts (int, optional): Number of search prompts per question. Defaults to 10.
            search_engine (str, optional): Search engine context. Defaults to "Semantic Scholar".
            llm_model (str, optional): LLM model name. Defaults to "gpt-4.1".
            state (Optional[QuestionState], optional): Existing QuestionState object.
            messages (Optional[List[List[Dict[str, str]]]], optional):
                Pre-generated messages. Usually left None.
        """
        self.questions: List[str] = questions
        self.llm_client: Any = llm_client
        self.num_prompts: int = num_prompts
        self.search_engine: str = search_engine
        self.llm_model: str = llm_model

        # Create or reuse a QuestionState object
        if state:
            self.state = state
        else:
            self.state = QuestionState(insights = self._make_state())

        # Messages are generated later by `message_maker`
        self.messages: Optional[List[List[Dict[str, str]]]] = messages

    def _make_state(self) -> pd.DataFrame:
        """
        Build the initial insights dataframe with question IDs and text.

        Returns:
            pd.DataFrame: DataFrame with `question_id` and `question_text`.
        """
        state = pd.DataFrame()
        question_ids: List[str] = [f"question_{i}" for i in range(len(self.questions))]
        state["question_id"] = question_ids
        state["question_text"] = self.questions
        return state

    def message_maker(self) -> List[List[Dict[str, str]]]:
        """
        Generate LLM messages for each research question.

        Returns:
            List[List[Dict[str, str]]]: List of messages per question.
        """
        sys_prompt: str = Prompts().question_make_sys_prompt(
            search_engine=self.search_engine,
            num_prompts=self.num_prompts,
        )

        # Build user prompts from question IDs and texts
        user_prompts: List[str] = []
        for question_id, question_text in zip(
            self.state.insights["question_id"], self.state.insights["question_text"]
        ):
            user_prompt: str = f"**QUESTION**\n{question_id}: {question_text}"
            user_prompts.append(user_prompt)

        # Wrap prompts into the LLM message format
        messages: List[List[Dict[str, str]]] = []
        for prompt in user_prompts:
            message = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ]
            messages.append(message)

        # Store internally (not in state to save space/memory)
        self.messages = messages

        return self.messages

    def searchstring_maker(self) -> List[str]:
        """
        Query the LLM to generate search strings for each question,
        update the state, and persist it to pickle.

        Returns:
            List[str]: Generated search strings across all questions.
        """
        # Ensure messages exist
        if self.messages is None:
            self.message_maker()

        # Initialize empty results dataframe
        search_strings_df = pd.DataFrame(columns=["question_id", "search_string"])

        # Iterate over all questions in state
        for index, question_id in enumerate(self.state.insights["question_id"]):
            message = self.messages[index]
            print(f"Generating prompts for question {index + 1} of {self.state.insights.shape[0]}")

            # Call LLM
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=message,
                response_format={"type": "json_object"},
            )

            # Parse JSON response
            response_data: Dict[str, List[str]] = json.loads(response.choices[0].message.content)
            # Use internal question_id instead of LLM key
            llm_prompts = list(response_data.values())[0]

            # Append results to dataframe
            search_strings_df = pd.concat(
                [
                    search_strings_df,
                    pd.DataFrame(
                        {
                            "question_id": [question_id] * len(llm_prompts),
                            "search_string": llm_prompts,
                        }
                    ),
                ],
                ignore_index=True,
            )

        # Add unique IDs for each search string
        search_strings_df["search_string_id"] = [
            f"search_string_{i}" for i in range(search_strings_df.shape[0])
        ]

        # Merge back into the insights dataframe
        self.state.insights = search_strings_df.merge(
            self.state.insights, how="left", on="question_id"
        )

        # Save updated state object
        self.state.save(os.path.join(STATE_SAVE_LOCATION, "01_search_strings"))

        # Return search strings for testing/debugging
        return self.state.insights["search_string"].to_list()

class AcademicLit:
    def __init__(self, 
                 state: Optional["QuestionState"] = None, 
                 search_strings: Optional[pd.DataFrame] = None) -> None:
        """
        Initialize the AcademicLit class with a validated state or search_strings.

        Args:
            state (Optional[QuestionState]): Existing QuestionState object.
            search_strings (Optional[pd.DataFrame]): DataFrame with search strings.
        """
        # Deepcopy ensures this class has its own copy of state
        self.state = deepcopy(
            validate_format(
                state=state,
                injected_value=search_strings,
                state_required_cols=["question_id", "question_text", "search_string", "search_string_id"],
                injected_required_cols=["question_id", "question_text", "search_string", "search_string_id"]
            )
        )

    def search_crossref(self, num_results: int = 20) -> pd.DataFrame:
        """
        Search CrossRef for each search string in state and update the state.

        Args:
            num_results (int): Maximum number of papers to retrieve per search string.

        Returns:
            pd.DataFrame: Updated state.insights with CrossRef results.
        """
        works: Works = Works()
        output: Dict[str, List] = {
            "search_string": [],
            "paper_title": [],
            "paper_author": [],
            "paper_date": [],
            "paper_doi": []
        }

        for search_string in self.state.search_strings['search_string']:
            try:
                results = works.query(search_string).sort('relevance').rows(num_results)
            except HTTPError as e:
                if e.response.status_code == 503:
                    print("Received 503 from CrossRef API, retrying after a short delay...")
                    time.sleep(5 + random() * 5)  # Wait between 5-10 seconds before retrying
                    results = works.query(search_string).sort('relevance').rows(num_results)
   
            for item in results:
                # Paper title
                output["paper_title"].append(item['title'][0] if 'title' in item else 'No title found')

                # Paper authors, safely handling missing given/family names
                output["paper_author"].append(
                    [i["family"] + ", " + (i["given"][0] if i.get("given") else "") 
                     for i in item.get("author", [])] or ['No author found']
                )

                # Publication year, prefer print then online, pd.NA if missing
                if 'published-print' in item and 'date-parts' in item['published-print']:
                    year: Optional[int] = item['published-print']['date-parts'][0][0]
                elif 'published-online' in item and 'date-parts' in item['published-online']:
                    year = item['published-online']['date-parts'][0][0]
                else:
                    year = pd.NA
                output["paper_date"].append(year)

                # DOI, pd.NA if missing
                output["paper_doi"].append(item['DOI'] if 'DOI' in item else pd.NA)

                output["search_string"].append(search_string)

        output_df: pd.DataFrame = pd.DataFrame(output)
        output_df["search_engine"] = "CrossRef"
        output_df["paper_id"] = [f"crossref_paper_{i + 1}" for i in range(output_df.shape[0])]

        # Merge results into state
        self.state.insights = self._merge_search_results_with_state(output_df)
        self.state.save(STATE_FILE_LOCATION)

        return self.state.insights

    def search_CORE(self, num_results: int = 20) -> pd.DataFrame:
        """
        Search CORE API for each search string in state and update the state.

        Args:
            num_results (int): Maximum number of papers to retrieve per search string.

        Returns:
            pd.DataFrame: Updated state.insights with CORE results.
        """
        core: OACore = OACore(api_key=os.getenv("CORE_API_KEY"))
        output: Dict[str, List] = {
            "search_string": [],
            "paper_title": [],
            "paper_author": [],
            "paper_date": [],
            "paper_doi": []
        }

        for search_string in self.state.search_strings['search_string']:
            results: dict = core.search(query=search_string, page_size=num_results)

            for item in results.get('results', []):
                # Paper title
                output["paper_title"].append(item.get('title', 'No title found'))

                # Authors, safely handling missing family/given names
                output["paper_author"].append(
                    [i["familyName"] + ", " + (i["givenName"][0] if i.get("givenName") else "") 
                     for i in item.get("authors", [])] or ['No author found']
                )

                # Publication year (first 4 chars of published_date)
                year: Optional[int] = int(item['published_date'][:4]) if 'published_date' in item else pd.NA
                output["paper_date"].append(year)

                # DOI, pd.NA if missing
                output["paper_doi"].append(item.get('doi', pd.NA))
                output["search_string"].append(search_string)
                time.sleep(0.6) # To respect CORE API rate limits

        output_df: pd.DataFrame = pd.DataFrame(output)
        output_df["search_engine"] = "CORE"
        output_df["paper_id"] = [f"core_paper_{i + 1}" for i in range(output_df.shape[0])]

        # Merge results into state
        self.state.insights = self._merge_search_results_with_state(output_df)
        self.state.save(STATE_FILE_LOCATION)

        return self.state.insights

    def _merge_search_results_with_state(self, search_results: pd.DataFrame) -> pd.DataFrame:
        """
        Merge search results with the question information in state.

        Args:
            search_results (pd.DataFrame): DataFrame containing search results.

        Returns:
            pd.DataFrame: Merged DataFrame with duplicates removed.
        """
        if "search_engine" in self.state.insights:
            # Merge to bring in question info and concatenate with existing results
            output_df: pd.DataFrame = search_results.merge(
                self.state.insights[["question_id", "question_text", "search_string_id", "search_string"]],
                how="left",
                on="search_string"
            )
            merged_output: pd.DataFrame = pd.concat([self.state.insights, output_df], ignore_index=True)
        else:
            # If first engine, just merge to add question info
            merged_output = search_results.merge(
                self.state.insights, how="left", on="search_string"
            )

        merged_output = merged_output.drop_duplicates().reset_index(drop=True)
        return merged_output
    
class DOI:
    """
    Retrieve DOIs and open-access download links for papers stored in a QuestionState
    object or provided as a DataFrame.
    """

    def __init__(
        self, 
        state: Optional["QuestionState"] = None, 
        papers: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Initialize DOI retriever.

        Args:
            state: A pre-existing QuestionState object containing paper metadata.
            papers: A DataFrame containing paper metadata (used if state is None).
        """
        # Validate and set up state
        self.state = deepcopy(
            validate_format(
            state=state,
            injected_value=papers,
            state_required_cols=[
                "question_id", "question_text", "search_string_id", "search_string", 
                "search_engine", "doi", "paper_id", "paper_title", "paper_author", "paper_date"
            ],
            injected_required_cols=[
                "question_id", "question_text", "paper_id", 
                "paper_title", "paper_author", "paper_date"
            ]
            )
            )

        # Ensure folder exists for pickle
        os.makedirs(os.path.dirname(STATE_FILE_LOCATION), exist_ok=True)

        # Store search strings for DOI lookup
        self.search_string: List[str] = self._create_search_string()

    def _create_search_string(self) -> List[str]:
        """
        Concatenates paper title, authors, and year into search strings for DOI lookups.

        Returns:
            List[str]: A list of search strings for each paper.
        """
        if self.state.insights.empty:
            return []

        df = self.state.insights[["paper_title", "paper_author", "paper_date"]].copy()
        df["search_string"] = (
        df["paper_title"].astype(str) + " " +
        df["paper_author"].astype(str) + " " +
        df["paper_date"].astype(str)
        )

        search_string = df["search_string"].tolist()
        return search_string

    @staticmethod
    def call_alex(search_string: str) -> Optional[str]:
        """
        Queries the OpenAlex API with a search string to retrieve a DOI.

        Args:
            search_string: A string composed of title, author(s), and year.

        Returns:
            Optional[str]: The DOI string if found, otherwise None.
        """
        url = "https://api.openalex.org/works"
        params = {"search": search_string, "per-page": 1}
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                items = response.json().get("results", [])
                if items:
                    doi_url = items[0].get("doi")
                    if doi_url:
                        return doi_url.removeprefix("https://doi.org/")
        except requests.exceptions.RequestException:
            pass

        time.sleep(1)  # Prevent hitting API rate limits
        return None

    def get_doi(self) -> List[Optional[str]]:
        """
        Retrieves DOIs for all papers in the current state using OpenAlex.

        Returns:
            List[Optional[str]]: A list of DOIs corresponding to papers.
        """
        dois = self.state.insights["doi"]

        if not self.search_string:
            print("No papers available to retrieve DOIs.")
            self.state.insights["doi"] = []
            return []

        for idx, (string, doi) in enumerate(zip(self.search_string, dois), start=1):
            print(f"Retrieving DOI {idx} of {len(self.search_string)}")
            if not pd.isna(doi):
                continue  # Skip if DOI already exists from the AcademicLit search
            else:
                doi_result = self.call_alex(string)
                dois[idx - 1] = doi_result

        self.state.insights["doi"] = dois

        return dois

    def get_download_link(self) -> List[Optional[str]]:
        """
        Retrieves open-access PDF download links for each paper via Unpywall.

        Returns:
            List[Optional[str]]: A list of open-access PDF download links (or None if unavailable).
        """
        # Ensure DOI column exists
        self.state = validate_format(
            state=self.state,
            injected_value=self.state.insights,
            state_required_cols=[
                "question_id", "question_text", "search_string_id", "search_string",
                "search_engine","paper_id", "paper_title", "paper_author", "paper_date", "doi"
            ],
            injected_required_cols=[
                "question_id", "question_text", "paper_id",
                "paper_title", "paper_author", "paper_date", "doi"
            ]
        )

        download_links: List[Optional[str]] = []

        for idx, doi in enumerate(self.state.insights.get("doi", []), start=1):
            print(f"Retrieving downlod link for paper {idx} of {self.state.insights.shape[0]}")

            if not doi:
                download_links.append(None)
                continue

            try:
                unpay = Unpywall.get_json(doi)
                oa_locations = unpay.get("oa_locations", [])
                if not oa_locations:
                    download_links.append(None)
                    continue
            except Exception as e:
                download_links.append(f"Error: {e}")
                continue

            for loc in oa_locations:
                url = loc.get("url_for_pdf")
                if url:
                    try:
                        response = requests.head(url, allow_redirects=True, timeout=10)
                        if (
                            response.status_code == 200
                            and "application/pdf" in response.headers.get("Content-Type", "")
                        ):
                            download_links.append(url)
                            break
                    except requests.exceptions.RequestException:
                        continue
            else:
                download_links.append(None)

        self.state.insights["download_link"] = download_links
        self.state.save(STATE_SAVE_LOCATION)

        return download_links

class GreyLiterature:
    """
    A class for retrieving and managing grey literature using an LLM and live web search.

    Grey literature is defined here as reports, policy briefs, working papers, and case
    studies published by think tanks, INGOs, multilateral organizations, and other 
    research institutions.

    The class integrates with a QuestionState object that tracks research questions
    and insights, ensuring that retrieved grey literature is associated with the 
    correct research question IDs.
    """

    # Default path for caching grey literature results
    def __init__(
        self,
        llm_client: Any,  # Client interface for interacting with the LLM API
        state: Optional["QuestionState"] = None,  # Current research state (can be injected)
        questions: Optional[List[str]] = None,    # User-defined research questions
        ai_model: str = "o3-deep-research",       # LLM model to use
        GREY_LIT_PICKLE_FILE: str = os.path.join(os.getcwd(), "data", "pickles", "grey_lit.pkl"), # The pickle location for the valid processed json response from the LLM
        GREY_LIT_RAW_PICKLE_FILE: str = os.path.join(os.getcwd(), "data", "pickles", "grey_lit_raw.pkl") # If the LLM fails to return a valid json the raw output gets saved here - as this is an expensive call
    ) -> None:
        """
        Initialize the GreyLiterature object.

        Args:
            llm_client (Any): Client for interacting with the language model.
            state (Optional[QuestionState]): QuestionState object holding research state.
            questions (Optional[List[str]]): User-defined research questions.
            ai_model (str): Name of the LLM model to use.
        """

        # If questions are provided directly, format them into a DataFrame
        if questions:
            question_id: List[str] = [f"Question_{i}" for i in range(len(questions))]
            questions = pd.DataFrame({
                "question_id": question_id,
                "question_text": questions
            })

        # Validate the state and inject the questions if provided
        self.state: "QuestionState" = deepcopy(
            validate_format(
                state=state,
                injected_value=questions,
                state_required_cols=[
                    "question_id", "question_text", "search_string_id", "search_string",
                    "search_engine","paper_id", "paper_title", "paper_author", "paper_date", "doi", "download_link"
                    ],
                injected_required_cols=["question_id", "question_text"]
                )
        )

        self.llm_client: Any = llm_client
        self.ai_model: str = ai_model
        self.GREY_LIT_PICKLE_FILE = GREY_LIT_PICKLE_FILE
        self.GREY_LIT_RAW_PICKLE_FILE = GREY_LIT_RAW_PICKLE_FILE


    def get_grey_lit(self, timeout=1200) -> Optional[pd.DataFrame]:
        """
        Retrieve grey literature relevant to the research questions using the LLM.

        Steps:
        1. Build a prompt from research questions.
        2. Call the LLM with web search capability.
        3. Parse the JSON output from the LLM.
        4. Merge results with existing QuestionState using `question_id`.
        5. Save updated state to disk.

        Returns:
            Optional[pd.DataFrame]: Subset of state with grey literature results
            (where `paper_id` starts with "grey_lit_"), or None if parsing fails.
        """

        # Build question strings: "question_id: question_text"
        question_strings = (
            self.state.insights[["question_id", "question_text"]]
            .drop_duplicates()
            .assign(combined=lambda df: df["question_id"].astype(str) + ": " + df["question_text"])
            ["combined"]
            .to_list()
        )

        # Build LLM prompt
        prompt: str = Prompts().grey_lit_retrieve(questions=question_strings)

        now = datetime.datetime.now()
        end_time = now + datetime.timedelta(seconds=timeout + 10)

        print(
            f"Undertaking AI-assisted research. Process will finish by {end_time.strftime('%Y-%m-%d %H:%M:%S')}."
            " If not finished by then, the system may have hung."
        )

        # Call the LLM
        response = call_reasoning_model(prompt=prompt, llm_client=self.llm_client, ai_model=self.ai_model, timeout=timeout)

        print("Seeking valid json format from the LLM...")
        # First ask an LLM to clean it
        clean_response = llm_json_clean(response, Prompts().grey_literature_format_check(), self.llm_client, "gpt-4o")

        # Then check whether the list of strings - for the authors has come in correctky and the file will save as a parquet
        success, error, result = json_format_check(clean_response)
        if not success:
            print(error)
            return result
        else:
            grey_lit = result

            # Prefix paper_id with "grey_lit_"
            grey_lit["paper_id"] = [f"grey_lit_{i}" for i in range(len(grey_lit))]

            # Merge ONLY on canonical `question_id` to get original question_text
            grey_lit = grey_lit.merge(
                self.state.insights[["question_id", "question_text"]].drop_duplicates(),
                on="question_id",
                how="left"
            )

            # Update state
            self.state.insights = pd.concat([self.state.insights, grey_lit], ignore_index=True)
            self.state.save(STATE_FILE_LOCATION)

            # Return grey literature subset
            return self.state.insights[self.state.insights["paper_id"].str.startswith("grey_lit_")]

class Literature:
    """
    A class to manage literature (including grey literature) for research questions,
    detect exact and fuzzy duplicates, and export files for manual checking.

    Workflow:
    1. Split literature by question_id.
    2. Generate a string for duplicate detection.
    3. Drop exact duplicates.
    4. Detect fuzzy duplicates using pairwise string similarity.
    5. Export potential matches for manual verification.
    6. Update QuestionState with cleaned results.
    """

    FUZZY_CHECK_PATH: str = os.path.join(os.getcwd(), "data", "fuzzy_check")
    os.makedirs(FUZZY_CHECK_PATH, exist_ok=True)

    def __init__(self, state: "QuestionState", literature: Optional[pd.DataFrame] = None) -> None:
        
        self.state: "QuestionState" = deepcopy(
            validate_format(
            state=state,
            injected_value=literature,
            state_required_cols=[
                "question_id", "question_text", "search_string_id", "search_string",
                "paper_id", "paper_title", "paper_author", "paper_date",
                "doi", "download_link"
            ],
            injected_required_cols=[
                "question_id", "paper_id", "paper_title", "paper_author",
                "paper_date", "doi", "download_link"
            ]
            )
        )

        # Split literature into a list of DataFrames per question_id
        self.question_dfs: List[pd.DataFrame] = self._splitter()

    def _splitter(self) -> List[pd.DataFrame]:
        """
        Split the literature by question_id and generate a string for duplicate detection.
        Only question_id is needed; question_text is not included here.
        """
        dfs: List[pd.DataFrame] = [
            self.state.insights[self.state.insights["question_id"] == qid].copy()
            for qid in self.state.insights["question_id"].drop_duplicates()
        ]

        for df in dfs:
            authors_str = df["paper_author"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
            title_str = df["paper_title"].astype(str)
            date_str = df["paper_date"].astype(str)

            # Concatenate for duplicate checking
            df["duplicate_check_string"] = authors_str + " " + title_str + " " + date_str

        return dfs

    def drop_exact_duplicates(self) -> List[pd.DataFrame]:
        for df in self.question_dfs:
            df.drop_duplicates(subset="duplicate_check_string", keep="first", inplace=True)
        return self.question_dfs

    def _get_fuzzy_match(self, similarity_threshold: int = 90) -> List[List[Tuple[str, str]]]:
        fuzzy_duplicates_list: List[List[Tuple[str, str]]] = []

        for df in self.question_dfs:
            strings = df["duplicate_check_string"].tolist()
            fuzzy_scores = process.cdist(strings, strings, scorer=fuzz.ratio)
            unique_fuzzy_matches: List[Tuple[str, str]] = []

            for i, row in enumerate(fuzzy_scores):
                for j, score in enumerate(row):
                    if i < j and score >= similarity_threshold:
                        unique_fuzzy_matches.append((strings[i], strings[j]))

            fuzzy_duplicates_list.append(unique_fuzzy_matches)

        print("Pairwise fuzzy score calculated.")
        return fuzzy_duplicates_list

    def _get_similar_groups(self) -> List[pd.DataFrame]:
        fuzzy_groups_list: List[pd.DataFrame] = []
        fuzzy_duplicates_list = self._get_fuzzy_match()

        for possible_duplicates, df in zip(fuzzy_duplicates_list, self.question_dfs):
            graph = nx.Graph()
            graph.add_edges_from(possible_duplicates)
            groups = list(nx.connected_components(graph))

            grouped_matches = []
            matched_strings = set()

            for i, group in enumerate(groups, start=1):
                for string in group:
                    grouped_matches.append({"duplicate_check_string": string, "sim_group": i})
                    matched_strings.add(string)

            # Assign -1 to strings with no matches
            for string in df["duplicate_check_string"]:
                if string not in matched_strings:
                    grouped_matches.append({"duplicate_check_string": string, "sim_group": -1})

            groups_df = pd.DataFrame(grouped_matches)
            fuzzy_groups_list.append(groups_df)

        return fuzzy_groups_list

    def get_fuzzy_matches(self) -> None:
        fuzzy_groups_list = self._get_similar_groups()

        for index, (fuzzy_group_df, df) in enumerate(zip(fuzzy_groups_list, self.question_dfs)):
            df_for_manual_check = fuzzy_group_df.merge(
                df, 
                how="left",
                on="duplicate_check_string"
            )
            df_for_manual_check.to_csv(
                os.path.join(self.FUZZY_CHECK_PATH, f"question{index + 1}.csv"),
                index=False
            )

        print(
            f"All fuzzy matches exported to {self.FUZZY_CHECK_PATH}. "
            "Check and remove any true duplicates manually. "
            "Only save as .csv to ensure update_state() works correctly."
        )

    def update_state(self, path_to_files: Optional[str] = None) -> pd.DataFrame:
        path_to_files = path_to_files or self.FUZZY_CHECK_PATH
        files_to_import = [
            os.path.join(path_to_files, f)
            for f in os.listdir(path_to_files)
            if f.lower().endswith(".csv") or f.lower().endswith(".xlsx")
        ]

        dfs: List[pd.DataFrame] = []
        for file in files_to_import:
            if file.lower().endswith(".csv"):
                dfs.append(pd.read_csv(file))
            else:
                dfs.append(pd.read_excel(file))
        
        # The list of authors needs to come in as a string from the csv so hanlde that with ast.literal_eval()
        for df in dfs:
            if "paper_author" in df.columns:
                df["paper_author"] = df["paper_author"] \
                    .apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
        
        self.state.insights = pd.concat(dfs, ignore_index=True)

        if "duplicate_check_string" in self.state.insights.columns:
            self.state.insights.drop(columns="duplicate_check_string", inplace=True)

        if "sim_group" in self.state.insights.columns:
            self.state.insights.drop(columns="sim_group", inplace=True)

        self.state.save(STATE_FILE_LOCATION)
        return self.state.insights

class AiLiteratureCheck:
    """
    Class to check the completeness of literature for a set of research questions
    using an LLM. Takes a QuestionState object and optionally a DataFrame of papers
    and outputs missing literature suggested by the AI.
    """

    def __init__(
        self,
        llm_client: Any,
        ai_model: str = "o3-deep-research",
        state: Optional["QuestionState"] = None,
        papers: Optional[pd.DataFrame] = None, 
        AI_LIT_PICKLE_FILE = os.path.join(os.getcwd(), "data", "pickles", "ai_lit.pkl"),
        AI_LIT_RAW_PICKLE_FILE = os.path.join(os.getcwd(), "data", "pickles", "ai_lit_raw.pkl")
    ) -> None:
        """
        Initialize the AI Literature Check.

        Args:
            llm_client: An instance of the language model client (e.g., OpenAI client).
            ai_model: The name of the LLM model to use.
            state: QuestionState object containing current literature data.
            papers: Optional DataFrame with literature to inject if state is None.
        """
        self.llm_client: Any = llm_client
        self.ai_model: str = ai_model
        self.AI_LIT_PICKLE_FILE = AI_LIT_PICKLE_FILE
        self.AI_LIT_RAW_PICKLE_FILE = AI_LIT_RAW_PICKLE_FILE

        # Validate that the state or injected papers contain all required columns
        self.state: "QuestionState" = deepcopy(
            validate_format(
            state=state,
            injected_value=papers,
            state_required_cols=[
                "question_id", "question_text", "search_string_id", "search_string",
                "paper_id", "paper_title", "paper_author", "paper_date",
                "doi", "download_link"
            ],
            injected_required_cols=[
                "question_id", "question_text", "paper_id", "paper_title", "paper_author",
                "paper_date", "doi", "download_link"
            ]
            )
        )

        # Preprocess current literature into JSON for LLM prompt insertion
        self.json_for_prompt_insertion: str = self._clean_data_for_prompt_insertion()

    def _clean_data_for_prompt_insertion(self) -> str:
        """
        Converts the literature DataFrame into JSON suitable for LLM prompt insertion.
        Groups papers by question_id and question_text, flattening into a list of paper dicts.

        Returns:
            str: JSON string of grouped literature for input to the LLM.
        """
        df = self.state.insights[[
            "question_id", "question_text", "paper_id", "paper_author", "paper_date", "paper_title"
        ]]

        json_list = [
            {
                "question_id": qid,
                "question_text": qtext,
                "papers": group[["paper_id", "paper_author", "paper_date", "paper_title"]]
                        .to_dict(orient="records")
            }
            for (qid, qtext), group in df.groupby(["question_id", "question_text"], sort=False)
            ]

        return json.dumps(json_list, indent=2)

    def ai_literature_check(self, timeout = 1200) -> Optional[pd.DataFrame]:
        """
        Uses the LLM to identify missing literature for each research question.
        Parses the LLM JSON output, flattens it, merges with the state, and
        assigns unique AI literature paper IDs.

        Returns:
            Optional[pd.DataFrame]: DataFrame of AI-suggested missing papers with columns:
            ["paper_id", "paper_title", "paper_author", "paper_date"].
            Returns None if LLM call fails or invalid output is returned.
        """
        # Generate the prompt using preprocessed JSON
        prompt: str = Prompts().ai_literature_retrieve(
            questions_papers_json=self.json_for_prompt_insertion
        )
        
        # Send request to the language model
        now = datetime.datetime.now()
        end = now + datetime.timedelta(seconds = timeout + 10)
        print(f"Undertaking AI assisted literature check. This may take some time. If you do not see a result by {end.strftime("%Y-%m-%d, %H:%M:%S")}, the process has hung. You should try again.")

        response = call_reasoning_model(prompt=prompt, llm_client=self.llm_client, ai_model=self.ai_model, timeout=timeout)
        
        print("Checking json format of LLM output...")
        clean_response = llm_json_clean(response, Prompts().ai_literature_format_check(), self.llm_client, "gpt-4o")

        success, error, result = json_format_check(clean_response)
        if not success:
            self.llm_response = clean_response
            print(error)
            return(None)
        else:
            ai_lit = result

        if ai_lit.shape[0] == 0:
            print("No missing papers returned by the LLM.")
            return pd.DataFrame()

        # Merge back with question metadata for context
        ai_lit = ai_lit.merge(
            self.state.insights[["question_id", "question_text", "search_string_id", "search_string"]],
            how="left",
            on="question_id"
        )

        # Assign unique AI paper IDs
        ai_lit["paper_id"] = [f"ai_lit_{i}" for i in range(ai_lit.shape[0])]

        # Append AI literature to state
        self.state.insights = pd.concat([self.state.insights, ai_lit], ignore_index=True)

        # Save
        self.state.insights["paper_date"] = pd.to_numeric(self.state.insights["paper_date"], errors="coerce").astype("Int64") #Clean up the paper_date file in case there are any strings which will break parquet
        self.state.save(STATE_FILE_LOCATION)

        # Return only the new AI-suggested papers
        return self.state.insights.loc[
            self.state.insights["paper_id"].str.contains("ai_lit_"),
            ["paper_id", "paper_title", "paper_author", "paper_date"]
        ]

class DownloadManager:
    """
    Class to manage downloading of papers listed in a QuestionState object.
    Downloads are organized by sanitized question_id and paper_id to ensure
    filesystem-safe filenames, while maintaining traceability to original IDs.
    """

    def __init__(
        self,
        state: "QuestionState" = None,
        papers: Optional[pd.DataFrame] = None,
        DOWNLOAD_LOCATION: str = os.path.join(os.getcwd(), "data", "docs")
    ) -> None:
        """
        Initialize the Downloader.

        Args:
            state: QuestionState object containing literature data.
            papers: Optional DataFrame of literature to inject.
            DOWNLOAD_LOCATION: Base directory to save downloaded files.
        """
        # Validate that the state or injected papers contain all required columns
        self.state: "QuestionState" = deepcopy(
            validate_format(
            state=state,
            injected_value=papers,
            state_required_cols=[
                "question_id", "question_text", "search_string_id", "search_string",
                "paper_id", "paper_title", "paper_author", "paper_date", "doi",
                "download_link"
            ],
            injected_required_cols=[
                "question_id", "question_text",
                "paper_id", "paper_title", "paper_author", "paper_date",
                "download_link"
            ]
            )
        )

        # Check if download_status variable is in the passed state if its not create with 0 values, assuming no downloads have happened yet, if it does exsist, simply use it. 
        if "download_status" not in self.state.insights.columns:
            self.state.insights["download_link"] = 0
        else: 
            pass

        # Ensure the base download folder exists
        self.DOWNLOAD_LOCATION: str = DOWNLOAD_LOCATION
        os.makedirs(self.DOWNLOAD_LOCATION, exist_ok=True)

        # Preserve original IDs and sanitize for filesystem-safe filenames
        self.state.insights["messy_question_id"] = self.state.insights["question_id"]
        self.state.insights["messy_paper_id"] = self.state.insights["paper_id"]
        self.state.insights["question_id"] = self.state.insights["question_id"].apply(self._sanitize_filename)
        self.state.insights["paper_id"] = self.state.insights["paper_id"].apply(self._sanitize_filename)

        # write the insights to csv
        self.state.write_to_csv(save_location= self.DOWNLOAD_LOCATION, 
                                write_full_text=False, write_chunks=False)
        
        print(
            f"Architecture for downloading papers has been created at {self.DOWNLOAD_LOCATION}.\n"
            f"You sould manually download files and update thier status in the file at {os.path.join(self.DOWNLOAD_LOCATION, "insights.csv")}. "
            "Assuming you do not change the files location the easiest way to do this is to call DownloadManager.update()\n"
            f"Note when saving these files you MUST SAVE THEM IN THE FOLDER CORRESPONDING TO THIER QUESTION ID. You should also ensure the filenames match the paper_id in the form paper_id.[relevant extension]. " 
            "Matching filenames with paper_ids is not neccesary but will allow you to track papers back to search prompts. You can add papers to these folders that are not in your "
            )

    def _create_download_folder(self) -> None:
        """
        Create subfolders for each sanitized question_id to organize downloaded files.
        """
        for qid in self.state.insights["question_id"].unique():
            os.makedirs(os.path.join(self.DOWNLOAD_LOCATION, qid), exist_ok=True)

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """
        Sanitize a string to be a valid filename by removing illegal filesystem characters.

        Args:
            filename: Original filename string.

        Returns:
            Sanitized filename string.
        """
        sanitized = re.sub(r'[\\/:*?"<>|]', "_", filename)
        return sanitized.strip()
    
    def update(self):
        # This convenience function just calls the from csv method of the questionstate
        self.state = self.state.from_csv(filepath=os.path.join(self.DOWNLOAD_LOCATION))
        # And updates the state object on file by saving
        self.state.save(STATE_FILE_LOCATION)
        return(self.state.insights)

# THIS WAS ALL TOO FRAGILE TO MAKE WORK SO I BAILED ON IT AND RESORTED TO MANUAL DOWNLOADING
#     def download_files(self) -> pd.DataFrame:
#         """
#         Attempt to download all files in the state DataFrame. Tracks download status
#         and local filenames. Updates state and writes a CSV with download results.

#         Returns:
#             DataFrame containing columns ['paper_id', 'download_status'] with updated statuses.
#         """
#         # Ensure subfolders exist
#         self._create_download_folder()

#         # Initialize download tracking columns
#         if "download_status" not in self.state.insights.columns:
#             self.state.insights["download_status"] = 0
#         if "filename" not in self.state.insights.columns:
#             self.state.insights["filename"] = np.nan

#         # Iterate through each row and attempt download
#         for idx, row in self.state.insights.iterrows():
#             url: str = row["download_link"]
#             status: int = row["download_status"]
#             qid: str = row["question_id"]
#             pid: str = row["paper_id"]

#             print(f"Downloading file {idx + 1} of {self.state.insights.shape[0]}")

#             if status == 0:
#                 if pd.notna(url) and url != "NA":
#                     try:
#                         response = requests.get(url, stream=True, timeout=10)
#                         response.raise_for_status()

#                         file_path = os.path.join(self.DOWNLOAD_LOCATION, qid, f"{pid}.pdf")
#                         with open(file_path, "wb") as f:
#                             for chunk in response.iter_content(chunk_size=8192):
#                                 f.write(chunk)

#                         self.state.insights.at[idx, "filename"] = file_path
#                         self.state.insights.at[idx, "download_status"] = 1
#                     except Exception as e:
#                         print(f"Failed to download {url}: {e}")
#                         self.state.insights.at[idx, "filename"] = np.nan
#                         self.state.insights.at[idx, "download_status"] = 0
#                 else:
#                     self.state.insights.at[idx, "filename"] = np.nan
#                     self.state.insights.at[idx, "download_status"] = 0
#             else:
#                 self.state.insights.at[idx, "download_status"] = 1

#         # Save download status CSV for inspection
#         download_status_csv = os.path.join(self.DOWNLOAD_LOCATION, "download_status.csv")
#         self.state.insights.to_csv(download_status_csv, index=False)

#         print(
#             f"Attempted downloads complete. Inspect the results here: {download_status_csv}.\n"
#             "For files that failed to download, open this CSV, update the 'download_link' as needed, and save it.\n"
#             "Then reload the updated CSV into a QuestionState using:\n"
#             "    state = QuestionState.load_from_csv('path/to/download_status.csv')\n"
#             "After that, pass the new state to the Downloader and retry downloads:\n"
#             "    downloader = Downloader(state=state)\n"
#             "Filenames correspond to sanitized question_id and paper_id, preserving traceability."
#         )
#         # Save the state
#         self.state.save(STATE_FILE_LOCATION)
#         return self.state.insights[["paper_id", "download_status"]]

class PaperAttainmentTriage:
    """
    Class to triage papers that failed to download (hard-to-get) and prioritize
    manual retrieval based on semantic similarity between research questions and paper titles.
    """

    def __init__(
        self,
        state: "QuestionState",
        client: Any,
        embedding_model: str = "text-embedding-3-small",
        save_location: str = os.path.join(os.getcwd(), "data", "hard_to_get_papers.csv"),
        hard_to_get_papers: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Initialize PaperAttainmentTriage.

        Args:
            state: QuestionState object containing literature data.
            client: OpenAI or similar embedding client.
            embedding_model: Name of the embedding model.
            save_location: CSV path to save the hard-to-get papers.
            hard_to_get_papers: Optional pre-filtered DataFrame of failed downloads.
        """
        # Validate the state structure
        self.state: "QuestionState" = deepcopy(
            validate_format(
            state=state,
            state_required_cols=[
                "question_id", "question_text", "search_string_id", "search_string",
                "paper_id", "paper_title", "paper_author", "paper_date", "doi",
                "download_link", "download_status", "messy_question_id", "messy_paper_id"
            ],
            injected_value=None,
            injected_required_cols=[]
            )
        )

        self.client: Any = client
        self.embedding_model: str = embedding_model
        self.save_location: str = save_location

        # Filter hard-to-get papers (failed downloads)
        self.hard_to_get_papers: pd.DataFrame = (
            hard_to_get_papers if hard_to_get_papers is not None 
            else self.state.insights[self.state.insights["download_status"] == 0].copy()
        )

    def _generate_question_embeddings(self) -> pd.DataFrame:
        """
        Generate embeddings for unique research questions.

        Returns:
            DataFrame with columns ['question_text', 'question_embedding'].
        """
        questions = self.hard_to_get_papers["question_text"].drop_duplicates()
        embeddings = []

        for question in questions:
            response = self.client.embeddings.create(
                input=question,
                model=self.embedding_model
            )
            embeddings.append(response.data[0].embedding)

        df = pd.DataFrame({
            "question_text": questions,
            "question_embedding": embeddings
        })

        self.question_embeddings: pd.DataFrame = df
        return df

    def _generate_title_embeddings(self) -> pd.DataFrame:
        """
        Generate embeddings for all hard-to-get paper titles.

        Returns:
            DataFrame with columns ['paper_title', 'title_embedding'].
        """
        titles = self.hard_to_get_papers["paper_title"]
        embeddings: List[Any] = []

        for idx, title in enumerate(titles):
            print(f"Generating embedding for title {idx + 1} of {len(titles)}")
            response = self.client.embeddings.create(
                input=title,
                model=self.embedding_model
            )
            embeddings.append(response.data[0].embedding)

        df = pd.DataFrame({
            "paper_title": titles,
            "title_embedding": embeddings
        })

        self.title_embeddings: pd.DataFrame = df
        return df

    def generate_embeddings(self) -> pd.DataFrame:
        """
        Generate embeddings for both questions and titles and merge them into one DataFrame.

        Returns:
            DataFrame of hard-to-get papers with question and title embeddings.
        """
        print("Generating question embeddings...")
        q_df = self._generate_question_embeddings()
        print("Generating title embeddings...")
        t_df = self._generate_title_embeddings()

        merged_df = self.hard_to_get_papers.merge(
            q_df, how="left", on="question_text"
        ).merge(
            t_df, how="left", on="paper_title"
        )

        self.embeddings_df: pd.DataFrame = merged_df
        return merged_df

    @staticmethod
    def calc_cosine_sim(embedding1: pd.Series, embedding2: pd.Series) -> List[float]:
        """
        Calculate cosine similarity between two series of embeddings.

        Args:
            embedding1: Series of embeddings (one per row).
            embedding2: Series of embeddings.

        Returns:
            List of cosine similarity values.
        """
        emb1 = np.vstack(embedding1.to_numpy())
        emb2 = np.vstack(embedding2.to_numpy())
        dot_product = np.sum(emb1 * emb2, axis=1)
        norms = np.linalg.norm(emb1, axis=1) * np.linalg.norm(emb2, axis=1)
        return (dot_product / norms).tolist()

    @staticmethod
    def moving_average_filter(x: Union[pd.Series, list], window: int = 5) -> List[float]:
        """
        Apply a moving average smoothing to a series or list.

        Args:
            x: Data to smooth.
            window: Rolling window size.

        Returns:
            Smoothed data as a list.
        """
        if isinstance(x, list):
            x = pd.Series(x)
        return x.rolling(window=window, center=False).mean().tolist()

    @staticmethod
    def locate_knee(y: pd.Series) -> List[float]:
        """
        Locate the knee/elbow point in a descending series using KneeLocator.

        Args:
            y: Series of values (e.g., smoothed cosine similarities).

        Returns:
            List of knee_y values repeated for each item in y.
        """
        y_sorted = y.sort_values(ascending=False)
        x = list(range(len(y_sorted)))

        # Handle degenerate or empty input
        if len(y_sorted) == 0 or y_sorted.isna().all():
            return [np.nan for _ in y_sorted]

        kl = KneeLocator(x=x, y=y_sorted, direction="decreasing", curve="concave")

        # If no knee detected, replace None with np.nan
        knee_y = kl.knee_y if kl.knee_y is not None else np.nan

        return [knee_y for _ in y_sorted]

    def triage_papers(
        self,
        low_threshold: float = 0.35,
        medium_threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Classify hard-to-get papers into 'low', 'medium', or 'high' priority for manual retrieval.

        Args:
            low_threshold: Cosine similarity threshold for low priority.
            medium_threshold: Cosine similarity threshold for medium priority.

        Returns:
            DataFrame of hard-to-get papers with rankings and cosine similarity.
        """
        # Cosine similarity between question and title embeddings
        self.hard_to_get_papers["cosine_sim"] = self.calc_cosine_sim(
            self.embeddings_df["question_embedding"],
            self.embeddings_df["title_embedding"]
        )

        # Smooth the cosine similarity
        self.hard_to_get_papers["cosine_sim_smooth"] = self.moving_average_filter(
            self.hard_to_get_papers["cosine_sim"]
        )

        # Count papers per research question
        self.hard_to_get_papers["count"] = self.hard_to_get_papers.groupby("question_id")["paper_id"].transform("count")

        # Compute knee/elbow for each research question
        self.hard_to_get_papers["knee"] = self.hard_to_get_papers.groupby("question_id")["cosine_sim_smooth"].transform(self.locate_knee)

        # Initial ranking based on low threshold
        self.hard_to_get_papers["paper_ranking"] = np.where(
            self.hard_to_get_papers["cosine_sim"] <= low_threshold, "low", pd.NA
        )

        # Count-based ranking overrides
        self.hard_to_get_papers["paper_ranking"] = self.hard_to_get_papers["paper_ranking"].mask(
            (self.hard_to_get_papers["count"] <= 10) & (self.hard_to_get_papers["cosine_sim"] > medium_threshold),
            "high"
        )
        self.hard_to_get_papers["paper_ranking"] = self.hard_to_get_papers["paper_ranking"].mask(
            (self.hard_to_get_papers["count"] <= 10) &
            (self.hard_to_get_papers["cosine_sim"] > low_threshold) &
            (self.hard_to_get_papers["cosine_sim"] <= medium_threshold),
            "medium"
        )

        # Knee-based ranking
        self.hard_to_get_papers["paper_ranking"] = self.hard_to_get_papers["paper_ranking"].mask(
            self.hard_to_get_papers["paper_ranking"].isna() &
            (self.hard_to_get_papers["cosine_sim"] > self.hard_to_get_papers["knee"]),
            "high"
        )

        # Remaining papers get medium ranking
        self.hard_to_get_papers["paper_ranking"] = self.hard_to_get_papers["paper_ranking"].mask(
            self.hard_to_get_papers["paper_ranking"].isna(),
            "medium"
        )

        # Merge rankings back into main state
        self.state.insights = self.state.insights.merge(
            self.hard_to_get_papers[["paper_id", "cosine_sim", "paper_ranking"]],
            how="left",
            on="paper_id"
        )

        # Save to CSV for manual review
        self.state.insights.to_csv(self.save_location, index=False)
        print(
            f"The list of hard-to-get papers can be viewed here: {self.save_location}.\n"
            f"Manually attain the papers that you can and save them in the relevant question folder: {os.path.join(os.getcwd(), 'data', 'docs')}.\n"
            f"Update this file so that download status reflects papers that you manually downloaded.\n"
            f"Ensure manually downloaded papers follow the naming convention 'paper_id.pdf' matching this file.\n"
            "Once you have updated the file, you should reload with .update_state() - assuming you have not moved the file from where it was saved."
        )

    def update_state(self):
        # This is simply a wrapper for state.from_csv() which makes it intutive to update the state of the class after manually editing the csv
        self.state = self.state.from_csv(filepath=self.save_location)

class Ingestor:
    """
    Class to ingest PDF or HTML papers into a QuestionState object.
    Validates papers against known question_ids and populates state.full_text.

    Attributes:
        state: QuestionState object containing literature metadata.
        file_path: Directory containing PDF/HTML files to ingest.
        llm_client: Client for calling the LLM.
        ai_model: Model name to use for LLM.
        confirm_read: Optional; set to "c" to skip ingestion error confirmation.
        ingestion_errors: List of file paths that failed ingestion.
    """

    def __init__(
        self,
        llm_client: Any,
        ai_model: str,
        state: QuestionState = None,
        papers: pd.DataFrame = None,
        confirm_read: Optional[str] = None,
        file_path: str = os.path.join(os.getcwd(), "data", "docs"),
    ) -> None:
        """Initialize Ingestor and validate state/papers format."""
        self.state = deepcopy(
            validate_format(
                state=state,
                injected_value=papers,
                state_required_cols=[
                    "question_id", "question_text", "search_string_id", "search_string",
                    "paper_id", "paper_title", "paper_author", "paper_date", "doi",
                    "download_link", "download_status", "messy_question_id", "messy_paper_id"
                ],
                injected_required_cols=["question_id", "question_text"]
            )
        )

        self.state.enforce_canonical_question_text()

        self.file_path: str = file_path
        self.llm_client: Any = llm_client
        self.ai_model: str = ai_model
        self.confirm_read: Optional[str] = confirm_read
        self.ingestion_errors: List[str] = []

    def _list_files(self) -> List[str]:
        """Recursively list all PDF and HTML files in the target directory."""
        list_of_files: List[str] = []
        for root, _, files in os.walk(self.file_path):
            for file in files:
                if Path(file).suffix.lower() in [".pdf", ".html"]:
                    list_of_files.append(os.path.join(root, file))
        return list_of_files

    def _ingest_pdf(self, path: str) -> List[str]:
        """Extract text from all pages of a PDF file."""
        with pymupdf.open(path) as doc:
            return [doc[i].get_text() for i in range(doc.page_count)]

    @staticmethod
    def _html_cleaner(html_content: str) -> str:
        """Clean HTML content by removing structural noise and returning plain text."""
        soup = BeautifulSoup(html_content, "html.parser")
        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            tag.decompose()
        body = soup.find('body')
        if body:
            return body.get_text(separator='\n', strip=True)
        return ""

    @staticmethod
    def _html_chunker(clean_html: str, token_limit: int = 16000) -> List[str]:
        """Split HTML text into chunks if it exceeds the token limit."""
        if len(clean_html) == 0:
            return [""]
        elif len(clean_html) > token_limit:
            chunks: List[str] = []
            start = 0
            end = token_limit
            while start < len(clean_html):
                chunks.append(clean_html[start:end])
                start += token_limit
                end += token_limit
            return chunks
        else:
            return [clean_html]

    def _llm_parse_html(self, html_list: List[str], prompt: str) -> List[str]:
        """Call the LLM to extract meaningful content from HTML chunks."""
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
        """Read PDF or HTML file and return list of page texts or processed chunks."""
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
        Ingest all papers and populate state.full_text.
        Returns a DataFrame with columns ['paper_path', 'pages', 'paper_id', 'question_id', 'full_text'].
        """
        list_of_papers_by_page: List[List[str]] = []
        ingestion_status: List[int] = []
        self.ingestion_errors = []

        list_of_files = self._list_files()
        valid_question_ids = set(self.state.insights["question_id"].values)

        for count, file in enumerate(list_of_files, start=1):
            print(f"Ingesting paper {count} of {len(list_of_files)}...")
            question_id = os.path.basename(os.path.dirname(file))
            if question_id in valid_question_ids:
                try:
                    pages = self._paper_ingestor(file)
                    list_of_papers_by_page.append(pages)
                    ingestion_status.append(1)
                except Exception as e:
                    list_of_papers_by_page.append([str(e)])
                    self.ingestion_errors.append(file)
                    ingestion_status.append(0)
            else:
                list_of_papers_by_page.append(["Error: Paper not affiliated with Question id"])
                self.ingestion_errors.append(file)
                ingestion_status.append(0)

        # Confirm ingestion errors
        if self.ingestion_errors:
            self.confirm_read = self.confirm_read or ""
            while self.confirm_read != "c":
                self.confirm_read = input(
                    "Ingestion errors occurred. Examine .ingestion_errors and state.full_text.\n"
                    "Hit 'c' to confirm having read this message:\n"
                ).lower()

        # Update ingestion status in state.insights
        ingestion_status_df = pd.DataFrame({
            "paper_id": [os.path.splitext(os.path.basename(path))[0] for path in list_of_files],
            "question_id": [os.path.basename(os.path.dirname(path)) for path in list_of_files],
            "ingestion_status": ingestion_status
        })
        self.state.insights = self.state.insights.merge(
            ingestion_status_df, how="outer", on=["question_id", "paper_id"]
        )

        # Build full_text DataFrame
        full_text = pd.DataFrame({
            "paper_path": list_of_files,
            "pages": list_of_papers_by_page
        })
        full_text["paper_id"] = [os.path.splitext(os.path.basename(path))[0] for path in list_of_files]
        full_text["question_id"] = [os.path.basename(os.path.dirname(path)) for path in list_of_files]
        full_text["full_text"] = ["".join(pages) for pages in full_text["pages"]]

        # Drop pages from full_text as they take up memory and are not needed:
        full_text.drop(columns=["pages"], inplace=True) 

        self.state.full_text = full_text
        return self.state.full_text

    def _get_metadata(self, question_id: str, paper_id: str, text: str) -> dict[str, Any]:
        """Call the LLM to extract metadata from the first three pages of a paper."""
        sys_prompt = Prompts().get_metadata()
        user_prompt = f"question_id: {question_id}\npaper_id: {paper_id}\nTEXT:\n{text}"
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response = self.llm_client.chat.completions.create(
            model=self.ai_model,
            messages=messages,
            response_format={"type": "json_object"}
        )
        response_dict = json.loads(response.choices[0].message.content)

        # Validate keys
        required_keys = ["question_id", "paper_id", "paper_title", "paper_author", "paper_date"]
        for key in required_keys:
            if key not in response_dict:
                raise KeyError(f"Metadata extraction failed: missing key '{key}'")

        # Ensure authors is a list
        if isinstance(response_dict["paper_author"], str):
            response_dict["paper_author"] = ast.literal_eval(response_dict["paper_author"])

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

    def update_metadata(self) -> pd.DataFrame:
        """Update metadata for papers missing it by calling the LLM."""
        metadata_check_df = self.state.insights[
            ["question_id", "paper_id", "paper_title", "paper_author", "paper_date"]
        ].merge(
            self.state.full_text[["question_id", "paper_id", "pages"]],
            how="left",
            on=["question_id", "paper_id"]
        )

        for idx, row in metadata_check_df.iterrows():
            print(f"Checking metadata for paper {idx + 1} of {self.state.insights.shape[0]}...")
            author = row["paper_author"]
            if pd.isna(author) or author == "NA":
                question_id = row["question_id"]
                paper_id = row["paper_id"]
                text = row["full_text"][:5000] if row["full_text"] else ""
                metadata = self._get_metadata(question_id, paper_id, text)
                metadata_check_df.at[idx, "paper_title"] = metadata["paper_title"]
                metadata_check_df.at[idx, "paper_author"] = metadata["paper_author"]  
                metadata_check_df.at[idx, "paper_date"] = metadata["paper_date"]

        self.state.insights = (
            self.state.insights
            .drop(["paper_title", "paper_author", "paper_date"], axis=1)
            .merge(metadata_check_df, how="left", on=["question_id", "paper_id"])
        )
        # Drop "pages" column if it exists - i needed it to get the metadata (from the first three pages) but after doing this i don't want it clogging up the df anymore
        # "pages" should only be in state.full_text not state.insights
        if "pages" in self.state.insights.columns:
            self.state.insights.drop(columns=["pages"], inplace=True)

        self.state.save(STATE_FILE_LOCATION)

        return self.state.insights

    def chunk_papers(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 120,
        length_function=len,
        separators: Optional[List[str]] = None,
        is_separator_regex: bool = False
    ) -> None:
        """Split full_text into nested chunks and flatten for downstream processing."""
        if separators is None:
            separators = ["\n\n", "\n", " ", ""]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            separators=separators,
            is_separator_regex=is_separator_regex
        )

        full_text_list = self.state.full_text["full_text"].to_list()
        chunks_list: List[List[str]] = [text_splitter.split_text(text) for text in full_text_list]

        # Create the chunks state from the full_text state
        self.state.full_text["chunks"] = chunks_list
        self.state.chunks = self.state.full_text[["question_id", "paper_id", "chunks"]].explode("chunks").reset_index(drop=True).copy()
        self.state.chunks.rename(columns={"chunks": "chunk_text"}, inplace=True)   
        self.state.chunks["chunk_id"] = self.state.chunks.groupby(["question_id", "paper_id"]).cumcount()

        # Chunks from full_text as its now joined by paper and question id
        self.state.full_text.drop(columns=["chunks"], inplace=True)

        # Save the updated state
        self.state.save(STATE_FILE_LOCATION)

class Insights:
    def __init__(
        self,
        state: "QuestionState",
        llm_client: Any,
        ai_model: str, 
        pickle_path: str = os.path.join(os.getcwd(), "data", "pickles"), 
        chunk_insights_pickle_file: str="chunk_insights.pkl", 
        meta_insights_pickle_file: str="meta_insights.pkl"
    ) -> None:
        """
        Class for extracting insights (both chunk-level and meta/paper-level) 
        from a corpus of academic papers and grey literature using an LLM.

        Args:
            state (QuestionState): 
                Container for all relevant state data including chunks, 
                full text, and insights tables.
            llm_client (Any): 
                Client instance for calling the LLM API (e.g. OpenAI client).
            ai_model (str): 
                Model name/ID to be used for completions.
        """
        self.llm_client: Any = llm_client
        self.ai_model: str = ai_model
        self.pickle_path: str = pickle_path
        os.makedirs(self.pickle_path, exist_ok=True)

        self.chunk_insights_pickle_file = chunk_insights_pickle_file
        self.meta_insights_pickle_file = meta_insights_pickle_file


        # Ensure state has all required columns before processing
        self.state = deepcopy(
            validate_format(
            state=state,
            injected_value=None,
            state_required_cols=[
                "question_id", "question_text", "search_string_id", "search_string",
                "paper_id", "paper_title", "paper_author", "paper_date", "doi",
                "download_link", "download_status", "messy_question_id", "messy_paper_id",
                "ingestion_status"
            ],
            injected_required_cols=None
            )
        )

        self.state.enforce_canonical_question_text()

    def get_chunk_insights(self, chunk_state= None, insights = None, count_start = None) -> pd.DataFrame:
       
        if os.path.exists(os.path.join(self.pickle_path, self.chunk_insights_pickle_file)):
            recover = None
            while recover not in ['r', 'n']:
                recover = input(
                    f"Insights already exist on file at {os.path.join(self.pickle_path, self.chunk_insights_pickle_file)}. "
                    "Do you wish to recover/restore or regenerate insights?\n"
                    "Hit 'r' to recover/restore from file, or 'n' to generate new insights (this will overwrite existing file):\n"
                    ).lower()
            if recover == 'r':
                self.recover_chunk_insights_generation()
            else:
                print("Overwriting existing chunk insights pickle file...")
                self.generate_chunk_insights()

    def generate_chunk_insights(self, chunk_state= None, insights = None, count_start = None) -> pd.DataFrame:
        """
        Extract insights from each text chunk using the LLM.
        Each chunk is processed individually, with the research question
        and other RQs as context. Insights are traced back to 
        (chunk_id, paper_id, question_id).

        Returns:
            pd.DataFrame: Updated `state.insights` with new insights appended.
        """
        if chunk_state is None:
            chunk_state = self.state.chunks
        if insights is None:
            insights: List[Dict[str, Any]] = []
        if count_start is None:
            count_start = 0
        
        # All unique research questions, to provide context
        rqs: List[str] = self.state.insights["question_text"].unique().tolist()

        # Merge chunk text with metadata (author, date, etc.)
        temp_state_df: pd.DataFrame = chunk_state.merge(
            self.state.insights[["question_id", "paper_id", "question_text", "paper_author", "paper_date"]],
            how="left",
            on=["question_id", "paper_id"]
        )

        # Iterate over each chunk
        for idx, (df_index, row) in enumerate(temp_state_df.iterrows()):
            print(f"Processing chunk {idx + 1 + count_start} of {temp_state_df.shape[0]}...")

            # Extract fields from row
            question: str = row["question_text"]
            question_id: str = row["question_id"]
            paper_id: str = row["paper_id"]
            chunk_text = row["chunk_text"] if pd.notna(row["chunk_text"]) else ""
            chunk_id: int = int(row["chunk_id"])

            # Generate the citation accounting for NA values in authors and date
            authors = row["paper_author"]
            if isinstance(authors, (list, np.ndarray)):
                citation = " ".join(authors)
            elif pd.isna(authors):
                citation = ""
            else:
                citation = str(authors)
            date = row["paper_date"] if not pd.isna(row["paper_date"]) else ""
            citation = f"{citation} {date}"
           
            # Prepare other RQs for context
            other_research_questions: str = " - " + "\n - ".join(
                [rq for rq in rqs if rq not in (None, question)] 
                )

            # Encode text safely for JSON
            safe_chunk_text: str = json.dumps(chunk_text, ensure_ascii=False)
            safe_citation: str = json.dumps(citation, ensure_ascii=False)
            safe_other_rqs: str = json.dumps(other_research_questions, ensure_ascii=False)

            # Build prompts
            sys_prompt: str = Prompts().gen_chunk_insights()
            user_prompt: str = (
                f"CURRENT RESEARCH QUESTION:\n{question}\n\n"
                f"TEXT CHUNK (chunk_id: {chunk_id}):\n{safe_chunk_text} - {safe_citation}\n"
                f"OTHER RESEARCH QUESTIONS (for context only):\n{safe_other_rqs}"
            )

            fall_back = {
                "chunk_id":"",
                "question_id": "",
                "paper_id": "",
                "insight":[]
            }

            response_dict = call_chat_completion(ai_model = self.ai_model,
                                 llm_client = self.llm_client,
                                 sys_prompt = sys_prompt,
                                 user_prompt = user_prompt,
                                 return_json = True, 
                                 fall_back=fall_back)
            # Ensure the chunk_id is included in the response
            response_dict["chunk_id"] = chunk_id
            response_dict["question_id"] = question_id
            response_dict["paper_id"] = paper_id
            # Ensure insight key exists
            if "insight" not in response_dict:
                response_dict["insight"] = []
            # Ensure insight is a list
            if isinstance(response_dict["insight"], list):
                pass
            elif isinstance(response_dict["insight"], str):
                response_dict["insight"] = [response_dict["insight"]]
            else:
                response_dict["insight"] = []
            # Append to insights list
            insights.append(response_dict)
            with open(os.path.join(self.pickle_path, self.chunk_insights_pickle_file), "wb") as f:
                pickle.dump(insights, f)

        # Convert insights list to DataFrame
        print("Converting insights to DataFrame and merging into state...")
        chunk_insights_df: pd.DataFrame = pd.DataFrame(insights)
        print(f"Dropping cols for first merge...")
        # Merge new insights into chunks - first drop existing insight columns if present (these can be created by previous runs of recover_chunk_insights_generation)        
        if "insight" in self.state.chunks.columns:
            self.state.chunks.drop(columns=["insight"], inplace=True)
        print("First merge...")
        self.state.chunks = self.state.chunks.merge(
            chunk_insights_df, 
            how="left", 
            on=["paper_id", "chunk_id", "question_id"]
        )
        print(f"Dropping cols for second merge...")
        # Merge into global insights table - first drop existing insight columns if present (these can be created by previous runs of recover_chunk_insights_generation)
        for col in ["chunk_id", "insight"]:
            if col in self.state.insights.columns:
                self.state.insights.drop(columns=[col], inplace=True)
        print("Second merge...")
        self.state.insights = self.state.insights.merge(
            self.state.chunks[["question_id", "paper_id", "chunk_id", "insight"]], # Make sure chunks does not come across to insights
            how="left",
            on=["question_id", "paper_id"]
        )

        # Now drop insights from chunks to keep state clean - insights are linked back to chunks via question_id, paper_id, chunk_id in state.insights
        self.state.chunks.drop(columns=["insight"], inplace=True)

        print("Ensuring insights are lists before exploding...")
        # intend to explode insights so each insight is its own row, but before that i have to convert all NaN insights (from papers that did not get downloaded) to empty lists
        # Also have found that pd.dataframes might pollute insights - possibly from parquet load. So lets just convert all pd.Series to lists
        self.state.insights["insight"] = self.state.insights["insight"].apply(self.ensure_list)
        print("Exploding insights so each insight is its own row...")
        self.state.insights = self.state.insights.explode("insight")


        # Note i don't save here as i only save at the end of the class's operations. This is cleaner for the user. 
        # Also since i have the recover function in place, once the insights are generated for this class it should be quick to recreate to this point
        return self.state.insights
    
    def recover_chunk_insights_generation(self):
        print("Opening pickle file to recover chunk insights generation...")
        with open(os.path.join(self.pickle_path, self.chunk_insights_pickle_file), "rb") as f:
            recover_chunk_insights = pickle.load(f)
        
        start = len(recover_chunk_insights)
        print(f"Resuming chunk insights generation from chunk {start}...")
        self.generate_chunk_insights(chunk_state = self.state.chunks.iloc[start:], insights=recover_chunk_insights, count_start=start)


    def get_meta_insights(self, max_token_length = 100000) -> pd.DataFrame:
        """
        Generate 'meta-insights' — arguments that span multiple chunks within 
        the same paper. Each paper is processed once, combining all chunk insights 
        and the full text.

        Returns:
            pd.DataFrame: DataFrame of meta-insights appended to state.insights.

        Raises:
            ValueError: If chunk insights do not exist prior to running.
        """
        # Must run chunk insights first
        if "insight" not in self.state.insights.columns:
            raise ValueError(
                "Meta-insights cannot be created prior to generating chunk insights. "
                "Please run .get_chunk_insights before .get_meta_insights."
            )
        
        meta_insights: List[Dict[str, Any]] = []
        # All research questions for context
        rqs: List[str] = [
            f"{row['question_id']}: {row['question_text']}"
            for _, row in self.state.insights[["question_id", "question_text"]].iterrows()
        ]

        # Process each paper
        for idx, paper_id in enumerate(self.state.insights["paper_id"].unique()):
            # Skip if paper was not ingested (and therefore won't exist in state.full_text  )
            ingestion_status = self.state.insights[self.state.insights["paper_id"] == paper_id]["ingestion_status"].iloc[0]
            if pd.isna(ingestion_status) or ingestion_status == 0:
                continue
            print(f"Processing meta-insight for paper {idx + 1} of {len(self.state.insights['paper_id'].unique())}...")
            # Get paper full text
            paper_content: str = (
                self.state.full_text
                .loc[self.state.full_text["paper_id"] == paper_id, "full_text"]
                .iloc[0]
            )

            token_count = self.estimate_tokens(paper_content, self.ai_model)
            if token_count > max_token_length:
                paper_content_list = self.string_breaker(paper_content, max_token_length=max_token_length)
            else:
                paper_content_list = [paper_content]
              
            for paper_content in paper_content_list:
                # Collect metadata from insights table
                paper_df: pd.DataFrame = self.state.insights[self.state.insights["paper_id"] == paper_id]
                question_id = paper_df['question_id'].iloc[0]
                authors = paper_df['paper_author'].iloc[0]
                if isinstance(authors, list):
                    author_str = ", ".join(authors)
                elif pd.isna(authors):
                    author_str = ""
                else:
                    author_str = str(authors)
                date = paper_df['paper_date'].iloc[0]
                date_str = "" if pd.isna(date) else str(date)
                title = paper_df['paper_title'].iloc[0]
                title_str = "" if pd.isna(title) else str(title)
                metadata = f"{author_str}, {date_str}, {title_str}"

                # Current and other RQs
                current_rq: str = f"{paper_df['question_id'].iloc[0]}: {paper_df['question_text'].iloc[0]}"
                other_rqs: List[str] = list(set([rq for rq in rqs if rq != current_rq])) # Get unique list of Other RQs

                # Collate all chunk insights for this paper - flatten list of lists (should be one per row, but this is defensive) and join with newlines
                insights_text: str = "\n".join(insight_string for insight in paper_df["insight"] if isinstance(insight, list) for insight_string in insight)

                # Build prompt
                user_prompt: str = (
                    "SPECIFIC RESEARCH QUESTION FOR CONSIDERATION\n"
                    f"{current_rq}\n"
                    f"PAPER CONTENT ({paper_id}):\n"
                    f"Metadata: {metadata}\n"
                    f"{paper_content}\n"
                    "CHUNK INSIGHTS\n"
                    f"{insights_text}\n"
                    "OTHER RESEARCH QUESTIONS IN THE REVIEW\n"
                    f"{other_rqs}\n\n"
                )
                sys_prompt: str = Prompts().gen_meta_insights()

                # Empty dict for fallback
                fall_back = {
                    "paper_id": "",
                    "insight": []
                }   
                # call LLM
                response_dict = call_chat_completion(ai_model = self.ai_model,
                                                    llm_client = self.llm_client,
                                                    sys_prompt = sys_prompt,
                                                    user_prompt = user_prompt,
                                                    return_json = True, 
                                                    fall_back=fall_back)
                
                response_dict["paper_id"] = paper_id
                response_dict["question_id"] = question_id
                # Ensure insight key exists
                if "insight" not in response_dict:
                    response_dict["insight"] = []
                # Ensure insight is a list
                if isinstance(response_dict["insight"], list):
                    pass
                elif isinstance(response_dict["insight"], str):
                    response_dict["insight"] = [response_dict["insight"]]
                else:
                    response_dict["insight"] = []
                # Now append to the overall meta insights
                meta_insights.append(response_dict)
                with open(os.path.join(self.pickle_path, "meta_insights.pkl"), "wb") as f:
                    pickle.dump(meta_insights, f)
            
        
        # Convert to DataFrame
        meta_insights_df: pd.DataFrame = pd.DataFrame(meta_insights)
        
        # We want to eventually concat meta insights with insights, so we get all the columns neccesary to make meta insights compatible with insights
        # Make a temp copy of state.insights to drop unneccesary columns and then to merge with meta insights
        # Make copy
        temp_insights = deepcopy(self.state.insights)
        
        # Drop columns that will duplicate or are unneccesary
        cols_to_drop = [col for col in ["chunk_id", "insight"] if col in temp_insights.columns]
        temp_insights = temp_insights.drop(columns=cols_to_drop)
        
        # Drop duplicates so we have one row per (paper_id, question_id)
        # First have to convert paper_author lists to strings so we can drop duplicates - or else unhashable in pandas
        temp_insights["paper_author"] = temp_insights["paper_author"].apply(lambda x: "||-||-||".join(x))
        temp_insights = temp_insights.drop_duplicates()
        # Now convert paper_author strings back to lists
        temp_insights["paper_author"] = temp_insights["paper_author"].apply(lambda x: x.split("||-||-||"))

        # Merge meta insights into state.insights so meta insights have all the same columns as insights
        meta_insights_df = meta_insights_df.merge(
            temp_insights, how="left", on=["paper_id", "question_id"])

        # Prepare for exploding insights into separate rows
        meta_insights_df["insight"] = meta_insights_df["insight"].apply(self.ensure_list)
        # Explode meta insights so each insight is its own row
        meta_insights_df = meta_insights_df.explode("insight")
        # Create chunk_id column to identify meta insights
        meta_insights_df["chunk_id"] = [f"meta_insight_{pid}" for pid in meta_insights_df["paper_id"]]

        # Concat new meta insights
        self.state.insights = pd.concat(
            [self.state.insights, meta_insights_df], 
            ignore_index=True
        )
        
        # Add insight_id as i need this for joining in subsequent steps
        self.state.insights["insight_id"] = self.state.insights.index.astype(str)

        # Ensure chunk_id is string type - neccesary as earlier chunk ids were integers, now they have "meta insight_{paper_id}" strings too
        self.state.insights["chunk_id"] = self.state.insights["chunk_id"].astype(str)

        self.state.save(os.path.join(STATE_SAVE_LOCATION, "10_insights"))

        return meta_insights_df

    @staticmethod
    def ensure_list(x):
        if isinstance(x, list):
            return x
        if isinstance(x, np.ndarray):
            return x.tolist()
        if pd.isna(x):
            return []
        return [x]  # fallback for any other type
    
    @staticmethod
    def estimate_tokens(text, model):
        """Estimate token count for a given text and model using tiktoken."""
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    
    @staticmethod
    def string_breaker(text, max_token_length):
        """Break a long string into a list of strings each less than max length."""
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
    
    def recover_meta_insights_generation(self):
        print("Opening pickle file to recover meta insights generation...")
        with open(os.path.join(self.pickle_path, self.meta_insights_pickle_file), "rb") as f:
            recover_meta_insights = pickle.load(f)
        
        start = len(recover_meta_insights)
        print(f"Resuming meta insights generation from paper {start}...")
        self.get_meta_insights()
    
class Clustering:
    """
    Manage embedding, dimensionality reduction, clustering, and cluster evaluation
    for insights associated with research questions, while safely handling empty insights.
    """

    def __init__(
        self,
        state: QuestionState,
        llm_client: Any,
        embedding_model: str,
        embedding_dims: int = 1024,
        embeddings_pickle_path: str = os.path.join(os.getcwd(), "data", "pickles", "insight_embeddings.pkl")
    ):
        self.state = deepcopy(
            validate_format(
            state=state,
            injected_value=None,
            state_required_cols=[
                "question_id", "question_text", "search_string_id", "search_string",
                "paper_id", "paper_title", "paper_author", "paper_date", "doi",
                "download_link", "download_status", "messy_question_id", "messy_paper_id",
                "ingestion_status", "chunk_id", "insight"
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

    
    @staticmethod
    def _strip_author_parenthetical(row):
        first_author = re.escape(row["paper_author"][0])
        # Remove (FirstAuthor ... ) greedily
        pattern = r"\(" + first_author + r".*?\)"
        # Always a list with one string
        if len(row["insight"]) > 0:
            insight_string = row["insight"][0]
        else: 
            return("")
        cleaned = re.sub(pattern, "", insight_string).strip()

        return cleaned

    def _gen_valid_embeddings_df(self):
        """
        Get the DataFrame of insights that are non-empty after stripping author parentheticals.
        Updates the self.state.insights with a new column 'no_author_insight_string'.
        Returns: pd.DataFrame
        """
        # Apply row-wise to your DataFrame
        self.state.insights["no_author_insight_string"] = self.state.insights.apply(self._strip_author_parenthetical, axis=1)

        out = self.state.insights[
            self.state.insights["no_author_insight_string"] != ""
            ].copy()

        return out
    
    def embed_insights(self) -> np.ndarray:
        """
        Generate embeddings for non-empty insights only.
        Returns:
            np.ndarray: 2D array of embeddings for valid insights.
        """
        # Check if embeddings pickle exists
        if os.path.exists(self.embeddings_pickle_path):
            recover = None
            while recover not in ['r', 'n']:
                recover = input(
                    f"Embeddings pickle found at '{self.embeddings_pickle_path}'. "
                    "Do you wish to reload them or regenerate embeddings?\n"
                    "Hit 'n' to generate new embeddings (this will overwrite existing pickle), or 'r' to reload from file:\n"
                ).lower()
            if recover == 'r':
                self.load_embeddings()
            elif recover == 'n':
                print("Overwriting existing embeddings pickle...")
                
                # Generate embeddings for valid insights only
                insight_embeddings = []
                for idx, insight in enumerate(self.valid_embeddings_df["no_author_insight_string"], start=1):
                    print(f"Embedding insight {idx} of {self.valid_embeddings_df.shape[0]}")
                    response = self.llm_client.embeddings.create(
                        input=insight,
                        model=self.embedding_model,
                        dimensions=self.embedding_dims
                    )
                    insight_embeddings.append(response.data[0].embedding)


                self.insight_embeddings_array = np.vstack(insight_embeddings)
                self.valid_embeddings_df["full_insight_embedding"] = [emb.tolist() for emb in self.insight_embeddings_array]

                self._save_embeddings()  # safe pickle save
                return self.insight_embeddings_array

    def _save_embeddings(self):
        """Save embeddings safely, creating folder if it does not exist."""
        os.makedirs(os.path.dirname(self.embeddings_pickle_path), exist_ok=True)
        with open(self.embeddings_pickle_path, "wb") as f:
            pickle.dump(self.insight_embeddings_array, f)
        print(f"Embeddings safely saved to '{self.embeddings_pickle_path}'.")

    def load_embeddings(self):
        """Load embeddings safely if the pickle exists."""
        if not os.path.exists(self.embeddings_pickle_path):
            raise FileNotFoundError(f"No embeddings pickle found at {self.embeddings_pickle_path}")
        with open(self.embeddings_pickle_path, "rb") as f:
            data = pickle.load(f)
        self.insight_embeddings_array = data
        print("Linking embeddings back to valid embeddings DataFrame...")
        self.valid_embeddings_df["full_insight_embedding"] = [emb.tolist() for emb in self.insight_embeddings_array]    
        print(f"Embeddings loaded from '{self.embeddings_pickle_path}'.")
        return self.insight_embeddings_array

    def reduce_dimensions(
        self, full_embeddings: np.array = None, n_neighbors: int = 15, min_dist: float = 0.25, n_components: int = 10,
        metric: str = "cosine", random_state: int = 42, update_attributes: bool = True
    ) -> np.ndarray:
        
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
        Calculate the silhouette score for the current reduced embeddings,
        using research question IDs as cluster labels.

        Args:
            rq_exclude (list[str], optional): 
                List of question_id strings to exclude from the silhouette calculation.
                If provided, any rows with question_id in this list will be excluded.

        Returns:
            float: The silhouette score (higher is better cluster separation).
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
        Grid search over UMAP dimensionality reduction parameters, evaluating each
        combination using silhouette score (optionally excluding certain questions).

        IMPORTANT: This tuning is being done to check how well the insights cluster according
        to the research questions they were derived for. This is intended as a proxy for their
        ability to identify meaningful semantic structure and therefore the likelihood they can
        generalize to new, unseen data (the clusters within research questions). So this function
        runs over all the research questions. It is not run per question.

        Args:
            n_neighbors_list (list[int]): List of UMAP n_neighbors values to try.
            min_dist_list (list[float]): List of UMAP min_dist values to try.
            n_components_list (list[int]): List of UMAP n_components (output dims) to try.
            metric_list (list[str]): List of UMAP distance metrics to try.
            rq_exclude (list[str], optional): 
                List of question_id strings to exclude from silhouette scoring.
                Useful for ignoring questions that drive overlap or noise.

        Returns:
            None. Results are stored in self.umap_param_tuning_results as a DataFrame.
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
    def cluster(embedding_matrix, min_cluster_size: int = 5, metric: str = "euclidean", cluster_selection_method: str = "eom"):
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric=metric,
            cluster_selection_method=cluster_selection_method
        )

        cluster_labels = clusterer.fit_predict(embedding_matrix)
        cluster_probs = clusterer.probabilities_

        return cluster_labels, cluster_probs
    
    @staticmethod
    def calc_davies_bouldain_score(embeddings_matrix, cluster_labels):
        mask = cluster_labels != -1
        num_outliers = np.sum(~mask)
        filtered_embeddings = embeddings_matrix[mask]
        filtered_labels = cluster_labels[mask]
        if len(set(filtered_labels)) < 2:
            return(pd.NA)
        db_score = davies_bouldin_score(filtered_embeddings, filtered_labels)
        return db_score, num_outliers

    def tune_hdbscan_params(self,
                            min_cluster_sizes: list[int] = [5, 10, 15, 20],
                            metrics: list[str] = ["euclidean", "manhattan"],
                            cluster_selection_methods: list[str] = ["eom", "leaf"]
                            ) -> None:
      
        param_grid = list(itertools.product(min_cluster_sizes, metrics, cluster_selection_methods))
        rqs = self.valid_embeddings_df["question_id"].unique()
        total_runs = len(param_grid) * len(rqs)
        data = [self.rq_valid_embeddings_dfs[rq] for rq in rqs]
        results = []
        for idx, (d, rq) in enumerate(zip(data, rqs)):
            print(f"Tuning HDBSCAN for {rq}...(run {idx + 1} of {total_runs})")
            embeddings_matrix = np.vstack(d["reduced_insight_embedding"].to_list())
            for min_cluster_size, metric, cluster_selection_method in param_grid:

                cluster_labels, _ = self.cluster(embeddings_matrix, min_cluster_size=min_cluster_size, metric=metric, cluster_selection_method=cluster_selection_method)
                db_score, num_outliers = self.calc_davies_bouldain_score(embeddings_matrix, cluster_labels)
                results.append({
                    "question_id": rq,
                    "min_cluster_size": min_cluster_size,
                    "metric": metric,
                    "cluster_selection_method": cluster_selection_method,
                    "db_score": db_score,
                    "num_outliers": num_outliers
                    })

        results_df = pd.DataFrame(results)
        self.hdbscan_tuning_results = results_df.sort_values(["question_id","db_score"], ascending=True)
        print(self.hdbscan_tuning_results)


    def generate_clusters(self, clustering_param_dict: dict) -> pd.DataFrame:
        """
        Generate clusters for each research question using HDBSCAN.
        Updates self.state.insights with cluster labels and probabilities.

        Args:
            min_cluster_size (int): Minimum cluster size for HDBSCAN.
            metric (str): Distance metric for HDBSCAN.
            cluster_selection_method (str): Cluster selection method for HDBSCAN.
        """
        
        rqs = self.valid_embeddings_df["question_id"].unique()
        # Check if clustering_param_dict has entries for all rqs
        if len(clustering_param_dict) != len(rqs):
            use_default = None
            while use_default not in ['y', 'n']:
                use_default = input(
                    f"You did not enter specific clustering parameters for each research question. "
                    "Do you want to use default parameters? (y/n): "
                ).lower()
                if use_default == 'n':
                    raise KeyboardInterrupt("Please rerun and provide clustering parameters for each research question.")
                else:
                    params = [clustering_param_dict.get(rq, {"min_cluster_size": 5, "metric": "euclidean", "cluster_selection_method": "eom"}) for rq in rqs]
        else:
            params = [clustering_param_dict[rq] for rq in rqs]
        
        data = [self.rq_valid_embeddings_dfs[rq] for rq in rqs]

        offset = 0  # Initialize offset for cluster label adjustment

        for d, rq, param in zip(data, rqs, params):
            embeddings_matrix = np.vstack(d["reduced_insight_embedding"].to_list())
            cluster_labels, cluster_probs = self.cluster(
                embedding_matrix=embeddings_matrix,
                min_cluster_size=param["min_cluster_size"],
                metric=param["metric"],
                cluster_selection_method=param["cluster_selection_method"]
            )

            d["cluster"] = cluster_labels
            d["cluster_prob"] = cluster_probs
            # Translate cluster labels stating from 1, with 1 being the largest 
            # 1. Get cluster sizes (excluding -1)
            cluster_sizes = d[d["cluster"] != -1].groupby("cluster").size().sort_values(ascending=False)
            # 2. Map old cluster labels to new ones (largest=1, next=2, etc.)
            label_map = {old: i+1+offset for i, old in enumerate(cluster_sizes.index)}
            # 3. Apply mapping, keep -1 as is
            d["cluster"] = d["cluster"].apply(lambda x: label_map[x] if x in label_map else -1)
            # 4. Update offset for next DataFrame
            if cluster_sizes.size > 0:
                offset = max(label_map.values())

        summary_df = [self.make_cum_prop_cluster_table(d) for d in data]
        summary_df = [df.assign(question_id=rq) for df, rq in zip(summary_df, rqs)]
        self.cum_prop_cluster = pd.concat(summary_df)
        clustered_df = pd.concat(data)

        self.state.insights = self.state.insights.merge(
            clustered_df[["question_id", "paper_id", "chunk_id", "insight_id", "cluster", "cluster_prob", "full_insight_embedding", "reduced_insight_embedding"]],
            on=["question_id", "paper_id", "chunk_id", "insight_id"],
            how="left"
        )

        return self.state.insights

    @staticmethod
    def make_cum_prop_cluster_table(df: pd.DataFrame) -> pd.DataFrame:
        """
        Summarize the DataFrame by cluster, showing count, proportion, and cumulative proportion.
        Outliers (cluster == -1) are moved to the end.
        """
        # Exclude rows with missing cluster
        df = df.dropna(subset=["cluster"]).copy()

        # Count size of each cluster (excluding outliers)
        main = df[df["cluster"] != -1].groupby("cluster").size().sort_values(ascending=False)
        # Relabel clusters so largest is 1, next is 2, etc.
        label_map = {old: i+1 for i, old in enumerate(main.index)}
        df["cluster"] = df["cluster"].apply(lambda x: label_map[x] if x in label_map else -1)

        # Count again with new labels (including outliers)
        summary = (
            df.groupby("cluster")
            .size()
            .reset_index(name="count")
            .sort_values(["cluster"], key=lambda col: col.where(col != -1, 999))
            .reset_index(drop=True)
        )

        # Calculate proportion and cumulative proportion
        summary["prop"] = summary["count"] / summary["count"].sum()
        summary["cum_prop"] = summary["prop"].cumsum()

        # Move outlier (-1) to the end
        main_clusters = summary[summary["cluster"] != -1]
        outliers = summary[summary["cluster"] == -1]
        summary = pd.concat([main_clusters, outliers], ignore_index=True)

        return summary


    # def generate_clusters(
    #     self, min_cluster_size: int = 5, metric: str = "euclidean", cluster_selection_method: str = "eom"
    # ) -> pd.DataFrame:
    #     if self.reduced_insight_embeddings_array.size == 0:
    #         raise ValueError("Reduced embeddings not available. Run .embed_insights() and .reduce_dimensions() first.")

    #     clusterer = hdbscan.HDBSCAN(
    #         min_cluster_size=min_cluster_size,
    #         metric=metric,
    #         cluster_selection_method=cluster_selection_method
    #     )

    #     self.valid_embeddings_df["reduced_insight_embeddings"] = [row.tolist() for row in self.reduced_insight_embeddings_array]

    #     clustered_dfs = []
    #     for rq in self.valid_embeddings_df["question_id"].unique():
    #         print(f"Generating clusters for {rq}...")
    #         rq_df = self.valid_embeddings_df[self.valid_embeddings_df["question_id"] == rq].copy()
    #         embeddings_matrix = np.vstack(rq_df["reduced_insight_embeddings"].to_list())
    #         cluster_labels = clusterer.fit_predict(embeddings_matrix)
    #         cluster_probs = clusterer.probabilities_

    #         rq_df["cluster"] = cluster_labels
    #         rq_df["cluster_prob"] = cluster_probs
    #         clustered_dfs.append(rq_df)   

    #     clustered_df = pd.concat(clustered_dfs)

    #     # In case the user is re-running generate clusters to adjust parameters, remove the old cluster assignments
    #     for col in ["cluster", "cluster_prob"]:
    #         if col in self.state.insights.columns:
    #             self.state.insights.drop(columns=[col], inplace=True)   

    #     self.state.insights = self.state.insights.merge(
    #         clustered_df[["question_id", "paper_id", "chunk_id", "cluster", "cluster_prob"]],
    #         on=["question_id", "paper_id", "chunk_id"],
    #         how="left"
    #     )

    #     # 1. Calculate counts and prop per question_id and cluster
    #     cum_prop_cluster = (
    #         self.state.insights.dropna(subset=["cluster"])
    #         .groupby(["question_id", "cluster"])
    #         .size()
    #         .reset_index(name="count")
    #     )

    #     # 2. Calculate proportions within each question_id
    #     cum_prop_cluster["prop"] = cum_prop_cluster.groupby("question_id")["count"].transform(lambda x: x / x.sum())

    #     # 3. Move -1 to the end and calculate cumsum within each question_id
    #     def move_outlier_and_cumsum(df):
    #         outlier = df[df["cluster"] == -1]
    #         main = df[df["cluster"] != -1].sort_values("count", ascending=False)
    #         df_sorted = pd.concat([main, outlier], ignore_index=True)
    #         df_sorted["cum_prop"] = df_sorted["prop"].cumsum()
    #         return df_sorted

    #     cum_prop_cluster = cum_prop_cluster.groupby("question_id", group_keys=False).apply(move_outlier_and_cumsum)

    #     self.cum_prop_cluster = cum_prop_cluster

    #     print("Clusters generated; -1 indicates outliers. Empty insights remain with NaN clusters.")

    #     return self.cum_prop_cluster

    def clean_clusters(self, final_cluster_count: dict = None) -> pd.DataFrame:
        """
        Selects the top N clusters (by size) for each research question, marking all other clusters as outliers (-1).
        Updates self.state.insights with a new column 'selected_cluster' and saves the result.

        Args:
            final_cluster_count (dict): Dictionary mapping question_id to the number of clusters to keep for that question.
                                        Example: {'question_0': 3, 'question_1': 2, ...}

        Returns:
            pd.DataFrame: Updated insights DataFrame with 'selected_cluster' column.
        """
        if final_cluster_count is None:
            self.state.save(os.path.join(STATE_SAVE_LOCATION, "11_clusters"))
            return(self.state.insights)

        else:
            rqs = self.state.insights["question_id"].unique()
            if len(rqs) != len(final_cluster_count):
                raise ValueError(
                    "final_cluster_count must specify the number of clusters to keep for each research question."
                )

            selected_clusters_list = []

            # Loop over each research question
            for rq in self.state.insights["question_id"].unique():
                # Filter insights for the current research question
                current_rq_df = self.state.insights[self.state.insights["question_id"] == rq].copy()
                # Count the size of each cluster (excluding outliers)
                cluster_sizes = current_rq_df.dropna(subset=["cluster"]).groupby("cluster").size().sort_values(ascending=False)

                # Get the number of clusters to keep for this question
                n_keep = final_cluster_count.get(rq, 0)
                # Get the cluster labels of the top N clusters (excluding outlier cluster -1)
                top_clusters = cluster_sizes[cluster_sizes.index != -1].head(n_keep).index.tolist()

                # Mark clusters to keep, others (and outliers) set to -1
                current_rq_df["selected_cluster"] = np.where(
                    current_rq_df["cluster"].isin(top_clusters),
                    current_rq_df["cluster"],
                    -1
                )

                selected_clusters_list.append(current_rq_df)

            # Concatenate all research questions back together
            self.state.insights = pd.concat(selected_clusters_list)
            # Save the updated DataFrame to disk
            self.state.save(os.path.join(STATE_SAVE_LOCATION, "11_clusters"))
            return self.state.insights
        
            

class Summarize:
    def __init__(self,
                 state: Any,
                 llm_client: Any,
                 ai_model: str,
                 paper_output_length: int,  # Approximate total paper length in words
                 summary_save_location: str = None, 
                 state_save_location: str = os.path.join(STATE_SAVE_LOCATION, "12_summarize"), 
                 insight_embedding_path = os.path.join(os.getcwd(), "data", "pickles", "insight_embeddings.pkl")):
        """
        Class to handle summarization of clustered insights.

        Args:
            state: Object holding insights (expects DataFrame `state.insights`).
            llm_client: Client to interact with LLM API.
            ai_model: Model name or identifier for LLM.
            paper_output_length: Total word length for paper; used to proportion cluster summaries.
            summaries_pickle_path: Optional path to pickle the resulting summaries DataFrame.
        """
        self.state: Any = deepcopy(state)
        self.llm_client: Any = llm_client
        self.ai_model: str = ai_model
        self.paper_output_length: int = paper_output_length
        self.summary_save_location: str = summary_save_location or os.path.join(SUMMARY_SAVE_LOCATION, "parquet")
        self.state_save_location: str = state_save_location
        
        if not os.path.exists(insight_embedding_path):
            raise FileNotFoundError(f"Insight embeddings pickle not found at {insight_embedding_path}. Please run clustering first or amend the path to where you pickled your insight embeddings.")
        else:
            with open(insight_embedding_path, "rb") as f:
                self.insight_embeddings_array: np.ndarray = pickle.load(f)

        if not os.path.exists(self.summary_save_location):
            os.makedirs(self.summary_save_location, exist_ok=True)

    def _calculate_summary_length(self) -> pd.DataFrame:
        """
        Calculate approximate word length for each cluster relative to the total paper.

        Returns:
            DataFrame of insights with additional 'length_str' column for prompting the LLM.
        """
        # Count number of insights per cluster
        length_df: pd.DataFrame = (
            self.state.insights
            .dropna(subset = ["cluster"]) # remove any cases where chunks revealed no insights and therefore have no cluster
            .groupby(["question_id", "cluster"])
            .size()
            .reset_index(name="count")
        )

        # Compute proportion of total insights and allocate word length per cluster
        length_df["prop"] = length_df["count"] / length_df["count"].sum()
        length_df["length"] = length_df["prop"] * self.paper_output_length
        length_df["length_str"] = np.where(
            length_df["length"] > 2800,
            "2800 words (approx 4000 tokens)",
            length_df["length"].astype(int).astype(str) + " words"
        )

        # Merge length info back to original insights DataFrame
        insights_with_length: pd.DataFrame = self.state.insights.merge(
            length_df[["question_id", "cluster", "length_str"]],
            how="left",
            on=["question_id", "cluster"]
        )

        return insights_with_length

    def _calculate_centroid(self, col="full_insight_embedding"):
        rows = []
        for rq, d in self.state.insights.groupby("question_id", sort=False):
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
    
    
    def summarize(self) -> Summaries:
        """
        Generate summaries for all clusters across all research questions.

        Returns:
            Summaries object containing a DataFrame of cluster summaries.
        """
        
        if os.path.exists(os.path.join(os.path.join(self.summary_save_location, "summaries.parquet"))):
            recover = None
            while recover not in ['r', 'n']:
                recover = input("Summaries already exist on disk. Do you want to recover (r) or generate new ones (n)? (r/n): ").lower()
            if recover == 'r':
                self.summaries = pd.read_parquet(os.path.join(self.summary_save_location, "summaries.parquet"))
                return self.summaries
            else:
                print("Re-running cleaning of summaries...")

        # We are going to send the insights to the LLM in the order of the shortest path, so that the most similar clusters are summarized close together
        # This will add coherence to the final paper when the summaries are stitched together
        # It will also aid in the applicaion of the sliding window for summary clean up
        shortest_paths = self._estimate_shortest_path()
        
        # Add calculated lengths to insights
        self.state.insights = self._calculate_summary_length()
        
        raw_summaries_list: List[str] = []

        total_clusters = len(self.state.insights.groupby("question_id")["cluster"].nunique(dropna=False))
        count = 1

        # Loop over unique research questions
        for rq_id in self.state.insights["question_id"].unique():
            rq_df: pd.DataFrame = self.state.insights[self.state.insights["question_id"] == rq_id].copy()
            rq_text: str = rq_df["question_text"].iloc[0]

            # Loop over clusters for this research question - in shortest path order
            for cluster in shortest_paths[rq_id]["order"]:
                print(f"Summarizing cluster {cluster} for research question {rq_id} ({count} of {total_clusters})...")
                count += 1
                # Skip any cases where chunks might have had no insights (and therefore no cluster)
                if pd.isna(cluster) or cluster == "NA":
                    continue

                cluster_df: pd.DataFrame = rq_df[rq_df["cluster"] == cluster]
                length_str: str = cluster_df["length_str"].iloc[0]
                # get the insights, they are list of single strings. So make sure they are valid string to send to the LLM
                insights: List[str] = cluster_df["insight"].apply(
                    lambda x: x if isinstance(x, str) else (
                        x[0] if isinstance(x, list) and len(x) == 1 and isinstance(x[0], str) else None
                    )).tolist()
                
                if any(i is None for i in insights):
                    raise ValueError("Insight format error: each insight must be a string or a single-item list containing a string.")

                # Build user prompt for LLM
                user_prompt: str = (
                    f"Research question id: {rq_id}\n"
                    f"Research question text: {rq_text}\n"
                    "PRECEDING CLUSTER SUMMARIES (for context only; may be empty):\n"
                    f"{'\n'.join(raw_summaries_list) if raw_summaries_list else ''}\n"
                    f"Cluster: {cluster}\n"
                    "INSIGHTS:\n" +
                    "\n".join(insights)
                )

                # Build system prompt from predefined method
                sys_prompt: str = Prompts().summarize(summary_length=length_str)

                messages: List[dict] = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ]

                # Call LLM
                response: Any = self.llm_client.chat.completions.create(
                    model=self.ai_model,
                    messages=messages,
                    response_format={"type": "json_object"},
                )

                # Store raw JSON string from LLM
                raw_summaries_list.append(response.choices[0].message.content)

        # Parse JSON responses safely
        clean_summaries_list: List[dict] = []
        for idx, summary in enumerate(raw_summaries_list):
            try:
                clean_summaries_list.append(json.loads(summary))
            except json.JSONDecodeError:
                print(f"JSON decode failed for summary at index: {idx}")

        # Convert to DataFrame
        summaries_df: pd.DataFrame = pd.DataFrame(clean_summaries_list)


        print(
            f"Summaries saved here: {self.summary_save_location}\n"
            "Returned object is a Summaries instance. Access via `variable.summaries`.\n"
            f"Or load later with: Summaries.from_parquet('{self.summary_save_location}')"
        )

        self.summaries = summaries_df

        # Save summaries as this is LLM output we may want to reuse later - save as parquet
        os.makedirs(self.summary_save_location, exist_ok=True)
        self.summaries.to_parquet(os.path.join(self.summary_save_location, "summaries.parquet"))

        return summaries_df
    
    def identify_themes(self, save_file_name="summary_themes.parquet") -> pd.DataFrame:
        if not hasattr(self, "summaries"):
            raise ValueError("No summaries found. Please run summarize() first.")

        save_dir = self.summary_save_location
        save_path = os.path.join(save_dir, save_file_name)

        if os.path.exists(save_path):
            recover = None
            while recover not in ['r', 'n']:
                recover = input("Themed summaries already exist on disk. Recover (r) or generate new (n)? ").lower()
            if recover == 'r':
                self.summary_themes = pd.read_parquet(save_path)  # fix
                return self.summary_themes
            else:
                print("Re-running theming of summaries...")

        out_pdfs = []

        for question_id, rq_df in self.summaries.groupby("question_id", sort=False):
            question_text = rq_df["question_text"].iloc[0]
            summary_text = "\n\n".join(rq_df["summary"].tolist())

            user_prompt = (
                f"Research question id: {question_id}\n"
                f"Research question text: {question_text}\n"
                "SUMMARY TEXT:\n"
                f"{summary_text}\n"
            )
            sys_prompt = Prompts().llm_theme_id()

            fall_back = {"question_id": question_id, "themes": [], "other_bucket_rules": ""}

            resp = call_chat_completion(
                sys_prompt=sys_prompt,
                user_prompt=user_prompt,
                llm_client=self.llm_client,
                ai_model=self.ai_model,
                return_json=True,
                fall_back=fall_back,
            )

            # enforce schema even if empty
            themes = resp.get("themes") or []
            themes_df = pd.DataFrame(themes, columns=["id", "label", "criteria"])

            other_bucket_rules = (resp.get("other_bucket_rules") or "").strip()
            other_df = pd.DataFrame(
                [{"id": "other", "label": "Other", "criteria": other_bucket_rules}],
                columns=["id", "label", "criteria"],
            )

            out_row = pd.concat([themes_df, other_df], ignore_index=True)
            out_row["question_id"] = question_id
            out_row["question_text"] = question_text
            out_pdfs.append(out_row)

        output = pd.concat(out_pdfs, ignore_index=True, sort=False)

        self.summary_themes = output
        os.makedirs(save_dir, exist_ok=True)
        self.summary_themes.to_parquet(save_path)

        return self.summary_themes

    def populate_themes(self, save_file_name="populated_themes.parquet") -> pd.DataFrame:
        # utils
        def build_frozen_block(frozen_content: list[dict]) -> str:
            if not frozen_content:
                return "(none)\n"
            parts = []
            for t in frozen_content:
                parts.append(
                    f"Theme ID: {t.get('theme_id','')}\n"
                    f"Label: {t.get('label','')}\n"
                    f"Criteria: {t.get('criteria','')}\n"
                    f"Content:\n{t.get('contents','')}\n"
                    "--- END THEME ---"
                )
            return "\n".join(parts) + "\n"

        def build_remaining_themes_block(rq_df: pd.DataFrame, current_theme_id: str, processed_ids: set[str]) -> str:
            # remaining = all themes for this RQ not yet processed, excluding the current one
            rem = rq_df.loc[~rq_df["id"].isin(processed_ids | {current_theme_id}), ["label", "criteria"]]
            if rem.empty:
                return "(none)\n"
            parts = []
            for _, r in rem.iterrows():
                parts.append(
                    f"Theme label: {r['label']}\n"
                    f"Criteria: {r['criteria']}\n"
                    "--- END THEME ---"
                )
            return "\n".join(parts) + "\n"

        # guard
        if not hasattr(self, "summary_themes"):
            raise ValueError("No summary themes found. Please run identify_themes() first.")

        save_dir = self.summary_save_location
        save_path = os.path.join(save_dir, save_file_name)

        if os.path.exists(save_path):
            recover = None
            while recover not in ['r', 'n']:
                recover = input("Populated themes already exist on disk. Recover (r) or generate new (n)? ").lower()
            if recover == 'r':
                self.populated_themes = pd.read_parquet(save_path)
                return self.populated_themes
            else:
                print("Re-running population of themes...")

        out_rows = []
        total_themes = len(self.summary_themes)
        counter = 0

        # iterate per research question
        for question_id, rq_df in self.summary_themes.groupby("question_id", sort=False):
            # reset frozen content per question to avoid leakage
            frozen_content: list[dict] = []
            processed_ids: set[str] = set()

            # source text for this RQ
            summary_text_list = self.summaries.loc[self.summaries["question_id"] == question_id, "summary"].tolist()
            summary_text = "\n\n".join(summary_text_list)

            # iterate themes for this question in the given order
            for _, row in rq_df.iterrows():
                counter += 1
                print(f"Populating theme {counter} of {total_themes}")

                question_text = row["question_text"]
                theme_id = row["id"]
                theme_label = row["label"]
                theme_criteria = row["criteria"]

                frozen_block = build_frozen_block(frozen_content)
                remaining_theme_block = build_remaining_themes_block(rq_df, theme_id, processed_ids)

                sys_prompt = Prompts().populate_themes()
                user_prompt = (
                    f"Research question id: {question_id}\n"
                    f"Research question text: {question_text}\n"
                    "FROZEN CONTENT (read-only; text already assigned to themes):\n"
                    f"{frozen_block}"
                    "---CURRENT THEME TO POPULATE:---\n"
                    f"Theme ID: {theme_id}\n"
                    f"Theme label: {theme_label}\n"
                    f"Criteria: {theme_criteria}\n\n"
                    "CLUSTER SUMMARY TEXT (source material):\n"
                    f"{summary_text}\n\n"
                    "--- THEMES STILL TO PROCESS (context only):---\n"
                    f"{remaining_theme_block}"
                )

                fall_back = {"question_id": question_id, "theme_id": theme_id, "assigned_content": ""}

                resp = call_chat_completion(
                    sys_prompt=sys_prompt,
                    user_prompt=user_prompt,
                    llm_client=self.llm_client,
                    ai_model=self.ai_model,
                    return_json=True,
                    fall_back=fall_back,
                )

                assigned = (resp.get("assigned_content") or "").strip()

                out_row = {
                    "question_id": question_id,
                    "question_text": question_text,
                    "theme_id": theme_id,
                    "label": theme_label,
                    "criteria": theme_criteria,
                    "contents": assigned,
                }
                out_rows.append(out_row)

                # update frozen and processed sets
                frozen_content.append(out_row)
                processed_ids.add(theme_id)

        output = pd.DataFrame(
            out_rows, columns=["question_id", "question_text", "theme_id", "label", "criteria", "contents"]
        )

        self.populated_themes = output
        os.makedirs(save_dir, exist_ok=True)
        self.populated_themes.to_parquet(save_path)
        return self.populated_themes
        