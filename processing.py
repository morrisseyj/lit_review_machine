# Import the two projevct modules
from outputs import QuestionState, Summaries
from prompts import Prompts

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
from rapidfuzz.process import cdist
import numpy as np
import networkx as nx
from semanticscholar import SemanticScholar, SemanticScholarException
import datetime
import pymupdf
from kneed import KneeLocator
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
import umap
from sklearn.metrics import silhouette_score
import hdbscan
from docx import Document


STATE_FILE_LOCATION = os.path.join(os.getcwd(), "data", "pickles", "state.pkl")
  
from typing import Optional, List
import pandas as pd
import numpy as np

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
                    or if required columns are missing.
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
        return state

    else:
        # Create new state from injected DataFrame
        state = QuestionState(injected_value)
        if not set(injected_required_cols).issubset(state.insights.columns):
            raise ValueError(
                "Injected dataframe does not contain all required fields. "
                f"Expected at least {injected_required_cols}."
            )
        # Add missing columns (from state_required but not in injected)
        for field in state_required_cols:
            if field not in injected_required_cols:
                state.insights[field] = np.nan
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
        search_engine: str = "Google Scholar",
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
            search_engine (str, optional): Search engine context. Defaults to "Google Scholar".
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
        self.state: QuestionState = state or QuestionState(self._make_state())

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
        if self.messages is None:
            raise RuntimeError("Messages must be generated first with `message_maker()`.")

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
            question_id, search_strings = list(response_data.items())[0]

            # Append results to dataframe
            search_strings_df = pd.concat(
                [
                    search_strings_df,
                    pd.DataFrame(
                        {
                            "question_id": [question_id] * len(search_strings),
                            "search_string": search_strings,
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
        self.state.save(STATE_FILE_LOCATION)

        # Return search strings for testing/debugging
        return self.state.insights["search_string"].to_list()

class AcademicLit:
    """
    Class to retrieve academic literature for given search strings using the Semantic Scholar API.

    Attributes:
        sch (Any): Semantic Scholar client instance.
        num_results (int): Maximum number of papers to return per search string.
        semantic_scholar_api_key (Optional[str]): API key for Semantic Scholar.
        state (QuestionState): Holds search strings and retrieved papers.
    """

    def __init__(
        self,
        sch: Any = None,
        num_results: int = 50,
        semantic_scholar_api_key: Optional[str] = None,
        state: Optional[QuestionState] = None,
        search_strings: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Initialize AcademicLit.

        Args:
            sch (Any, optional): Semantic Scholar client instance. Will create default if None.
            num_results (int, optional): Max papers per search string. Defaults to 50.
            semantic_scholar_api_key (Optional[str], optional): API key to avoid rate limits.
            state (Optional[QuestionState], optional): Pre-existing QuestionState with search_strings.
            search_strings (Optional[pd.DataFrame], optional): Tidy DataFrame with columns
                ["question_id", "question_text", "search_string"] if state is not provided.
        """

        # Initialize Semantic Scholar client
        if sch is None:
            from semanticscholar import SemanticScholar
            sch = SemanticScholar()
        self.sch = sch

        # Enforce max results limit
        if num_results > 100:
            raise ValueError("num_results can't be more than 100.")
        self.num_results = num_results

        
        self.state = validate_format(
            state = state, 
            injected_value=search_strings, 
            state_required_cols=["question_id", "question_text", "search_string", "search_string_id"],
            injected_required_cols=["question_id", "question_text", "search_string", "search_string_id"]
            )

        # Check or prompt for API key
        if semantic_scholar_api_key is None:
            api_key_check = ""
            while api_key_check not in ["y", "n"]:
                api_key_check = input(
                    "You have not set a Semantic Scholar API key. "
                    "It is strongly suggested to acquire one to avoid rate limits.\n"
                    "You can acquire a key here: https://www.semanticscholar.org/product/api\n"
                    "You can proceed without a key but may experience rate limits.\n"
                    "Would you like to proceed? (y/n): "
                ).strip().lower()
                if api_key_check == "n":
                    raise ValueError(
                        "API key not provided. User chose to abort creation."
                    )
            self.semantic_scholar_api_key = None
        else:
            self.semantic_scholar_api_key = semantic_scholar_api_key

    def get_papers(self) -> pd.DataFrame:
        """
        Retrieve papers from Semantic Scholar for each search string in the state.

        Returns:
            pd.DataFrame: DataFrame containing retrieved papers with metadata.

        Raises:
            RuntimeError: If too many requests are sent or exponential backoff fails.
        """
        pubs = []
        search_string_count = 0

        # Iterate over each search string
        for search_string, search_string_id in zip(
            self.state.insights["search_string"], self.state.insights["search_string_id"]
        ):
            search_string_count += 1
            print(f"Retrieving papers for search string {search_string_count} of {self.state.insights.shape[0]}")

            retry_counter = 0
            back_off_exp = 0
            back_off = 1.2 ** back_off_exp
            success = False
            paper_count = 0

            while retry_counter < 10:
                try:
                    semantic_results = self.sch.search_paper(search_string, limit=self.num_results)

                    for result in semantic_results:
                        authors = []
                        if hasattr(result, "author") and result.author:
                            for a in result.author:
                                if a is not None and "name" in a:
                                    authors.append(a["name"])
                        pub_df = pd.DataFrame({
                            "search_string_id": [search_string_id],
                            "paper_id": [f"paper_{search_string_count}_{paper_count}"],
                            "paper_title": [result.title if hasattr(result, "title") else None],
                            "paper_author": [authors],
                            "paper_date": [result.year if hasattr(result, "year") else None]
                        })
                        pubs.append(pub_df)
                        paper_count += 1

                    success = True
                    break  # Exit retry loop if successful

                except Exception as e:
                    if "429" in str(e):
                        print("429 error, retrying...")
                        time.sleep(back_off)
                        retry_counter += 1
                        back_off_exp += 1
                        back_off = 1.2 ** back_off_exp
                    else:
                        raise e

            if not success:
                raise RuntimeError(
                    "Too many requests sent to Semantic Scholar. Could not resolve with exponential backoff."
                )

            time.sleep(1)  # Avoid hitting rate limits too quickly

        # Combine all paper DataFrames into one
        pubs_df = pd.concat(pubs, ignore_index=True) if pubs else pd.DataFrame(
            columns=["search_string_id", "paper_id", "paper_title", "paper_author", "paper_date"]
        )

        # Merge retrieved papers back into state
        merged_df = pubs_df.merge(self.state.insights, how="left", on="search_string_id")

        # Preserve original column order of state.insights
        original_columns = self.state.insights.columns.tolist()
        self.state.insights = merged_df.reindex(columns=original_columns)

        # Save updated state
        self.state.save(STATE_FILE_LOCATION)

        return pubs_df

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
        self.state = validate_format(
            state=state,
            injected_value=papers,
            state_required_cols=[
                "question_id", "question_text", "search_string_id", "search_string", 
                "paper_id", "paper_title", "paper_author", "paper_date"
            ],
            injected_required_cols=[
                "question_id", "question_text", "paper_id", 
                "paper_title", "paper_author", "paper_date"
            ]
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
        search_string = df.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        return search_string.to_list()

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
        dois: List[Optional[str]] = []

        if not self.search_string:
            print("No papers available to retrieve DOIs.")
            self.state.insights["doi"] = []
            return []

        for idx, string in enumerate(self.search_string, start=1):
            print(f"Retrieving DOI {idx} of {len(self.search_string)}")
            doi_result = self.call_alex(string)
            dois.append(doi_result)

        self.state.insights["doi"] = dois
        self.state.save(STATE_FILE_LOCATION)

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
                "paper_id", "paper_title", "paper_author", "paper_date", "doi"
            ],
            injected_required_cols=[
                "question_id", "question_text", "paper_id",
                "paper_title", "paper_author", "paper_date", "doi"
            ]
        )

        download_links: List[Optional[str]] = []

        for doi in self.state.insights.get("doi", []):
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
        self.state.save(STATE_FILE_LOCATION)

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
    GREY_LIT_PICKLE_FILE: str = os.path.join(os.getcwd(), "data", "pickles", "grey_lit.pkl")

    def __init__(
        self,
        llm_client: Any,  # Client interface for interacting with the LLM API
        state: Optional["QuestionState"] = None,  # Current research state (can be injected)
        questions: Optional[List[str]] = None,    # User-defined research questions
        ai_model: str = "o3-deep-research",       # LLM model to use
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
        self.state: "QuestionState" = validate_format(
            state=state,
            injected_value=questions,
            state_required_cols=["question_id", "question_text", "search_string_id", "search_string"],
            injected_required_cols=["question_id", "question_text"]
        )

        self.llm_client: Any = llm_client
        self.ai_model: str = ai_model

    def get_grey_lit(self) -> Optional[pd.DataFrame]:
        """
        Retrieve grey literature relevant to the research questions using the LLM.

        The method:
        1. Builds a prompt from the research questions.
        2. Calls the LLM with web search capability.
        3. Parses the LLM JSON output into a DataFrame.
        4. Merges results with existing QuestionState.
        5. Saves updated state to disk.

        Returns:
            Optional[pd.DataFrame]: A DataFrame containing the grey literature results
            (subset of state where `paper_id` starts with "grey_lit_"), or None if
            retrieval/parsing fails.
        """

        # Concatenate question_id and question_text into a single string for context
        question_strings: pd.Series = self.state.insights["question_id"].str.cat(
            self.state.insights["question_text"], sep=" "
        )

        # Build LLM prompt
        prompt: str = Prompts().grey_lit_retrieve_prompt(
            questions=question_strings.to_list()
        )

        print("Undertaking AI assisted research. This process may take some time.")

        # Send request to the LLM
        try:
            response: Any = self.llm_client.responses.create(
                model=self.ai_model,
                input=prompt,
                tools=[{"type": "web_search_preview"}],
                response_format={"type": "json_object"}
            )
        except Exception as e:
            print(f"Call to Open AI failed. Error: {e}")
            return None

        # Parse JSON output from the LLM
        try:
            response_dict: Dict[str, Any] = json.loads(response.output_text)
        except Exception as e:
            print(
                f"Open AI failed to return a valid JSON. Error: {e}. "
                "self.grey_literature remains unpopulated. To populate it you will need to re-run self.get_grey_lit(). "
                "Note this is an expensive call. Make sure your prompt is giving the correct instructions to return a JSON."
            )
            return None

        # Convert LLM responses into DataFrames
        llm_responses: List[pd.DataFrame] = [pd.DataFrame(value) for value in response_dict.values()]
        grey_lit: pd.DataFrame = pd.concat(llm_responses, ignore_index=True)

        # Prefix paper_id with "grey_lit_" to distinguish from other IDs
        grey_lit["paper_id"] = [f"grey_lit_{i}" for i in grey_lit["paper_id"]]

        # Merge back with original research questions for context
        grey_lit = grey_lit.merge(
            self.state.insights[["question_id", "question_text"]].drop_duplicates(),
            on="question_id",
            how="left"
        )

        # Update state with grey literature results
        self.state.insights = pd.concat([self.state.insights, grey_lit], ignore_index=True)
        self.state.save(STATE_FILE_LOCATION)

        # Return only the grey literature subset
        return self.state.insights[self.state.insights["paper_id"].str.contains("grey_lit_")]

class Literature:
    """
    A class to manage literature (including grey literature) for research questions,
    detect exact and fuzzy duplicates, and export files for manual checking.

    Workflow:
    1. Split literature by question.
    2. Generate a string for duplicate detection.
    3. Detect exact duplicates.
    4. Detect fuzzy duplicates using pairwise string similarity.
    5. Export potential matches for manual verification.
    6. Update QuestionState with cleaned results.
    """

    # Path to save CSV files containing fuzzy matches per question
    FUZZY_CHECK_PATH: str = os.path.join(os.getcwd(), "data", "fuzzy_check")
    os.makedirs(FUZZY_CHECK_PATH, exist_ok=True)

    def __init__(
        self,
        state: "QuestionState",
        literature: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Initialize the Literature object.

        Args:
            state (QuestionState): Current research state object.
            literature (Optional[pd.DataFrame]): Optional pre-loaded literature DataFrame.
        """
        # Validate the state or inject the literature DataFrame
        self.state: "QuestionState" = validate_format(
            state=state,
            injected_value=literature,
            state_required_cols=[
                "question_id", "question_text", "search_string_id", "search_string",
                "paper_id", "paper_title", "paper_author", "paper_date",
                "paper_doi", "download_link"
            ],
            injected_required_cols=[
                "question_id", "question_text", "paper_id", "paper_title", "paper_author",
                "paper_date", "paper_doi", "download_link"
            ]
        )

        # Split literature into a list of DataFrames per question
        self.question_dfs: List[pd.DataFrame] = self._splitter()

    def _splitter(self) -> List[pd.DataFrame]:
        """
        Split the literature by question_id and generate a string for duplicate detection.

        Returns:
            List[pd.DataFrame]: A list of DataFrames, one per research question.
        """
        dfs: List[pd.DataFrame] = [
            self.state.insights[self.state.insights["question_id"] == qid]
            for qid in self.state.insights["question_id"].drop_duplicates()
        ]

        for df in dfs:
            # Concatenate author, title, and date into a string for exact and fuzzy duplicate detection
            df["duplicate_check_string"] = df["paper_author"].str.cat(df[["paper_title", "paper_date"]], sep=" ")

        return dfs

    def drop_exact_duplicates(self) -> List[pd.DataFrame]:
        """
        Drop exact duplicates within each question's literature.

        Returns:
            List[pd.DataFrame]: Updated list of DataFrames with duplicates removed.
        """
        for df in self.question_dfs:
            df.drop_duplicates(subset="duplicate_check_string", keep="first", inplace=True)
        return self.question_dfs

    def _get_fuzzy_match(self, similarity_threshold: int = 90) -> List[List[Tuple[str, str]]]:
        """
        Identify potential fuzzy duplicates within each question.

        Args:
            similarity_threshold (int): Minimum similarity score (0-100) to consider a match.

        Returns:
            List[List[Tuple[str, str]]]: A list of lists of fuzzy duplicate string pairs per question.
        """
        fuzzy_duplicates_list: List[List[Tuple[str, str]]] = []

        for df in self.question_dfs:
            fuzzy_scores = process.cdist(
                df["duplicate_check_string"],
                df["duplicate_check_string"],
                scorer=fuzz.ratio,
                score_cutoff=similarity_threshold
            )

            unique_fuzzy_matches: List[Tuple[str, str]] = []
            for index_x, index_y, score in fuzzy_scores:
                if index_x < index_y:  # Avoid self-matches and reversed duplicates
                    unique_fuzzy_matches.append(
                        (df["duplicate_check_string"].iloc[index_x],
                         df["duplicate_check_string"].iloc[index_y])
                    )

            fuzzy_duplicates_list.append(unique_fuzzy_matches)

        print("Pairwise fuzzy score calculated.")
        return fuzzy_duplicates_list

    def _get_similar_groups(self) -> List[pd.DataFrame]:
        """
        Convert fuzzy duplicates into similarity groups for manual inspection.

        Returns:
            List[pd.DataFrame]: List of DataFrames with `sim_group` assigned per string.
        """
        fuzzy_groups_list: List[pd.DataFrame] = []
        print("Calculating pairwise fuzzy matching scores...")
        fuzzy_duplicates_list = self._get_fuzzy_match()

        for possible_duplicates, df in zip(fuzzy_duplicates_list, self.question_dfs):
            # Build graph with edges for potential duplicates
            graph = nx.Graph()
            graph.add_edges_from(possible_duplicates)
            groups = list(nx.connected_components(graph))

            grouped_matches = []

            for i, group in enumerate(groups, start=1):
                for string in group:
                    grouped_matches.append({"duplicate_check_string": string, "sim_group": i})

            # Assign -1 to strings with no matches
            matched_strings = {string for group in groups for string in group}
            for string in df["duplicate_check_string"]:
                if string not in matched_strings:
                    grouped_matches.append({"duplicate_check_string": string, "sim_group": -1})

            groups_df = pd.DataFrame(grouped_matches)
            fuzzy_groups_list.append(groups_df)

        return fuzzy_groups_list

    def get_fuzzy_matches(self) -> None:
        """
        Export CSV files for manual verification of fuzzy matches per question.

        CSV files include `duplicate_check_string`, similarity group, and original literature metadata.
        Warns users to avoid saving as Excel files.
        """
        fuzzy_groups_list = self._get_similar_groups()

        for index, (fuzzy_group_df, df) in enumerate(zip(fuzzy_groups_list, self.question_dfs)):
            df_for_manual_check = fuzzy_group_df.merge(df, how="left", on="duplicate_check_string")
            df_for_manual_check.to_csv(os.path.join(self.FUZZY_CHECK_PATH, f"question{index + 1}.csv"), index=False)

        print(
            "All fuzzy matches have been identified by question. "
            f"You can find them at {self.FUZZY_CHECK_PATH}. "
            "You should manually check the potential matches by group and delete any true duplicates. "
            "NOTE: papers with no potential matches have group number -1. "
            "IMPORTANT: Only save files as .csv. Saving as .xlsx will not be recognized by update_state(). "
            "Save the .csv files and call .update_state() on your Literature class to update the state."
        )

    def update_state(self, path_to_files: Optional[str] = None) -> pd.DataFrame:
        """
        Update the QuestionState with manually verified literature CSVs or Excel files.

        Args:
            path_to_files (Optional[str]): Directory containing files to import. Defaults to FUZZY_CHECK_PATH.

        Returns:
            pd.DataFrame: Updated `state.insights` DataFrame.
        """
        path_to_files = path_to_files or self.FUZZY_CHECK_PATH

        # Filter for CSV or XLSX files
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

        # Concatenate all files into state
        self.state.insights = pd.concat(dfs, ignore_index=True)

        # Drop helper column
        if "duplicate_check_string" in self.state.insights.columns:
            self.state.insights.drop(columns="duplicate_check_string", inplace=True)

        # Save updated state
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
        ai_model: str,
        state: Optional["QuestionState"] = None,
        papers: Optional[pd.DataFrame] = None
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

        # Validate that the state or injected papers contain all required columns
        self.state: "QuestionState" = validate_format(
            state=state,
            injected_value=papers,
            state_required_cols=[
                "question_id", "question_text", "search_string_id", "search_string",
                "paper_id", "paper_title", "paper_author", "paper_date",
                "paper_doi", "download_link"
            ],
            injected_required_cols=[
                "question_id", "question_text", "paper_id", "paper_title", "paper_author",
                "paper_date", "paper_doi", "download_link"
            ]
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

        json_list = (
            df.groupby(["question_id", "question_text"])
              .apply(lambda x: x[["paper_id", "paper_author", "paper_date", "paper_title"]].to_dict(orient="records"))
              .reset_index(name="papers")
              .to_dict(orient="records")
        )

        return json.dumps(json_list, indent=2)

    def ai_literature_check(self) -> Optional[pd.DataFrame]:
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
        prompt: str = Prompts().ai_literature_check_prompt(
            questions_papers_json=self.json_for_prompt_insertion
        )
        print("Undertaking AI assisted literature check. This may take some time.")

        # Send request to the language model
        try:
            response: Any = self.llm_client.responses.create(
                model=self.ai_model,
                input=prompt,
                tools=[{"type": "web_search_preview"}],
                response_format={"type": "json_object"},
            )
        except Exception as e:
            print(f"Call to LLM failed. Error: {e}")
            return None

        # Parse JSON output from the LLM
        try:
            response_dict: Dict[str, Any] = json.loads(response.output_text)
        except Exception as e:
            print(
                f"LLM failed to return valid JSON. Error: {e}. "
                "AI literature remains unpopulated. To populate, re-run ai_literature_check(). "
                "Note: This is an expensive call."
            )
            return None

        # Flatten dictionary: question_id -> list of paper dicts
        flat_list: List[Dict[str, Any]] = []
        for question_id, papers in response_dict.items():
            for paper in papers:
                paper["question_id"] = question_id
                flat_list.append(paper)

        if not flat_list:
            print("No missing papers returned by the LLM.")
            return pd.DataFrame()

        ai_lit: pd.DataFrame = pd.DataFrame(flat_list)

        # Merge back with question metadata for context
        ai_lit = ai_lit.merge(
            self.state.insights[["question_id", "question_text", "search_string_id", "search_string"]],
            how="left",
            on="question_id"
        )

        # Assign unique AI paper IDs
        ai_lit["paper_id"] = "ai_lit_" + ai_lit["paper_id"].astype(str)

        # Append AI literature to state
        self.state.insights = pd.concat([self.state.insights, ai_lit], ignore_index=True)
        self.state.save(STATE_FILE_LOCATION)

        # Return only the new AI-suggested papers
        return self.state.insights.loc[
            self.state.insights["paper_id"].str.contains("ai_lit_"),
            ["paper_id", "paper_title", "paper_author", "paper_date"]
        ]

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
        self.state: "QuestionState" = validate_format(
            state=state,
            state_required_cols=[
                "question_id", "question_text", "search_string_id", "search_string",
                "paper_id", "paper_title", "paper_author", "paper_date", "paper_doi",
                "download_link", "download_status", "messy_question_id", "messy_paper_id"
            ],
            injected_value=None,
            injected_required_cols=[]
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
            response = self.client.embedding.create(
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
            response = self.client.embedding.create(
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
        kl = KneeLocator(x=x, y=y_sorted, direction="decreasing", curve="concave")
        return [kl.knee_y for _ in y_sorted]

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
            self.hard_to_get_papers["cosine_sim"] <= low_threshold, "low", np.nan
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
            f"Ensure manually downloaded papers follow the naming convention 'paperid.pdf' matching this file."
        )

        return self.hard_to_get_papers


class Downloader:
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
        self.state: "QuestionState" = validate_format(
            state=state,
            injected_value=papers,
            state_required_cols=[
                "question_id", "question_string", "search_string_id", "search_string",
                "paper_id", "paper_title", "paper_author", "paper_date", "paper_doi",
                "download_link"
            ],
            injected_required_cols=[
                "question_id", "question_string",
                "paper_id", "paper_title", "paper_author", "paper_date",
                "download_link"
            ]
        )

        # Ensure the base download folder exists
        self.DOWNLOAD_LOCATION: str = DOWNLOAD_LOCATION
        os.makedirs(self.DOWNLOAD_LOCATION, exist_ok=True)

        # Preserve original IDs and sanitize for filesystem-safe filenames
        self.state.insights["messy_question_id"] = self.state.insights["question_id"]
        self.state.insights["messy_paper_id"] = self.state.insights["paper_id"]
        self.state.insights["question_id"] = self.state.insights["question_id"].apply(self._sanitize_filename)
        self.state.insights["paper_id"] = self.state.insights["paper_id"].apply(self._sanitize_filename)

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

    def download_files(self) -> pd.DataFrame:
        """
        Attempt to download all files in the state DataFrame. Tracks download status
        and local filenames. Updates state and writes a CSV with download results.

        Returns:
            DataFrame containing columns ['paper_id', 'download_status'] with updated statuses.
        """
        # Ensure subfolders exist
        self._create_download_folder()

        # Initialize download tracking columns
        if "download_status" not in self.state.insights.columns:
            self.state.insights["download_status"] = 0
        if "filename" not in self.state.insights.columns:
            self.state.insights["filename"] = np.nan

        # Iterate through each row and attempt download
        for idx, row in self.state.insights.iterrows():
            url: str = row["download_link"]
            status: int = row["download_status"]
            qid: str = row["question_id"]
            pid: str = row["paper_id"]

            print(f"Downloading file {idx + 1} of {self.state.insights.shape[0]}")

            if status == 0:
                if pd.notna(url) and url != "NA":
                    try:
                        response = requests.get(url, stream=True, timeout=10)
                        response.raise_for_status()

                        file_path = os.path.join(self.DOWNLOAD_LOCATION, qid, f"{pid}.pdf")
                        with open(file_path, "wb") as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)

                        self.state.insights.at[idx, "filename"] = file_path
                        self.state.insights.at[idx, "download_status"] = 1
                    except Exception as e:
                        print(f"Failed to download {url}: {e}")
                        self.state.insights.at[idx, "filename"] = np.nan
                        self.state.insights.at[idx, "download_status"] = 0
                else:
                    self.state.insights.at[idx, "filename"] = np.nan
                    self.state.insights.at[idx, "download_status"] = 0
            else:
                self.state.insights.at[idx, "download_status"] = 1

        # Save download status CSV for inspection
        download_status_csv = os.path.join(self.DOWNLOAD_LOCATION, "download_status.csv")
        self.state.insights.to_csv(download_status_csv, index=False)

        print(
            f"Attempted downloads complete. Inspect the results here: {download_status_csv}.\n"
            "For files that failed to download, open this CSV, update the 'download_link' as needed, and save it.\n"
            "Then reload the updated CSV into a QuestionState using:\n"
            "    state = QuestionState.load_from_csv('path/to/download_status.csv')\n"
            "After that, pass the new state to the Downloader and retry downloads:\n"
            "    downloader = Downloader(state=state)\n"
            "Filenames correspond to sanitized question_id and paper_id, preserving traceability."
        )

        return self.state.insights[["paper_id", "download_status"]]


class Ingestor:
    """
    Class to ingest PDF papers into a QuestionState object.
    It reads PDF files from a given directory, validates that each
    paper is associated with a known question_id, and populates
    state.full_text with the text of each paper.

    Attributes:
        state: QuestionState object containing literature metadata.
        file_location: Directory containing PDF files to ingest.
        llm_client: Client for calling the LLM.
        ai_model: Model name to use for LLM.
        confirm_read: Optional; set to "c" to skip the ingestion error confirmation prompt.
        ingestion_errors: List of file paths that failed ingestion.
    """

    def __init__(
        self,
        state: "QuestionState",
        papers: pd.DataFrame,
        file_location: str,
        llm_client: Any,
        ai_model: str,
        confirm_read: Optional[str] = None,  # Set to "c" to skip confirmation prompt 
    ) -> None:
        """Initialize the Ingestor and validate the state/papers format."""
        self.state = validate_format(
            state=state,
            injected_value=papers,
            state_required_cols=[
                "question_id", "question_text", "search_string_id", "search_string",
                "paper_id", "paper_title", "paper_author", "paper_date", "paper_doi",
                "download_link", "download_status", "messy_question_id", "messy_paper_id"
            ],
            injected_required_cols=[
                "question_id", "question_text"
            ]
        )

        self.file_location: str = file_location
        self.llm_client: Any = llm_client
        self.ai_model: str = ai_model
        self.confirm_read: Optional[str] = confirm_read
        self.ingestion_errors: List[str] = []

    @staticmethod
    def list_files(path: str) -> List[str]:
        """Recursively list all files in a given directory."""
        list_of_files: List[str] = []
        for root, _, files in os.walk(path):
            for file in files:
                list_of_files.append(os.path.join(root, file))
        return list_of_files

    @staticmethod
    def paper_ingestor(path: str) -> List[str]:
        """Read all pages from a PDF file using pymupdf."""
        with pymupdf.open(path) as doc:
            return [doc[i].get_text() for i in range(doc.page_count)]

    def ingest_papers(self) -> pd.DataFrame:
        """
        Ingest all PDF papers and populate state.full_text.
        Returns a DataFrame with ['paper_path', 'pages', 'paper_id', 'question_id', 'full_text'].
        """
        list_of_papers_by_page: List[List[str]] = []
        ingestion_status: List[int] = []
        self.ingestion_errors = []

        # List all PDF files in the target directory
        list_of_files = self.list_files(self.file_location)
        valid_question_ids = set(self.state.insights["question_id"].values)

        # Process each file
        for count, file in enumerate(list_of_files, start=1):
            print(f"Ingesting paper {count} of {len(list_of_files)}.")
            question_id = os.path.basename(os.path.dirname(file))

            if question_id in valid_question_ids:
                try:
                    pages = self.paper_ingestor(file)
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

        # Confirm ingestion errors if any
        if self.ingestion_errors:
            self.confirm_read = self.confirm_read or ""
            while self.confirm_read != "c":
                self.confirm_read = input(
                    "Ingestion errors occurred. Examine .ingestion_errors and state.full_text.\n"
                    "Hit 'c' to confirm having read this message:\n"
                ).lower()

        # Update state.insights ingestion status
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

        self.state.full_text = full_text
        return self.state.full_text

    def _get_metadata(self, question_id: str, paper_id: str, text: str) -> Dict[str, Any]:
        """
        Call the LLM to extract metadata from the first three pages.
        Throws an error if the model output does not contain all required fields.
        """
        sys_prompt = Prompts.get_metadata()
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

        # Ensure output contains all expected keys
        required_keys = ["question_id", "paper_id", "paper_title", "paper_author", "paper_date"]
        for key in required_keys:
            if key not in response_dict:
                raise KeyError(f"Metadata extraction failed: missing key '{key}'")
        if not isinstance(response_dict["paper_author"], list):
            raise TypeError(f"'paper_author' should be a list, got {type(response_dict['paper_author'])}")

        return response_dict

    def update_metadata(self) -> pd.DataFrame:
        """
        Update metadata in state.insights by calling LLM for papers
        missing metadata. Merges the result back into insights and saves state.
        """
        # Combine insights and full_text to check missing metadata
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
                text = "".join(row["pages"][:3])
                metadata = self._get_metadata(question_id, paper_id, text)
                metadata_check_df.loc[idx, ["paper_title", "paper_author", "paper_date"]] = [
                    metadata["paper_title"], metadata["paper_author"], metadata["paper_date"]
                ]

        print("Metadata check complete. Saving state...")
        self.state.insights = (
            self.state.insights
            .drop(["paper_title", "paper_author", "paper_date"], axis=1)
            .merge(metadata_check_df, how="left", on=["question_id", "paper_id"])
        )

        return self.state.insights
    
    @staticmethod
    def _flatten_chunks(nested_chunk_df: pd.DataFrame) -> pd.DataFrame:
        """
        Flatten nested chunks into a DataFrame with one chunk per row.

        Args:
            nested_chunk_df: DataFrame containing columns ['question_id', 'paper_id', 'chunks']

        Returns:
            A flattened DataFrame with columns ['question_id', 'paper_id', 'chunk_id', 'chunk_text']
        """
        flattened_chunks: List[dict] = []

        for _, row in nested_chunk_df.iterrows():
            if not row.get("chunks"):
                continue
            for i, chunk in enumerate(row["chunks"]):
                flattened_chunks.append({
                    "question_id": row["question_id"],
                    "paper_id": row["paper_id"],
                    "chunk_id": i,
                    "chunk_text": chunk
                })

        return pd.DataFrame(flattened_chunks)

    def chunk_papers(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 120,
        length_function=len,
        separators: List[str] = None,
        is_separator_regex: bool = False
    ) -> None:
        """
        Split full text of papers into chunks for downstream processing.
        Stores nested chunks in state.full_text and flattened chunks in state.chunks.

        Args:
            chunk_size: Maximum size of each chunk.
            chunk_overlap: Number of overlapping characters between chunks.
            length_function: Function to measure length of text.
            separators: Optional list of separators to use for splitting.
            is_separator_regex: Whether the separators are regex patterns.
        """
        # Use a safe default hierarchy of separators if none provided
        if separators is None:
            separators = ["\n\n", "\n", " ", ""]  # split by paragraphs, lines, spaces, then fallback

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            separators=separators,
            is_separator_regex=is_separator_regex
        )

        # Apply splitting to full_text of each paper
        full_text_list = self.state.full_text["full_text"].to_list()
        chunks_list: List[List[str]] = [text_splitter.split_text(text) for text in full_text_list]

        # Store nested chunks in full_text
        self.state.full_text["chunks"] = chunks_list

        # Flatten chunks for easy reference with chunk_id
        flattened_chunks = self._flatten_chunks(self.state.full_text)
        self.state.chunks = flattened_chunks

class Insights:
    def __init__(
        self,
        state: "QuestionState",
        llm_client: Any,
        ai_model: str
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
        self.state: "QuestionState" = state
        self.llm_client: Any = llm_client
        self.ai_model: str = ai_model

        # Ensure state has all required columns before processing
        self.state = validate_format(
            QuestionState=state,
            injected_value=None,
            state_required_cols=[
                "question_id", "question_text", "search_string_id", "search_string",
                "paper_id", "paper_title", "paper_author", "paper_date", "paper_doi",
                "download_link", "download_status", "messy_question_id", "messy_paper_id",
                "ingestion_status"
            ],
            injected_required_cols=None
        )

    def get_chunk_insights(self) -> pd.DataFrame:
        """
        Extract insights from each text chunk using the LLM.
        Each chunk is processed individually, with the research question
        and other RQs as context. Insights are traced back to 
        (chunk_id, paper_id, question_id).

        Returns:
            pd.DataFrame: Updated `state.insights` with new insights appended.
        """
        # All unique research questions, to provide context
        rqs: List[str] = self.state.insights["question_text"].unique().tolist()

        # Merge chunk text with metadata (author, date, etc.)
        temp_state_df: pd.DataFrame = self.state.chunks.merge(
            self.state.insights[["question_id", "question_text", "paper_author", "paper_date"]],
            how="left",
            on="question_id"
        )

        insights: List[Dict[str, Any]] = []

        # Iterate over each chunk
        for idx, row in temp_state_df.iterrows():
            print(f"Processing chunk {idx + 1} of {temp_state_df.shape[0]}...")

            # Extract fields from row
            question: str = row["question_text"]
            chunk_text: str = row["chunk_text"]
            citation: str = " ".join(row["paper_author"]) + " " + row["paper_date"]
            chunk_id: int = int(row["chunk_id"])

            # Prepare other RQs for context
            other_research_questions: str = " - " + "\n - ".join([rq for rq in rqs if rq != question])

            # Encode text safely for JSON
            safe_chunk_text: str = json.dumps(chunk_text, ensure_ascii=False)
            safe_citation: str = json.dumps(citation, ensure_ascii=False)
            safe_other_rqs: str = json.dumps(other_research_questions, ensure_ascii=False)

            # Build prompts
            sys_prompt: str = Prompts.gen_chunk_insights()
            user_prompt: str = (
                f"CURRENT RESEARCH QUESTION:\n{question}\n\n"
                f"TEXT CHUNK (chunk_id: {chunk_id}):\n{safe_chunk_text} - {safe_citation}\n"
                f"OTHER RESEARCH QUESTIONS (for context only):\n{safe_other_rqs}"
            )

            messages: List[Dict[str, str]] = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ]

            # Call LLM API
            response: Any = self.llm_client.chat.completions.create(
                model=self.ai_model,
                messages=messages,
                response_format={"type": "json_object"}
            )

            # Parse JSON response and keep traceability via chunk_id
            response_dict: Dict[str, Any] = json.loads(response.choices[0].message.content)
            response_dict["chunk_id"] = chunk_id
            insights.append(response_dict)

        # Convert insights list to DataFrame
        chunk_insights_df: pd.DataFrame = pd.DataFrame(insights)

        # Merge new insights into chunks
        self.state.chunks = self.state.chunks.merge(chunk_insights_df, how="left", on="chunk_id")

        # Merge into global insights table
        self.state.insights = self.state.insights.merge(
            self.state.chunks[["question_id", "paper_id", "chunk_id", "insight"]],
            how="left",
            on=["question_id", "paper_id"]
        )

        # One row per extracted insight
        self.state.insights = self.state.insights.explode("insight")

        # Save to disk (STATE_FILE_LOCATION assumed to be defined globally)
        self.state.save(STATE_FILE_LOCATION)

        return self.state.insights

    def get_meta_insights(self) -> pd.DataFrame:
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

        # All research questions for context
        rqs: List[str] = [
            f"{row['question_id']}: {row['question_text']}"
            for _, row in self.state.insights[["question_id", "question_text"]].iterrows()
        ]

        meta_insights: List[Dict[str, Any]] = []

        # Process each paper
        for paper_id in self.state.insights["paper_id"].unique():
            # Get paper full text
            paper_content: str = (
                self.state.full_text
                .loc[self.state.full_text["paper_id"] == paper_id, "full_text"]
                .iloc[0]
            )

            # Collect metadata from insights table
            paper_df: pd.DataFrame = self.state.insights[self.state.insights["paper_id"] == paper_id]
            metadata: str = (
                f"{', '.join(paper_df['paper_author'].iloc[0])}, "
                f"{paper_df['paper_date'].iloc[0]}, "
                f"{paper_df['paper_title'].iloc[0]}"
            )

            # Current and other RQs
            current_rq: str = f"{paper_df['question_id'].iloc[0]}: {paper_df['question_text'].iloc[0]}"
            other_rqs: List[str] = [rq for rq in rqs if rq != current_rq]

            # Collate all chunk insights for this paper
            insights_text: str = "\n".join(paper_df["insight"].dropna().astype(str).tolist())

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
            sys_prompt: str = Prompts.gen_meta_insights()

            messages: List[Dict[str, str]] = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ]

            # Call LLM
            response: Any = self.llm_client.chat.completions.create(
                model=self.ai_model,
                messages=messages,
                response_format={"type": "json_object"},
            )

            response_dict: Dict[str, Any] = json.loads(response.choices[0].message.content)
            meta_insights.append(response_dict)

        # Convert to DataFrame
        meta_insights_df: pd.DataFrame = pd.DataFrame(meta_insights)

        # Merge back into state.insights
        meta_insights_df = (
            self.state.insights
            .drop(columns=["chunk_id", "insight"], errors="ignore")
            .merge(meta_insights_df, how="left", on="paper_id")
            .assign(chunk_id="meta_insight")
            .explode("insight")
        )

        # Append new meta insights
        self.state.insights = pd.concat(
            [self.state.insights, meta_insights_df], 
            ignore_index=True
        )

        return meta_insights_df
            
class Clusters:
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
        self.state = validate_format(
            QuestionState=state,
            injected_value=None,
            state_required_cols=[
                "question_id", "question_text", "search_string_id", "search_string",
                "paper_id", "paper_title", "paper_author", "paper_date", "paper_doi",
                "download_link", "download_status", "messy_question_id", "messy_paper_id",
                "ingestion_status", "chunk_id", "insight"
            ],
            injected_required_cols=None
        )
        self.llm_client = llm_client
        self.embedding_model = embedding_model
        self.embedding_dims = embedding_dims
        self.embeddings_pickle_path = embeddings_pickle_path

        self.valid_embeddings_df: pd.DataFrame = pd.DataFrame()
        self.insight_embeddings_array: np.ndarray = np.array([])
        self.reduced_insight_embeddings_array: np.ndarray = np.array([])
        self.cum_prop_cluster: pd.DataFrame = pd.DataFrame()

    def embed_insights(self) -> np.ndarray:
        """
        Generate embeddings for non-empty insights only.
        Returns:
            np.ndarray: 2D array of embeddings for valid insights.
        """
        self.valid_embeddings_df = self.state.insights[
            self.state.insights["insight"].notna() & (self.state.insights["insight"].str.strip() != "")
        ].copy()

        insight_embeddings = []
        for idx, insight in enumerate(self.valid_embeddings_df["insight"], start=1):
            print(f"Embedding insight {idx} of {self.valid_embeddings_df.shape[0]}")
            response = self.llm_client.embedding.create(
                input=insight,
                model=self.embedding_model,
                dimensions=self.embedding_dims
            )
            insight_embeddings.append(response.data[0].embedding)

        self.insight_embeddings_array = np.vstack(insight_embeddings)
        self.save_embeddings()  # safe pickle save
        return self.insight_embeddings_array

    def save_embeddings(self):
        """Save embeddings safely, creating folder if it does not exist."""
        os.makedirs(os.path.dirname(self.embeddings_pickle_path), exist_ok=True)
        with open(self.embeddings_pickle_path, "wb") as f:
            pickle.dump({"insight_embeddings_array": self.insight_embeddings_array}, f)
        print(f"Embeddings safely saved to '{self.embeddings_pickle_path}'.")

    def load_embeddings(self):
        """Load embeddings safely if the pickle exists."""
        if not os.path.exists(self.embeddings_pickle_path):
            raise FileNotFoundError(f"No embeddings pickle found at {self.embeddings_pickle_path}")
        with open(self.embeddings_pickle_path, "rb") as f:
            data = pickle.load(f)
        self.insight_embeddings_array = data["insight_embeddings_array"]
        print(f"Embeddings loaded from '{self.embeddings_pickle_path}'.")
        return self.insight_embeddings_array

    def reduce_dimensions(
        self, n_neighbors: int = 15, min_dist: float = 0.25, n_components: int = 10,
        metric: str = "cosine", random_state: int = 42
    ) -> np.ndarray:
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric,
            random_state=random_state
        )
        self.reduced_insight_embeddings_array = reducer.fit_transform(self.insight_embeddings_array)
        return self.reduced_insight_embeddings_array

    def calc_silhoette(self, rq_exclude: list = None) -> float:
        if self.reduced_insight_embeddings_array.size == 0:
            raise ValueError("Reduced embeddings not available. Run .embed_insights() and .reduce_dimensions() first.")

        sil_df = self.valid_embeddings_df.copy()
        sil_df["reduced_insight_embeddings"] = [row.tolist() for row in self.reduced_insight_embeddings_array]

        if rq_exclude:
            sil_df = sil_df[~sil_df["question_id"].isin(rq_exclude)]

        score = silhouette_score(
            X=np.vstack(sil_df["reduced_insight_embeddings"].to_list()),
            labels=sil_df["question_id"].to_numpy(),
            metric="euclidean"
        )
        print(f"Silhouette score: {score}")
        return score

    def generate_clusters(
        self, min_cluster_size: int = 5, metric: str = "euclidean", cluster_selection_method: str = "eom"
    ) -> pd.DataFrame:
        if self.reduced_insight_embeddings_array.size == 0:
            raise ValueError("Reduced embeddings not available. Run .embed_insights() and .reduce_dimensions() first.")

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric=metric,
            cluster_selection_method=cluster_selection_method
        )

        self.valid_embeddings_df["reduced_insight_embeddings"] = [row.tolist() for row in self.reduced_insight_embeddings_array]

        clustered_dfs = []
        for rq in self.valid_embeddings_df["question_id"].unique():
            print(f"Generating clusters for {rq}...")
            rq_df = self.valid_embeddings_df[self.valid_embeddings_df["question_id"] == rq].copy()
            embeddings_matrix = np.vstack(rq_df["reduced_insight_embeddings"].to_list())
            cluster_labels = clusterer.fit_predict(embeddings_matrix)
            cluster_probs = clusterer.probabilities_

            rq_df["cluster"] = cluster_labels
            rq_df["cluster_prob"] = cluster_probs
            clustered_dfs.append(rq_df)

        clustered_df = pd.concat(clustered_dfs)
        self.state.insights = self.state.insights.merge(
            clustered_df[["chunk_id", "cluster", "cluster_prob"]],
            on="chunk_id",
            how="left"
        )

        self.cum_prop_cluster = (
            self.state.insights.dropna(subset=["cluster"])
            .groupby("cluster")
            .size()
            .reset_index(name="count")
            .assign(cum_prop=lambda x: x["count"] / x["count"].sum())
        )

        print("Clusters generated; -1 indicates outliers. Empty insights remain with NaN clusters.")
        return self.state.insights

    def clean_clusters(self, final_cluster_count: dict) -> pd.DataFrame:
        selected_clusters_list = []

        for rq in self.state.insights["question_id"].unique():
            current_rq_df = self.state.insights[self.state.insights["question_id"] == rq].copy()
            cluster_sizes = current_rq_df.dropna(subset=["cluster"]).groupby("cluster").size().sort_values(ascending=False)

            n_keep = final_cluster_count.get(rq, 0)
            top_clusters = cluster_sizes[cluster_sizes.index != -1].head(n_keep).index.tolist()

            current_rq_df["selected_cluster"] = np.where(
                current_rq_df["cluster"].isin(top_clusters),
                current_rq_df["cluster"],
                -1
            )

            selected_clusters_list.append(current_rq_df)

        self.state.insights = pd.concat(selected_clusters_list)
        self.state.save(STATE_FILE_LOCATION)
        return self.state.insights

class Summarize:
    def __init__(self,
                 state: Any,
                 llm_client: Any,
                 ai_model: str,
                 paper_output_length: int,  # Approximate total paper length in words
                 summaries_pickle_path: str = None):
        """
        Class to handle summarization of clustered insights.

        Args:
            state: Object holding insights (expects DataFrame `state.insights`).
            llm_client: Client to interact with LLM API.
            ai_model: Model name or identifier for LLM.
            paper_output_length: Total word length for paper; used to proportion cluster summaries.
            summaries_pickle_path: Optional path to pickle the resulting summaries DataFrame.
        """
        self.state: Any = state
        self.llm_client: Any = llm_client
        self.ai_model: str = ai_model
        self.paper_output_length: int = paper_output_length
        self.summaries_pickle_path: str = summaries_pickle_path or os.path.join(
            os.getcwd(), "data", "pickles", "summaries.pkl"
        )

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

    def summarize(self) -> Summaries:
        """
        Generate summaries for all clusters across all research questions.

        Returns:
            Summaries object containing a DataFrame of cluster summaries.
        """
        # Add calculated lengths to insights
        self.state.insights = self._calculate_summary_length()
        
        raw_summaries_list: List[str] = []

        # Loop over unique research questions
        for rq_id in self.state.insights["question_id"].unique():
            rq_df: pd.DataFrame = self.state.insights[self.state.insights["question_id"] == rq_id]
            rq_text: str = rq_df["question_text"].iloc[0]

            # Loop over clusters for this research question
            for cluster in rq_df["cluster"].unique():
                # Skip any cases where chunks might have had no insights (and therefore no cluster)
                if pd.isna(cluster) or cluster == "NA":
                    continue

                cluster_df: pd.DataFrame = rq_df[rq_df["cluster"] == cluster]
                length_str: str = cluster_df["length_str"].iloc[0]
                insights: List[str] = cluster_df["insight"].to_list()

                # Build user prompt for LLM
                user_prompt: str = (
                    f"Research question id: {rq_id}\n"
                    f"Research question text: {rq_text}\n"
                    f"Cluster: {cluster}\n"
                    "INSIGHTS:\n" +
                    "\n".join(insights)
                )

                # Build system prompt from predefined method
                sys_prompt: str = Prompts.summarize(summary_length=length_str)

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

        # Convert to DataFrame and sort
        summaries_df: pd.DataFrame = pd.DataFrame(clean_summaries_list)
        summaries_df = summaries_df.sort_values(
            ["question_id", "cluster"],
            key=lambda col: col.where(col != -1, 999)
        ).reset_index(drop=True)

        # Ensure directory exists and pickle the DataFrame
        os.makedirs(os.path.dirname(self.summaries_pickle_path), exist_ok=True)
        with open(self.summaries_pickle_path, "wb") as f:
            pickle.dump(summaries_df, f)

        print(
            f"Summaries pickled here: {self.summaries_pickle_path}\n"
            "Returned object is a Summaries instance. Access via `variable.summaries`.\n"
            f"Or load later with: Summaries.from_pickle('{self.summaries_pickle_path}')"
        )

        return Summaries(summaries_df)



