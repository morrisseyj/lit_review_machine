# Import project libraries
from lit_review_machine.prompts import Prompts

# Import general libraries
import pandas as pd
from docx import Document
import pickle
from typing import Optional, Dict, Any, List, Union
import os
import json
import ast
import os
import pickle
from typing import List, Optional, Union
import pandas as pd


class QuestionState:
    """
    Container for managing research pipeline state.

    This class keeps track of:
      1. Insights dataframe - traces insights back to the `question_id`.
      2. Full-text dataframe - links full text to a `paper_id`.

    It provides methods to save and load the entire state object as a pickle,
    and to initialize a state from a CSV containing literature data.
    """

    def __init__(
        self,
        state: Optional[List[pd.DataFrame]] = None
    ) -> None:
        """
        Initialize a QuestionState instance.

        Args:
            state (Optional[List[pd.DataFrame]]):
                A list of one or two pandas DataFrames:
                  - [0] Insights dataframe (empty by default).
                  - [1] Full-text dataframe (added automatically if missing).
                If None, initializes with an empty insights dataframe and
                an empty full-text dataframe.
        """
        if state is None:
            state = [pd.DataFrame()]

        if len(state) == 1:
            # Add the full_text dataframe with explicit schema
            state.append(pd.DataFrame(columns=["paper_id", "paper_full_text"]))
            self.state: List[pd.DataFrame] = state
        elif len(state) == 2:
            self.state = state
        else:
            raise ValueError(
                "state must contain 1 or 2 items.\n"
                "- The first item should be the dataframe tracing insights "
                "to question_id.\n"
                "- The optional second item should be the dataframe linking "
                "full text to paper_id."
            )

        # Convenient attributes
        self.insights: pd.DataFrame = self.state[0]
        self.full_text: pd.DataFrame = self.state[1]

        # Placeholder for flattened chunks
        if not hasattr(self, 'chunks'):
            self.chunks: pd.DataFrame = pd.DataFrame(
                columns=["question_id", "paper_id", "chunk_id", "chunk_text"]
            )

    def save(self, save_location: str) -> None:
        """
        Save the entire QuestionState object as a pickle file.

        Args:
            save_location (str): Path where the pickle file will be saved.
        """
        directory_path: str = os.path.dirname(save_location)
        os.makedirs(directory_path, exist_ok=True)

        with open(save_location, "wb") as file:
            pickle.dump(self, file)

    def write_to_csv(
        self,
        folder_path: str = os.path.join(os.getcwd(), "outputs"),
        write_insights: bool = True,
        write_full_text: bool = True,
        write_chunks: bool = True
    ) -> None:
        """
        Write selected QuestionState dataframes to CSV files in a folder.

        Args:
            folder_path (str): Folder where CSV files will be saved. Default is
                               'question_state_export'.
            write_insights (bool): If True, writes self.insights.csv
            write_full_text (bool): If True, writes self.full_text.csv
            write_chunks (bool): If True, writes self.chunks.csv
        """
        os.makedirs(folder_path, exist_ok=True)

        if write_insights:
            insights_file = os.path.join(folder_path, "insights.csv")
            self.insights.to_csv(insights_file, index=False)

        if write_full_text:
            full_text_file = os.path.join(folder_path, "full_text.csv")
            self.full_text.to_csv(full_text_file, index=False)

        if write_chunks:
            chunks_file = os.path.join(folder_path, "chunks.csv")
            self.chunks.to_csv(chunks_file, index=False)

    @classmethod
    def load_from_pickle(cls, filepath: str) -> "QuestionState":
        """
        Load a QuestionState object from a pickle file.

        Args:
            filepath (str): Path to the pickle file.

        Returns:
            QuestionState: Loaded object.

        Raises:
            FileNotFoundError: If no file exists at the given path.
            TypeError: If the pickle does not contain a QuestionState object.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No pickle found at {filepath}")

        with open(filepath, "rb") as f:
            obj: Union["QuestionState", object] = pickle.load(f)

        if not isinstance(obj, cls):
            raise TypeError("Pickle file does not contain a QuestionState object")

        return obj

    @staticmethod
    def _strict_literal_eval(value):
        if pd.isna(value):
            return []
        try:
            return ast.literal_eval(value)
        except (ValueError, TypeError, SyntaxError) as e:
            # Raise a specific error for malformed literal
            raise ValueError(
                f"Fatal Error: Failed to evaluate literal in 'paper_author' column. "
                f"Ensure ALL entries are strictly formatted as a Python list of strings, "
                f"e.g., ['Author A', 'Author B']. The offending value was: '{value}'. "
                "An easy way to do this is to pass the column from your csv to an LLM (web version is fine) and ask it to format it correctly before pasting back."
                f"Original error: {e.__class__.__name__}"
            ) from e

    @classmethod
    def load_from_csv(cls, filepath: str, encoding="utf-8") -> "QuestionState":
        """
        Loads data robustly. Handles encoding error with custom message.
        Handles 'paper_author' literal format error with custom message, no local function.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No CSV found at {filepath}")

        # --- 1. Attempt File Read with Encoding Error Handling ---
        try:
            df: pd.DataFrame = pd.read_csv(filepath, encoding=encoding)
        except UnicodeDecodeError as e:
            # Custom error for the user to fix the encoding argument
            raise UnicodeDecodeError(
                f"Failed to read CSV with encoding '{encoding}'. "
                f"Please check your file's true encoding (e.g., 'cp1252' for ANSI) "
                f"and try again by setting the 'encoding' parameter. Original error: {e}"
            ) from e
        
        # --- 2. Minimal Validation ---
        required_columns = ["search_string_id"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"The CSV is missing required columns for insights: {missing_columns}"
            )

        # --- 3. Strict Literal Evaluation with Custom Error Catch ---
        if "paper_author" in df.columns:
            try:
                # Apply the lambda directly.
                # If ast.literal_eval fails for any row, the exception will propagate here.
                df["paper_author"] = df["paper_author"].apply(
                    lambda x: ast.literal_eval(x) if pd.notna(x) else []
                )
            except (ValueError, TypeError, SyntaxError) as e:
                # Catch the exception that bubbled up from the .apply() method and re-raise the custom message.
                raise ValueError(
                    f"Fatal Error: Failed to evaluate literal in 'paper_author' column. "
                    f"Ensure ALL non-empty entries are strictly formatted as a Python list of strings, "
                    f"e.g., ['Author A', 'Author B']. "
                    "An easy way to do this is to pass the entire 'paper_author' column data to an LLM (web version is fine) and ask it to format it correctly before pasting back. "
                    f"Original error: {e.__class__.__name__}"
                ) from e

        return cls(state=[df])

class Summaries:
    def __init__(
        self, 
        summaries: pd.DataFrame, 
        llm_client: Any,
        ai_model: str,
        summary_string: Optional[str] = None, 
        ai_peer_review: Optional[Dict] = None,
        output_save_location: str = os.path.join(os.getcwd(), "results")
    ):
        """
        Wrapper for a DataFrame of summaries and tools for AI-assisted peer review.
        
        Args:
            summaries: DataFrame with columns ['question_id', 'question_text', 'cluster', 'cluster_summary'].
            llm_client: LLM client for interacting with the AI model.
            ai_model: Name of the deep research model to use.
            summary_string: Optional; pre-computed concatenated summary string.
            ai_peer_review: Optional; stores the AI peer review output as a dictionary.
            output_save_location: Directory to save Word documents.
        """
        self.summaries: pd.DataFrame = summaries
        self.llm_client = llm_client
        self.ai_model = ai_model
        self.summary_string: Optional[str] = summary_string
        self.ai_peer_review: Optional[Dict] = ai_peer_review
        self.output_save_location: str = output_save_location

    @classmethod
    def from_pickle(cls, filepath: str) -> "Summaries":
        """
        Load summaries from a pickle file containing a DataFrame.

        Args:
            filepath: Path to the pickle file.

        Returns:
            Summaries instance with loaded DataFrame.
        """
        with open(filepath, "rb") as f:
            df: pd.DataFrame = pickle.load(f)
        return cls(df)
    
    def get_summary_string(self, output_result: bool = True) -> Optional[str]:
        """
        Concatenate cluster summaries into a single string per research question,
        ordered by question_id and cluster.

        Args:
            output_result: If True, return the concatenated string; else only sets self.summary_string.

        Returns:
            Concatenated summary string if output_result is True; else None.
        """
        # Ensure DataFrame is sorted by question_id and cluster for stable ordering
        self.summaries = self.summaries.sort_values(by=["question_id", "cluster"])

        output_string = ""
        for qid in self.summaries["question_id"].unique():
            qtext = self.summaries.loc[self.summaries["question_id"] == qid, "question_text"].iloc[0]
            question_df = self.summaries[self.summaries["question_id"] == qid]

            question_string = (
                f"Research question id: {qid}\n"
                f"Research question text: {qtext}\n"
                "Review:\n"
                f"{'\n\n'.join(question_df['cluster_summary'].tolist())}\n\n"
            )
            output_string += question_string

        self.summary_string = output_string

        if output_result:
            return output_string

    def get_ai_peer_review(self, output_length: int = 5000, max_tokens: int = 10000) -> Optional[Dict]:
        """
        Request an AI peer review of the concatenated summaries.

        Args:
            output_length: Suggested maximum word count for the review.
            max_tokens: Hard limit on token usage for the AI model.

        Returns:
            AI review as a dictionary. If the review exceeds token budget, 'error' key may appear.
        """
        # Populate summary string if not already done
        if self.summary_string is None:
            self.get_summary_string(output_result=False)

        prompt = Prompts.ai_peer_review(
            lit_review=self.summary_string, 
            output_length=output_length, 
            max_tokens=max_tokens
        )

        print("This call to deep research may take some time...")

        try:
            response = self.llm_client.responses.create(
                model=self.ai_model,
                input=prompt,
                tools=[{"type": "web_search_preview"}],
                response_format={"type": "json_object"}
            )
        except Exception as e:
            print(f"Deep research call failed: {e}")
            return None
        
        self.raw_ai_output: str = response.output_text

        try:
            ai_peer_review: Dict = json.loads(self.raw_ai_output)
            self.ai_peer_review = ai_peer_review
        except json.JSONDecodeError:
            print("AI service did not return valid JSON. Examine 'raw_ai_output'.")
            return self.raw_ai_output

        if "error" in ai_peer_review:
            print("Review could not be completed within the token limit. Check 'ai_peer_review' for details.")

        return ai_peer_review
    
    def summary_to_doc(self, filename: str = "literature_review.docx") -> str:
        """
        Save the concatenated summary string as a Word document.

        Args:
            filename: Name of the output .docx file.

        Returns:
            Path message indicating where the document was saved.
        """
        document = Document()
        document.add_heading("HILT Literature Review", level=0)

        for qid in self.summaries["question_id"].unique():
            qtext = self.summaries.loc[self.summaries["question_id"] == qid, "question_text"].iloc[0]
            question_df = self.summaries[self.summaries["question_id"] == qid]
            cluster_summary_string = "\n".join(question_df["cluster_summary"])

            document.add_heading(f"{qid}: {qtext}", level=1)
            for para in cluster_summary_string.split("\n"):
                if para.strip():
                    document.add_paragraph(para)
            
        path = os.path.join(self.output_save_location, filename)
        os.makedirs(self.output_save_location, exist_ok=True)
        document.save(path)
        return f"AI peer review document created. Saved here: {path}"

    def peer_review_to_doc(self, filename: str = "ai_peer_review.docx") -> str:
        """
        Save the AI peer review output as a Word document.

        Args:
            filename: Name of the output .docx file.

        Returns:
            Path message indicating where the document was saved.

        Raises:
            ValueError: If ai_peer_review has not yet been generated.
        """
        if self.ai_peer_review is None:
            raise ValueError("AI peer review has not yet been completed. Please call .get_ai_peer_review() first.")
        
        document = Document()
        document.add_heading(f"AI peer review - generated by {self.ai_model}", level=0)
        
        for key, value in self.ai_peer_review.items():
            document.add_heading(key, level=1)
            for para in value.split("\n"):
                if para.strip():
                    document.add_paragraph(para)

        path = os.path.join(self.output_save_location, filename)
        os.makedirs(self.output_save_location, exist_ok=True)
        document.save(path)
        return f"AI peer review document created. Saved here: {path}"