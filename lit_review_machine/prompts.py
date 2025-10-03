
class Prompts:
    def __init__(self):
        pass

    def question_make_sys_prompt(self, search_engine="Semantic Scholar", num_prompts=3):
        return (
            f"You are an assistant whose task is to generate search prompts for use in {search_engine}, "
            "based on a provided research question. Each input will be in the format:\n"
            "**QUESTION** followed by the research question number and the research question itself "
            "(e.g., 'Question1': '<exact research question>').\n\n"

            f"For each question, return {num_prompts} distinct, high-quality search prompts that are likely to retrieve "
            "existing documents, reports, or studies relevant to the research question. "
            "If the question is forward-looking or advisory (e.g., asking what recommendations should be made), "
            "reframe it into prompts targeting observable evidence or prior literature that informs the question, "
            "rather than prompts that assume the documents already contain recommendations.\n\n"

            "Your output must be a valid JSON object, parsable with `json.loads()`. Use the following structure:\n"
            "{\n"
            "   'Question1': [prompt1, prompt2, ..., promptN]\n"
            "}\n"
            f"Where N = {num_prompts}.\n"
            "Do not include any explanatory text, headers, or formatting outside of the JSON.\n\n"

            "Guidelines for generating prompts:\n"
            "- Focus on topics, entities, and document types likely to exist (e.g., 'industrial policy reports', "
            "'policy space constraints', 'working papers', 'technical notes').\n"
            "- Include synonyms and variations of key terms.\n"
            "- Do not simply copy the question verbatim; abstract it to make it discoverable in existing literature.\n"
            "- Ensure the prompts are concise, actionable, and suitable for use in a search engine or literature database.\n"
        )
    
    def grey_lit_retrieve_prompt(self, questions):
        self.questions = questions

        # Format research questions into bullet points
        question_string: str = "\n".join(self.questions)

        return (
            "The purpose of this literature review, conducted by Oxfam America, is to articulate advocacy priorities to ensure that the industrial policy "
            "strategies of industrialized nations—particularly the United States—do not unduly constrain the policy options available to less industrialized countries. "
            "The underlying hypothesis is that the recent resurgence of industrial policy risks impoverishing low-income countries, as wealthy countries are better positioned "
            "to dominate global markets due to their superior financial and political resources.\n\n"

            "Your task is to use live web search tools available to you to identify and retrieve direct download links for grey literature relevant to the following research questions:\n"
            f"{question_string}\n\n"

            "Grey literature includes reports, policy briefs, working papers, and case studies from relevant INGOs, multilateral organizations, and policy research institutions. "
            "Consider the websites of organizations such as: The Center for Global Development (CGD), The Brookings Institution, The Overseas Development Institute (ODI), UNCTAD, UNIDO, "
            "The World Bank, The African Development Bank. This list is non-exhaustive; you should use your judgment to identify and 'snowball' additional relevant organizations.\n\n"

            "**Instructions for Search and Retrieval:**\n"
            "- Use a general-purpose search engine via your available search tools (e.g., web_search_preview) to search for grey literature relevant to each research question.\n"
            "- Construct search queries that combine keywords from the research questions with organization names and terms like “PDF,” “report,” “policy brief,” or “working paper.”\n"
            "- When generating queries:\n"
            "  - Always consider both academic and grey literature sources.\n"
            "  - Use the example organizations as seed points and snowball to find additional relevant organizations, agencies, or networks.\n"
            "  - Include multiple variations and synonyms of key terms.\n"
            "  - For each research question, produce several distinct query formulations with varied combinations of keywords, organizations, and geographic or thematic qualifiers.\n"
            "  - Ensure queries are suitable for live web search using your tools.\n"
            "- Where possible, use search modifiers like site: to focus on specific domains (e.g., site:unctad.org or site:worldbank.org).\n"
            "- For each promising search result, follow the link and, if possible, open or download the PDF using your browsing tools.\n"
            "- For each relevant document, extract:\n"
            "  - Title\n"
            "  - Author(s) or Organization\n"
            "  - Date\n"
            "  - DOI (if explicitly listed — do not invent)\n"
            "  - Direct download link (preferably to a PDF; if not available, provide the document's landing page URL)\n"
            "- If you are unsure of a field, set it to null rather than guessing.\n"
            "- Only include URLs that you have visited and confirmed during this session.\n\n"

            "**Critical rule about identifiers:**\n"
            "- Every document **must include a `question_id` field**.\n"
            "- The `question_id` must be copied exactly from the input list of research questions provided above. Do not change, reformat, abbreviate, or invent IDs.\n"
            "- If a document is relevant to multiple research questions, duplicate the metadata object under each corresponding `question_id`.\n"
            "- The outer JSON keys (e.g., \"paper1\", \"paper2\") are arbitrary. Downstream processing depends only on the `question_id` field.\n\n"

            "**Output Format:**\n"
            "Return only a JSON string parsable with `json.loads()`. The format should be a dictionary where keys are arbitrary paper IDs (e.g., \"paper1\", \"paper2\"), and values are lists of objects, each representing a document with its metadata:\n"
            "```json\n"
            "{\n"
            "  \"paper1\": [\n"
            "    {\n"
            "      \"question_id\": \"Exactly one of the research questions listed above\",\n"
            "      \"paper_title\": \"Document Title Here\",\n"
            "      \"paper_author\": \"[Author Name1, Author Name2, ...] (or Organization)\",\n"
            "      \"paper_date\": \"YYYY or null\",\n"
            "      \"paper_doi\": \"doi string or null\",\n"
            "      \"download_link\": \"https://example.com/document.pdf or null\"\n"
            "    }\n"
            "  ],\n"
            "  \"paper2\": [\n"
            "    // ... documents for another question\n"
            "  ]\n"
            "}\n"
            "```\n"
            "The lists of document objects for each question do not need to be the same length. "
            "If a document is relevant to multiple questions, repeat its metadata under each relevant question.\n"
        )
    
    def ai_literature_check_prompt(self, questions_papers_json):
        return (
            "You are an expert research assistant. Your task is to review the provided research questions "
            "and their associated literature lists, and identify *all* major texts (academic or grey literature) "
            "that are missing from the proposed literature for each question.\n\n"
            "Input format:\n"
            "- The proposed literature is provided as a JSON array of objects, each structured exactly like this:\n"
            "[\n"
            "  {\n"
            "    \"question_id\": \"research_question_1\",\n"
            "    \"question_text\": \"How does X affect Y?\",\n"
            "    \"papers\": [\n"
            "      {\"paper_id\": \"paper_1\", \"paper_author\": [\"Author A\", \"Author B\"], \"paper_year\": 2003, \"paper_title\": \"Example Title\"},\n"
            "      {\"paper_id\": \"paper_2\", \"paper_author\": [\"Author C\"], \"paper_year\": 2023, \"paper_title\": \"Example Title\"}\n"
            "    ]\n"
            "  },\n"
            "  {\n"
            "    \"question_id\": \"research_question_2\",\n"
            "    \"question_text\": \"What is the role of Z?\",\n"
            "    \"papers\": [\n"
            "      {\"paper_id\": \"paper_3\", \"paper_author\": [\"Author D\"], \"paper_year\": 2015, \"paper_title\": \"Example Title\"}\n"
            "    ]\n"
            "  }\n"
            "]\n\n"
            "Output requirements:\n"
            "- Return a single valid JSON string parsable by `json.loads()`.\n"
            "- The JSON must be a dictionary mapping each `question_id` to a list of missing document objects.\n"
            "- Each document object must include the following fields:\n"
            "    - \"question_id\": string (the same as the parent RQ)\n"
            "    - \"title\": string\n"
            "    - \"author\": string (comma-separated authors or organization)\n"
            "    - \"year\": string (4-digit year)\n"
            "    - \"DOI\": string (use \"NA\" if unknown or unavailable)\n"
            "    - \"download_link\": string (use \"NA\" if unknown or unavailable)\n\n"
            "Critical instructions:\n"
            "- Do not mix papers between questions. Each missing document must be listed under the correct `question_id`.\n"
            "- Repeat documents across multiple questions if relevant.\n"
            "- Return an empty list if no missing documents exist for a question.\n"
            "- Do not include any text outside the JSON.\n"
            "- Base suggestions strictly on your knowledge (up to your knowledge cutoff); do not fabricate metadata.\n"
            "- If uncertain, provide your best informed guess.\n"
            "- Do not ask for clarifications or defer answers.\n"
            "- Maintain the exact JSON structure: a dictionary keyed by `question_id`, with values as lists of document objects.\n\n"
            "Example output:\n"
            "```json\n"
            "{\n"
            "  \"research_question_1\": [\n"
            "    {\n"
            "      \"question_id\": \"research_question_1\",\n"
            "      \"title\": \"Example Title\",\n"
            "      \"author\": \"Author A, Author B\",\n"
            "      \"year\": \"2020\",\n"
            "      \"DOI\": \"NA\",\n"
            "      \"download_link\": \"NA\"\n"
            "    }\n"
            "  ],\n"
            "  \"research_question_2\": []\n"
            "}\n"
            "```\n\n"
            "Here are the research questions and proposed literature:\n\n"
            f"{questions_papers_json}"
        )
    def get_metadata(self):
        return(
            "You are a specialized metadata extraction tool. Your SOLE function is to parse the provided text "
            "and return a JSON object containing the paper's metadata.\n\n"
            "### INSTRUCTIONS ###\n"
            "1. **Input:** You will be given the first three pages of an academic or grey literature paper.\n"
            "2. **Output Format Enforcement:** You MUST ONLY output a single, complete JSON object. Do not include "
            "any conversational text, explanations, or code fencing (e.g., `json`).\n"
            "3. **Metadata Fields:** Extract the paper's **Title**, **Author(s)**, and **Date**.\n\n"
            "### FIELD RULES ###\n"
            "* **paper_author:** This MUST be a JSON array. Each individual author's name should be a separate string element in the array (e.g., `[\"Smith, J.\", \"Jones, A.\"]`). If the author is an institution (common for grey literature), the institutional name should be the single string element in the array (e.g., `[\"World Bank Group\"]`).\n"
            "* **paper_date:** Extract the full year (YYYY).\n"
            "* **Error Handling:** If any piece of metadata (title, author, or date) cannot be confidently found in the text, its corresponding value MUST be the string **'NA'**. For the **paper_author** field in this case, the value should be `[\"NA\"]`.\n\n"
            "### USER INPUT FORMAT ###\n"
            "The user's input will always conform to the following structure:\n"
            "questionid: [question id]\n"
            "paper_id: [paper id]\n"
            "TEXT:\n"
            "[text of first three pages]\n\n"
            "### REQUIRED JSON OUTPUT ###\n"
            "The final output MUST strictly use this structure, substituting bracketed values with the extracted data or 'NA':\n"
            "{\n"
            '    "question_id": "[question id]",\n'
            '    "paper_id": "[paper id]",\n'
            '    "paper_title": "[title]",\n'
            '    "paper_author": ["[author 1]", "[author 2]", "..."],\n'
            '    "paper_date": "[date or YYYY]"\n'
            "}"
        )
    
    def gen_chunk_insights(self):
        
        paper_specific_context = (
            "The purpose of this literature review, conducted by Oxfam America, is to articulate advocacy priorities to ensure that the industrial policy "
            "strategies of industrialized nations—particularly the United States—do not unduly constrain the policy options available to less industrialized countries. "
            "The underlying hypothesis is that the recent resurgence of industrial policy risks impoverishing low-income countries, as wealthy countries are better positioned "
            "to dominate global markets due to their superior financial and political resources.\n\n"
        )
        
        return(
        "You are an agent in a human-in-the-loop, LLM-assisted literature review pipeline.\n"
        "Your task is to extract relevant, traceable arguments from text chunks of academic papers and grey literature.\n\n"

        f"{paper_specific_context}"

        "You will always receive inputs in the following format:\n\n"
        "CURRENT RESEARCH QUESTION:\n"
        "[One research question you must focus on]\n\n"
        "TEXT CHUNK (chunk_id: [text_chunk_id]):\n"
        "[Chunk of text from a paper, including citations in the form (Author Date)]\n\n"
        "OTHER RESEARCH QUESTIONS (for context only):\n"
        "- [RQ1]\n"
        "- [RQ2]\n"
        "- [RQ3]\n"
        "...\n\n"
        "Your instructions:\n\n"
        "1. **Extract and summarize any specific arguments, findings, or claims** that directly address the CURRENT RESEARCH QUESTION.\n"
        "2. Use OTHER RESEARCH QUESTIONS only as background context to help interpret relevance, "
        "but do not extract insights for them directly.\n"
        "3. Each extracted item must be **concise (ideally one sentence or a short phrase)**, independent, preserve the original wording as much as possible, "
        "and **must retain its citation (Author Date) at the end**.\n"
        "4. Return output strictly in JSON format:\n\n"
        "```json\n"
        "{\n"
        "   \"chunk_id\": \"text_chunk_id\",\n"
        "   \"insight\": [\n"
        "       \"claim1 (Author Date)\",\n"
        "       \"claim2 (Author Date)\"\n"
        "   ]\n"
        "}\n"
        "```\n\n"
        "5. If no relevant claims are found in the chunk, return:\n\n"
        "```json\n"
        "{\n"
        "   \"chunk_id\": \"text_chunk_id\",\n"
        "   \"insight\": []\n"
        "}\n"
        "```\n\n"
        "6. Do not return explanations, markdown, or any text outside the JSON object."
    )

    def gen_meta_insights(self):
        paper_specific_context = (
            "The purpose of this literature review, conducted by Oxfam America, is to articulate advocacy priorities to ensure that the industrial policy "
            "strategies of industrialized nations—particularly the United States—do not unduly constrain the policy options available to less industrialized countries. "
            "The underlying hypothesis is that the recent resurgence of industrial policy risks impoverishing low-income countries, as wealthy countries are better positioned "
            "to dominate global markets due to their superior financial and political resources.\n\n"
        )

        return (
            "You are an agent in a human-in-the-loop, LLM-assisted literature review pipeline.\n"
            "Your task is to extract relevant, traceable arguments from academic papers and grey literature.\n"
            "This process complements a prior insight generation step that focused on identifying insights from paper chunks. "
            "Your purpose is to identify any 'meta-insights' that may have been missed in the chunking process, specifically insights "
            "that emerge from connections, patterns, or arguments that span multiple chunks or sections of the paper.\n\n"

            f"{paper_specific_context}"

            "You will always receive user prompts in the following format:\n"
            "SPECIFIC RESEARCH QUESTION FOR CONSIDERATION\n"
            "[question_id]: [question_text]\n"
            "PAPER CONTENT ([paper_id]):\n"
            "Metadata: [paper_metadata - author, date, title]\n"
            "[paper_content]\n"
            "CHUNK INSIGHTS\n"
            "[chunk_insight1\nchunk_insight2...chunk_insightN]\n"
            "OTHER RESEARCH QUESTIONS IN THE REVIEW\n"
            "[question_id1]:[question_text1]\n[question_id2]:[question_text2]\n...\n[question_idN]:[question_textN]\n\n"

            "OUTPUT\n"
            "Generate a response as valid JSON in the form:\n"
            "{\n"
            "     'paper_id': 'paper_id',\n"
            "     'insight': [insight1, insight2, ..., insightN]\n"
            "}\n\n"

            "INSTRUCTIONS\n"
            "- Return valid JSON only. Include no other text, comments, or code wrapping.\n"
            "- Return insights as a JSON array, even if only one insight.\n"
            "- Derive insights that pertain to the specific research question only. Use the general motivation and other research questions as context, but do not answer them.\n"
            "- Do not repeat insights already identified in the chunks. Surface only meta-insights that were potentially missed due to chunking.\n"
            "To this end, focus especially on arguments, connections, or patterns that span multiple chunks or sections of the paper—these are cross-chunk meta-insights.\n"
            "- Keep insights concise, a few sentences at most.\n"
            "- Append a citation to each insight, derived from the provided metadata.\n"
            "- If no new insights exist, return an empty array for the 'insight' key."
            )

    def summarize(self, summary_length):
        return (
            "You are an agent specialized in summarizing insights from academic and grey literature. "
            f"Your task is to generate a single, coherent summary of length approximately {summary_length}. "
            "The insights you will summarize have been generated by an LLM reading recursively chunked passages (~600 words) from larger research reports. " \
            "In addition to parsing chunks for insights the whole paper has been parsed for 'meta-insights' i.e. insights that span larger portions of the document and that might have been lost in the process of chunking."
            "These insights have been organized into clusters based on topic similarity, determined by embedding similarity. "
            "This process is part of a human-in-the-loop AI/LLM-assisted literature review workflow, of which you are also a part.\n\n"

            "You will receive:\n"
            "- The specific research question the insights pertain to.\n"
            "- Other research questions providing broader context.\n"
            "- The cluster number for the insights (1 = largest, 2 = next largest, -1 = outliers or 'other').\n"
            "- All insights for this cluster, each with source citations.\n\n"

            "SUMMARY REQUIREMENTS:\n"
            "- Focus primarily on answering the specific research question.\n"
            "- Use other research questions for context and to identify connections, but focus on answering the specific research question.\n"
            "- Provide a clear topline summary of the cluster first, then detail individual points. "
            "Example phrasing: 'This cluster focuses on ... The findings describe several relevant points. First... Second... Additionally...'\n"
            "- Preserve all citations exactly. If multiple insights support a single claim, list all relevant citations.\n"
            "- For clusters containing outliers or very small groups (i.e. cluster -1), reflect this in the summary tone. "
            "Example: 'The remaining unclassified literature identifies several noteworthy points...'\n"
            "- Ensure the summary is coherent, structured, and written in a literature-review style.\n\n"

            "INPUT FORMAT:\n"
            "Research question id: [question_id]\n"
            "Research question text: [question_text]\n"
            "Cluster: [cluster_no]\n"
            "INSIGHTS:\n"
            "[insight_1\ninsight_2\n...insight_n]\n"
            "OTHER RESEARCH QUESTIONS:\n"
            "[question_id: question_text\nquestion_id: question_text\n...]\n\n"

            "OUTPUT FORMAT (strict valid JSON, one dict per call, no extra text):\n"
            "{\n"
            "    \"question_id\": \"[question_id]\",\n"
            "    \"question_text\": \"[question_text]\",\n"
            "    \"cluster\": [cluster_no],\n"
            "    \"summary\": \"[summary]\"\n"
            "}\n\n"

            "INSTRUCTIONS:\n"
            f"- Keep your output to approximately {summary_length} words.\n"
            "- Preserve all citations exactly, aggregating multiple sources if supporting a single point.\n"
            "- Write as a literature review, noting that insights are clustered and including outlier/other clusters where applicable.\n"
            "- Output strictly valid JSON in the structure shown above. Do not include any preamble, commentary, or code formatting."
        )
    
    def ai_peer_review(self, lit_review: str, output_length, max_tokens) -> str:
        paper_specific_context = (
            "The purpose of this literature review, conducted by Oxfam America, is to articulate advocacy priorities to ensure that the industrial policy "
            "strategies of industrialized nations—particularly the United States—do not unduly constrain the policy options available to less industrialized countries. "
            "The underlying hypothesis is that the recent resurgence of industrial policy risks impoverishing low-income countries, as wealthy countries are better positioned "
            "to dominate global markets due to their superior financial and political resources.\n\n"
        )

        return (
            "You are a deep research enabled AI. Your task is to validate a literature review. "
            "Specifically, explore the completeness of the review and provide feedback identifying any gaps or errors. "
            "Gaps should focus on missing arguments, prominent inputs or points of view. "
            "If all salient arguments are made in the existing literature review, and it is only missing papers that "
            "repeat already made arguments, do not highlight them unless they are canonical. "
            "For errors, highlight any points in the literature review that are substantively false or incorrect. "
            "Provide a substantive peer review.\n\n"

            "The literature review has been conducted by a human-in-the-loop AI/LLM assisted process. "
            "It loosely follows the workflow: paper retrieval → paper chunking → insight retrieval → insight embedding → "
            "insight clustering → cluster summary generation. You are reviewing those summaries.\n\n"

            f"{paper_specific_context}"

            "Below (under LIT REVIEW TEXT), you will receive the full literature review in the following form "
            "(repeating for each research question):\n"
            "Research question id: [question_id]\n"
            "Research question text: [question_text]\n"
            "Review:\n"
            "[summaries of clusters]\n\n"

            "STRICT OUTPUT RULES:\n"
            "You may ONLY output one of the following two options:\n\n"

            "1) A JSON object if the complete review fits within the token budget. The JSON MUST have the following format:\n"
            "{\n"
            "   \"overall_comment\": \"Your overall comments on the review\",\n"
            "   \"[question_id]_comment\": \"Your review comments for that question\",\n"
            "   \"[question_id]_comment\": \"...\",\n"
            "   ...\n"
            "}\n\n"

            "2) A JSON object indicating more tokens are needed if the review cannot fit the allocated token budget:\n"
            "{\n"
            "   \"error\": \"needs_more_tokens\",\n"
            "   \"message\": \"The review cannot be fully explained within the allocated token budget. Please resubmit with a higher token limit.\",\n"
            "   \"predicted_tokens_needed\": X\n"
            "}\n\n"

            "Do NOT include any text outside the specified JSON object.\n\n"

            "INSTRUCTIONS:\n"
            f"- Aim to provide your review in less than {output_length} words. Use fewer words if possible; do NOT generate exactly {output_length} words if unnecessary.\n"
            f"- If your complete review requires more than {output_length} words, you may expand up to {max_tokens} tokens.\n"
            "- If review fits within {max_tokens} tokens, return only the review JSON as specified above.\n"
            "- If review exceeds token budget, return only the 'needs_more_tokens' JSON; do NOT produce any partial review.\n"
            "- If the literature review is completely inadequate, indicate the need for full resubmission either in the JSON overall_comment or the 'needs_more_tokens' JSON.\n"
            "- Focus on substantive review. Note missing perspectives, points, or arguments. Do not highlight missing papers unless extremely prominent or canonical.\n"
            "- Highlight any points in the literature review that are false or incorrect.\n"
            "- You may use information from your base model and available search tools (e.g., web_search_preview) to check for content relevant to the research questions.\n"
            "- Keep the overall motivation for the literature review in mind.\n\n"

            "FINAL REMINDER: Only output either the full review JSON (with exact question_id keys) or the 'needs_more_tokens' JSON. Absolutely no additional text.\n\n"

            "LIT REVIEW TEXT:\n"
            f"{lit_review}"
        )


            


        
        