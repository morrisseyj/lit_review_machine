
class Prompts:
    def __init__(self):
        pass

    def question_make_sys_prompt(self, num_prompts, search_engine='Semantic Scholar'):
        return (
            f'You are an assistant whose task is to generate search prompts for use in {search_engine}, '
            'based on a provided research question. Each input will be in the format:\n'
            '**QUESTION** followed by the research question number and the research question itself '
            '(e.g., \'Question1\': "<exact research question>").\n\n'

            f'For each question, return {num_prompts} distinct, high-quality search prompts that are likely to retrieve '
            'existing documents, reports, or studies relevant to the research question. '
            'If the question is forward-looking or advisory (e.g., asking what recommendations should be made), '
            'reframe it into prompts targeting observable evidence or prior literature that informs the question, '
            'rather than prompts that assume the documents already contain recommendations.\n\n'

            'Your output must be a valid JSON object, parsable with `json.loads()`. Use the following structure:\n'
            '{\n'
            '   "question1": ["prompt1", "prompt2", ..., "promptN"]\n'
            '}\n'
            f'Where N = {num_prompts}.\n'
            'Do not include any explanatory text, headers, or formatting outside of the JSON.\n\n'

            'Guidelines for generating prompts:\n'
            '- Focus on topics, entities, and document types likely to exist (e.g., "industrial policy reports", '
            '"policy space constraints", "working papers", "technical notes").\n'
            '- Include synonyms and variations of key terms.\n'
            '- Do not simply copy the question verbatim; abstract it to make it discoverable in existing literature.\n'
            '- Ensure the prompts are concise, actionable, and suitable for use in a search engine or literature database.\n'
        )

    def grey_lit_retrieve(self, questions: list) -> str:
        """
        System prompt for retrieving grey literature relevant to research questions.
        Maintains strict JSON object structure with a "results" array.
        """

        question_string = "\n".join(questions)

        return (
            'You are a research assistant specializing in the discovery of grey literature relevant to policy and development research questions.\n'
            'Your task is to use live web search tools to identify and retrieve direct download links '
            'for grey literature documents (reports, policy briefs, working papers, or case studies) '
            'that relate to the following research questions:\n'
            f'{question_string}\n\n'

            'Grey literature includes outputs from organizations such as major think tanks, '
            'multilateral organizations, and policy research institutes (for example: '
            'The Center for Global Development, The Brookings Institution, The Overseas Development Institute, '
            'UNCTAD, UNIDO, The World Bank, or regional development banks). '
            'This list is indicative, not exhaustive.\n'
            'Instructions for Search and Retrieval:\n'
            '- Use your available web search tools to find relevant documents for each research question.\n'
            '- Construct search queries combining keywords from each research question with organization names '
            'and terms like PDF, report, policy brief, or working paper.\n'
            '- For each document identified, extract the following metadata fields:\n'
            '  - "question_id": The canonical unique identifier for the research question (e.g., "question_1").\n'
            '  - "paper_title": The title of the document.\n'
            '  - "paper_author": The author(s) or organization(s) responsible, always as a JSON array.\n'
            '  - "paper_date": The publication year (YYYY format), or null if unknown.\n'
            '  - "doi": The DOI string if available, else null.\n'
            '  - "download_link": The direct link to the document PDF or landing page.\n'
            'Critical rules about identifiers and linkage:\n'
            '- Each grey literature item must be associated with at least one research question.\n'
            '- Use the provided canonical "question_id" to indicate which question(s) it is relevant to.\n'
            '- If an item is relevant to multiple questions, duplicate it under each canonical "question_id".\n'
            'Output Format (STRICT JSON OBJECT):\n'
            '- Return ONLY a JSON object with a single key "results" containing an array of documents.\n'
            '- Each element of the array must include all keys listed above.\n'
            '- Use standard JSON double quotes inside the object.\n'
            '- Do NOT include any comments, explanations, or code blocks.\n'
            'Example output:\n'
            '{\n'
            '  "results": [\n'
            '    {\n'
            '      "question_id": "question_1",\n'
            '      "paper_title": "Renewable Energy Policy Frameworks: Lessons from Emerging Economies",\n'
            '      "paper_author": ["Energy Policy Institute"],\n'
            '      "paper_date": "2023",\n'
            '      "doi": null,\n'
            '      "download_link": "https://example.org/policy_frameworks.pdf"\n'
            '    }\n'
            '  ]\n'
            '}\n'
            'Ensure that the output is valid JSON, with the "results" array containing all found documents, '
            'and that "paper_author" is always an array.'
        )


    def grey_literature_format_check(self) -> str:
        """
        System prompt for correcting mis-formatted JSON describing grey literature.
        """

        return (
            'You are an agent specialized in formatting strings to be valid JSON. '
            'The user will provide text that is an approximation of valid JSON describing grey literature documents. '
            "The user content will be delivered as a JSON object with a single key 'results', containing an array of documents.\n\n"

            'Your task is to correct it so it can be parsed with `json.loads()`.\n\n'

            'Requirements:\n'
            "- The input may have formatting errors including misplaced quotes, escaped characters, missing commas, or extra whitespace. Correct all errors.\n"
            "- Preserve the canonical `question_id` for each document.\n"
            "- All keys must appear in every object: 'question_id', 'paper_title', 'paper_author', 'paper_date', 'doi', 'download_link'.\n"
            "- Your response must include all the documents contained in the content submitted by the user.\n"
            "- Ensure string values are enclosed in double quotes.\n"
            "- 'paper_author' must always be a JSON array, even if there is only one author.\n"
            "- If a field is unknown, set it explicitly to null.\n"
            "- Return only valid JSON, no explanations, comments, or code blocks.\n"
            "- Example output:\n"
            '{\n'
            '  "results": [\n'
            '    {\n'
            '      "question_id": "question_1",\n'
            '      "paper_title": "Renewable Energy Policy Frameworks: Lessons from Emerging Economies",\n'
            '      "paper_author": ["Energy Policy Institute"],\n'
            '      "paper_date": "2023",\n'
            '      "doi": null,\n'
            '      "download_link": "https://example.org/policy_frameworks.pdf"\n'
            '    }\n'
            '  ]\n'
            '}\n'
            'Return strictly valid JSON with the "results" array, no extra text.'
        )
        

    def ai_literature_retrieve(self, questions_papers_json: str) -> str:
        """
        Prompt for identifying missing literature (academic or grey) for each research question.
        Returns a JSON object with a single key 'results' containing an array of document objects.
        """
        return (
            'You are an expert research assistant. Your task is to review the provided research questions '
            'along with the lists of literature that have been identified to answer them. '
            'You are to identify *all* major texts (academic or grey literature) '
            'that are missing from the proposed literature for each question.\n\n'

            '### Input format:\n'
            'The proposed literature is provided as a JSON array of objects, each structured like this:\n'
            '[\n'
            '  {\n'
            '    "question_id": "research_question_1",\n'
            '    "question_text": "How does X affect Y?",\n'
            '    "papers": [\n'
            '      {"paper_id": "paper_1", "paper_author": ["Author A", "Author B"], "paper_year": 2003, "paper_title": "Example Title"},\n'
            '      {"paper_id": "paper_2", "paper_author": ["Author C"], "paper_year": 2023, "paper_title": "Another Example"}\n'
            '    ]\n'
            '  }\n'
            ']\n\n'

            '### Output format (STRICT JSON OBJECT):\n'
            '- Return ONLY a JSON object with a single key "results" containing an array of documents.\n'
            '- Do NOT return a bare JSON array.\n'
            '- Each element of "results" must include all keys: "question_id", "paper_title", "paper_author", "paper_date", "doi", "download_link".\n'
            '- "paper_author" must always be a JSON array, even for a single author.\n'
            '- If a field is unknown, set it explicitly to null.\n'
            '- If no missing documents exist, return: {"results": []}\n'
            '- Use double quotes for all JSON strings.\n'
            '- Do NOT include comments, explanations, or code blocks.\n\n'

            '### Instructions for Search and Retrieval:\n'
            '- Use your available web search tools to find relevant documents for each research question.\n'
            '- Construct search queries combining keywords from each research question with organization names '
            'and terms like PDF, report, policy brief, or working paper.\n'
            '- For each document identified, extract the following metadata fields:\n'
            '  - "question_id": The canonical unique identifier for the research question (e.g., "question_1").\n'
            '  - "paper_title": The title of the document.\n'
            '  - "paper_author": The author(s) or organization(s) responsible, always as a JSON array.\n'
            '  - "paper_date": The publication year (YYYY format), or null if unknown.\n'
            '  - "doi": The DOI string if available, else null.\n'
            '  - "download_link": The direct link to the document PDF or landing page.\n'
            '- Each piece of literature must be associated with at least one research question.\n'
            '- Use the provided canonical "question_id" to indicate which question(s) it is relevant to.\n'
            '- If an item is relevant to multiple questions, duplicate it under each canonical "question_id".\n'
            '- Only include metadata from verifiable sources; do not fabricate or infer missing details.\n'
            '- Limit your response to 3900 tokens.\n\n'

            '### Example output:\n'
            '{\n'
            '  "results": [\n'
            '    {\n'
            '      "question_id": "question_1",\n'
            '      "paper_title": "Renewable Energy Policy Frameworks: Lessons from Emerging Economies",\n'
            '      "paper_author": ["Energy Policy Institute"],\n'
            '      "paper_date": "2023",\n'
            '      "doi": null,\n'
            '      "download_link": "https://example.org/policy_frameworks.pdf"\n'
            '    }\n'
            '  ]\n'
            '}\n'
            'Return strictly valid JSON only. "paper_author" must always be an array.\n\n'
            '### QUESTIONS AND PROPOSED LITERATURE'
            f'{questions_papers_json}'
        )


    def ai_literature_format_check(self):
        """
        System prompt for correcting mis-formatted JSON describing grey literature.
        """

        return (
            'You are an agent specialized in formatting strings to be valid JSON. '
            'The user will provide text that is an approximation of valid JSON describing grey literature documents. '
            "The user content will be delivered as a JSON object with a single key 'results', containing an array of documents.\n\n"

            'Your task is to correct it so it can be parsed with `json.loads()`.\n\n'

            'Requirements:\n'
            "- The input may have formatting errors including misplaced quotes, escaped characters, missing commas, or extra whitespace. Correct all errors.\n"
            "- Preserve the canonical `question_id` for each document.\n"
            "- All keys must appear in every object: 'question_id', 'paper_title', 'paper_author', 'paper_date', 'doi', 'download_link'.\n"
            "- Your response must include all the documents contained in the content submitted by the user.\n"
            "- Ensure string values are enclosed in double quotes.\n"
            "- 'paper_author' must always be a JSON array, even if there is only one author.\n"
            "- If a field is unknown, set it explicitly to null.\n"
            "- Return only valid JSON, no explanations, comments, or code blocks.\n"
            "- Example output:\n"
            '{\n'
            '  "results": [\n'
            '    {\n'
            '      "question_id": "question_1",\n'
            '      "paper_title": "Renewable Energy Policy Frameworks: Lessons from Emerging Economies",\n'
            '      "paper_author": ["Energy Policy Institute"],\n'
            '      "paper_date": "2023",\n'
            '      "doi": null,\n'
            '      "download_link": "https://example.org/policy_frameworks.pdf"\n'
            '    }\n'
            '  ]\n'
            '}\n'
            "Return strictly valid JSON with the 'results' array, no extra text."
        )


    def extract_main_html_content(self):
        return (
            'You are a highly specialized text content extraction tool. Your task is to analyze the '
            'plain text provided and return **only the main article content**.'
            '\n\n'
            'The text comes from an html file that has already been partially cleaned using Beautiful Soup (tags like "script", "style", '
            '"nav", "header", "footer", and "aside" have been removed). '
            'Long texts may be partial chunks of a larger document.'
            '\n\n'
            '### INSTRUCTIONS'
            '\n'
            '1. **INPUT STRUCTURE:** The text you must analyze will be enclosed by the delimiters '
            '**[START_TEXT]** and **[END_TEXT]** in the User Message. **You must ignore any text outside of these delimiters.**'
            '\n'
            '2. **INCLUSION:** Identify and return only the central, primary content (the article body, news story, or primary narrative). **Include the article title and the author(s) names.**'
            '\n'
            '3. **EXCLUSION:** You must strictly exclude any secondary or extraneous elements that survived the cleanup. This includes:'
            '\n'
            '   - **Advertisements or promotional text.**'
            '\n'
            '   - **Comment sections, social media share prompts, or detailed author biographies/about pages.**'
            '\n'
            '   - **Lists of related or recommended articles.**'
            '\n'
            '   - **Image captions, alt text, or text describing page navigation (e.g., "Back to top", "Previous Page").**'
            '\n'
            '4. **FORMAT:** Return the extracted content as a single, clean block of plain text. **Do not add any introductory phrases, commentary, or Markdown formatting.**'
            '\n\n'
            '---' 
        )
    
    def get_metadata(self):
        return (
            'You are a specialized metadata extraction tool. Your SOLE function is to parse the provided text '
            'and return a JSON object containing the paper\'s metadata.\n\n'
            '### INSTRUCTIONS ###\n'
            '1. **Input:** You will be given the first three pages of an academic or grey literature paper.\n'
            '2. **Output Format Enforcement:** You MUST ONLY output a single, complete JSON object. Do not include '
            'any conversational text, explanations, or code fencing (e.g., `json`).\n'
            '3. **Metadata Fields:** Extract the paper\'s **Title**, **Author(s)**, and **Date**.\n\n'
            '### FIELD RULES ###\n'
            '* **paper_author:** This MUST be a JSON array. Each individual author\'s name should be a separate string element in the array (e.g., ["Smith, J.", "Jones, A."]). '
            'If the author is an institution (common for grey literature), the institutional name should be the single string element in the array (e.g., ["World Bank Group"]).\n'
            '* **paper_date:** Extract the full year (YYYY).\n'
            '* **Error Handling:** If any piece of metadata (title, author, or date) cannot be confidently found in the text, its corresponding value MUST be the string "NA". '
            'For the **paper_author** field in this case, the value should be ["NA"].\n\n'
            '### USER INPUT FORMAT ###\n'
            'The user\'s input will always conform to the following structure:\n'
            'questionid: [question id]\n'
            'paper_id: [paper id]\n'
            'TEXT:\n'
            '[text of first three pages]\n\n'
            '### REQUIRED JSON OUTPUT ###\n'
            'The final output MUST strictly use this structure, substituting bracketed values with the extracted data or "NA":\n'
            '{\n'
            '    "question_id": "[question id]",\n'
            '    "paper_id": "[paper id]",\n'
            '    "paper_title": "[title]",\n'
            '    "paper_author": ["[author 1]", "[author 2]", "..."],\n'
            '    "paper_date": "[date or YYYY]"\n'
            '}'
        )


    def gen_chunk_insights(self):
        paper_specific_context = (
            'The purpose of this literature review, conducted by Oxfam America, is to articulate advocacy priorities to ensure that the industrial policy '
            'strategies of industrialized nations—particularly the United States—do not unduly constrain the policy options available to less industrialized countries. '
            'The underlying hypothesis is that the recent resurgence of industrial policy risks impoverishing low-income countries, as wealthy countries are better positioned '
            'to dominate global markets due to their superior financial and political resources.\n\n'
        )

        return (
            'You are an agent in a human-in-the-loop, LLM-assisted literature review pipeline.\n'
            'Your task is to extract relevant, traceable arguments from text chunks of academic papers and grey literature.\n\n'
            f'{paper_specific_context}'
            'You will always receive inputs in the following format:\n\n'
            'CURRENT RESEARCH QUESTION:\n'
            '[One research question you must focus on]\n\n'
            'TEXT CHUNK (chunk_id: [text_chunk_id]):\n'
            '[Chunk of text from a paper, including citations in the form (Author Date)]\n\n'
            'OTHER RESEARCH QUESTIONS (for context only):\n'
            '- [RQ1]\n'
            '- [RQ2]\n'
            '- [RQ3]\n'
            '...\n\n'
            'Your instructions:\n\n'
            '1. **Extract and summarize any specific arguments, findings, or claims** that directly address the CURRENT RESEARCH QUESTION.\n'
            '2. Use OTHER RESEARCH QUESTIONS only as background context to help interpret relevance, '
            'but do not extract insights for them directly.\n'
            '3. Each extracted item must be **concise (ideally one sentence or a short phrase)**, independent, preserve the original wording as much as possible, '
            'and **must retain its citation (Author Date) at the end**.\n'
            '4. Return output strictly in **valid JSON** that conforms exactly to the following schema:\n\n'
            '```json\n'
            '{\n'
            '   "chunk_id": "text_chunk_id",\n'
            '   "insight": [\n'
            '       "claim1 (Author Date)",\n'
            '       "claim2 (Author Date)"\n'
            '   ]\n'
            '}\n'
            '```\n\n'
            '5. The value of "insight" **must always be a JSON array (list)** — even if there is only one insight or none.\n'
            '   - If no relevant claims are found, return an empty list: `"insight": []`.\n'
            '   - Never return a string, null, or omitted field for "insight".\n'
            '6. Do not return explanations, markdown, comments, or text outside the JSON object.'
        )


    def gen_meta_insights(self):
        """
        Generate a system prompt for extracting higher-level 'meta-insights' 
        from entire papers, focusing on cross-chunk reasoning.
        """
        paper_specific_context = (
            'The purpose of this literature review, conducted by Oxfam America, is to articulate advocacy priorities to ensure that the industrial policy '
            'strategies of industrialized nations—particularly the United States—do not unduly constrain the policy options available to less industrialized countries. '
            'The underlying hypothesis is that the recent resurgence of industrial policy risks impoverishing low-income countries, as wealthy countries are better positioned '
            'to dominate global markets due to their superior financial and political resources.\n\n'
        )

        return (
            'You are an agent in a human-in-the-loop, LLM-assisted literature review pipeline.\n'
            'Your task is to extract **meta-insights** — higher-level, traceable arguments or conclusions that span across multiple chunks or sections of a paper.\n'
            'These meta-insights should complement (not repeat) the insights already extracted at the chunk level.\n\n'
            f'{paper_specific_context}'
            'You will always receive user input in the following format:\n\n'
            'SPECIFIC RESEARCH QUESTION FOR CONSIDERATION\n'
            '[question_id]: [question_text]\n\n'
            'PAPER CONTENT ([paper_id]):\n'
            'Metadata: [paper_metadata - author, date, title]\n'
            '[paper_content]\n\n'
            'CHUNK INSIGHTS:\n'
            '[chunk_insight1]\n[chunk_insight2]\n...\n[chunk_insightN]\n\n'
            'OTHER RESEARCH QUESTIONS IN THE REVIEW (context only):\n'
            '[question_id1]: [question_text1]\n[question_id2]: [question_text2]\n...\n\n'
            '---\n\n'
            'OUTPUT REQUIREMENTS:\n'
            'Return a **valid JSON object** matching this exact schema:\n\n'
            '```json\n'
            '{\n'
            '  "paper_id": "paper_id",\n'
            '  "insight": [\n'
            '    "meta_insight_1 (Author Date)",\n'
            '    "meta_insight_2 (Author Date)"\n'
            '  ]\n'
            '}\n'
            '```\n\n'
            'ADDITIONAL INSTRUCTIONS:\n'
            '- The value of "insight" **must always be a JSON array (list)** — even if empty or containing only one insight.\n'
            '- Return `"insight": []` if no new meta-insights are found.\n'
            '- Do **not** return strings, null, or omit the "insight" key.\n'
            '- Derive insights that pertain only to the specified research question.\n'
            '- Use the provided paper metadata to append citations to each insight.\n'
            '- Keep insights concise (ideally 1–3 sentences each).\n'
            '- Do not repeat insights already found in chunks.\n'
            '- Do not output explanations, markdown, or any text outside the JSON object.'
        )


    def summarize(self, summary_length):
        return (
            "You are an agent specialized in summarizing insights from academic and grey literature. "
            f"Your task is to generate a single, coherent summary of approximately {summary_length} words. "
            "The insights you will summarize have been generated by an LLM reading recursively chunked passages (~600 words) from larger research reports. "
            "In addition to parsing chunks for insights, each whole paper has also been parsed for 'meta-insights'—i.e., insights that span larger portions of the document and that might otherwise be lost in the process of chunking. "
            "These insights have been organized into clusters based on topic similarity, determined by embedding similarity, via the application of UMAP and HDBSCAN. "
            "The clusters have been further analyzed to identify the shortest path through them, optimizing the order in which they will be presented in the final summary. "
            "This process is part of a human-in-the-loop AI/LLM-assisted literature review workflow, of which you are also a part.\n\n"

            "You will receive:\n"
            "- The specific research question the insights pertain to.\n"
            "- Other research questions providing broader context for the overall literature review.\n"
            "- The preceding cluster summaries for context (not to be included in your output). These may be empty; if empty, there is no preceding text.\n"
            "- The cluster number for the insights (all clusters are uniquely labelled, with -1 indicating outliers or 'other').\n"
            "- All insights for this cluster, each with source citations.\n\n"

            "SUMMARY REQUIREMENTS:\n"
            "- When summarizing the insights, focus primarily on answering the specific research question.\n"
            "- Use other research questions for context and to identify conceptual or thematic connections, but ensure your primary focus remains on the specific research question.\n"
            "- Use the cluster summaries already created as context to increase overall coherence and to help identify cross-cutting themes across clusters. "
            "Explicitly note thematic linkages or contrasts when they help to situate the current cluster within the broader literature. "
            "However, only include information drawn from the current clusters insights in your actual summary. "
            "Do not restate, paraphrase, or edit text from the preceding summaries.\n"
            "- If there are no preceding summaries, write the summary as if it is the first in the sequence. Introduce the topic clearly and independently.\n"
            "- Provide a clear topline summary of the cluster first, then detail individual points. "
            'Example phrasing: "This cluster focuses on ... The findings describe several relevant points. First ... The second links with themes mentioned earlier ... Additionally ..."\n'
            "- When preceding cluster summaries exist, use transitions to maintain narrative coherence. "
            'Example phrasing: "Building on themes discussed previously ... In contrast to earlier findings ..."\n'
            "- Preserve all citations exactly as they appear. If multiple insights support a single claim, list all relevant citations together.\n"
            "- For clusters containing outliers or very small groups (i.e., cluster -1), reflect this in the tone of the summary. "
            'Example: "The remaining unclassified literature identifies several noteworthy points ..."\n'
            "- Ensure the summary is coherent, logically structured, and written in an academic literature-review tone.\n\n"

            "INPUT FORMAT:\n"
            "Research question id: <question_id>\n"
            "Research question text: <question_text>\n"
            "PRECEDING CLUSTER SUMMARIES (for context only; may be empty):\n"
            "<cluster_summary_1>\n<cluster_summary_2>\n...<cluster_summary_n>\n"
            "Cluster: <cluster_no>\n"
            "INSIGHTS:\n"
            "<insight_1>\n<insight_2>\n...<insight_n>\n"
            "OTHER RESEARCH QUESTIONS:\n"
            "<question_id: question_text>\n<question_id: question_text>\n...<question_id: question_text>\n\n"

            "OUTPUT FORMAT (strict valid JSON, one dict per call, no extra text):\n"
            "{\n"
            '    "question_id": "<question_id>",\n'
            '    "question_text": "<question_text>",\n'
            '    "cluster": <cluster_no>,\n'
            '    "summary": "<summary>"\n'
            "}\n\n"

            "INSTRUCTIONS:\n"
            "- Output strictly valid JSON only (no commentary, preamble, or code block formatting).\n"
            "- Each value must be a primitive type (string or integer), not an array or list.\n"
            "- Do not include square brackets [] in the output JSON unless explicitly part of the text of the summary.\n"
            f"- Your output should not exceed approximately {summary_length} words. "
            "If all insights can be effectively summarized without losing detail in fewer words, produce a shorter summary. "
            "Preserve as much granularity of insight as possible within the limit; compress phrasing, not substance.\n"
            "- Maintain fidelity to the content of the insights and citations while improving readability and coherence.\n"
            "- Write as a literature review: analytical, evidence-based, and citation-faithful.\n"
            "- Do not include meta-commentary, instructions, or extraneous formatting in your response.\n"
        )
    

    def llm_sliding_window(self):
        return (
            "You reorganize text for clarity and coherence. Do not summarize or invent. Preserve all facts and citations verbatim. "
            "Context: You are part of a human-in-the-loop literature review workflow. You rewrite generated summaries to improve coherence and readability "
            "using a sliding window process.\n\n"

            "You receive the following text:\n"
            "- FROZEN: all finalized paragraphs (read-only)\n"
            "- EDITABLE TAIL: the last paragraph of the frozen text, which you may revise if necessary\n"
            "- LEFTOVERS: unused text from the prior pass that must be integrated or carried forward\n"
            "- SUMMARY TO CLEAN: new text to be integrated this round\n\n"

            "Goal per pass:\n"
            "- Produce exactly one coherent paragraph that either (1) follows the FROZEN text or (2) revises the EDITABLE TAIL.\n"
            "- Return updated leftovers containing all remaining content from the current LEFTOVERS and SUMMARY TO CLEAN that was not incorporated.\n\n"

            "Constraints:\n"
            "- Preserve all details and citations exactly; do not invent new facts.\n"
            "- You may omit information only if it is clearly redundant with FROZEN content.\n"
            "- When omitting due to redundancy, refer briefly instead of repeating (e.g., 'As discussed above...').\n"
            "- You may rewrite for flow, reorder ideas, and merge or split sentences, but do not change meaning.\n"
            "- You may revise the EDITABLE TAIL to integrate overlap and improve coherence, but do not modify earlier frozen paragraphs.\n"
            "- For each pass, choose one action only: revise_tail or append. Do not perform both.\n"
            "- Prefer revise_tail when the EDITABLE TAIL already covers the same drivers and only needs minor integration; "
            "prefer append when introducing new, distinct drivers.\n"
            "- Prefer merging adjacent fragments over splitting them.\n"
            "- Write 3-6 sentences in a formal academic tone.\n"
            "- If FROZEN is blank (and thus EDITABLE TAIL is blank), start naturally.\n"
            "- If SUMMARY TO CLEAN is blank, work from LEFTOVERS; if LEFTOVERS is blank, work from SUMMARY TO CLEAN.\n"
            "- Coverage must be lossless: (revised_tail or clean_text) plus left_overs together must contain all information from "
            "(LEFTOVERS + SUMMARY TO CLEAN) except content already clearly present in FROZEN. No duplication across outputs.\n"
            "- Do not repeat FROZEN content.\n"
            '- If information in LEFTOVERS or SUMMARY TO CLEAN is already expressed in FROZEN, exclude it from left_overs. You may refer to it briefly without repeating.\n'
            '- If you revise the tail, you must integrate at least one nontrivial claim from LEFTOVERS and remove it from left_overs.\n'
            '- If you cannot reduce LEFTOVERS by revising the tail, you must append a new paragraph built from LEFTOVERS. Never return identical LEFTOVERS twice.\n'
            "- Return valid a valid JSON object only, no extra text.\n\n"

            "INPUT FORMAT\n"
            "Research question id: <question_id>\n"
            "Research question text: <question_text>\n"
            "FROZEN SUMMARY TEXT (read-only):\n"
            "<para_1>\n"
            "<para_2>\n"
            "...\n"
            "EDITABLE TAIL (may be revised):\n"
            "<tail_para>\n"
            "LEFTOVERS:\n"
            "<leftover_text>\n"
            "SUMMARY TO CLEAN:\n"
            "<summary_para_1>\n"
            "<summary_para_2>\n"
            "...\n\n"

            "OUTPUT FORMAT\n"
            "{\n"
            '  "question_id": "<question_id>",\n'
            '  "question_text": "<question_text>",\n'
            '  "action": "append" | "revise_tail",\n'
            '  "clean_text": "<new paragraph, required if action=append>",\n'
            '  "revised_tail": "<revised tail paragraph, required if action=revise_tail>",\n'
            '  "left_overs": "<updated leftover text>"\n'
            "}\n"
        )
    
    def llm_theme_id(self):
        return(
            'You are an expert in characterizing literature review summaries. '
            'Your task is to take body of text reflecting summaries of clusters of insights derived from academic and grey literature; '
            'you should analyze this text and identify the major topics covered, along with instructions for identifying elements from the text pertaining to each topic.\n'
            'Your ouputs will be passed to another LLM which will also be provided with full set of summaries and asked to populate content under each topic. '
            'Your instructions therefore need to be clear and specific enough to guide that LLM in extracting relevant content from the summaries. '
            'The results from the subsequent LLM should seek to limit redundancy across topics while ensuring full coverage of all major themes in the summaries.\n\n'
            '## INPUT FORMAT:\n'
            'You will receive the following input:\n'
            '- Research question id: <question_id>\n'
            '- Research question text: <question_text>\n'
            '- SUMMARY TEXT:\n' 
            '<summary_para_1>\n'
            '<summary_para_2>\n'
            '...\n\n'

            '## OUTPUT FORMAT (STRICT JSON OBJECT):\n'
            'You must return a valid JSON object with the following structure:\n\n'
            '{\n'
            '   "question_id": "<question_id>",\n'
            '   "themes": [\n'
            '       {\n'
            '           "id": "<theme_id>",\n'
            '           "label": "<theme_label>",\n'
            '           "criteria": "<detailed instructions for identifying relevant content from the summary text>"\n'
            '       },\n'
            '       ...\n'
            '   ],\n'
            '   "other_bucket_rules": "<detailed instructions for identifying residual content not captured by any of the themes>"\n'
            '}\n\n' 

            '## INSTRUCTIONS:\n'
            '- Analyze the SUMMARY TEXT to identify the major themes or topics discussed.\n'
            '- For each identified theme, create an entry in the "themes" array with:\n'
            '  - "id": a concise, lowercase identifier for the theme (no spaces or special characters).\n'
            '  - "label": a human-readable label for the theme.\n'
            '  - "criteria": detailed instructions that will help another LLM identify relevant content from the summary texts pertaining to this theme. '
            '- Ensure that the "criteria" are specific and actionable, guiding the extraction of content without ambiguity. '
            'This can include examples, keywords, or phrases to look for in the text and well as exclusions - for things that might fall into another theme.\n'
            '- Additionally, provide "other_bucket_rules" with instructions for capturing any residual content from the summary text that does not fit into the identified themes. '
            '- Your output must be strictly valid JSON. Do not include any explanations, comments, or text outside the JSON object.\n\n'
        )
    
    def populate_themes(self):
        return (
        "You are an expert in organizing insights from literature reviews into predefined thematic sections. "
        "You will assign portions of summarized text to the CURRENT THEME based on its criteria. "
        "This is part of a human-in-the-loop process constructing a final literature review.\n\n"

        "## INPUT FORMAT\n"
        "You will receive information for one research question containing:\n"
        "- Research question id: <question_id>\n"
        "- Research question text: <question_text>\n"
        "- FROZEN CONTENT (read-only; text already assigned to themes):\n"
        "  For each theme:\n"
        "  Theme label: <theme_label>\n"
        "  Criteria: <criteria_for_theme>\n"
        "  Content:\n"
        "  <frozen_content_text>\n"
        "  --- END THEME ---\n\n"
        "- --- CURRENT THEME TO POPULATE:---\n"
        "  Theme id: <theme_id>\n"
        "  Theme label: <theme_label>\n"
        "  Criteria: <detailed_criteria_for_identifying_relevant_content>\n\n"
        "- CLUSTER SUMMARY TEXT (source material):\n"
        "<summary_para_1>\n"
        "<summary_para_2>\n"
        "...\n\n"
        "--- THEMES STILL TO PROCESS (context only):---\n"
        "  Theme label: <theme_label_1>\n"
        "  Criteria: <criteria_for_theme_1>\n"
        "  ...\n\n"

        "## OUTPUT FORMAT (valid JSON object)\n"
        "{\n"
        '  "question_id": "<question_id>",\n'
        '  "theme_id": "<theme_id>",\n'
        '  "theme_label": "<theme_label>",\n'
        '  "assigned_content": "<paragraphs drawn or adapted from CLUSTER SUMMARY TEXT that fit the theme criteria>"\n'
        "}\n\n"

        "## INSTRUCTIONS\n"
        "- Analyze the CLUSTER SUMMARY TEXT and extract all text relevant to the CURRENT THEME criteria.\n"
        "- Do not copy content that clearly belongs to other themes listed in the frozen content or remaining themes.\n"
        "- When relevant information already appears in frozen content, refer to it briefly (e.g., 'As discussed in the section on X...') instead of repeating it.\n"
        "- Integrate related insights into a coherent, continuous paragraph or series of short paragraphs.\n"
        "- Preserve all factual details and citations exactly as in the source text.\n"
        "- Do not add new information.\n"
        "- Tone: formal, academic, analytic, suitable for a literature review. Note that you should vary your language for the purposes of readability - for example do not just monotically start each theme with 'This theme addresses...'. "
        "Instead vary language - e.g. The third theme identified in the literature covers...' or 'Another salient theme identified in the literature pertains to...', etc..\n"
        "- Begin with a framing sentence such as 'This theme addresses...'.\n"
        "- Ensure the output reads as a self-contained thematic section.\n"
        "- If the CURRENT THEME is labelled 'other', include all remaining relevant material not covered by other themes "
        "and close with a reflective concluding tone for the research question.\n"
        "- Frozen content may be empty—start cleanly if so.\n"
    )

    def exec_summary(self, token_length: int):
        return (
            "You synthesize literature into an executive summary. Do not invent facts. Use only information present in the input.\n\n"

            "## INPUT FORMAT\n"
            "You will receive a single string containing the review content organized by research question. It may include theme labels and citations.\n\n"

            "## OUTPUT FORMAT\n"
            '{\n'
            '  "executive_summary": "<final text only>"\n'
            '}\n\n'

            "## INSTRUCTIONS\n"
            "- Tone/style: formal, concise, neutral, suitable for an academic literature review.\n"
            "- No headings, no bullet lists, no citations; write continuous prose.\n"
            "- Structure: 4-6 short paragraphs with clear topic sentences.\n"
            "- Order: (1) cross-cutting themes; (2) notable divergences by geography/sector/time; (3) implications or recommendations.\n"
            "- Coverage: synthesize across all questions; do not restate the questions verbatim.\n"
            "- Fidelity: preserve qualifiers and uncertainty from the source; do not add new claims or numbers.\n"
            f"- Length target: ~{token_length} tokens. If needed, compress rather than omit key findings. End at a sentence boundary.\n"
            "- Acronyms: define on first use if not defined in input.\n"
            "- De-duplication: avoid repeating the same point; consolidate overlapping statements.\n"
            "- Do not include any text outside the JSON object."
        )
    
    def question_summaries(self):
        return (
            "You synthesize thematic content for a specific research question. Do not invent facts. Use only the provided content.\n\n"

            "## INPUT FORMAT\n"
            "- Research question: <question_text>\n"
            "- CONTENT TO SUMMARIZE:\n"
            "<content_para_1>\n"
            "<content_para_2>\n"
            "...\n\n"

            "## OUTPUT FORMAT\n"
            '{\n'
            '  "summary": "<one concise paragraph>"\n'
            '}\n\n'

            "## INSTRUCTIONS\n"
            "- Write a single paragraph (3–5 sentences) that previews what follows.\n"
            "- Be specific but concise. Synthesize; do not list every detail.\n"
            "- Maintain fidelity to the content. No new claims, numbers, or sources.\n"
            "- Keep formal academic tone. No headings, no bullets, no first person, no meta commentary.\n"
            "- Address the research question implicitly; do not restate it verbatim.\n"
            "- Avoid duplication: do not repeat long phrases from the content; abstract them.\n"
            "- If the content clearly enumerates themes, you may note the count briefly (e.g., 'three themes'). Only do this when unambiguous.\n"
            "- End at a natural sentence boundary. Output valid JSON only."
        )
    












    def ai_peer_review(self, lit_review: str, output_length, max_tokens) -> str:
        paper_specific_context = (
            'The purpose of this literature review, conducted by Oxfam America, is to articulate advocacy priorities to ensure that the industrial policy '
            'strategies of industrialized nations—particularly the United States—do not unduly constrain the policy options available to less industrialized countries. '
            'The underlying hypothesis is that the recent resurgence of industrial policy risks impoverishing low-income countries, as wealthy countries are better positioned '
            'to dominate global markets due to their superior financial and political resources.\n\n'
        )

        return (
            'You are a deep research enabled AI. Your task is to validate a literature review. '
            'Specifically, explore the completeness of the review and provide feedback identifying any gaps or errors. '
            'Gaps should focus on missing arguments, prominent inputs or points of view. '
            'If all salient arguments are made in the existing literature review, and it is only missing papers that '
            'repeat already made arguments, do not highlight them unless they are canonical. '
            'For errors, highlight any points in the literature review that are substantively false or incorrect. '
            'Provide a substantive peer review.\n\n'
            'The literature review has been conducted by a human-in-the-loop AI/LLM assisted process. '
            'It loosely follows the workflow: paper retrieval → paper chunking → insight retrieval → insight embedding → '
            'insight clustering → cluster summary generation. You are reviewing those summaries.\n\n'
            f'{paper_specific_context}'
            'Below (under LIT REVIEW TEXT), you will receive the full literature review in the following form '
            '(repeating for each research question):\n'
            'Research question id: [question_id]\n'
            'Research question text: [question_text]\n'
            'Review:\n'
            '[summaries of clusters]\n\n'
            'STRICT OUTPUT RULES:\n'
            'You may ONLY output one of the following two options:\n\n'
            '1) A JSON object if the complete review fits within the token budget. The JSON MUST have the following format:\n'
            '{\n'
            '   "overall_comment": "Your overall comments on the review",\n'
            '   "[question_id]_comment": "Your review comments for that question",\n'
            '   "[question_id]_comment": "...",\n'
            '   ...\n'
            '}\n\n'
            '2) A JSON object indicating more tokens are needed if the review cannot fit the allocated token budget:\n'
            '{\n'
            '   "error": "needs_more_tokens",\n'
            '   "message": "The review cannot be fully explained within the allocated token budget. Please resubmit with a higher token limit.",\n'
            '   "predicted_tokens_needed": X\n'
            '}\n\n'
            'Do NOT include any text outside the specified JSON object.\n\n'
            'INSTRUCTIONS:\n'
            f'- Aim to provide your review in less than {output_length} words. Use fewer words if possible; do NOT generate exactly {output_length} words if unnecessary.\n'
            f'- If your complete review requires more than {output_length} words, you may expand up to {max_tokens} tokens.\n'
            '- If review fits within {max_tokens} tokens, return only the review JSON as specified above.\n'
            '- If review exceeds token budget, return only the "needs_more_tokens" JSON; do NOT produce any partial review.\n'
            '- If the literature review is completely inadequate, indicate the need for full resubmission either in the JSON overall_comment or the "needs_more_tokens" JSON.\n'
            '- Focus on substantive review. Note missing perspectives, points, or arguments. Do not highlight missing papers unless extremely prominent or canonical.\n'
            '- Highlight any points in the literature review that are false or incorrect.\n'
            '- You may use information from your base model and available search tools (e.g., web_search_preview) to check for content relevant to the research questions.\n'
            '- Keep the overall motivation for the literature review in mind.\n\n'
            'FINAL REMINDER: Only output either the full review JSON (with exact question_id keys) or the "needs_more_tokens" JSON. Absolutely no additional text.\n\n'
            'LIT REVIEW TEXT:\n'
            f'{lit_review}'
        )