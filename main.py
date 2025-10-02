

import LitReview
from dotenv import load_dotenv
import os
import json
from openai import OpenAI
from unpywall.utils import UnpywallCredentials


#---------
import importlib
importlib.reload(LitReview)



#---------

load_dotenv() 
OPEN_API_KEY = os.getenv("OPENAI_API_KEY")

llm_client = OpenAI(api_key=OPEN_API_KEY)



questions = [
    "What drivers account for the resurgence of industrial policy in both highly industrial and industrializing countries?", 
    "How have the definitions and approaches to industrial policy shifted from the post-World War II period to present day? And, more specifically, how have these shifts considered (if at all) inclusive and sustainable growth, and respect of human rights and gender equality?",
    "What challenges and constraints do less-industrialized countries face in realizing effective industrial policy?",
    "What key recommendations can Oxfam make to more industrialized countries so that their industrial policies do less harm to industrialized countries?",
    "What key recommendations can Oxfam make to rich countries to advance reforms among different transnational institutions, so as to increase the policy space available to less-industrialized countries, to a point that it is, at least, comparable with the policy space afforded to industrialized nations?"
]

prompts = LitReview.ScholarPrompt(questions=questions, 
                                  llm_client=llm_client, 
                                  num_prompts=1)



academic_lit = LitReview.AcademicLit(scholar_prompt_class=prompts, num_results=5)
academic_lit.get_papers()





literature.get_stub_pubs()
literature.stub_pub_list
literature.stub_pub_cleaner()
literature.get_pubs()

literature.clean_stub_pub

search_str = "The resurgence of industrial policies in the age of advanced manufacturing Mateus Labrunie 2020"
doi = LitReview.DOI.call_alex(search_str)


from semanticscholar import SemanticScholar
# Initialize the Semantic Scholar client
sch = SemanticScholar()
results = sch.search_paper(prompts.prompts[0]["Question1"][0])
for result in results:
    result.keys()
    result.title
    result.authors
    result.year
    result.openAccessPdf
    result.code
