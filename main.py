

from lit_review_machine import processing, outputs, prompts
from dotenv import load_dotenv
import os
from openai import OpenAI



#---------
import importlib
from lit_review_machine import processing, outputs, prompts

importlib.reload(prompts)
importlib.reload(outputs)
importlib.reload(processing)
#---------

# Access env variables
load_dotenv()
# Securely load our open ai API key  
OPEN_API_KEY = os.getenv("OPENAI_API_KEY")
# Create the LLM client 
llm_client = OpenAI(api_key=OPEN_API_KEY)

# Input our questions that will drive the review
questions = [
    "What drivers account for the resurgence of industrial policy in both highly industrial and industrializing countries?", 
    "How have the definitions and approaches to industrial policy shifted from the post-World War II period to present day? And, more specifically, how have these shifts considered (if at all) inclusive and sustainable growth, and respect of human rights and gender equality?",
    "What challenges and constraints do less-industrialized countries face in realizing effective industrial policy?",
    "What key recommendations can Oxfam make to more industrialized countries so that their industrial policies do less harm to industrialized countries?",
    "What key recommendations can Oxfam make to rich countries to advance reforms among different transnational institutions, so as to increase the policy space available to less-industrialized countries, to a point that it is, at least, comparable with the policy space afforded to industrialized nations?"
]

#############
# Get Search terms
#############

# Initialize the ScholarSearchString class that we will use to generate search terms
search_terms = processing.ScholarSearchString(questions=questions, 
                                              llm_client=llm_client,
                                              num_prompts=2)

# Call the LLM to populate the search terms
search_terms.searchstring_maker()

# Examine the results
search_terms.state.insights

#############
# Identify academic literature 
#############

# Initialize the AcademicLit class with the state from the ScholarSearchString class 
# You can optionally pass your semantic scholar API key via param 'semantic_scholar_api_key'. You will get a warning if you don't
academic_lit = processing.AcademicLit(state = search_terms.state)

# Get the papers from semantic search based on the search terms contained in state
# num_results (default 20) is the number of search results for each search term so total results for the project will be: num_prompts * num_results * number_of_questions (though note this may include duplicates - that will be handled later)

##### THIS IS NOT WORKING DUE TO THE NEED FOR AN API KEY
papers = academic_lit.get_papers(num_results = 5)
# SO I MANUALLY DOWNLOAD THE CSV, INSERT SOME PAPERS BASED ON THE SEARCHES AND NOW LOAD THE CSV TO KEEP TESTING
academic_lit.state.write_to_csv(os.path.join(os.getcwd(), "outputs"))
papers_state = outputs.QuestionState.load_from_csv(os.path.join(os.getcwd(), "outputs", "insights.csv"), encoding = 'cp1252')
#NOW I KEEP TESTING BY PUSHING THIS INTO THE NEXT CLASS

# Instantiate the DOI class witht the question state that would have come from acacdemic lit
doi = processing.DOI(state = papers_state)
# Get the DOI for all papers - needed to try and identify open source versions
doi.get_doi()

# Instantiate grey literature class witht the state from doi
grey_literature = processing.GreyLiterature(state=doi.state, llm_client=llm_client)
# Undertake the deep research search for grey literature
grey_literature.get_grey_lit()





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
