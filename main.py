

import json
import pickle
from turtle import pd
import hdbscan



from lit_review_machine import processing, outputs, prompts, utils
from dotenv import load_dotenv
import os
from openai import OpenAI
from unpywall import Unpywall
import os




#---------
import importlib
from lit_review_machine import processing, outputs, prompts, utils

importlib.reload(prompts)
importlib.reload(outputs)
importlib.reload(processing)
importlib.reload(utils)
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
#academic_lit.state.write_to_csv(os.path.join(os.getcwd(), "outputs"))
papers_state = outputs.QuestionState.from_csv(os.path.join(os.getcwd(), "outputs"), encoding = 'cp1252')
#NOW I KEEP TESTING BY PUSHING THIS INTO THE NEXT CLASS

# Instantiate the DOI class witht the question state that would have come from acacdemic lit
doi = processing.DOI(state = papers_state)
# Get the DOI for all papers - needed to try and identify open source versions
doi.get_doi()
# We want to use unpaywall to get the download links so we create the environment variable here
os.environ["UNPAYWALL_EMAIL"] ="james.morrissey@oxfam.org"
doi.get_download_link()

# Instantiate grey literature class with the state from doi
grey_literature = processing.GreyLiterature(state=doi.state, llm_client=llm_client)
# Undertake the deep research search for grey literature
grey_literature.get_grey_lit()

# Instantiate the Literature class to handle duplicates
literature = processing.Literature(grey_literature.state)
# Drop the exact duplicates - will return a dataframe for each question with unique papers. 
literature.drop_exact_duplicates()
# Identify all the approximate matched for manual review
literature.get_fuzzy_matches()
# Now we edit the csv to drop all approximate matches, and update the state with that csv
literature.update_state()

# Instantiate the AiLiteratureCheck class
ai_literature = processing.AiLiteratureCheck(state = literature.state, llm_client=llm_client)
# Undertake the ai_literature check
ai_literature.ai_literature_check()

# To make sure we get all the papers we want from the LLM we don't encourage it to address 
# duplicates - in case this stopped it duplicating papers across questions which is what we want. As such
# more duplicates can be introduced. Lets hanlde them again with the literature class:
# First we instantiate
de_dup_lit = processing.Literature(state = ai_literature.state)

# Then dedup as above:
de_dup_lit.drop_exact_duplicates()
de_dup_lit.get_fuzzy_matches()
de_dup_lit.update_state()

# Proceed with downloading the files
# Instantiate the downloader
downloads = processing.DownloadManager(state=de_dup_lit.state)
# After manually undertaking the downloads
downloads.update()

# Now we look at the papers that are a priority. 
# Initialize the class
paper_triage = processing.PaperAttainmentTriage(state = downloads.state, 
                                                client=llm_client)
# First we generate the embeddings
paper_triage.generate_embeddings()
# THen we calculate the cosine similarities between them
paper_triage.triage_papers()
# Update the state after making any further changes to the downloads
paper_triage.update_state()

# Now we move to ingest all the papers - initialize the ingestor class
ingestor = processing.Ingestor(state=paper_triage.state, 
                               llm_client=llm_client, 
                               ai_model="gpt-4o")
# ingest the papers
ingestor.ingest_papers()
# Get the metatdata for any papers that were added by the user - i.e. the metadat was not pulled from the search run as part of the pipeline
ingestor.update_metadata()
# Chink the ingested papers
ingestor.chunk_papers()

# Now we get the insights - initialize the insights class:
insights = processing.Insights(state=ingestor.state, 
                               llm_client=llm_client, 
                               ai_model="gpt-4o")

# insights = processing.Insights(state=outputs.QuestionState.from_parquet(os.path.join(processing.STATE_SAVE_LOCATION, "09_ingestor")), 
#                               llm_client=llm_client, 
#                               ai_model="gpt-4o")

# Now we get the insights
insights.get_chunk_insights()
# If the llm fails for some reason, we can recover using the following:
# insights.recover_chunk_insights_generation() - this will be offered automatically if you call get_chunk_insights again and it had previously partially completed
# Get the meta insights - from the full text (i.e. overcome the idea that insights from chunks might miss larger context)
insights.get_meta_insights()

# Now we move to clustering 
# Initialize clustering class
clusters = processing.Clustering(state=insights.state, 
                                   llm_client=llm_client, 
                                   embedding_model="text-embedding-3-small")


clusters = processing.Clustering(state=outputs.QuestionState.from_parquet(os.path.join(processing.STATE_SAVE_LOCATION, "10_insights")),
                                 llm_client=llm_client, 
                                 embedding_model="text-embedding-3-small")

clusters.state.chunks.drop(columns=["insight"], inplace=True)
clusters.state.full_text.drop(columns=["pages", "chunks"], inplace=True)
clusters.state.insights["insight"] = clusters.state.insights["insight"].apply(
    lambda x: x[0] if isinstance(x, list) and len(x) == 1 else ("" if isinstance(x, list) and len(x) == 0 else x)
)

clusters.valid_embeddings_df["insight"] = clusters.state.insights["insight"].apply(
    lambda x: x[0] if isinstance(x, list) and len(x) == 1 else ("" if isinstance(x, list) and len(x) == 0 else x)
) ######## HAVE TO GET RID OF THE REF TO INDEX[0] IN THE VALID EMBEDDINGS CREATION

# Create the embeddings of the insights
clusters.embed_insights()
# I can view the embeddings here: 
# clusters.insight_embeddings_array

# Tune the UMAP parameters
# This will take some time as its reducing the dimensions and calculating silhouette scores for many combinations of UMAP params
clusters.tune_umap_params(rq_exclude=["question_3"])
# View the results sorted by silhouette score and save to html - helpful if you want to see diminshing returns
clusters.umap_param_tuning_results.to_html("umap_param_tuning_results.html")
# Reduce the dimensions of the embeddings using the params that make most sense above
clusters.reduce_dimensions(n_neighbors=75, min_dist=0.0, n_components=10, metric="euclidean")
# You can inspect the reduced embeddings here:
# clusters.reduced_insight_embeddings_array
# You can also produce the silhoette score for your reduced embeddings - note if you don't exclude_rq and you did during tuning you will see a different result
clusters.calc_silhouette()

# Now we move to clustering with HDBSCAN
# First we can tune the parameters - depending on how many combinations you try this may take a while
clusters.tune_hdbscan_params(min_cluster_sizes=[5, 10, 15, 20],
                             metrics=["euclidean", "manhattan"],
                             cluster_selection_methods=["eom", "leaf"])

# Useful to save the results to html to inspect - you want low db score and few outliers
clusters.hdbscan_tuning_results.to_html("hdbscan_tuning_results.html")

# Finally we generate the clusters based on the tuning results
# Here we pass the hdbscan parameters that we want for each question - as a dictionary
clusters.generate_clusters(clustering_param_dict={
    "question_0": {"min_cluster_size": 15, "metric": "euclidean", "cluster_selection_method": "eom"},
    "question_1": {"min_cluster_size": 20, "metric": "euclidean", "cluster_selection_method": "eom"},
    "question_2": {"min_cluster_size": 10, "metric": "euclidean", "cluster_selection_method": "leaf"},
    "question_3": {"min_cluster_size": 15, "metric": "manhattan", "cluster_selection_method": "eom"},
    "question_4": {"min_cluster_size": 10, "metric": "euclidean", "cluster_selection_method": "eom"}
    }) 
# Look at the cumulative proportions by cluster to decide how many groups we want - useful to write to html
clusters.cum_prop_cluster.to_html("cum_prop_cluster.html")
# Now clean the clusters to the number of clusters you want - you can inspect the cumulative proportions to help decide this
clusters.clean_clusters()

# Finally move to generating the summaries

summarize = processing.Summarize(state=clusters.state, 
                                 llm_client=llm_client, 
                                 ai_model="gpt-4o", 
                                 paper_output_length=8000)

summarize = processing.Summarize(state=outputs.QuestionState.load(os.path.join(processing.STATE_SAVE_LOCATION, "11_clusters")),
                                 llm_client=llm_client,
                                 ai_model="gpt-4o",
                                 paper_output_length=8000)

# Undertake the summarization of the clusters
summarize.summarize()
# Examine the summaries dataframe if ouAccess the df of summaries
summarize.summaries
# Identify themes across the summaries
summarize.identify_themes()
# Examine the themes
summarize.summary_themes
# Now populate the themes with the summaries
summarize.populate_themes()
 # inspect the populated themes
summarize.populated_themes

# Now we can initalize the Summaries class by passing either one of the summary_dfs (either with themese or without) to the initilaizer. 
# This class is used to manipulate and out the summaries in different formats
summary = outputs.Summaries(summaries = summarize.populated_themes, 
                            llm_client=llm_client, 
                            ai_model="gpt-4o")

# Generate the executive summary for the document
summary.gen_executive_summary()
# Generate the question summaries
summary.gen_question_summaries()
# Generate the final document
summary.summary_to_doc(paper_title="Industrial Policy: What a resurgence means for Oxfam")



#----------------

 summary.summaries = summary.summaries.sort_values(by=["question_id", "cluster"])

        output_string = ""
        for qid in summary.summaries["question_id"].unique():
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



# Convert to DataFrame
with open(os.path.join('C:\\Users\\jmorrissey\\Documents\\python_projects\\lit_review_machine\\data\\pickles', 'meta_insights.pkl'), "rb") as f:
    meta_insights = pickle.load(f)

meta_insights_df: pd.DataFrame = pd.DataFrame(meta_insights)

# We want to eventually concat meta insights with insights, so we get all the columns neccesary to make meta insights compatible with insights
# Make a temp copy of state.insights to drop unneccesary columns and then to merge with meta insights
# Make copy
temp_insights = deepcopy(insights.state.insights)

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
meta_insights_df["insight"] = meta_insights_df["insight"].apply(processing.Insights.ensure_list)
# Explode meta insights so each insight is its own row
meta_insights_df = meta_insights_df.explode("insight")
#.explode converts to str so i convert back to list for consistency
meta_insights_df["insight"] = meta_insights_df["insight"].apply(lambda x: [] if pd.isna(x) else ([x] if isinstance(x, str) else x))
# Create chunk_id column to identify meta insights
meta_insights_df["chunk_id"] = [f"meta_insight_{pid}" for pid in meta_insights_df["paper_id"]]

# Concat new meta insights
insights.state.insights = pd.concat(
    [insights.state.insights, meta_insights_df], 
    ignore_index=True
)


# Add insight_id as i need this for joining in subsequent steps
insights.state.insights["insight_id"] = insights.state.insights.index.astype(str)

# Ensure chunk_id is string type - neccesary as earlier chunk ids were integers, now they have "meta insight_{paper_id}" strings too
insights.state.insights["chunk_id"] = insights.state.insights["chunk_id"].astype(str)

insights.state.save(os.path.join(processing.STATE_SAVE_LOCATION, "10_insights"))














dupes = clusters.state.insights.duplicated(subset=["question_id", "paper_id", "chunk_id"], keep=False)
print("Duplicates in state.insights:", clusters.state.insights[dupes])



clusterer = hdbscan.HDBSCAN(
            min_cluster_size=5,
            metric="euclidean",
            cluster_selection_method="eom"
        )

clusters.valid_embeddings_df["reduced_insight_embeddings"] = [row.tolist() for row in clusters.reduced_insight_embeddings_array]

clustered_dfs = []
for rq in clusters.valid_embeddings_df["question_id"].unique():
    print(f"Generating clusters for {rq}...")
    rq_df = clusters.valid_embeddings_df[clusters.valid_embeddings_df["question_id"] == rq].copy()
    embeddings_matrix = np.vstack(rq_df["reduced_insight_embeddings"].to_list())
    cluster_labels = clusterer.fit_predict(embeddings_matrix)
    cluster_probs = clusterer.probabilities_

    rq_df["cluster"] = cluster_labels
    rq_df["cluster_prob"] = cluster_probs
    clustered_dfs.append(rq_df)   

clustered_df = pd.concat(clustered_dfs)
self.state.insights = clusters.state.insights.merge(
    clustered_df[["question_id", "paper_id", "chunk_id", "cluster", "cluster_prob"]],
    on=["question_id", "paper_id", "chunk_id"],
    how="left"
)







with open('C:\\Users\\jmorrissey\\Documents\\python_projects\\lit_review_machine\\data\\pickles\\meta_insights.pkl', "rb") as f:
    x = pickle.load(f)

meta_insights_df = pd.DataFrame(x)

temp_insights = deepcopy(insights.state.insights)


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
meta_insights_df["insight"] = meta_insights_df["insight"].apply(processing.Insights.ensure_list)
# Explode meta insights so each insight is its own row
meta_insights_df = meta_insights_df.explode("insight")
meta_insights_df["insight"] = meta_insights_df["insight"].apply(lambda x: [] if pd.isna(x) else ([x] if isinstance(x, str) else x))
# Create chunk_id column to identify meta insights
meta_insights_df["chunk_id"] = [f"meta_insight_{pid}" for pid in meta_insights_df["paper_id"]]

# Concat new meta insights
insights.state.insights = pd.concat(
    [insights.state.insights, meta_insights_df], 
    ignore_index=True
)
# Ensure chunk_id is string type - neccesary as earlier chunk ids were integers, now they have "meta insight_{paper_id}" strings too
insights.state.insights["chunk_id"] = insights.state.insights["chunk_id"].astype(str)
insights.state.insights["insight"] = insights.state.insights["insight"].apply(lambda x: [] if pd.isna(x) else ([x] if isinstance(x, str) else x))

insights.state.save(os.path.join(processing.STATE_SAVE_LOCATION, "10_insights"))

return meta_insights_df



#----------------------




SEP = "||-||-||"

def fix_paper_author(x):
    if isinstance(x, list):
        return x
    if pd.isna(x) or x == "" or x == "NA":
        return []
    s = str(x)
    # If it's a string of single characters separated by SEP, join them
    if SEP in s:
        parts = s.split(SEP)
        # If all parts are single characters or dashes, it's a corrupted name
        if all(len(p) <= 2 for p in parts):
            return [''.join(parts).replace('-', '').replace('||', '').strip()]
        # Otherwise, it's a list of authors
        return [p.strip() for p in parts if p.strip()]
    # Otherwise, just wrap in a list
    return [s.strip()]

insights.state.insights["paper_author"] = insights.state.insights["paper_author"].apply(fix_paper_author)



#---------

clusters.state.insights.drop(columns=["cluster", "cluster_prob"]).drop_duplicates()
clusters.state.insights.shape

clusters.insight_embeddings_array.shape
clusters.valid_embeddings_df.shape

# Use a unique separator unlikely to appear in your data
SEP = "||-||-||"

df = qs.insights.copy()

# Convert lists to strings for deduplication
for col in ["insight", "paper_author"]:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: SEP.join(x) if isinstance(x, list) else str(x))

df.drop(columns=["chunk_id", "insight"], inplace=True)

# Drop duplicates
df = df.drop_duplicates()

# Convert back to lists
for col in ["insight", "paper_author"]:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: x.split(SEP) if isinstance(x, str) and x else [])

#qs.insights = df

qs_again = outputs.QuestionState(insights= df, 
                                 full_text=qs.full_text,
                                 chunks=qs.chunks.drop(columns=["insight"]).drop_duplicates()
                                 )

qs_again.save(os.path.join(processing.STATE_SAVE_LOCATION, "09_ingestor"))

qs = outputs.QuestionState.from_parquet(os.path.join(processing.STATE_SAVE_LOCATION, "09_ingestor"))

qs.save(os.path.join(processing.STATE_SAVE_LOCATION, "09_ingestor"))

# Debugging get_meta_insights
insights = processing.Insights(state=outputs.QuestionState.from_parquet(processing.STATE_FILE_LOCATION),
                               llm_client=llm_client,
                               ai_model="gpt-4o")

qs = outputs.QuestionState.from_parquet(processing.STATE_FILE_LOCATION)

qs_clean = qs.insights.drop(columns=["chunk_id", "insight"]).drop_duplicates()

qs.insights["chunk_id"].str.contains("meta_insight").sum()
qs.full_text
qs.chunks





clusters.valid_embeddings_df




with open('C:\\Users\\jmorrissey\\Documents\\python_projects\\lit_review_machine\\data\\pickles\\meta_insights.pkl', "rb") as f:
    x = pickle.load(f)

meta_insights_df = pd.DataFrame(x)

temp_insights = deepcopy(insights.state.insights)
# Drop columns that will duplicate or are unneccesary
for col in ["chunk_id", "insight"]:
    if col in temp_insights.columns:
        temp_insights.drop(columns=[col], inplace=True)

temp_insights["paper_author"] = temp_insights["paper_author"].apply(lambda x: "||-||-||".join(x))

# Drop duplicates so we have one row per (paper_id, question_id)
temp_insights = temp_insights.drop_duplicates()
temp_insights["paper_author"] = temp_insights["paper_author"].apply(lambda x: x.split("||-||-||"))




meta_insights_df = meta_insights_df.merge(
    temp_insights, how="left", on=["paper_id", "question_id"])


meta_insights_df["insight"] = meta_insights_df["insight"].apply(self.ensure_list)

meta_insights_df = meta_insights_df.explode("insight")
        # Create chunk_id column to identify meta insights
meta_insights_df["chunk_id"] = [f"meta_insight_{pid}" for pid in meta_insights_df["paper_id"]]
meta_insights_df["chunk_id"] = meta_insights_df["chunk_id"].astype(str)

        # Append new meta insights
insights.state.insights = pd.concat(
    [insights.state.insights, meta_insights_df], 
    ignore_index=True
)

insights.state.insights["chunk_id"] = insights.state.insights["chunk_id"].astype(str)

insights.state.save(processing.STATE_FILE_LOCATION)


rqs = [
    f"{row['question_id']}: {row['question_text']}"
    for _, row in insights.state.insights[["question_id", "question_text"]].iterrows()
    ]

paper_id = insights.state.insights["paper_id"].unique()[31]



paper_content: str = (
                insights.state.full_text
                .loc[insights.state.full_text["paper_id"] == paper_id, "full_text"]
                .iloc[0]
            )


paper_df = insights.state.insights[insights.state.insights["paper_id"] == paper_id]
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

current_rq: str = f"{paper_df['question_id'].iloc[0]}: {paper_df['question_text'].iloc[0]}"
other_rqs = list(set([rq for rq in rqs if rq != current_rq]))
insights_text: str = "\n".join([str(x) for x in paper_df["insight"].dropna().tolist() if len(x) > 0])

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


"\n".join([i[0] for i in paper_df["insight"] if len(i) > 0])


print("\n".join(insight_string for insight in paper_df["insight"] if isinstance(insight, list) for insight_string in insight))

.dropna().tolist() if len(i) > 0



with open('C:\\Users\\jmorrissey\\Documents\\python_projects\\lit_review_machine\\data\\pickles\\chunk_insights.pkl', "rb") as f:
    x = pickle.load(f)
x_df = pd.DataFrame(x)


insights.state.insights.drop(columns=["pages"], inplace=True)

insights.state.save()

print(insights.state.insights["paper_author"].apply(type).value_counts())
print(insights.state.insights["insight"].apply(type).value_counts())
print(insights.state.full_text["pages"].apply(type).value_counts())
print(insights.state.full_text["chunks"].apply(type).value_counts())
print(insights.state.chunks["insight"].apply(type).value_counts())


def is_list_of_lists(val):
    return isinstance(val, list) and any(isinstance(i, list) for i in val)

print(insights.state.chunks["insight"].apply(is_list_of_lists).value_counts())
print(insights.state.insights["paper_author"].apply(is_list_of_lists).value_counts())
print(insights.state.insights["insight"].apply(is_list_of_lists).value_counts())
print(insights.state.full_text["pages"].apply(is_list_of_lists).value_counts())
print(insights.state.full_text["chunks"].apply(is_list_of_lists).value_counts())



for i, d in enumerate(x):
    d["question_id"] = insights.state.chunks.iloc[i]["question_id"]
    d["paper_id"] = insights.state.chunks.iloc[i]["paper_id"]

with open('C:\\Users\\jmorrissey\\Documents\\python_projects\\lit_review_machine\\data\\pickles\\chunk_insights.pkl', "wb") as f:
    pickle.dump(x, f)

insights.recover_chunk_insights_generation()


for d in x:
    if "insight" not in d:
        print(d)

#Check there are insights for some chunks
len([d["insight"] for d in x if d["insight"] != []])

d[""]

#Check x will write to df
x_df = pd.DataFrame(x)

# clean up after previous runs
for col in ["chunk_id", "insight"]:
    if col in insights.state.insights.columns:
        insights.state.insights.drop(columns=[col], inplace=True)

for col in ["insight_x", "insight_y"]:
    if col in insights.state.chunks.columns:
        insights.state.chunks.drop(columns=[col], inplace=True)

insights.state.chunks.shape
#Check if x_df will merge with insights.state.chunks
x_df_merged_chunks = insights.state.chunks.merge(x_df, on=["question_id", "paper_id", "chunk_id"], how="left")
x_df_merged_chunks.shape

x_df_merged_chunks_insights = insights.state.insights.merge(x_df_merged_chunks, on=["question_id", "paper_id"], how="left")
x_df_merged_chunks_insights[pd.isna(x_df_merged_chunks_insights["chunk_id"])]

x_df_merged_chunks_insights



insights = processing.Insights(state=outputs.QuestionState.from_parquet(processing.STATE_FILE_LOCATION, new=False),
                               llm_client=llm_client,
                               ai_model="gpt-4o")

insights.state.insights.drop(columns = ["pages"], inplace=True)

y.state.insights.drop(columns=["chunk_text"], inplace=True)
y.state.save(save_location=processing.STATE_FILE_LOCATION)
state_df_dict = {}
files = os.listdir(processing.STATE_FILE_LOCATION)
for file in files:
    full_path = os.path.join(processing.STATE_FILE_LOCATION, file)
    df = pd.read_parquet(full_path)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")] #Remove unnamed cols
    state_df_dict[Path(file).stem] = df


qs = outputs.QuestionState(
    insights=state_df_dict["insights"],
    full_text=state_df_dict.get("full_text", None),
    chunks=state_df_dict.get("chunks", None)
)

print(qs.insights["paper_author"].apply(type).value_counts())
print(qs.full_text["pages"].apply(type).value_counts())
print(qs.full_text["chunks"].apply(type).value_counts())

qs.normalize_list_columns(["paper_author", "insight", "chunks", "pages"])

y = pd.read_parquet(os.path.join(processing.STATE_FILE_LOCATION, "chunks.parquet")).loc[:, ~df.columns.str.contains("^Unnamed")]

outputs.QuestionState.from_parquet(processing.STATE_FILE_LOCATION)



first_chunk = insights.state.chunks.iloc[0]
chunk_text = first_chunk["chunk_text"]
question_id = first_chunk["question_id"]

# Get the corresponding question_text from the canonical mapping
question_text = insights.state.insights.loc[
    insights.state.insights["question_id"] == question_id, "question_text"
].dropna().iloc[0]

# Build the prompt using your Prompts class
sys_prompt = prompts.Prompts().gen_chunk_insights()
user_prompt = (
    f"CURRENT RESEARCH QUESTION:\n{question_text}\n\n"
    f"TEXT CHUNK:\n{chunk_text}\n"
)

# Call the LLM directly and print the raw response
response = llm_client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    ], 
    response_format={"type": "json_object"}
    )

import json
print("RAW LLM RESPONSE:", response.choices[0].message.content)
x = json.loads(response.choices[0].message.content)

ingestor.state = processing.QuestionState.from_parquet(processing.STATE_FILE_LOCATION)

with open('C:\\Users\\jmorrissey\\Documents\\python_projects\\lit_review_machine\\data\\docs\\question_1\\grey_lit_4.html', "r", encoding="utf-8") as f:
    html_content = f.read()

clean_html = processing.Ingestor._html_cleaner(html_content)
html_chunks = processing.Ingestor._html_chunker(clean_html)
out = processing.Ingestor._llm_parse_html(html_list = html_chunks, prompt=prompts.Prompts().extract_main_html_content())

paper_triage = processing.PaperAttainmentTriage(state = processing.QuestionState.from_csv(os.path.join(os.getcwd(), "data")), 
                                                client=llm_client) 

os.getcwd()
# HERE
downloads.state.save(processing.STATE_FILE_LOCATION)
downloads = processing.DownloadManager(state = outputs.QuestionState.from_parquet(processing.STATE_FILE_LOCATION))

outputs.QuestionState.from_csv(filepath = "C:/Users/jmorrissey/Documents/python_projects/lit_review_machine/data/docs")
os.listdir("C:/Users/jmorrissey/Documents/python_projects/lit_review_machine/data/docs")

grey_literature.state.insights["paper_id"]

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
