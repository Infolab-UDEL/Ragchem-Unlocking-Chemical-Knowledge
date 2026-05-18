# streamlit run rag_fronted.py
from datetime import datetime
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import io
import os
import csv
import re
import json
import copy
import requests
import numpy as np
import rank_gpt
import pandas as pd
import streamlit as st
from model_hadler import ModelHandler
from promptLibrary import PromptLibrary
from collections import defaultdict
import streamlit.components.v1 as components
from streamlit_extras.stateful_button import button
import uuid

import spacy
from spacy import displacy
from spacy_streamlit.util import get_html

SPACY_VERSION = tuple(map(int, spacy.__version__.split(".")))
NER_ATTRS = ["text", "label_", "start", "end", "start_char", "end_char"]


def cleanDocument(doc):
    # This regex pattern captures square brackets with numbers inside, allowing optional spaces and commas.
    pattern = r"\[\s*\d+\s*(,\s*\d+\s*)*\]"
    cleaned_text = re.sub(pattern, "", doc)
    return cleaned_text.strip()


def generate_snippet(text, max_length=100):
    if len(text) <= max_length:
        return text
    return text[:max_length].rsplit(" ", 1)[0] + "..."


# Function to extract the top N highest occurrences for each key
def get_top_n_occurrences(data, n=4):
    top_occurrences = {}
    for category, values in data.items():
        # Sort dictionary by values (occurrences) in descending order
        sorted_items = sorted(values.items(), key=lambda x: x[1], reverse=True)
        # Get the top N items
        top_occurrences[category] = sorted_items[:n]
    return top_occurrences


# Define a callback function
def on_selection_change():
    print(st.session_state.nTPBS)
    # st.write("The selection has changed!")
    # st.write(f"Selected value: {st.session_state.nTPBS}")


# Function to log questions into a CSV file
def log_question_to_csv(user_name):
    with open("questions_log.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now(), user_name])


# Function to log questions into a CSV file
def logs_feedbacks(data2Send):
    df = pd.json_normalize(data2Send)
    if not os.path.isfile("feedback_log.csv"):
        df.to_csv("feedback_log.csv", mode="w", index=False)
    else:
        df.to_csv("feedback_log.csv", mode="a", header=False, index=False)


def search(
    text: str, index_name: str, model_name: str, method_name, n_answers: int, checkboxStatus: bool, environment: str
):
    url = "http://127.0.0.1:5001/search"
    data = {
        "text": text,
        "index_name": index_name,
        "model_name": model_name,
        "method_name": method_name,
        "n_answers": n_answers,
        "environment_name": environment,
    }

    # Simple client-side cache to avoid repeated identical requests
    cache_key = f"search::{text}::{index_name}::{model_name}::{method_name}::{n_answers}::{environment}"
    cache = st.session_state.get("cache", {})
    if cache_key in cache:
        return text, cache[cache_key]

    try:
        response = requests.post(url, json=data, timeout=20)
        response.raise_for_status()
        result = response.json().get("result", [])
        cache[cache_key] = result
        st.session_state["cache"] = cache
        return text, result
    except requests.exceptions.HTTPError as e:
        st.error(f"Search failed: {e}")
        return text, []
    except requests.exceptions.RequestException as e:
        st.error(f"Search request error: {e}")
        return text, []


def search_dense(
    text: str, index_name: str, model_name: str, method_name, n_answers: int, checkboxStatus: bool, environment: str
):
    url = "http://127.0.0.1:5001/dense"
    data = {
        "text": text,
        "index_name": index_name,
        "model_name": model_name,
        "method_name": method_name,
        "n_answers": n_answers,
        "environment_name": environment,
    }

    cache_key = f"dense::{text}::{n_answers}::{environment}"
    cache = st.session_state.get("cache", {})
    if cache_key in cache:
        return text, cache[cache_key]

    try:
        response = requests.post(url, json=data, timeout=25)
        response.raise_for_status()
        result = response.json().get("result", [])
        cache[cache_key] = result
        st.session_state["cache"] = cache
        return text, result
    except requests.exceptions.HTTPError as e:
        st.error(f"Dense search failed: {e}")
        return text, []
    except requests.exceptions.RequestException as e:
        st.error(f"Dense search request error: {e}")
        return text, []


def display_answer(text):
    print(text)
    print("-----text----")
    st.markdown(
        f"""
        <div style='border: 1px solid #ccc; background-color: rgb(240,242,246); padding: 10px; max-height: 200px; overflow-y: auto; color: black;'>
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def extracted_entities(doc, ModelType):
    # texts = [t.strip() for t in doc.split('\n') if t]
    texts = doc
    data = {"text": texts, "model": ModelType}
    response = requests.post("http://infochain.ece.udel.edu:5000/ner", json=data)
    if response.status_code != 200:
        print(("unknown error during prediction"))
    else:
        response = response.json()
    return response


def darken_color(hex_color: str, factor: float = 0.4) -> str:
    # Convert hex to RGB
    hex_color = hex_color.lstrip("#")
    rgb = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

    # Apply the darkening factor
    darker_rgb = tuple(max(0, int(c * (1 - factor))) for c in rgb)

    # Convert back to hex
    return "#{:02x}{:02x}{:02x}".format(*darker_rgb)


def visualize_ner(
    doc: Union[spacy.tokens.Doc, List[Dict[str, str]]],
    *,
    labels: Sequence[str] = tuple(),
    attrs: List[str] = NER_ATTRS,
    show_table: bool = True,
    title: Optional[str] = "Named Entities",
    colors: Dict[str, str] = {},
    key: Optional[str] = None,
    manual: bool = False,
    displacy_options: Optional[Dict] = None,
    marco_option=True,
):
    """
    Visualizer for named entities.

    doc (Doc, List): The document to visualize.
    labels (list): The entity labels to visualize.
    attrs (list):  The attributes on the entity Span to be labeled. Attributes are displayed only when the show_table
    argument is True.
    show_table (bool): Flag signifying whether to show a table with accompanying entity attributes.
    title (str): The title displayed at the top of the NER visualization.
    colors (Dict): Dictionary of colors for the entity spans to visualize, with keys as labels and corresponding colors
    as the values. This argument will be deprecated soon. In future the colors arg need to be passed in the displacy_options arg
    with the key "colors".
    key (str): Key used for the streamlit component for selecting labels.
    manual (bool): Flag signifying whether the doc argument is a Doc object or a List of Dicts containing entity span
    information.
    displacy_options (Dict): Dictionary of options to be passed to the displacy render method for generating the HTML to be rendered.
      See https://spacy.io/api/top-level#displacy_options-ent.
    """
    if not displacy_options:
        displacy_options = dict()
    if colors:
        displacy_options["colors"] = colors

    if title:
        st.header(title)

    if manual:
        if show_table:
            st.warning("When the parameter 'manual' is set to True, the parameter 'show_table' must be set to False.")
        if not isinstance(doc, list):
            st.warning(
                "When the parameter 'manual' is set to True, the parameter 'doc' must be of type 'list', not 'spacy.tokens.Doc'."
            )
    else:
        labels = labels or [ent.label_ for ent in doc.ents]

    if not labels:
        st.warning("The parameter 'labels' should not be empty or None.")
    else:
        html = displacy.render(
            doc,
            style="ent",
            options=displacy_options,
            manual=manual,
        )
        style = "<style>mark.entity { display: inline-block }</style>"
        if marco_option:
            colH2.write(f"{style}{get_html(html)}", unsafe_allow_html=True)
        else:
            st.write(f"{style}{get_html(html)}", unsafe_allow_html=True)


def display_docs(result, ner_extraction, environment, expander_option):
    print("onix si si ")
    # respect a results limit (for load more behaviour)
    results_limit = st.session_state.get("results_limit", 10)
    scores_docs = ["" for x in range(results_limit)]
    for i, line in enumerate(result):
        if i >= results_limit:
            break

        if "error" in line.keys():
            st.warning(line["error"], icon="⚠️")
            st.divider()
            break
        else:
            # print("*********************************************")
            # print(line)
            if st.session_state.optionSorting == "keyword matching":
                score = line["relevance"]
                line = line["fields"]
                if environment == "production":
                    st.markdown(f"<strong style='font-size: 20px;'>{line['title']}</strong>", unsafe_allow_html=True)
                if environment == "test":
                    st.markdown(
                        f"<strong style='font-size: 20px;'>{line['title-full']}</strong>", unsafe_allow_html=True
                    )

            else:
                if environment == "production":
                    score = line["relevance"]

                if environment == "test":
                    score = line["similarity"]

                line = line["fields"]
                st.markdown(f"<strong style='font-size: 20px;'>{line['title']}</strong>", unsafe_allow_html=True)

            # st.markdown(f"<strong style='font-size: 12px;'>{line['title-full']}</strong>", unsafe_allow_html = True)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.caption(f"Link: http://doi.org/{line['doi']}")
            with col2:
                st.caption(f"Publisher: {line.get('publisher', '') }")
            with col3:
                st.caption(f"Rank: {i+1}")
            with col4:
                st.caption(f"Score: {score}")
            # with col5:
            #     st.caption(f"Relevance: {line['relevance']}")
            if st.session_state.optionSorting == "keyword matching":
                if expander_option == True:
                    with st.expander("See the full snippet of text"):
                        if environment == "production":
                            modified_full_text = line["content"].replace("<hi>", "<u><b>").replace("</hi>", "</b></u>")
                        if environment == "test":
                            modified_full_text = (
                                line["content-full"].replace("<hi>", "<u><b>").replace("</hi>", "</b></u>")
                            )

                        # modified_full_text_ner = line["content-full"].replace("<hi>", "").replace("</hi>", "")
                        visualize_ner(
                            [ner_extraction["pred"][i]],
                            labels=ner_extraction["labels"].keys(),
                            colors=ner_extraction["labels"],
                            title=None,
                            show_table=False,
                            manual=True,
                            marco_option=False,
                        )
                    # st.html(modified_full_text)
                    # ner (modified_full_text_ner , st.session_state.optionModel)
            else:
                if expander_option == True:
                    with st.expander("See the full snippet of text"):
                        modified_full_text = line["content"]
                        # st.html(modified_full_text)
                        visualize_ner(
                            [ner_extraction["pred"][i]],
                            labels=ner_extraction["labels"].keys(),
                            colors=ner_extraction["labels"],
                            title=None,
                            show_table=False,
                            manual=True,
                            marco_option=False,
                        )
            if expander_option == False:
                visualize_ner(
                    [ner_extraction["pred"][i]],
                    labels=ner_extraction["labels"].keys(),
                    colors=ner_extraction["labels"],
                    title=None,
                    show_table=False,
                    manual=True,
                    marco_option=False,
                )
            else:
                colText, colThumbs = st.columns([9, 1])
                # print(line)
                with colText:
                    # st.markdown(line["content"], unsafe_allow_html = False)
                    if st.session_state.optionSorting == "keyword matching":
                        modified_text = line["content"].replace("<hi>", "<u><b>").replace("</hi>", "</b></u>")
                        st.html(modified_text)
                    else:
                        snippet = generate_snippet(line["content"], max_length=700)
                        modified_text = snippet.replace("<hi>", "<u><b>").replace("</hi>", "</b></u>")
                        st.html(modified_text)

                    # st.html("<strong>Recycling</strong>")
                with colThumbs:
                    scores_docs[i] = st.feedback("thumbs", key=f"thumbs_{i}_{st.session_state.nTPBS}")
                    if scores_docs[i] is not None:
                        if scores_docs[i]:
                            st.markdown(f":green[{sentiment_mapping_text[scores_docs[i]]}]")
                        else:
                            st.markdown(f":red[{sentiment_mapping_text[scores_docs[i]]}]")

                        # Add copy and open DOI actions (small HTML component for clipboard access)
                        snippet = modified_text if 'modified_text' in locals() else line.get("content", "")
                        doi = line.get("doi", "")
                        key_id = f"doc_actions_{i}_{uuid.uuid4().hex}"
                        # JSON-encode strings to safely embed into JS handlers
                        js_snippet = json.dumps(snippet)
                        js_doi = json.dumps("https://doi.org/" + doi)
                        html_comp = f"""
                        <div style='display:flex; gap:8px; margin-top:6px;'>
                            <button onclick="navigator.clipboard.writeText({js_snippet})" style='padding:6px 8px;'>Copy snippet</button>
                            <button onclick="window.open({js_doi},'_blank')" style='padding:6px 8px;'>Open DOI</button>
                        </div>
                        """
                        components.html(html_comp, height=48, key=key_id)

            st.divider()

            # Create two columns: one for the content and the other for the submit button

    colX, colY = st.columns([3, 1])  # Adjust the ratio to control the space distribution

    with colY:
        # Submit button aligned to the right
        if st.button("Submit", use_container_width=True, key="submit"):
            # Save information (e.g., save feedback in session_state or a file)
            # st.success("Thanks for your help.")
            if "AnswerGenAI" in st.session_state:
                st.toast("Collecting Data")
                feedbacksJson = {
                    "timestamp": datetime.now(),
                    "query": input_query,
                    "generateAI": st.session_state.AnswerGenAI,
                    "scoreFeedback": st.session_state.scoreAns,
                    "feedbackText": user_comment,
                }
                for i, score in enumerate(scores_docs):
                    feedbacksJson[f"score_{i}"] = score
                time.sleep(0.9)
                st.toast("Sending Data")
                logs_feedbacks(feedbacksJson)
                time.sleep(0.9)
                st.toast("Thanks for you help!", icon="🎉")

            else:
                st.warning("Please provide feedback before submitting.")

    return result


def get_word_in_entities(multi_answ):
    for pred in multi_answ["pred"]:
        text = pred["text"]
        entities = pred["ents"]
        for ents in entities:
            word = text[ents["start"] : ents["end"]]
            ents["word"] = word
            multi_answ["labels"][f"{ents['label']}_Query"] = darken_color(multi_answ["labels"].get(ents["label"]))
            ents["label"] = f"{ents['label']}_Query"


def correlated_entities(multi_answ):
    organized_data = defaultdict(lambda: defaultdict(int))
    for pred in multi_answ["pred"]:
        text = pred["text"]
        entities = pred["ents"]
        pred["words"] = {}
        for ents in entities:
            word = text[ents["start"] : ents["end"]]
            label = ents["label"]
            organized_data[label][word] += 1
            ents["word"] = word

            if label in pred["words"]:
                pred["words"][label].append(word)
            else:
                pred["words"][label] = []
                pred["words"][label].append(word)
        #        # Convert defaultdict to normal dict
    entities_ex = {label: dict(words) for label, words in organized_data.items()}
    print("**CHECK")
    # print(multi_answ)
    # Get top 4 occurrences for each category
    top_k_per_category = get_top_n_occurrences(entities_ex, n=5)

    return top_k_per_category, entities_ex


def modify_color_entities_docs(Query_Entities, Docs_Entities):
    # Extract Query Entities (assumes there is only one "pred" list in Query_Entities)
    assert len(Query_Entities["pred"]) == 1, "Query should be just one"

    # Create a lookup set for Query Entity (word, label) pairs for quick matching
    print("deLTE")
    print(Query_Entities["pred"][0])
    query_entities_set = {
        (ents_Q["word"], ents_Q["label"].removesuffix("_Query")) for ents_Q in Query_Entities["pred"][0]["ents"]
    }
    print("## Query Entities Set")
    print(query_entities_set)
    for pred in Docs_Entities["pred"]:
        for ents in pred["ents"]:
            word = ents["word"]
            label = ents["label"]
            query_label = f"{label}_Query"

            # Check if the current (word, label) exists in the query set
            if (word, label) in query_entities_set:
                # Update label and darken color
                Docs_Entities["labels"][query_label] = darken_color(Docs_Entities["labels"].get(label))
                ents["label"] = query_label


################### RERANKING ############################


def rerank_docs_by_entities(user_query ,initial_docs, initial_extracted_entities, entities_selected_by_user , update_answer:bool =True):
    # entities_selected_by_user = {"Reactant":["glucose"], "Treatment":["calcination"]} #TEST
    print(entities_selected_by_user)
    my_list = [0] * len(initial_docs)
    print("=================")
    for index, extracted in enumerate(initial_extracted_entities["pred"]):
        print(extracted)
        for label in entities_selected_by_user.keys():
            list1 = extracted["words"].get(label, [])
            list2 = entities_selected_by_user[label]
            intersec = list(set(list1).intersection(list2))
            if len(intersec) > 0:
                my_list[index] = 1
                break
    # Get the indices sorted by the corresponding elements in list2
    sorted_indices = sorted(range(len(initial_extracted_entities["pred"])), key=lambda i: my_list[i], reverse=True)

    update_docs_after_ranking(sorted_indices, user_query, initial_docs, initial_extracted_entities , update_answer)


def reranking_docs_by_llm(user_query: str, initial_docs, initial_extracted_entities, entities, rank_gpt=False) -> None:

    # prompt  = PromptLibrary.get_prompt("reranking_llm", query, initial_docs)
    prompt = PromptLibrary.get_prompt("rank_gpt", query, initial_docs, debug=True)
    # handler = ModelHandler(model_name= "deepseek-r1:7b" , backend_type="ollama")
    handler = ModelHandler(model_name="gpt-4o-mini", backend_type="openai")

    response = handler.generate_answer(
        prompt,
        temperature=0,
    )

    # Find all numbers within square brackets
    numbers = re.findall(r"\[(\d+)\]", response)
    ranking_list = list(map(int, numbers))  # Convert strings to integers

    if not rank_gpt:
        ranking_list = list(map(lambda x: x - 1, ranking_list))
    update_docs_after_ranking(ranking_list, user_query, initial_docs, initial_extracted_entities)
    # print(f"Rank-LLM: {}")



def reranking_cohere_pre(user_query: str, documents: list , top_n) -> List:
    import cohere

    co = cohere.ClientV2(
        "bZfph2epGHhjVGrZAUyzznqf2Aa6yqOW16migPRa"
    )  # Get your free API key here: https://dashboard.cohere.com/api-keys
    results = co.rerank(model="rerank-v3.5", query=user_query, documents=documents, top_n=top_n)

    print("Cohore Rerank")

    ranking_list = [result.index for result in results.results]
    return ranking_list

def reranking_cohere(initial_extracted_entities, initial_docs, user_query: str, documents: list) -> None:
    num = len(documents)
    import cohere

    co = cohere.ClientV2(
        "bZfph2epGHhjVGrZAUyzznqf2Aa6yqOW16migPRa"
    )  # Get your free API key here: https://dashboard.cohere.com/api-keys
    results = co.rerank(model="rerank-v3.5", query=user_query, documents=documents, top_n=40)

    print("Cohore Rerank")

    ranking_list = [result.index for result in results.results]
    update_docs_after_ranking(ranking_list, user_query, initial_docs, initial_extracted_entities)


def update_docs_after_ranking(ranking_list, user_query, initial_docs, initial_extracted_entities, update_answer:bool = True):

    organized_list_docs = [initial_docs[indx] for indx in ranking_list]
    st.session_state.ranked_docs = organized_list_docs

    organized_list_entities = [initial_extracted_entities["pred"][indx - 1] for indx in ranking_list]
    st.session_state.ner_extraction["pred"] = organized_list_entities
    st.session_state.ranked_docs = organized_list_docs

    if on and update_answer == True:
        answer = generate_answer(user_query, organized_list_docs, environment)
        st.session_state.AnswerGenAI = answer       


def generate_answer(query: str, context, model: str = "openai" ,limit_num_docs:int =10 ):
    
    formatted_top_results = [
        {
            "title": article["fields"].get("title-full", article["fields"].get("title")),
            "description": article["fields"].get("content-full", article["fields"].get("content")),
            "doi": "https://doi.org/" + article["fields"]["doi"],
        }
        for article in context[0:limit_num_docs]
    ]
    libPrompt = PromptLibrary()
    prompt = libPrompt.get_prompt("generate_answer_openai", query, formatted_top_results , True)
    handler = ModelHandler(model_name="gpt-4o-mini", backend_type="openai")

    response = handler.generate_answer(
        prompt,
        temperature=0,
    )
    return response


###########################################################

default_session_state_values = {
    "nTPBS": 0,
    "scoreAns": -1,
    "results": "",
    "message": "",
    "AnswerGenAI": "",
    "feedbackText": "",
    "ON_STATE": False,
    "optionSorting": "semantic matching",
    "ner_extraction": "",
    "optionModel": "",
    "topk_entities": "",
    "colors_entities": {},
    "initial_results": "",
    "init_doc_ner_extraction": "",
    "query_ner_extraction": "",
    "user_question": "",
    "entities_in_query": {},
    "fulldocs": [],
    "ranked_docs":""
}

for key, value in default_session_state_values.items():
    if key not in st.session_state:
        st.session_state[key] = value


def change_name(name, modelNER):
    st.session_state["ON_STATE"] = True
    st.session_state["optionModel"] = modelNER


sentiment_mapping_text = ["No Relevant", "Relevant"]

# Initialize a key in session state for the message if it's not already set


##########################################################################################
#########                           FRONT-END                                   ##########
##########################################################################################

# Defaults values
index_name = "abstract"
method_name = "symmetric"
model_name = "bge"

n_texts_retrieval_step = 150
n_texts_reranking_step = 40

environment = "production"
expander_option = False
feedbackOption = False

st.set_page_config(page_title="Search Engine - Plastics", page_icon="💡", layout="wide")

with st.columns(3)[1]:
    st.image("SearchEngineLogo2.jpg")

# st.header("💡 Search Engine - MSEG 467/667: Plastics Sustainability", divider  ="blue" )
colH1, colH2, colH3 = st.columns([0.05, 0.9, 0.05])
colH2.subheader("Please enter your question or keywords below:")
input_query = colH2.text_input("", value="", autocomplete="on")
# with st.columns(3)[1]:
#    st.subheader("Please enter your question or keywords below:")
#    input_query = st.text_area("", value="")
# input_query = st.text_area("Please enter here the your question", value="")

if st.session_state.ON_STATE == True and input_query != "":
    # print("****NER***")
    # print(input_query)
    # print("****NER***2")
    # print(st.session_state.optionModel)
    query_entities = extracted_entities([input_query], st.session_state.optionModel)
    st.session_state.user_question = input_query
    st.session_state.colors_entities = query_entities["labels"]  ### This is the error
    # print(f"Query Entities: \n{query_entities}")
    get_word_in_entities(query_entities)
    st.session_state.query_ner_extraction = query_entities
    visualize_ner(
        query_entities["pred"],
        labels=query_entities["labels"].keys(),
        colors=query_entities["labels"],
        title=None,
        show_table=False,
        manual=True,
        marco_option=True,
    )

cOpt1, cOpt2, cOpt3, cOpt4 = colH2.columns([0.25, 0.25, 0.2, 0.3], vertical_alignment="top", gap="small")

cOpt1.caption("Sorting by?")
with cOpt1:
    option = st.radio("Sorting by?", ("semantic matching", "keyword matching"), label_visibility="collapsed")


cOpt2.caption("Choose NER model")
with cOpt2:
    optionModel = st.selectbox(
        "Choose NER model",
        ("Catalysis", "Solid State", "Wet Lab", "PcMSP", "CHEMU", "MsMention"),
        key="Model",
        label_visibility="collapsed",
    )

###C3
# c3.write(st.session_state.ON_STATE)
cOpt3.caption("Active AI Answer")
on = cOpt3.toggle("Feature", value=True, label_visibility="visible")

# collapse_option = st.selectbox("Collapse Results", ["True", "False"], key="a2a_collapse")
buttonSearch = cOpt4.button(
    label="Search",
    key="a2a_buttonRet",
    on_click=change_name,
    icon=":material/search:",
    use_container_width=True,
    args=[input_query, optionModel],
)

st.write ("Use dpasdaspo")
query = ""
if buttonSearch:
    st.session_state.optionSorting = option
    log_question_to_csv(input_query)

    if st.session_state.optionSorting == "keyword matching":
        print("Sparse........")
        query, docs = search(input_query, index_name, model_name, method_name, n_texts_retrieval_step, True, environment=environment)

    else:
        print("Dense........")
        query, docs = search_dense(
            input_query, index_name, model_name, method_name, n_texts_retrieval_step, True, environment=environment
        )
           
    # Trigger the NER
    full_text_docs = []

    for i, line in enumerate(docs):
        line = line["fields"]
        if st.session_state.optionSorting == "keyword matching" and environment == "test":
            line["content-full"] = cleanDocument(line["content-full"])
            full_text_docs.append(line["content-full"].replace("<hi>", "").replace("</hi>", ""))
        else: 
            line["content"] = cleanDocument(line["content"])
            full_text_docs.append(line["content"].replace("<hi>", "").replace("</hi>", ""))

    
    st.session_state.initial_results = docs #Use to rerank origninal DENSE AND RETRIEVAL
    #Reranking documents using Cohere
    ranked_list_docs = reranking_cohere_pre(query,full_text_docs , n_texts_reranking_step )
    
    #Reorder Documents
    ranked_docs = [docs[indx] for indx in ranked_list_docs]
    st.session_state.initital_ranked_docs = ranked_docs
    st.session_state.ranked_docs = ranked_docs
    

    ranked_text_docs = [full_text_docs[indx] for indx in ranked_list_docs]
    st.session_state.text_docs = ranked_text_docs
    
    docs_entities = extracted_entities(ranked_text_docs, st.session_state.optionModel)
    topK, entities_ex = correlated_entities(docs_entities)
    modify_color_entities_docs(query_entities, docs_entities)

    st.session_state.ner_extraction = docs_entities
    st.session_state.init_doc_ner_extraction = copy.deepcopy(docs_entities)
    st.session_state.topk_entities = topK
    st.session_state.nTPBS = st.session_state.nTPBS + 1
    

    print("******NER EXTRACTION*********")
    # print(entities_ex)
    # st.rerun()

st.text("✨ Answer generated by AI")

if query:
    with st.spinner("Generating answer, please wait..."):
        # a = generate_answer_llama(q, t)
        if on:
            answer = generate_answer(query, ranked_docs, environment ,limit_num_docs=10)
            st.session_state.AnswerGenAI = answer
        else:
            st.session_state.AnswerGenAI = ""

if st.session_state.AnswerGenAI:
    # if on:
    display_answer(st.session_state.AnswerGenAI)
    st.markdown("<br>", unsafe_allow_html=True)  # Add space

    if feedbackOption == True:
        st.markdown("**Please rate the answer generated , this will help us to improve the search engine:**")
        colFeed1, colFeed2 = st.columns([1, 5])
        with colFeed1:
            sentiment_mapping = ["one", "two", "three", "four", "five"]
            print(st.session_state.nTPBS)
            selected = st.feedback(
                "stars", key=f"startfeedback_{st.session_state.nTPBS}", on_change=on_selection_change
            )
            st.session_state.scoreAns = selected
        with colFeed2:
            if selected is not None:
                st.markdown(f"You selected {sentiment_mapping[selected]} star(s).")
        user_comment = st.text_area(
            "Please enter your comments below. Here are some questions to consider:    -**How helpful was this answer for your search? , -**Did the answer meet your expectations for clarity and completeness?, -**How well did this answer address your query?,  -**Did you find the response concise and easy to understand? , -**What could make this answer better?:",
            key=f"textFeedback_{st.session_state.nTPBS}",
        )

if st.session_state.ranked_docs:
    st.header("📑 Documents related", divider="blue")
    st.info(
        'Please provide feedback on the following snippet of text using the thumbs-up/thumbs-down buttons on the right side. -**Once you have provided your feedback, scroll to the end of the result and click the "Submit" button to save your responses. -**Don´t forget to press the button to ensure your feedback is recorded!',
        icon="ℹ️",
    )

    display_docs(st.session_state.ranked_docs, st.session_state.ner_extraction, environment, expander_option)


with st.sidebar:
    # st.write(topK)
    if st.session_state.topk_entities != "":
        st.subheader("Correlated Entities")
        st.write("Choose for each category the labels ")

        print("*/*/*//*/*/*/*//**/*//**/*/*//*/*/*/*/*")
        # print(st.session_state.topk_entities)
        categories = st.session_state.topk_entities
        num_elements = len(categories)
        options = {}
        i = 0
        for label, values in categories.items():
            list_values = [entities[0] for entities in values]
            # print(list_values)
            st.markdown(
                f'<p style="background-color:{st.session_state.colors_entities[label]};"><b>{label}</b></p>',
                unsafe_allow_html=True,
            )
            select_opt = st.multiselect(f"Select the {label}", list_values, key=label, label_visibility="collapsed")
            options[label] = select_opt
            # select_opt.append(options)
            st.markdown("<br>", unsafe_allow_html=True)
        st.session_state.options_entities = options
        col1Bre, col2Bre = st.columns(2, gap="small", vertical_alignment="center")
        col2Bre.button(
            label="Re-order",
            icon=":material/search:",
            use_container_width=True,
            on_click=rerank_docs_by_entities,
            args=(
                st.session_state.user_question,
                st.session_state.initital_ranked_docs,
                st.session_state.init_doc_ner_extraction,
                st.session_state.options_entities,
                False
            ),
        )
        col2Bre.button(
            label="RankLLM",
            icon=":material/manufacturing:",
            use_container_width=True,
            on_click=reranking_docs_by_llm,
            args=(
                st.session_state.user_question,
                st.session_state.initial_results,
                st.session_state.init_doc_ner_extraction,
            ),
            disabled=True,
        )
        col2Bre.button(
            label="RankLLM+Ent",
            icon=":material/manufacturing:",
            use_container_width=True,
            on_click=reranking_docs_by_llm,
            args=(
                st.session_state.user_question,
                st.session_state.initial_results,
                st.session_state.init_doc_ner_extraction,
                st.session_state.query_ner_extraction,
            ),
            disabled=True,
        )
        col2Bre.button(
            label="Rank Cohere",
            icon=":material/manufacturing:",
            use_container_width=True,
            on_click=reranking_cohere,
            args=(
                st.session_state.init_doc_ner_extraction,
                st.session_state.initial_results,
                st.session_state.user_question,
                st.session_state.fulldocs,
            ),
            disabled=True,
        )

        st.write(st.session_state.topk_entities)

st.divider()
st.markdown(
    """
    **If you encounter any issues or have questions, please contact us:**
    📧 [damianm@udel.edu](mailto:damianm@udel.edu)
    """
)
