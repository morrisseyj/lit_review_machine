import ast
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Union, Tuple
import json
import pyarrow as pa


def ensure_list_of_strings(val):
    """
    Normalize input to a list of strings.
    Handles lists, strings, NaN, and other types.
    Flattens list-of-lists if encountered.
    """
    # Flatten list-of-lists (e.g., [["Smith", "Jones"]])
    if isinstance(val, list) and len(val) == 1 and isinstance(val[0], list):
        val = val[0]
    if isinstance(val, list):
        return [str(v) for v in val if v is not None]
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                return [str(v) for v in parsed if v is not None]
        except Exception:
            pass
        return [val]
    if pd.isna(val):
        return []
    return [str(val)]

def populate_dict_recursively(data_dict, value_to_set):
    """
    Recursively sets the value of every leaf-node (non-dictionary) key 
    in a nested dictionary to the specified value.
    """
    for key, value in data_dict.items():
        if isinstance(value, dict):
            # If the value is a dictionary, recurse into it
            populate_dict_recursively(value, value_to_set)
        else:
            # If it's a leaf node (not a dictionary), set its value
            data_dict[key] = value_to_set
    return data_dict

def call_chat_completion(llm_client, ai_model, sys_prompt, user_prompt, return_json, fall_back: Dict):
  
    # Call the LLM
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    ]
    if return_json:
        print("Calling LLM for JSON response...")
        try:
            response = llm_client.chat.completions.create(
            model=ai_model,
            messages=messages,
            response_format={"type": "json_object"}
        )
        except Exception as e:
            print(f"Call to OpenAI failed. Error: {e}")
            error_dict = fall_back
            populate_dict_recursively(error_dict, e)
            error_json = json.dumps(error_dict)
            return(error_json)
    else:
        try:
            response = llm_client.chat.completions.create(
                model=ai_model,
                messages=messages
            )
        except Exception as e:
            print(f"Call to OpenAI failed. Error: {e}")
            error_dict = fall_back
            populate_dict_recursively(error_dict, e)
            error_json = json.dumps(error_dict)
            return(error_json)

    try:
        dict = json.loads(response.choices[0].message.content.strip())
        return(dict)  
    except Exception as e:
        print(f"LLM failed to return a json: {e}")
        error_dict = fall_back
        populate_dict_recursively(error_dict, e)
        return(error_dict)

def call_reasoning_model(prompt, llm_client, ai_model, timeout):
    # Call the LLM
    try:
        response=llm_client.responses.create(
            model=ai_model,
            input=prompt,
            tools=[{"type": "web_search_preview"}],
            timeout=timeout
        )
    except llm_client.error.Timeout as e:
        print(f"{e} Consider increasing timeout (default is 1200s).")
        return None
    except Exception as e:
        print(f"Call to OpenAI failed. Error: {e}")
        return None
    return response.output_text

def llm_json_clean(x, prompt, llm_client, ai_model):
    sys_prompt = prompt
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": x}
    ]

    response = llm_client.chat.completions.create(
                model=ai_model,
                messages=messages,
                response_format={"type": "json_object"}
            )
    
    return response.choices[0].message.content

def json_format_check(x):
    # Check if its valid json for loading
    try:
        response_dict = json.loads(x)
    except json.JSONDecodeError:
        error = "The LLM did not return valid json. Efforts to resolve this have failed. Try run the LLM call again."
        return False, error, x
    
    result_key = list(response_dict.keys())[0]
    #Convert the list of dicts to a df
    response_df = pd.DataFrame(response_dict[result_key]).reset_index(drop = True)
    
    # Loop to catch any level of escaping (i.e., "strings of strings")
    while True:
        # 1. Identify WHICH elements are still strings
        # This is the reliable check, NOT the column's overall dtype
        is_string_mask = response_df["paper_author"].apply(lambda val: isinstance(val, str))
        
        # 2. Stop condition: If NO elements are strings, we are done un-escaping
        if not is_string_mask.any():
            break

        try:
            # 3. Apply json.loads ONLY to the elements identified as strings
            response_df.loc[is_string_mask, "paper_author"] = (
                response_df.loc[is_string_mask, "paper_author"].apply(json.loads)
            )
        except json.JSONDecodeError:
            # If we fail to un-escape a string element, it's malformed JSON
            error = "The LLM failed to generate the paper authors as valid json after attempts to repair escaping. Try the LLM call again."
            return False, error, x
    
    # --- FINAL VALIDATION CHECK ---
    
    # Final Check: Ensure every element is definitely a list
    # The column's dtype will be 'object', so we must check the content
    if not all(isinstance(val, list) for val in response_df["paper_author"]):
        error = "The LLM did not return the paper authors as lists. This cannnot be fixed generically. Try call the LLM again"
        return False, error, x
    
    # Check whether this output will save to a parquet file
    response_df_check = response_df.copy()
    response_df_check["paper_author"] = response_df_check["paper_author"].apply(json.dumps)
    try:
        pa.Table.from_pandas(response_df_check)
    except Exception as e:
        error = "Despite producing a valid dataframe with authors as lists, the dataframe will fail when saving to parquet. Manually inspect the output to understand the issue"
        return False, error, x 

    return True, None, response_df