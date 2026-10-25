import numpy as np 
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_main_verb(sentence):
    doc = nlp(sentence)
    return next(
        (token.text for token in doc
         if token.dep_ == "ROOT" and token.pos_ == "VERB"),
        None
    )

def pre_process(df, dataset):
    # remove irrelevant columns
    columns_to_keep = [
    "subjectID"
    , "trial_property"
    , "response"
    , "type"
    , "item"
    , "art_aux"
    , "prep_adv"
    ,"sentence_fragment"
    , "left_label"
    , "pres_order"
    , "...22"
    , "name"
    , "par_no"
        ]

    df = df[columns_to_keep]

    # remove filler rows
    df = df[(df["trial_property"] == f"trial-{dataset}") & (df["type"] != "FILLER")]

    #pipeline assumption check
    assert (df[["subjectID", "item"]].value_counts() == 3).all(), \
        "Every subjectID, item combo needs to occur 3 times."
    
    # adjust left label score
    df["adjusted_response"] = np.where(
        df["left_label"].isin(["RR", "S"]),
        100 - df["response"],
        df["response"]
        )

    # drop rows where time out occurred
    starting_count = len(df)
    print(f"Started with {starting_count} rows...")
    df = df.dropna(subset=["response"]).reset_index(drop=True)
    number_dropped = starting_count - len(df)
    print(f"Ending with {len(df)}, dropping {number_dropped} rows.")    

    # get verb of interest
    df["verb_of_interest_itemized"] = df["sentence_fragment"].str.split().str[-1] + "_" + df["item"].astype(str)
    df["verb_of_interest"] = df["sentence_fragment"].str.split().str[-1] 

    df["matrix_verb"] = df["sentence_fragment"].map(
    {s: extract_main_verb(s)
     for s in df["sentence_fragment"].dropna().unique()}
    )

    df["matrix_verb_itemized"] = df["matrix_verb"] + "_" + df["item"]

    # average responses by participant then verb
    average_user_response = df.groupby(["subjectID","matrix_verb_itemized", "matrix_verb"])["adjusted_response"].mean().reset_index()
    average_verb_response = average_user_response.groupby(["matrix_verb_itemized", "matrix_verb"])["adjusted_response"].mean().sort_values().reset_index()

    return average_user_response, average_verb_response