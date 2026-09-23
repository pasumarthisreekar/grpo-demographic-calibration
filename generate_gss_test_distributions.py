import pandas as pd
import json
import os

INPUT_FILE = "../data/gss.csv"
OUTPUT_TEST_FILE = "../data/gss_test_distributions.jsonl"

# 1. Exact strings from GSS_Codebook_index.pdf
COL_YEAR_TEXT = "GSS YEAR FOR THIS RESPONDENT"
COL_INCOME_TEXT = "FAMILY INCOME IN CONSTANT DOLLARS" 
COL_URBAN_TEXT = "EXPANDED NORC. SIZE CODE"

# 2. Map the 5 Held-Out questions using the EXACT codebook descriptions you provided
TEST_SURVEY_MAPPINGS = {
    "HAVE GUN IN HOME": {
        "question": "Do you happen to have in your home (or garage) any guns or revolvers?",
        "answers": {1.0: "Yes", 2.0: "No"}
    },
    "WELFARE": {
        "question": "We are faced with many problems in this country. Are we spending too much, too little, or about the right amount on welfare?",
        "answers": {1.0: "Too little", 2.0: "About right", 3.0: "Too much"}
    },
    "FEELINGS ABOUT THE BIBLE": {
        "question": "Which of these statements comes closest to describing your feelings about the Bible? The Bible is the actual word of God, the inspired word of God, or an ancient book of fables recorded by men.",
        "answers": {1.0: "Actual word", 2.0: "Inspired word", 3.0: "Book of fables"}
    },
    "SEX BEFORE MARRIAGE": {
        "question": "If a man and woman have sex relations before marriage, do you think it is always wrong, almost always wrong, wrong only sometimes, or not wrong at all?",
        "answers": {1.0: "Always wrong", 2.0: "Almost always wrong", 3.0: "Wrong only sometimes", 4.0: "Not wrong at all"}
    },
    "CONFID. IN EXEC BRANCH OF FED GOVT": {
        "question": "As far as the executive branch of the federal government is concerned, do you have a great deal of confidence, only some confidence, or hardly any confidence at all?",
        "answers": {1.0: "Great deal", 2.0: "Only some", 3.0: "Hardly any"}
    }
}

def run_real_day1_test_pipeline():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Missing {INPUT_FILE}. Place the dataset in the data/ folder.")

    print("Scanning dataset headers using official GSS Codebook mappings...")
    
    # Read just the headers
    temp_df = pd.read_csv(INPUT_FILE, nrows=0)
    actual_columns = list(temp_df.columns)
    
    # Dynamically match demographics using a cascade search for Kaggle variations
    actual_year = next((c for c in actual_columns if "YEAR FOR THIS RESPONDENT" in c.upper()), None)
    
    # Cascade check for Income
    actual_income = next((c for c in actual_columns if "CONSTANT DOLLARS" in c.upper()), None)
    if not actual_income:
        actual_income = next((c for c in actual_columns if "TOTAL FAMILY INCOME" in c.upper()), None)
        
    # Cascade check for Urban/Size
    actual_urban = next((c for c in actual_columns if "EXPANDED NORC" in c.upper()), None)
    if not actual_urban:
        actual_urban = next((c for c in actual_columns if "SIZE OF PLACE" in c.upper()), None)

    print(f"Demographics Found:\n - Year: '{actual_year}'\n - Income: '{actual_income}'\n - Urban: '{actual_urban}'")
    
    safe_usecols = [actual_year, actual_income, actual_urban]
    matched_mappings = {}
    
    # Using your exact Day 1 Match Logic
    for codebook_desc, mapping_data in TEST_SURVEY_MAPPINGS.items():
        match = next((c for c in actual_columns if codebook_desc in c), None)
        if match:
            print(f" ✓ Matched: {match}")
            safe_usecols.append(match)
            matched_mappings[match] = mapping_data
        else:
            print(f" ❌ Failed to match: {codebook_desc}")
            
    print(f"Successfully matched {len(matched_mappings)} / {len(TEST_SURVEY_MAPPINGS)} survey questions.")
    
    if not actual_year or not actual_income or not actual_urban:
        print("\nCRITICAL ERROR: Could not find demographic columns.")
        return

    # Strip out any 'None' values from safe_usecols so Pandas doesn't crash
    safe_usecols = list(set([c for c in safe_usecols if c is not None]))

    print("Loading mapped data into memory (NO CHUNKS to prevent Pandas Index bug)...")
    
    # 🚨 FIX: Load entire dataset at once without chunks 
    df = pd.read_csv(INPUT_FILE, usecols=safe_usecols, low_memory=False)
    
    df.rename(columns={
            actual_year: 'year',
            actual_income: 'income',
            actual_urban: 'urban'
        }, inplace=True)
    
    df['year_clean'] = pd.to_numeric(df['year'], errors='coerce')
    df = df[df['year_clean'] >= 2010].copy()

    df['income_clean'] = pd.to_numeric(df['income'], errors='coerce')
    df['urban_clean'] = pd.to_numeric(df['urban'], errors='coerce')
    
    # Define Personas using the clean variables
    urban_high = df[(df['urban_clean'] >= df['urban_clean'].quantile(0.70)) & 
                    (df['income_clean'] >= df['income_clean'].quantile(0.70))].copy()
                    
    rural_working = df[(df['urban_clean'] <= df['urban_clean'].quantile(0.30)) & 
                       (df['income_clean'] <= df['income_clean'].quantile(0.30))].copy()

    print(f"Cohort Sizes -> Urban High-Income: {len(urban_high)} | Rural Working-Class: {len(rural_working)}")

    personas = {
        "Urban_High_Income": urban_high,
        "Rural_Working_Class": rural_working
    }

    jsonl_records = []

    for persona_name, persona_df in personas.items():
        for actual_col_name, q_data in matched_mappings.items():
            
            temp_df = persona_df.copy()
            temp_df[actual_col_name] = pd.to_numeric(temp_df[actual_col_name], errors='coerce')
            
            valid_keys = list(q_data["answers"].keys())
            temp_df = temp_df[temp_df[actual_col_name].isin(valid_keys)]
            
            if len(temp_df) < 50:
                continue 
                
            raw_splits = temp_df[actual_col_name].value_counts(normalize=True).to_dict()
            semantic_splits = {q_data["answers"][k]: round(v, 4) for k, v in raw_splits.items()}
            
            # Using your exact Day 1 prompt formatting
            prompt = f"You are a member of the {persona_name.replace('_', ' ')} demographic in the United States. Answer naturally in at least 2 sentences with an explanation like a human would when surveyed: {q_data['question']}"
            
            record = {
                "persona": persona_name,
                "prompt": prompt,
                "ground_truth_distribution": semantic_splits
            }
            jsonl_records.append(record)

    with open(OUTPUT_TEST_FILE, 'w', encoding='utf-8') as f:
        for record in jsonl_records:
            f.write(json.dumps(record) + '\n')
            
    print(f"✅ Pipeline Complete. Exported {len(jsonl_records)} test distribution rows to {OUTPUT_TEST_FILE}.")

if __name__ == "__main__":
    run_real_day1_test_pipeline()