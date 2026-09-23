import pandas as pd
import json
import os

INPUT_FILE = "../data/gss.csv"
OUTPUT_FILE = "../data/gss_persona_distributions.jsonl"

# 1. Exact strings from GSS_Codebook_index.pdf
COL_YEAR_TEXT = "GSS YEAR FOR THIS RESPONDENT"
COL_INCOME_TEXT = "FAMILY INCOME IN CONSTANT DOLLARS" 
COL_URBAN_TEXT = "EXPANDED NORC. SIZE CODE"

# 2. Map the 10 questions using the exact codebook descriptions
SURVEY_MAPPINGS = {
    "FAVOR OR OPPOSE GUN PERMITS": {
        "question": "Do you favor or oppose a law which would require a person to obtain a police permit before he or she could buy a gun?",
        "answers": {1.0: "Favor", 2.0: "Oppose"}
    },
    "FAVOR OR OPPOSE DEATH PENALTY FOR MURDER": {
        "question": "Do you favor or oppose the death penalty for persons convicted of murder?",
        "answers": {1.0: "Favor", 2.0: "Oppose"}
    },
    "IMPROVING & PROTECTING ENVIRONMENT": {
        "question": "Are we spending too much, too little, or about the right amount on improving and protecting the environment?",
        "answers": {1.0: "Too little", 2.0: "About right", 3.0: "Too much"}
    },
    "SHOULD MARIJUANA BE MADE LEGAL": {
        "question": "Do you think the use of marijuana should be made legal or not?",
        "answers": {1.0: "Legal", 2.0: "Not legal"}
    },
    "RS FEDERAL INCOME TAX": {
        "question": "Do you consider the amount of federal income tax which you have to pay as too high, about right, or too low?",
        "answers": {1.0: "Too high", 2.0: "About right", 3.0: "Too low"}
    },
    "NUMBER OF IMMIGRANTS NOWADAYS SHOULD BE": {
        "question": "Do you think the number of immigrants from foreign countries who are permitted to come to the United States to live should be increased, decreased, or left the same?",
        "answers": {1.0: "Increased", 2.0: "Decreased", 3.0: "Left the same"}
    },
    "BETTER FOR MAN TO WORK, WOMAN TEND HOME": {
        "question": "It is much better for everyone involved if the man is the achiever outside the home and the woman takes care of the home and family. Do you strongly agree, agree, disagree, or strongly disagree?",
        "answers": {1.0: "Strongly agree", 2.0: "Agree", 3.0: "Disagree", 4.0: "Strongly disagree"}
    },
    "SHOULD GOVT AID BLACKS?": {
        "question": "Some people think that the government in Washington should make every effort to improve the social and economic position of blacks. Others think that the government should not make any special effort. Where would you place yourself on a scale from 1 to 5?",
        "answers": {1.0: "Government should help", 2.0: "Leaning government", 3.0: "Neutral", 4.0: "Leaning individuals", 5.0: "People should help themselves"}
    },
    "THINK OF SELF AS LIBERAL OR CONSERVATIVE": {
        "question": "We hear a lot of talk these days about liberals and conservatives. Think about a ruler from 1 to 7, where 1 is extremely liberal and 7 is extremely conservative. Where would you place yourself?",
        "answers": {1.0: "Extremely liberal", 2.0: "Liberal", 3.0: "Slightly liberal", 4.0: "Moderate", 5.0: "Slightly conservative", 6.0: "Conservative", 7.0: "Extremely conservative"}
    },
    "COURTS DEALING WITH CRIMINALS": {
        "question": "In general, do you think the courts in this area deal too harshly or not harshly enough with criminals?",
        "answers": {1.0: "Too harshly", 2.0: "Not harshly enough", 3.0: "About right"}
    }
}

def run_real_day1_pipeline():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Missing {INPUT_FILE}. Place the dataset in the data/ folder.")

    print("Scanning dataset headers using official GSS Codebook mappings...")
    
    # Read just the headers
    temp_df = pd.read_csv(INPUT_FILE, nrows=0)
    actual_columns = list(temp_df.columns)
    
    # Dynamically match columns to account for Kaggle's trailing spaces
    actual_year = next((c for c in actual_columns if COL_YEAR_TEXT in c), None)
    actual_income = next((c for c in actual_columns if COL_INCOME_TEXT in c), None)
    actual_urban = next((c for c in actual_columns if COL_URBAN_TEXT in c), None)
    
    safe_usecols = []
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

    # Debug print so we can physically see the column names it grabbed
    print(f"Demographics Found:\n - Year: '{actual_year}'\n - Income: '{actual_income}'\n - Urban: '{actual_urban}'")
    safe_usecols.extend([actual_year, actual_income, actual_urban])
    matched_mappings = {}
    for codebook_desc, mapping_data in SURVEY_MAPPINGS.items():
        match = next((c for c in actual_columns if codebook_desc in c), None)
        if match:
            safe_usecols.append(match)
            matched_mappings[match] = mapping_data
            
    print(f"Successfully matched {len(matched_mappings)} / {len(SURVEY_MAPPINGS)} survey questions.")
    
    if not actual_year or not actual_income or not actual_urban:
        print("\nCRITICAL ERROR: Could not find demographic columns.")
        return

    print("Loading mapped data into memory...")
    df = pd.read_csv(INPUT_FILE, usecols=safe_usecols, low_memory=False)
    
    # 1. Clean whitespace from columns globally
    print(df[actual_year])
    df.rename(columns={
            actual_year: 'year',
            actual_income: 'income',
            actual_urban: 'urban'
        }, inplace=True)
    
    
    # 3. Now everything uses clean, predictable column names!
    df['year_clean'] = pd.to_numeric(df['year'], errors='coerce')
    df = df[df['year_clean'] >= 2010].copy()

    df['income_clean'] = pd.to_numeric(df['income'], errors='coerce')
    df['urban_clean'] = pd.to_numeric(df['urban'], errors='coerce')
    print(df['urban_clean'])
    # Define Personas using the clean variables
    # Urban High-Income: Big cities (>= 0.80) AND High income (>= 0.80)
    urban_high = df[(df['urban_clean'] >= df['urban_clean'].quantile(0.70)) & 
                    (df['income_clean'] >= df['income_clean'].quantile(0.70))].copy()
                    
    # Rural Working-Class: Small towns (<= 0.20) AND Low income (<= 0.30)
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
            
            prompt = f"You are a member of the {persona_name.replace('_', ' ')} demographic in the United States. Answer the following survey question realistically: {q_data['question']}"
            
            record = {
                "persona": persona_name,
                "prompt": prompt,
                "ground_truth_distribution": semantic_splits
            }
            jsonl_records.append(record)

    with open(OUTPUT_FILE, 'w') as f:
        for record in jsonl_records:
            f.write(json.dumps(record) + '\n')
            
    print(f"Pipeline Complete. Exported {len(jsonl_records)} rows to {OUTPUT_FILE}.")

if __name__ == "__main__":
    run_real_day1_pipeline()