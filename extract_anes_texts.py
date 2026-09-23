import pandas as pd
import json
import os

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
CSV_PATH = "../data/anes_timeseries_2020_csv_20220210.csv"
EXCEL_PATH = "../data/anes_timeseries_2020_redactedopenends_excel_20211118.xlsx"
OUTPUT_JSON = "anes_human_texts.json"
SHEET_NAME = "V202205"  # The "Most Important Problem" open-ended sheet

def extract_and_save_anes_texts():
    print("📥 Loading ANES demographic data...")
    # V200001: Respondent ID
    # V201617x: Total Family Income 
    # V202355: Urban/Rural Size (1=Rural, 2=Small Town, 3=Suburb, 4=City)
    df_demo = pd.read_csv(CSV_PATH, usecols=['V200001', 'V201617x', 'V202355'], low_memory=False)

    print(f"📥 Loading ANES open-ended text sheet ('{SHEET_NAME}')...")
    df_text = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)

    text_col = [c for c in df_text.columns if c != 'V200001'][0]
    df_text.rename(columns={text_col: 'verbatim_text'}, inplace=True)

    print("🔗 Merging and cleaning data...")
    merged = pd.merge(df_demo, df_text, on="V200001").dropna(subset=['verbatim_text'])
    merged['verbatim_text'] = merged['verbatim_text'].astype(str).str.strip()
    merged = merged[merged['verbatim_text'].str.len() > 10]

    # Clean out non-responses/refusals (ANES codes these as negative numbers)
    merged = merged[(merged['V201617x'] > 0) & (merged['V202355'] > 0)].copy()

    # ==========================================
    # 🎯 APPLY EXACT QUANTILE LOGIC FROM GSS PIPELINE
    # ==========================================
    
    # Calculate exact thresholds just like your GSS script
    inc_70 = merged['V201617x'].quantile(0.70)
    inc_30 = merged['V201617x'].quantile(0.30)
    
    urb_70 = merged['V202355'].quantile(0.70)
    urb_30 = merged['V202355'].quantile(0.30)

    # 1. Urban High Income: Top 30% Urban AND Top 30% Income
    urban_high = merged[
        (merged['V202355'] >= urb_70) & 
        (merged['V201617x'] >= inc_70)
    ]['verbatim_text'].tolist()

    # 2. Rural Working Class: Bottom 30% Urban AND Bottom 30% Income
    rural_working = merged[
        (merged['V202355'] <= urb_30) & 
        (merged['V201617x'] <= inc_30)
    ]['verbatim_text'].tolist()

    print(f"\n📊 Demographics matched via Quantiles:")
    print(f"  ✓ Urban High-Income (>= 70th Percentile): {len(urban_high)} humans extracted.")
    print(f"  ✓ Rural Working-Class (<= 30th Percentile): {len(rural_working)} humans extracted.")

    # ==========================================
    # 💾 EXPORT TO JSON
    # ==========================================
    human_samples = {
        "Urban_High_Income": urban_high,
        "Rural_Working_Class": rural_working
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(human_samples, f, indent=2)
        
    print(f"\n✅ Successfully saved raw extracted texts to '{OUTPUT_JSON}'.")

if __name__ == "__main__":
    extract_and_save_anes_texts()