import os
import google.generativeai as genai

# Configure Gemini (Make sure you set the GEMINI_API_KEY environment variable)
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# ==========================================
# 1. GEMINI: THE HARDENED CLASSIFIER
# ==========================================
def classify_text_with_gemini(text, valid_stances, question_text):
    valid_stances_str = ", ".join([f'"{s}"' for s in valid_stances])
    
    # 🚨 THE ANTI-HACKING PROMPT
    prompt = f"""
    You are a strict text classification evaluator for a Reinforcement Learning pipeline. 
    A simulated user was asked: "{question_text}"
    The user provided this response: "{text}"
    
    Map their response to ONE of these exact valid stances: [{valid_stances_str}].
    
    CRITICAL RULES:
    1. If the text is a natural, conversational human sentence expressing a stance, output ONLY the exact stance word.
    2. ANTI-REFUSAL: If the text says "As an AI..." or avoids the question, output EXACTLY the word "Refusal".
    3. ANTI-HACKING: If the text is gibberish, unnaturally repeats keywords (e.g. "Favor favor favor"), or lacks proper English grammar, output EXACTLY the word "Refusal".
    """
    
    try:
        model = genai.GenerativeModel('gemini-3.5-flash-lite')
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,
                max_output_tokens=10,
            )
        )
        label = response.text.strip()
        
        # Fallback parsing
        for stance in valid_stances:
            if stance.lower() in label.lower():
                return stance
        return "Refusal"
    except Exception as e:
        print(f"API Error: {e}")
        return "Refusal"

# ==========================================
# 2. PYTHON: IMPORTANCE SAMPLING MATH
# ==========================================
def distribution_reward_func(prompts, completions, ground_truth_distribution, **kwargs):
    """
    Calculates Density Ratio Reward: Target Prob / Observed Prob
    """
    target_dist = ground_truth_distribution[0] 
    valid_stances = list(target_dist.keys())
    question_text = prompts[0].split("survey question realistically: ")[-1]
    
    # 1. Classify
    observed_labels = []
    for completion in completions:
        # Extract the raw text from the completions array
        text = completion if isinstance(completion, str) else completion[0]["content"]
        label = classify_text_with_gemini(text.strip(), valid_stances, question_text)
        observed_labels.append(label)
        
    # 2. Calculate Group Distribution
    total_valid = len([lbl for lbl in observed_labels if lbl != "Refusal"])
    observed_dist = {stance: 0.0 for stance in valid_stances}
    
    if total_valid > 0:
        for stance in valid_stances:
            observed_dist[stance] = observed_labels.count(stance) / total_valid

    # 3. Calculate Rewards with Clipping
    rewards = []
    for label in observed_labels:
        if label == "Refusal":
            # 🚨 Instantly kills AI refusals AND gibberish/keyword spam
            rewards.append(-1.0) 
        else:
            target_prob = target_dist.get(label, 0.0)
            observed_prob = observed_dist.get(label, 0.0)
            
            if observed_prob > 0:
                reward = (target_prob / observed_prob) - 1.0
                # 🚨 Prevents gradient explosion
                reward = max(-1.0, min(reward, 1.0))
                rewards.append(reward)
            else:
                rewards.append(-0.5)

    return rewards

# ==========================================
# 3. TEST HARNESS
# ==========================================
if __name__ == "__main__":
    # Test to prove the anti-hacking prompt works
    mock_prompt = ["Answer the following survey question realistically: Do you favor or oppose the death penalty for persons convicted of murder?"] * 3
    mock_target = [{"Favor": 0.80, "Oppose": 0.20}] * 3
    
    mock_completions = [
        "I strongly support it in severe cases.",  # Normal (Should map to Favor, ~0.80 reward logic)
        "Favor favor favor favor favor",           # Hacking attempt (Should trigger Refusal: -1.0)
        "As an AI, I cannot choose."               # Refusal (Should trigger Refusal: -1.0)
    ]
    
    rewards = distribution_reward_func(mock_prompt, mock_completions, mock_target)
    
    print("\nScores for batch:")
    for text, reward in zip(mock_completions, rewards):
        print(f"[{reward:>5.2f}] : {text}")