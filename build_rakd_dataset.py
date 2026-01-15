import json
import pandas as pd
import os
from tqdm import tqdm


def load_json_as_dict(path, key_field='user_id'):
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Returning empty dict.")
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {item[key_field]: item for item in data}


def get_user_history(data_path):
    df = pd.read_csv(data_path, sep='\t', header=None,
                     names=['user_id', 'venue_id', 'cat_id', 'cat_name', 'lat', 'lon', 'offset', 'utc_time'],
                     encoding='latin-1')
    df['utc_time'] = pd.to_datetime(df['utc_time'], format="%a %b %d %H:%M:%S +0000 %Y")
    df = df.sort_values(by=['user_id', 'utc_time'])

    history_dict = {}
    for user_id, group in df.groupby('user_id'):
        pois = group['cat_name'].tolist()
        history_dict[user_id] = pois
    return history_dict


def construct_rakd_prompt(role_data, pref_data, candidates, history, target_poi):

    role_desc = "Unknown"
    if role_data:
        res = role_data.get('analysis_result', {})
        role_desc = f"{res.get('User_Role', '')}: {res.get('User_Profile', '')}"

    pref_desc = "Unknown"
    if pref_data:
        pref_desc = pref_data.get('final_long_term_preference', '')

    cand_str = "None"
    if candidates:
        cand_str = ", ".join([str(c) for c in candidates[:5]])  # 取前5个

    recent_history = history[-10:] if history else []
    hist_str = " -> ".join(recent_history)

    prompt = f"""Role: {role_desc}
User preference is {pref_desc}.
Candidate POIs: {cand_str}.
Check-in history: {hist_str}.
Recommendation:"""

    return prompt


def main():
    roles_path = "user_roles_output.json"
    prefs_path = "intra_city_preferences.json"
    distill_path = "user_distillation_candidates.json"
    raw_data_path = "nyc.txt"
    output_path = "data/ref/nyc/rakd_train.json"
    print("Loading auxiliary data...")
    roles_map = load_json_as_dict(roles_path)
    prefs_map = load_json_as_dict(prefs_path)
    distill_map = load_json_as_dict(distill_path)
    print("Loading raw history...")
    history_map = get_user_history(raw_data_path)
    final_data = []

    print("Constructing RAKD prompts...")
    for user_id, visits in tqdm(history_map.items()):
        if len(visits) < 5:
            continue
        target = visits[-1]
        context_history = visits[:-1]
        u_role = roles_map.get(user_id)
        u_pref = prefs_map.get(user_id)
        u_cands = distill_map.get(user_id, [])
        prompt_text = construct_rakd_prompt(u_role, u_pref, u_cands, context_history, target)
        target_str = f"{target}"

        sample = {
            "question": prompt_text,
            "answer": target_str
        }
        final_data.append(sample)


    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=4)

    print(f"Successfully generated {len(final_data)} samples to {output_path}")
    print("Example Prompt:\n" + final_data[0]['question'])


if __name__ == "__main__":
    main()