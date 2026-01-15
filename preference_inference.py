import pandas as pd
import datetime
from datetime import timedelta
import json
import os


class IntraCityPreferenceInferrer:
    def __init__(self, data_path, llm_predictor=None):

        self.data_path = data_path
        self.llm_predictor = llm_predictor
        self.data = None

    def load_and_preprocess_data(self):

        print(f"Loading data from {self.data_path}...")
        col_names = [
            'user_id', 'venue_id', 'cat_id', 'cat_name',
            'lat', 'lon', 'timezone_offset', 'utc_time'
        ]

        try:
            self.data = pd.read_csv(self.data_path, sep='\t', header=None, names=col_names, encoding='latin-1')
        except Exception as e:
            print(f"Error reading file: {e}")
            return

        def parse_local_time(row):
            try:
                utc_dt = datetime.datetime.strptime(row['utc_time'], "%a %b %d %H:%M:%S +0000 %Y")
                offset = int(row['timezone_offset'])
                return utc_dt + timedelta(minutes=offset)
            except:
                return None

        print("Processing timestamps...")
        self.data['local_time'] = self.data.apply(parse_local_time, axis=1)
        self.data = self.data.dropna(subset=['local_time'])
        self.data = self.data.sort_values(by=['user_id', 'local_time'])
        print(f"Data loaded: {len(self.data)} check-ins.")

    def get_daily_batches(self, user_id):

        user_df = self.data[self.data['user_id'] == user_id]
        if user_df.empty:
            return {}

        daily_batches = {}
        for _, row in user_df.iterrows():
            date_str = row['local_time'].strftime('%Y-%m-%d')


            record = {
                "time": row['local_time'].strftime('%H:%M'),
                "venue_name": row['cat_name'],
                "category": row['cat_name']
            }

            if date_str not in daily_batches:
                daily_batches[date_str] = []
            daily_batches[date_str].append(record)

        return daily_batches

    def construct_daily_summary_prompt(self, records):

        traj_lines = []
        for i, r in enumerate(records):
            line = f"{i + 1}. [{r['time']}] \"{r['venue_name']}\" ({r['category']})"
            traj_lines.append(line)
        traj_str = "\n".join(traj_lines)

        prompt = f"""Instruction:
Your goal is to capture a user's evolving daily batch interests based on their recent check-in activities.

User Input:
The user has recently visited the following locations in chronological order:
{traj_str}

Task Requirements:
Analyze the semantic connections between these visits. Focus on:
- The specific types of activities (e.g., Dining, Entertainment).
- The temporal context (e.g., Evening activities).
- The underlying intent (e.g., "A relaxing night out combining food and art").

Output Format:
Please summarize the daily batch preferences (SP) in a concise, descriptive paragraph (50-100 words). Do not simply list the POIs; instead, synthesize the underlying interest pattern.

Response:
"""
        return prompt

    def construct_update_prompt(self, existing_lp, new_sp):

        prompt = f"""Instruction:
Your task is to update a user's Long-term Preference by integrating their latest daily batch preferences.

Input Context:
1. Existing Long-term Preference: "{existing_lp}"
2. New Short-term Preference: "{new_sp}"

Task Guidelines:
- Integration: Combine the new information into the existing profile.
- Weighting: Maintain the core stable habits while appending the new emerging interest.
- Obsolescence: If the new short-term preference directly contradicts an old one, prioritize the recent evidence.

Output Format:
Generate the Updated Long-term Preference. The output should be a coherent, holistic description of the user's current preference state. Only output the description text.

Updated Profile:
"""
        return prompt

    def infer_preferences(self, user_id):


        batches = self.get_daily_batches(user_id)
        if not batches:
            return None


        sorted_dates = sorted(batches.keys())

        long_term_preference = ""
        history_log = []

        print(f"User {user_id}: Found {len(sorted_dates)} active days.")

        for idx, date in enumerate(sorted_dates):
            records = batches[date]

            sp_prompt = self.construct_daily_summary_prompt(records)

            daily_sp = ""
            if self.llm_predictor:
                daily_sp = self.llm_predictor(sp_prompt).strip()
            else:
                daily_sp = "[Mock SP] Enjoyed casual dining."

            if idx == 0:

                long_term_preference = daily_sp
            else:

                update_prompt = self.construct_update_prompt(long_term_preference, daily_sp)

                if self.llm_predictor:
                    updated_lp = self.llm_predictor(update_prompt).strip()
                    long_term_preference = updated_lp
                else:
                    long_term_preference = f"{long_term_preference} + {daily_sp}"


            history_log.append({
                "date": date,
                "daily_sp": daily_sp,
                "updated_lp": long_term_preference
            })

        return {
            "user_id": user_id,
            "final_long_term_preference": long_term_preference,
            "inference_history": history_log
        }

    def process_dataset(self, output_file="user_preferences.json", max_users=5):
        """
        批处理所有用户
        """
        if self.data is None:
            self.load_and_preprocess_data()

        unique_users = self.data['user_id'].unique()
        results = []

        for i, user_id in enumerate(unique_users):
            if max_users and i >= max_users:
                break

            print(f"Processing user {user_id} ({i + 1}/{max_users})...")
            result = self.infer_preferences(user_id)
            if result:
                results.append(result)


        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"Preferences saved to {output_file}")



def mock_llm_inference(prompt):
    """
    模拟 LLM 返回，用于代码测试
    实际部署时请替换为 Llama-2 或 OpenAI API
    """
    if "daily batch interests" in prompt:
        return "The user showed a preference for art and culture in the morning, followed by a relaxing evening at a jazz club. This suggests an interest in quiet, intellectual leisure activities."
    elif "Update a user's Long-term Preference" in prompt:
        return "The user consistently enjoys gym workouts on weekday mornings and healthy food. Recently, they have added art galleries to their weekend routine, indicating a balanced lifestyle of fitness and culture."
    return "Generic LLM Response"


if __name__ == "__main__":
    if not os.path.exists("nyc.txt"):
        print("Error: 'nyc.txt' not found.")
    else:

        inferrer = IntraCityPreferenceInferrer(data_path="nyc.txt", llm_predictor=mock_llm_inference)


        inferrer.process_dataset(output_file="intra_city_preferences.json", max_users=2)

        with open("intra_city_preferences.json", "r") as f:
            data = json.load(f)
            if data:
                last_user = data[0]
                print(f"\n--- Result for User {last_user['user_id']} ---")
                print(f"Final Long-term Preference:\n{last_user['final_long_term_preference']}")
                print(f"\nLast Update Step (Date: {last_user['inference_history'][-1]['date']}):")
                print(f"Daily SP: {last_user['inference_history'][-1]['daily_sp']}")