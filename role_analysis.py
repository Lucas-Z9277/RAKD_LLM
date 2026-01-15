import pandas as pd
import datetime
from datetime import timedelta
import json
import os


class InterCityUserRoleAnalyzer:
    def __init__(self, data_path, llm_predictor=None):

        self.data_path = data_path
        self.llm_predictor = llm_predictor
        self.data = None

    def load_and_preprocess_data(self):

        print(f"Loading data from {self.data_path}...")
        # 定义列名 (根据您提供的 nyc.txt 描述)
        # 1. User ID, 2. Venue ID, 3. Cat ID, 4. Cat Name, 5. Lat, 6. Lon, 7. Offset, 8. Time
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
                # 解析 UTC 时间字符串 (例如: Tue Apr 03 18:00:09 +0000 2012)
                utc_dt = datetime.datetime.strptime(row['utc_time'], "%a %b %d %H:%M:%S +0000 %Y")
                # 应用时区偏移 (offset 是分钟)
                offset = int(row['timezone_offset'])
                local_dt = utc_dt + timedelta(minutes=offset)
                return local_dt
            except Exception as e:
                return None


        print("Processing timestamps...")
        self.data['local_time'] = self.data.apply(parse_local_time, axis=1)

        self.data = self.data.dropna(subset=['local_time'])

        self.data = self.data.sort_values(by=['user_id', 'local_time'])
        print(f"Data loaded: {len(self.data)} check-ins.")

    def _get_time_slot_desc(self, dt):

        is_weekend = dt.weekday() >= 5
        day_type = "Weekend" if is_weekend else "Weekday"

        time_str = f"{dt.hour}:{dt.minute:02d}"

        return f"{day_type} {time_str}"

    def generate_schema_trajectory(self, user_id):

        user_df = self.data[self.data['user_id'] == user_id]

        if user_df.empty:
            return []

        schema_traj = []
        for _, row in user_df.iterrows():
            category = row['cat_name']
            time_slot = self._get_time_slot_desc(row['local_time'])

            item = f"<{category}, {time_slot}>"
            schema_traj.append(item)

        return schema_traj

    def construct_prompt(self, user_id, schema_traj):
        traj_str = ",\n".join(schema_traj)
        if len(schema_traj) > 50:
            traj_str = ",\n".join(schema_traj[:50]) + "\n... (more records)"

        prompt = f"""Instruction:
You are an urban computing analyst specializing in human mobility modeling. Your task is to analyze a user's cross-city movement patterns to infer their social role and generate a descriptive profile.

User Input:
Here is the schema trajectory for User [{user_id}]:
[
{traj_str}
]

Reasoning Tasks (Chain-of-Thought):
1. Analyze Frequency: Identify the most frequently visited categories and their associated time slots.
2. Infer Social Role: Based on the schema trajectory, deduce the most likely social role of this user (e.g., University Student, Corporate Employee, Tourist, Nightlife Enthusiast).
3. Generate Profile: Create a descriptive profile that summarizes their lifestyle, activity preferences, and potential future needs.

Output Constraints:
Please output the result in the following JSON format:
{{
    "User_Role": "A concise label for the social role (max 5 words).",
    "Role_Reasoning": "A brief explanation of why this role fits the data.",
    "User_Profile": "A comprehensive paragraph describing the user's behavioral patterns and general preferences based on the inferred role. This text should highlight cross-city transferable characteristics."
}}
"""
        return prompt

    def analyze_user(self, user_id):

        schema_traj = self.generate_schema_trajectory(user_id)
        if not schema_traj:
            return None

        prompt = self.construct_prompt(user_id, schema_traj)

        result_json = None
        if self.llm_predictor:
            response_text = self.llm_predictor(prompt)
            try:

                cleaned_text = response_text.replace("```json", "").replace("```", "").strip()
                result_json = json.loads(cleaned_text)
            except json.JSONDecodeError:
                print(f"Failed to parse JSON for user {user_id}. Raw response: {response_text[:100]}...")
                result_json = {"raw_response": response_text}

        return {
            "user_id": user_id,
            "schema_trajectory": schema_traj,
            "prompt": prompt,
            "analysis_result": result_json
        }

    def process_dataset(self, output_file="user_roles.json", max_users=5):

        if self.data is None:
            self.load_and_preprocess_data()

        unique_users = self.data['user_id'].unique()
        print(f"Found {len(unique_users)} unique users.")

        results = []

        for i, user_id in enumerate(unique_users):
            if max_users and i >= max_users:
                break

            print(f"Processing user {user_id} ({i + 1}/{max_users})...")
            analysis = self.analyze_user(user_id)
            if analysis:
                results.append(analysis)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"Analysis saved to {output_file}")


def mock_llm_inference(prompt):
    """
    模拟 LLM 的返回结果
    """
    return """
    {
        "User_Role": "Nightlife Enthusiast",
        "Role_Reasoning": "The user frequently visits Bars and Nightclubs on Weekends late at night (22:00+), while showing little activity during weekday mornings.",
        "User_Profile": "This user enjoys a vibrant nightlife, preferring social venues like bars and clubs. Their activity peaks during late weekend hours, suggesting a preference for leisure and entertainment over structured daytime routines. Transferable preferences include high-energy social spots and late-night dining options."
    }
    """


# 2. 运行主程序
if __name__ == "__main__":
    if not os.path.exists("nyc.txt"):
        print("请上传 nyc.txt 文件")
    else:

        analyzer = InterCityUserRoleAnalyzer(data_path="nyc.txt", llm_predictor=mock_llm_inference)

        analyzer.process_dataset(output_file="user_roles_output.json", max_users=3)

        with open("user_roles_output.json", "r") as f:
            data = json.load(f)
            if data:
                print("\n--- Generated Prompt Example ---")
                print(data[0]["prompt"])
                print("\n--- Simulated LLM Response ---")
                print(json.dumps(data[0]["analysis_result"], indent=2))