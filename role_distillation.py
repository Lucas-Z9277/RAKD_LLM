import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import os
from tqdm import tqdm


try:
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE_TRANSFORMER = True
except ImportError:
    HAS_SENTENCE_TRANSFORMER = False
    print("Warning: sentence-transformers not installed. Using mock embeddings.")


class RoleDistillationModule(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dim=256):
        super(RoleDistillationModule, self).__init__()

        self.student_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)  # 输出维度需与 Teacher 对齐
        )

        self.distill_loss_fn = nn.MSELoss()

    def forward(self, current_user_embedding, teacher_embedding):

        student_repr = self.student_encoder(current_user_embedding)

        loss = self.distill_loss_fn(student_repr, teacher_embedding)
        return loss, student_repr


class RoleDistillationRetrievalAugment:
    def __init__(self, roles_path, prefs_path, embedding_model_name='all-MiniLM-L6-v2', device='cpu'):

        self.device = device
        self.roles_data = self._load_json(roles_path)
        self.prefs_data = self._load_json(prefs_path)
        self.user_db = {}  # {user_id: {'role_vec': tensor, 'pref_vec': tensor, 'history': list}}


        if HAS_SENTENCE_TRANSFORMER:
            self.emb_model = SentenceTransformer(embedding_model_name).to(device)
        else:
            self.emb_model = None

    def _load_json(self, path):
        if not os.path.exists(path):
            print(f"Warning: {path} not found.")
            return []
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _get_text_embedding(self, text):

        if not text:
            return torch.zeros(768).to(self.device)  # 假设 dim=768

        if self.emb_model:
            with torch.no_grad():
                emb = self.emb_model.encode(text, convert_to_tensor=True)
            return emb
        else:
            # Mock
            return torch.randn(768).to(self.device)

    def build_index(self):

        print("Building retrieval index...")

        pref_map = {item['user_id']: item for item in self.prefs_data}

        for role_item in tqdm(self.roles_data, desc="Encoding Users"):
            user_id = role_item['user_id']

            role_result = role_item.get('analysis_result', {})
            role_text = f"{role_result.get('User_Role', '')}. {role_result.get('User_Profile', '')}"

            pref_item = pref_map.get(user_id, {})
            pref_text = pref_item.get('final_long_term_preference', '')

            role_vec = self._get_text_embedding(role_text)
            pref_vec = self._get_text_embedding(pref_text)

            self.user_db[user_id] = {
                'role_vec': role_vec,
                'pref_vec': pref_vec,
                'combined_vec': torch.cat([role_vec, pref_vec], dim=0) if role_vec.dim() > 0 else role_vec,  # 简单拼接策略
                'raw_role': role_text
            }

        print(f"Index built for {len(self.user_db)} users.")

    def retrieve_similar_roles(self, query_user_id, top_k=5):

        if query_user_id not in self.user_db:
            return [], []

        query_vec = self.user_db[query_user_id]['role_vec']  # 使用 Role 向量进行检索

        scores = []
        user_ids = []

        for uid, data in self.user_db.items():
            if uid == query_user_id:
                continue

            target_vec = data['role_vec']
            score = F.cosine_similarity(query_vec.unsqueeze(0), target_vec.unsqueeze(0))
            scores.append(score.item())
            user_ids.append(uid)

        scores = torch.tensor(scores)
        if len(scores) == 0:
            return [], []

        top_k = min(top_k, len(scores))
        top_scores, top_indices = torch.topk(scores, top_k)

        top_users = [user_ids[i] for i in top_indices]
        return top_users, top_scores.tolist()

    def get_teacher_signal(self, neighbor_ids):

        if not neighbor_ids:
            return None

        role_vecs = []
        pref_vecs = []

        for uid in neighbor_ids:
            data = self.user_db[uid]
            role_vecs.append(data['role_vec'])
            pref_vecs.append(data['pref_vec'])

        avg_role = torch.stack(role_vecs).mean(dim=0)
        avg_pref = torch.stack(pref_vecs).mean(dim=0)

        teacher_vec = torch.cat([avg_role, avg_pref], dim=0)
        return teacher_vec

    def generate_candidates(self, neighbor_ids):

        candidates = []

        return neighbor_ids



def main():
    roles_path = "user_roles_output.json"
    prefs_path = "intra_city_preferences.json"

    if not os.path.exists(roles_path) or not os.path.exists(prefs_path):
        print("请先运行 5.1 和 5.2 的代码生成 json 文件。")
        print("Generating dummy data for demonstration...")
        dummy_roles = [{"user_id": 1, "analysis_result": {"User_Role": "Student", "User_Profile": "Likes libraries"}},
                       {"user_id": 2, "analysis_result": {"User_Role": "Student", "User_Profile": "Likes coffee"}},
                       {"user_id": 3, "analysis_result": {"User_Role": "Worker", "User_Profile": "Likes bars"}}]
        with open(roles_path, 'w') as f: json.dump(dummy_roles, f)

        dummy_prefs = [{"user_id": 1, "final_long_term_preference": "Study hard"},
                       {"user_id": 2, "final_long_term_preference": "Study and drink coffee"},
                       {"user_id": 3, "final_long_term_preference": "Work hard play hard"}]
        with open(prefs_path, 'w') as f: json.dump(dummy_prefs, f)

    rdra = RoleDistillationRetrievalAugment(roles_path, prefs_path)
    rdra.build_index()

    model = RoleDistillationModule(embedding_dim=1536, hidden_dim=512)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    target_user_id = 1
    print(f"\nProcessing User {target_user_id}...")

    neighbors, scores = rdra.retrieve_similar_roles(target_user_id, top_k=2)
    print(f"Retrieved Neighbors: {neighbors}, Scores: {scores}")

    teacher_vec = rdra.get_teacher_signal(neighbors)

    if teacher_vec is not None:

        student_input = torch.randn(1536)
        optimizer.zero_grad()
        loss, student_output = model(student_input, teacher_vec)
        loss.backward()
        optimizer.step()

        print(f"Distillation Loss: {loss.item():.4f}")

        candidates = rdra.generate_candidates(neighbors)
        print(f"Generated Candidates (Neighbor IDs): {candidates}")
        print("You can now use these candidates to construct the Fine-tuning Prompt (Section 5.4).")


if __name__ == "__main__":
    main()