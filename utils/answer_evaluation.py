

from sentence_transformers import SentenceTransformer, util

sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

def answer_quality_score(candidate_answer, expected_answer):
    if not candidate_answer or not expected_answer:
        return 0.0
    emb1 = sbert_model.encode(candidate_answer, convert_to_tensor=True)
    emb2 = sbert_model.encode(expected_answer, convert_to_tensor=True)
    sim_score = util.pytorch_cos_sim(emb1, emb2).item()
    return sim_score