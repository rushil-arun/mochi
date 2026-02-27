from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-mpnet-base-v2")

expected = "Provide a tutorial on baking a cake."
generated = "Give instructions for how to make a cake."

embeddings = model.encode([expected, generated])
score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

print(score)