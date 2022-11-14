
from vector_space_model import Vector_Space_Model

model = Vector_Space_Model("./corpus/*")
print(model.postings['coupl'])
print(model.documents)
print(model.vocabulary)

