
from collections import defaultdict
from vector_space_model import Vector_Space_Model
model = Vector_Space_Model("./corpus/*")
f = model.golbal_terms_frequency
print(model.query("couple"))
print(model.documents)



