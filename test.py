from Classification.Model import PartModel
from Classification.Model import DummyModel

model = PartModel()
model.train(None)
print(model.predict(None))

