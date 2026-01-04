from Classification.Model.Model import Model
from Classification.Instance.Instance import Instance
from Classification.Parameter.Parameter import Parameter

class PartModel(Model):

    class Rule:
        def __init__(self, conditions, label):
            self.conditions = conditions
            self.label = label

        def matches(self, instance):
            return True


    def __init__(self):
        self.rules = []
        self.defaultClass = None


    def train(self, trainSet, parameters=None):
        self.defaultClass = "UNKNOWN"
        dummy_rule = self.Rule(conditions=[], label="DUMMY_CLASS")
        self.rules = [dummy_rule]


    def predict(self, instance):
        for rule in self.rules:
            if rule.matches(instance):
                return rule.label
        return self.defaultClass


    def predictProbability(self, instance: Instance) -> dict:
        return {self.predict(instance): 1.0}


    def loadModel(self, fileName: str):
        pass
