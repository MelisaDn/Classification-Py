from Classification.Model.Model import Model
from Classification.Instance.Instance import Instance
from Classification.InstanceList.InstanceList import InstanceList
from Classification.Model.DecisionTree.DecisionTree import DecisionTree
from Classification.Parameter.C45Parameter import C45Parameter
from Classification.Attribute.ContinuousAttribute import ContinuousAttribute
from Classification.Attribute.DiscreteAttribute import DiscreteAttribute
from Classification.Attribute.DiscreteIndexedAttribute import DiscreteIndexedAttribute



class PartModel(Model):

    class Rule:
        def __init__(self, conditions, label):
            self.conditions = conditions
            self.label = label

        def matches(self, instance: Instance) -> bool:
            for condition in self.conditions:
                if not condition.satisfy(instance):
                    return False
            return True

    def __init__(self):
        self.rules = []
        self.defaultClass = None

    def _extract_rule_from_tree(self, root):
        """
        Extracts a single rule from a decision tree
        by following one root-to-leaf path.
        """
        conditions = []
        node = root

        while not node.leaf:
            child = node.children[0]  # simple strategy
            conditions.append(child._DecisionNode__condition)
            node = child

        label = node._DecisionNode__class_label
        return self.Rule(conditions, label)

    def _remove_covered(self, trainSet: InstanceList, rule):
        """
        Removes instances covered by the given rule.
        """
        newSet = InstanceList()
        for i in range(trainSet.size()):
            instance = trainSet.get(i)
            if not rule.matches(instance):
                newSet.add(instance)
        return newSet
    
    def _condition_to_string(self, condition):
        attr = condition._DecisionCondition__attribute_index
        value = condition._DecisionCondition__value
        comp = condition._DecisionCondition__comparison

        # Discrete attribute
        if isinstance(value, DiscreteAttribute):
            return f"att[{attr}] = {value.getValue()}"

        # Discrete indexed attribute
        if isinstance(value, DiscreteIndexedAttribute):
            idx = value.getIndex()
            if idx == -1:
                return f"att[{attr}] = *"
            return f"att[{attr}] = {idx}"

        # Continuous attribute
        if isinstance(value, ContinuousAttribute):
            return f"att[{attr}] {comp} {value.getValue()}"

        return "UNKNOWN_CONDITION"


    def train(self, trainSet: InstanceList, parameters=None):

        self.rules = []

        # default class = majority class
        distribution = trainSet.classDistribution()
        self.defaultClass = distribution.getMaxItem()

        currentSet = trainSet

        while currentSet.size() > 0:
            tree = DecisionTree()
            tree.train(
                currentSet,
                C45Parameter(seed=1, prune=True, crossValidationRatio=0.2)
            )

            rule = self._extract_rule_from_tree(tree._DecisionTree__root)
            self.rules.append(rule)

            newSet = self._remove_covered(currentSet, rule)

            # safety check to avoid infinite loop
            if newSet.size() == currentSet.size():
                break

            currentSet = newSet

    def predict(self, instance: Instance) -> str:
        for rule in self.rules:
            if rule.matches(instance):
                return rule.label
        return self.defaultClass

    def predictProbability(self, instance: Instance) -> dict:
        return {self.predict(instance): 1.0}

    def loadModel(self, fileName: str):
        pass

    def printRules(self):
        for i, rule in enumerate(self.rules, 1):
            if rule.conditions:
                conds = " AND ".join(
                    self._condition_to_string(c) for c in rule.conditions
                )
                print(f"Rule {i}: IF {conds} THEN class = {rule.label}")
            else:
                print(f"Rule {i}: IF TRUE THEN class = {rule.label}")

