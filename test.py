from Classification.InstanceList.InstanceList import InstanceList
from Classification.Instance.Instance import Instance
from Classification.Model.RuleBased.PartModel import PartModel

# Loading dataset
def load_dataset_discrete(path):
    instance_list = InstanceList()

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(",")
            label = parts[-1]
            values = parts[:-1]

            instance = Instance(label)
            for v in values:
                instance.addDiscreteAttribute(v)

            instance_list.add(instance)

    return instance_list

def load_dataset_continuous(path):
    instance_list = InstanceList()

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(",")
            label = parts[-1]
            values = parts[:-1]

            instance = Instance(label)
            for v in values:
                instance.addContinuousAttribute(float(v))

            instance_list.add(instance)

    return instance_list


# run PART on dataset

trainSet = load_dataset_discrete("datasets/car.data")
print("Train size:", trainSet.size())

model = PartModel()
model.train(trainSet)

# Print and save the rules
print("Number of rules:", len(model.rules))
print("Accuracy:", model.test(trainSet).getAccuracy())
model.printRules()

with open("car_rules.txt", "w", encoding="utf-8") as f:
    for i, rule in enumerate(model.rules, 1):
        if rule.conditions:
            conds = " AND ".join(
                model._condition_to_string(c) for c in rule.conditions
            )
            f.write(f"Rule {i}: IF {conds} THEN class = {rule.label}\n")
        else:
            f.write(f"Rule {i}: IF TRUE THEN class = {rule.label}\n")

