from Classification.InstanceList.InstanceList import InstanceList
from Classification.Instance.Instance import Instance
from Classification.Model.RuleBased.PartModel import PartModel


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


# ---- run PART on iris ----

trainSet = load_dataset_discrete("datasets/car.data")
print("Train size:", trainSet.size())

model = PartModel()
model.train(trainSet)


print("Number of rules:", len(model.rules))
print("Accuracy:", model.test(trainSet).getAccuracy())
model.printRules()

