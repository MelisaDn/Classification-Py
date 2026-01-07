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


# run PART and choose the dataset loader based on the dataset
dataset = "winequality"
trainSet = load_dataset_continuous(f"datasets/{dataset}.data")
print("Train size:", trainSet.size())

train = InstanceList()
test = InstanceList()

for i in range(trainSet.size()):
    if i % 5 == 0:      # 20%
        test.add(trainSet.get(i))
    else:               # 80%
        train.add(trainSet.get(i))

model = PartModel()
model.train(train)

print("Train accuracy:", model.test(train).getAccuracy())
print("Test accuracy:", model.test(test).getAccuracy())
# model.printRules()
model.saveRules(f"partRules/{dataset}_rules.txt")
