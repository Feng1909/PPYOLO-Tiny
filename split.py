with open("train_tot.txt", "r") as f:
    train_all = f.readlines()
    train_split = []
    num = len(train_all)
    tot = 0
    for i in train_all:
        if tot % 100 == 0:
            train_split.append(i)
        tot += 1
    with open("train.txt", "w") as f2:
        for i in train_split:
            f2.writelines(i)
with open("eval_tot.txt", "r") as f:
    train_all = f.readlines()
    train_split = []
    num = len(train_all)
    tot = 0
    for i in train_all:
        if tot % 100 == 0:
            train_split.append(i)
        tot += 1
    with open("eval.txt", "w") as f2:
        for i in train_split:
            f2.writelines(i)