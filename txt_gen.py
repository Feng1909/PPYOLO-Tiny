import os
with open("train.txt", "w") as f:
    a = os.listdir("F:\\datasets\\train")
    for i in a:
        path_1 = "F:\\datasets\\train\\"
        path_2 = "F:\\labels\\"
        line = path_1+str(i)+" "+path_2+str(i[:-3])+'xml\n'
        f.writelines(line)
    tot = 0
    a = os.listdir("F:\\datasets\\val")
    with open("eval.txt", "w") as ff:
        for i in a:
            path_1 = "F:\\datasets\\val\\"
            path_2 = "F:\\labels\\"
            line = path_1+str(i)+" "+path_2+str(i[:-3])+'xml\n'
            ff.writelines(line)