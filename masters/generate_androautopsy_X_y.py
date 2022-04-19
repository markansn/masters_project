import os
import json
import time
import tqdm

def get_data_from_file(main_root, classification, data):
    for root, dirs, files in os.walk(main_root):
        for file in files:
            if "._" not in file and ".opseq" in file:
                # print(root + file)
                f = open(root + "/" + file, "r")
                lines = f.readlines()
                f.close()
                ops = {}
                for line in lines:
                    l = line.strip()
                    if l in ops:
                        ops[l] += 1
                    else:
                        ops[l] = 1
                file_name = file.split(".")[0]
                data[file_name] = [ops, classification]




def create_X_y():
    f = open("../blade/AA/X_y.json", "r")
    X_y = json.load(f)
    f.close()

    f = open("../blade/AA/autopsy_meta_old.json", "r")
    meta = json.load(f)
    f.close()

    X = []
    y = []
    meta_out = []
    for item in tqdm.tqdm(meta):
        sha = item["sha256"]
        if sha not in X_y:
            print("!! " + sha + " not in X_y")
        else:
            # print(X_y[sha])
            X.append(X_y[sha][0])
            y.append(X_y[sha][1])
            meta_out.append(item)

    f = open("../blade/AA/autopsy_X.json", "w+")
    json.dump(X, f)
    f.close()
    f = open("../blade/AA/autopsy_y.json", "w+")
    json.dump(y, f)
    f.close()
    f = open("../blade/AA/autopsy_meta_final.json", "w+")
    json.dump(meta_out, f)
    f.close()



# data = {}
#
# get_data_from_file("../blade/AndroAutopsy/badware/", 1, data)
# get_data_from_file("../blade/AndroAutopsy/goodware/", 0, data)
#
# f = open("../blade/AA/X_y.json", "w+")
# json.dump(data, f)
# f.close()
#
# time.sleep(5)
print("X_Y")
create_X_y()