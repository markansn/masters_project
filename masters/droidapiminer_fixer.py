import os
# with open ("../droidapiminer_data/old-apg-droidapiminer-meta.json") as f:
#     s = f.read()
#     s = s.replace("u'", "\"")
#     s = s.replace("'", "\"")
#
#     a = open("../droidapiminer_data/apg-droidapiminer-meta.json", "w")
#     a.write(s)
#
#     a.close()

for item in os.listdir("../droidapiminer_data/x-files/"):
    with open ("../droidapiminer_data/x-files/" + item) as f:
        s = f.read()
        s = s.replace("u'", "\"")
        s = s.replace("'", "\"")
        s = s.replace("set(", "")
        s = s[:-1]
        a = open("../droidapiminer_data/fixed-x-files2/" + item, "w+")
        a.write(s)

        a.close()
