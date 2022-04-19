import os
import csv
import datetime
import time
import json
print("t")
X = []
meta = []
y = []

months = ['Jan', 'Feb', 'Mar', 'Apr', "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
with open('../blade/AndroAutopsy/goodware_meta.csv', mode='r') as f:
	reader = csv.reader(f, delimiter=":")
	for row in reader:
		try:
			row[7] = row[7].strip()
			a = {}
			a["sha256"] = row[1].strip()

			year = row[7].split(" ")[2].strip()
			monthday = row[7].split(",")[0].split(" ")
			month = months.index(monthday[0]) + 1
			day = monthday[1]

			a["year"] = str(year)
			a["month"] = str(month)
			a["day"] = str(day)
			d = datetime.datetime(int(year), int(month), int(day))
			a["dex_date"] = int(time.mktime(d.timetuple()))
			meta.append(a)
		except Exception as e:
			print("goodware err")
		# goodware_meta = {rows[1]: rows[7] for rows in reader}
# print(goodware_meta)
with open('../blade/AndroAutopsy/badware_meta.csv', mode='r') as l:
	reader = csv.reader(l, delimiter=",")
	for row in reader:
		# print(row)
		try:
			a = {}
			d = datetime.datetime.strptime(row[7].split(" ")[0], "%Y-%m-%d")
			a["sha256"] = row[5]
			a["year"] = str(d.year)
			a["month"] = str(d.month)
			a["day"] = str(d.day)
			a["dex_date"] = int(time.mktime(d.timetuple()))
			meta.append(a)

		except:
			print("badware err")
		# goodware_meta = {rows[1]: rows[7] for rows in reader}

# print(badware_meta)
# f = open("../blade/AA/badware_meta.json", "w+")
# json.dump(badware_meta, f)
# f.close()



f = open("../blade/AA/autospy_meta .json", "w+")
json.dump(meta, f)
f.close()
# print(goodware_meta)
# print(badware_meta)
# for root, dirs, files in os.walk("../blade/AndroAutopsy/badware"):
# 	for file in files:
# 		sha = file.split(".")[0]
# 		if sha not in badware_meta:
# 			print(sha + " not found")
