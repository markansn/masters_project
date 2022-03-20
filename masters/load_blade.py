import os
import csv
import datetime
print("t")
X = []
meta = []
y = []
goodware_meta = []
badware_meta = []
months = ['Jan', 'Feb', 'Mar', 'Apr', "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
with open('../blade/AndroAutopsy/goodware_meta.csv', mode='r') as f:
	reader = csv.reader(f, delimiter=":")
	for row in reader:
		try:
			row[7] = row[7].strip()
			a = {}
			a["sha256"] = row[1].strip()
			year = row[7].split(" ")[1].strip()
			monthday = row[7].split(",")[0].split(" ")
			month = months.index(monthday[0]) + 1
			day = monthday[1]
			a["year"] = year
			a["month"] = month
			a["day"] = day
			goodware_meta.append(a)
		except:
			pass
		# goodware_meta = {rows[1]: rows[7] for rows in reader}
print(goodware_meta)
with open('../blade/AndroAutopsy/badware_meta.csv', mode='r') as f:
	reader = csv.reader(f, delimiter=",")
	for row in reader:
		try:
			badware_meta[row[5]] = row[7]
		except:
			pass
		# goodware_meta = {rows[1]: rows[7] for rows in reader}


# print(goodware_meta)
# print(badware_meta)
for root, dirs, files in os.walk("../blade/AndroAutopsy/badware"):
	for file in files:
		sha = file.split(".")[0]
		if sha not in badware_meta:
			print(sha + " not found")
