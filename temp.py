import os
import json
import csv
import time
import datetime as dt
path = "/home/workplace/HOnnotate_OXR/hand.json"
ctime = os.path.getctime(path)
dtime = dt.datetime.fromtimestamp(ctime).strftime('%Y-%m-%d %H:%M')
print(dtime)