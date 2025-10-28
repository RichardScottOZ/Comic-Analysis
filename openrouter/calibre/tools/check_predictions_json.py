
import os
import json
import pandas as pd

with open(r'E:\CalibreComics\test_dections\predictions.json', 'r', encoding='utf-8') as f:
    predictions = json.load(f)

print(len(predictions))
print(predictions.keys())
#quit()
for key in predictions:
    print(key, len(predictions[key]))
    print(predictions[key][0:3])
    #break

list25 = []
for i in predictions['images']:
    if '25th' in i['file_name']:
        pass
    list25.append(i)


df = pd.DataFrame(list25)
df.to_csv('images_25th.csv', index=False)
        #break



