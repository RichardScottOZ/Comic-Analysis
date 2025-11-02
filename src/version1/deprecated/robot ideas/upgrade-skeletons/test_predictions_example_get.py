import json

# Load the predictions file
with open('E:\\CalibreComics\\test_dections\\predictions.json', 'r') as f:
    predictions = json.load(f)

print(predictions.keys())

#for key in predictions:
    #print(key, predictions[key][0:3])

count = 0
for idx, l in enumerate(predictions['images']):
    count += 1
    #print('checking for abe')
    #print(l)
    if 'Abe Sapien 1 Dark and Terrible - Unknown_00023.jpg' in l['id']:
        print(l['id'])
        print(predictions['annotations'][idx])
        break

    if count > 2000:
        break
print(count)

count = 0
for idx, l in enumerate(predictions['annotations']):
    count += 1
    #print('checking for abe')
    if 'Abe Sapien 1 Dark and Terrible - Unknown_00023.jpg' in l['image_id']:
        print(l)

        print(l['id'])

    if idx > 100000:
        break

print(predictions['categories'])    
