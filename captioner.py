from os import listdir
from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import os 
import string

dir_path = os.path.dirname(os.path.realpath(__file__))

def extract_features(directory):
    model = VGG16()
    model.layers.pop()
    model=Model(inputs=model.input, outputs=model.layers[-1].output)

    print(model.summary)

    features = dict()

    for name in listdir(directory):
        filename = directory + "/" + name
        image = load_img(filename, target_size=(224,224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        
        feature = model.predict(image, verbose=0)
        image_id = name.split('.')[0]

        features[image_id] = feature
        print('>%s' % name)

    return features


def load_doc(filename):
    file = open(filename,'r')
    text= file.read()
    file.close()
    return text

def load_descriptions(doc):
    mapping=dict()

    for line in doc.split('\n'):
        tokens = line.split()
        if len(tokens)<2:
            continue

        image_id, image_desc = tokens[0], tokens[1:]
        image_id = image_id.split('.')[0]

        image_desc = ' '.join(image_desc)

        if image_id not in mapping:
            mapping[image_id] = list()

        mapping[image_id].append(image_desc)

    return mapping

def clean_descriptions(descriptions):

    table = str.maketrans('','', string.punctuation)
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):

            desc = desc_list[i]
            desc = desc.split()
            desc = [word.lower() for word in desc]
            desc = [w.translate(table) for w in desc]
            desc = [word for word in desc if len(word)>1]
            desc = [word for word in desc if word.isalpha()]
            desc_list[i] = ' '.join(desc)

def to_vocabulary(descriptions):
    all_desc = set()
    
    for key in descriptions.keys():
        [all_desc.update(d.split()) for d in descriptions[key]]
    
    return all_desc

def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + ' ' + desc)
    
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()





'''
directory = dir_path + "/dataset/Flicker8k_Dataset"
features = extract_features(directory)
print('Extracted Features: %d' % len(features))
# save to file
dump(features, open('features.pkl', 'wb'))
'''

filename = dir_path + "/dataset/Flickr8k_text/Flickr8k.token.txt"
# load descriptions
doc = load_doc(filename)
# parse descriptions
descriptions = load_descriptions(doc)
print('Loaded: %d ' % len(descriptions))
# clean descriptions
clean_descriptions(descriptions)
# summarize vocabulary
vocabulary = to_vocabulary(descriptions)
print('Vocabulary Size: %d' % len(vocabulary))
# save to file
save_descriptions(descriptions, 'descriptions.txt')