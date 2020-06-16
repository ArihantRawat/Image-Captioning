from os import listdir
from pickle import dump, load
from numpy import array,argmax
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model
from nltk.translate.bleu_score import corpus_bleu
from keras.utils import to_categorical
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
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

def get_photo_features(filename):
    # load the model
    model = VGG16()
    # re-structure the model
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    # load the photo
    image = load_img(filename, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # get features
    feature = model.predict(image, verbose=0)
    return feature



def load_doc(filename):
    file = open(filename,'r')
    text= file.read()
    file.close()
    return text

def load_descriptions(doc):
    mapping=dict()
    for line in doc.split('\n')[20460:]:
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


def load_set(filename):
    doc = load_doc(filename)
    dataset = list()
    for line in doc.split('\n'):
        if len(line)<1:
            continue
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return dataset

def load_clean_descriptions(filename, dataset):
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        image_id, image_desc = tokens[0], tokens[1:]
        if image_id in dataset:
            if image_id not in descriptions:
                descriptions[image_id] = list()
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
            descriptions[image_id].append(desc)
    return descriptions

def load_photo_features(filename, dataset):
    all_features = load(open(filename,'rb'))
    features = {k: all_features[k] for k in dataset}
    return features

def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def create_sequences(tokenizer, max_length, descriptions, photos, vocab_size):
    X1, X2, y = list(), list(), list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            seq = tokenizer.texts_to_sequences([desc])[0]
            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen = max_length)[0]
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                X1.append(photos[key][0])
                X2.append(in_seq)
                y.append(out_seq)

    return array(X1),array(X2),array(y)

def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)

def define_model(vocab_size,max_length):
    # feature extractor model
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256,activation='relu')(fe1)
    # sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size,256,mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    # decoder model
    decoder1 = add([fe2,se3])
    decoder2 = Dense(256,activation='relu')(decoder1)
    outputs = Dense(vocab_size,activation='softmax')(decoder2)
    # put it together [image,seq] [word]
    model = Model(inputs=[inputs1,inputs2],outputs=outputs)
    model.compile(loss='categorical_crossentropy',optimizer='adam')
    print(model.summary())
    #plot_model(model,to_file='model.png',show_shapes=True)
    return model


# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
    # seed the generation process
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = model.predict([photo, sequence], verbose=0)
        # convert probability to integer
        yhat = argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    return in_text

# evaluate the skill of the model
def evaluate_model(model, descriptions, photos, tokenizer, max_length):
        actual, predicted = list(), list()
        # step over the whole set
        for key, desc_list in descriptions.items():
            # generate description
            yhat = generate_desc(model, tokenizer, photos[key], max_length)
            # store actual and predicted
            references = [d.split() for d in desc_list]
            actual.append(references)
            predicted.append(yhat.split())
        # calculate BLEU score
        print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
        print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
        print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
        print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

# directory = dir_path + "/dataset/Flickr8k_Dataset"
# features = extract_features(directory)
# print('Extracted Features: %d' % len(features))
# # save to file
# dump(features, open('features.pkl', 'wb'))
#
#
# filename = dir_path + "/dataset/Flickr8k_text/Flickr8k.token.txt"
# # load descriptions
# doc = load_doc(filename)
# # parse descriptions
# descriptions = load_descriptions(doc)
# print("Loaded: ",len(descriptions))
# # clean descriptions
# clean_descriptions(descriptions)
# # summarize vocabulary
# vocabulary = to_vocabulary(descriptions)
# print("Vocabulary Size: ",len(vocabulary))
# # save to file
# save_descriptions(descriptions, 'descriptions.txt')

# # loading training data (6K)
# filename = dir_path + "/dataset/Flickr8k_text/Flickr_8k.trainImages.txt"
# train = load_set(filename)
# print("Dataset: ", len(train))
# # descriptions
# train_descriptions = load_clean_descriptions('descriptions.txt', train)
# print("Descriptions: train= ", len(train_descriptions))
# # photo features
# train_features = load_photo_features('features.pkl', train)
# print("Photos: train= ", len(train_features))
# # prepare tokenizer
# tokenizer = create_tokenizer(train_descriptions)
# dump(tokenizer, open('tokenizer.pkl','wb'))
# vocab_size = len(tokenizer.word_index) + 1
# print('Vocabulary Size: %d' % vocab_size)
# # determine the max sequence length
# max_length = max_length(train_descriptions)
# print('Description Length: %d' % max_length)
# # prepare sequences
# X1train, X2train, ytrain = create_sequences(tokenizer,max_length,train_descriptions,train_features,vocab_size)


# # loading dev data
# filename = dir_path + "/dataset/Flickr8k_text/Flickr_8k.devImages.txt"
# test = load_set(filename)
# print("Dataset: ", len(test))
# # descriptions
# test_descriptions = load_clean_descriptions('descriptions.txt', test)
# print("Descriptions: test= ", len(test_descriptions))
# # photo features
# test_features = load_photo_features('features.pkl', test)
# print("Photos: test= ", len(test_features))
# # prepare sequences
# X1test, X2test, ytest = create_sequences(tokenizer,max_length,test_descriptions,test_features,vocab_size)

# # prepare test set
# # load test set
# filename = dir_path + '/dataset/Flickr8k_text/Flickr_8k.testImages.txt'
# test = load_set(filename)
# print('Dataset: %d' % len(test))
# # descriptions
# test_descriptions = load_clean_descriptions('descriptions.txt', test)
# print('Descriptions: test=%d' % len(test_descriptions))
# # photo features
# test_features = load_photo_features('features.pkl', test)
# print('Photos: test=%d' % len(test_features))

# # fit model
# model = define_model(vocab_size,max_length)
# # define checkpoint callback
# filepath = dir_path + '/model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
# checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# # fit model
# model.fit([X1train,X2train], ytrain, epochs=20, verbose=2, callbacks=[checkpoint], validation_data=([X1test,X2test],ytest))

# # load the model
# filename = dir_path + '/model-ep003-loss3.725-val_loss3.998.h5'
# model = load_model(filename)
# # evaluate model
# evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)

# generate captions for new photos
tokenizer = load(open('tokenizer.pkl','rb'))
max_length = 28
filename = dir_path + '/model-ep003-loss3.725-val_loss3.998.h5'
model = load_model(filename)
image_path = dir_path + '/sample/'
for image in listdir(image_path):
    photo = get_photo_features(image_path + image)
    description = generate_desc(model,tokenizer,photo,max_length)
    print(description)