import torch.utils.data as data
import os.path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import codecs


class Texts(data.Dataset):
    def __init__(self, root, mode='train', transform=None, target_transform=None):
        self.get_tfidf()
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode

        if self.mode == 'train':
            self.data, self.target = self.vectorizer(os.path.join(root, 'train.txt'))
        elif self.mode == 'valid':
            self.data, self.target = self.vectorizer(os.path.join(root, 'valid.txt'))
        else:
            self.data, self.target = self.vectorizer(os.path.join(root, 'test.txt'))
        print(self.data.shape[0], self.target.shape[0])

    @staticmethod
    def escape(string):
        text = string
        for i in "1234567890-=`~!@#$%^&*()_+{}[]:\";\'<>?,./\\|«»—":
            text = text.replace(i, ' ')
        return text

    def get_tfidf(self):
        with codecs.open('./wikitext/train.txt', 'r', 'utf-8') as train, \
                codecs.open('./wikitext/test.txt', 'r', 'utf-8') as test, \
                codecs.open('./wikitext/valid.txt', 'r', 'utf-8') as valid:
            corpus = []
            target = []
            res = ''
            for i in train.readlines() + test.readlines() + valid.readlines():
                if i[0] == "=" or i[1] == "=":
                    corpus.append(" ".join(self.escape(res).split()))
                    res = ''
                    target.append(" ".join(self.escape(i).split()))
                else:
                    res += i
            corpus = corpus[1::]
            corpus.append(" ".join(res.split()))

        self.tfidf = TfidfVectorizer()
        self.tfidf.fit_transform([corpus[i] + target[i] for i in range(len(corpus))])

    def vectorizer(self, path):
        assert os.path.exists(path)
        with codecs.open(path, 'r', 'utf-8') as data:
            input = []
            target = []
            res = ''
            for i in data.readlines():
                if i[0] == "=" or i[1] == "=":
                    input.append(" ".join(self.escape(res).split()))
                    res = ''
                    target.append(" ".join(self.escape(i).split()))
                else:
                    res += i

            data.close()
        input.append(" ".join(res.split()))
        return self.tfidf.transform(input[1::]).toarray(), self.tfidf.transform(target).toarray()

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.data.shape[0]


