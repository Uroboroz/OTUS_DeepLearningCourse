import os
import torch
import torch.utils.data as data
import codecs

class Alphabet(object):
    def __init__(self):
        self.symbol2idx = {}
        self.idx2symbol = []
        self._len = 0
        
    def add_symbol(self, s):
        if s not in self.symbol2idx:
            self.idx2symbol.append(s)
            self.symbol2idx[s] = self._len
            self._len += 1
    
    def __len__(self):
        return self._len

    
class Texts(object):
    def __init__(self, path):
        self.dictionary = Alphabet()
        self.get = self.tokenize(os.path.join(path))


    def tokenize(self, path):
        """Tokenizes a text file."""
        print(path)
        assert os.path.exists(path)
        # Add symbol to the dictionary
        with codecs.open(path, 'r', 'utf-8') as f:
            tokens = 0
            for line in f:
                tokens += len(line)
                for s in line:
                    self.dictionary.add_symbol(s)

        # Tokenize file content
        with codecs.open(path, 'r', 'utf-8') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                for s in line:
                    ids[token] = self.dictionary.symbol2idx[s]
                    token += 1

        return ids
    
class TextLoader(data.Dataset):
    def __init__(self, mode, sequence_length=30):
        self.data = Texts('./wikitext/' + mode + '.txt').get
        self.sequence_length = sequence_length
        
    def __getitem__(self, i):
        seq_len = min(self.sequence_length, len(self.data) - 1 - i)
        data = self.data[i:i + seq_len]
        target = self.data[i + 1:i + 1 + seq_len].view(-1)
        return data, target

    def __len__(self):
        return int(self.data.shape[0])//self.sequence_length


