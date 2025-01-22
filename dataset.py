import random
import torch
import numpy as np
import copy
import torch.nn.utils.rnn as rnn_utils
from utils import read_peptides_from_file, tokenize


class GeneratorData(object):
    def __init__(self, training_data_path, tokens=None, start_token='<',
                 end_token='>', max_len=40, use_cuda=None):  # max_len=150
        """
        Constructor for the GeneratorData object.

        Parameters
        ----------
        training_data_path: str
            path to file with training dataset. Training dataset must contain
            a column with training strings. The file also may contain other
            columns.

        tokens: list (default None)
            list of characters specifying the language alphabet. Of left
            unspecified, tokens will be extracted from data automatically.

        start_token: str (default '<')
            special character that will be added to the beginning of every
            sequence and encode the sequence start.

        end_token: str (default '>')
            special character that will be added to the end of every
            sequence and encode the sequence end.

        max_len: int (default 150)
            maximum allowed length of the sequences. All sequences longer than
            max_len will be excluded from the training data.

        use_cuda: bool (default None)
            parameter specifying if GPU is used for computations. If left
            unspecified, GPU will be used if available

        """
        super(GeneratorData, self).__init__()

        data, read_successful = read_peptides_from_file(training_data_path)

        assert read_successful

        self.start_token = start_token
        self.end_token = end_token

        self.peptide_list = []
        for i in range(len(data)):
            if len(data[i]) <= max_len:

                self.peptide_list.append(data[i])

        self.peptide_num = len(self.peptide_list)

        self.all_characters, self.char2idx, self.char_num = tokenize(self.peptide_list, tokens)
        self.use_cuda = use_cuda
        if self.use_cuda is None:
            self.use_cuda = torch.cuda.is_available()

        self.idx = 0

    def __len__(self):
        return self.peptide_num


    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def reset(self):
        self.idx = 0
        random.shuffle(self.peptide_list)

    def next(self):


        chunk = self.peptide_list[self.idx]
        inp = self.char_tensor(chunk[:-1])
        target = self.char_tensor(chunk[1:])
        self.idx += 1
        if self.idx >= self.peptide_num:
            self.reset()

        if self.use_cuda:
            return inp.cuda(), target.cuda()
        else:
            return inp, target

    def load_dictionary(self, tokens, char2idx):
        self.all_characters = tokens
        self.char2idx = char2idx
        self.char_num = len(tokens)

    def char_tensor(self, string):

        tensor = torch.zeros(len(string)).long()
        for c in range(len(string)):
            # print('序列.....',string)
            tensor[c] = self.all_characters.index(string[c].lower())
        if self.use_cuda:
            return tensor.cuda()
        else:
            return tensor

    def get_random(self, num):
        data = []
        l = 0
        while (l < num):
            ran_idx = random.randint(0, len(self) - 1)
            sample = self.peptide_list[ran_idx][1:-1]
            data.append(sample)
            l = l + 1
        return data


class DiscriminatorData(object):
    def __init__(self, truth_data, fake_data, tokens=None, batch_size=64, use_cuda=None):
        """
        Constructor for the GeneratorData object.

        Parameters
        ----------
        truth_data: list
            list of truth_data. Every element is a peptides string.

        fake_data: list
            list of fake_data. Every element is a peptides string.


        tokens: list (default None)
            list of characters specifying the language alphabet. Of left
            unspecified, tokens will be extracted from data automatically.

        batch_size: int (default 64)
            size of learning batch

        use_cuda: bool (default None)
            parameter specifying if GPU is used for computations. If left
            unspecified, GPU will be used if available

        """
        super(DiscriminatorData, self).__init__()

        self.truth_data = truth_data
        self.fake_data = fake_data
        self.all_characters, self.char2idx, self.char_num = tokenize(None, tokens)
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        if self.use_cuda is None:
            self.use_cuda = torch.cuda.is_available()

        self.pairs = []

        self.pairs = self.pairs + list(
            zip([self.char_tensor(peptides) for peptides in self.truth_data], [1 for _ in range(len(self.truth_data))])) \
                     + list(
            zip([self.char_tensor(peptides) for peptides in self.fake_data], [0 for _ in range(len(self.fake_data))]))
        self.reset()
        self.indices = range(len(self.pairs))

    def __len__(self):
        return len(self.pairs)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def reset(self):
        self.idx = 0
        random.shuffle(self.pairs)

    def next(self):
        """
        Converts peptides into tensor of indices wrapped into torch.autograd.Variable.
        Args:
            string (str): input peptides string
        Returns:
            tokenized_string (torch.autograd.Variable(torch.tensor))
        """

        if self.idx >= len(self.pairs):
            raise StopIteration
        index = self.indices[self.idx:self.idx + self.batch_size]
        pairs = [self.pairs[i] for i in index]
        data, data_length, label = self.collate_fn(pairs)
        self.idx += self.batch_size
        return data, data_length, label

    def char_tensor(self, string):
        """
        Converts peptides into tensor of indices wrapped into torch.tensor.
        Args:
            string (str): input peptides string
        Returns:
            tokenized_string (torch.tensor)
        """
        tensor = torch.zeros(len(string)).long()
        for c in range(len(string)):
            tensor[c] = self.all_characters.index(string[c].lower())
        if self.use_cuda:
            return tensor.cuda()
        else:
            return tensor

    def update(self, truth_data=None, fake_data=None):
        """
        Constructor for the GeneratorData object.

        Parameters
        ----------
        truth_data: list
            list of truth_data. Every element is a peptides string.

        fake_data: list
            list of fake_data. Every element is a peptides string.

        Return
        ---------
        None

        """
        if truth_data is not None:
            self.truth_data = copy.deepcopy(truth_data)
        if fake_data is not None:
            self.fake_data = copy.deepcopy(fake_data)

        self.pairs = []

        self.pairs = self.pairs + list(
            zip([self.char_tensor(peptides) for peptides in self.truth_data], [1 for _ in range(len(self.truth_data))])) \
                     + list(
            zip([self.char_tensor(peptides) for peptides in self.fake_data], [0 for _ in range(len(self.fake_data))]))
        self.reset()
        self.indices = range(len(self.pairs))

    def collate_fn(self, pairs):
        batch_size = len(pairs)
        pairs.sort(key=lambda x: len(x[0]), reverse=True)
        data = [pair[0].cpu() for pair in pairs]
        data_length = [len(d) for d in data]
        if self.use_cuda:
            label = torch.tensor([pair[1] for pair in pairs]).float().view(batch_size, 1).cuda()
            data = torch.LongTensor(rnn_utils.pad_sequence(data, batch_first=True, padding_value=0)).cuda()
            data_length = torch.Tensor(data_length).cuda()
        else:
            label = torch.tensor([pair[1] for pair in pairs]).float().view(batch_size, 1)
            data = torch.LongTensor(rnn_utils.pad_sequence(data, batch_first=True, padding_value=0))
            data_length = torch.Tensor(data_length)
        return data, data_length, label