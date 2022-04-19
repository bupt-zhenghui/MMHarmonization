import os
import json
import torch
import torch.utils.data
from torch import nn
from d2l import torch as d2l


def read_data(path):
    img_caption_dic = {}
    with open(path, 'r') as f:
        line = f.readline()
        while line:
            img_name, caption = line.split(':')
            img_caption_dic[img_name] = caption
            line = f.readline()
    return img_caption_dic


class CaptionDataset(torch.utils.data.Dataset):
    def __init__(self, caption_dic, reg_dic, max_len, vocab=None):
        self.max_len = max_len
        self.vocab = vocab
        self.idx2img = {}
        self.all_token_ids, self.all_segments, self.valid_lens = self._preprocess(caption_dic, reg_dic)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx]), self.idx2img[idx]

    def __len__(self):
        return len(self.all_token_ids)

    def _preprocess(self, c_dic, r_dic):
        all_token_ids, all_segments, valid_lens = [], [], []
        for img, caption_sentence in c_dic.items():
            caption_token = d2l.tokenize([caption_sentence.lower()])[0]

            reg_token = d2l.tokenize([r_dic[img].lower()])[0]
            self.idx2img[len(all_token_ids)] = img
            # print(caption_token)

            token_ids, segments, valid_len = self._tokenize(caption_token, reg_token)
            all_token_ids.append(token_ids)
            all_segments.append(segments)
            valid_lens.append(valid_len)
        return (torch.tensor(all_token_ids, dtype=torch.long),
                torch.tensor(all_segments, dtype=torch.long),
                torch.tensor(valid_lens))

    def _truncate_pair_of_tokens(self, p_tokens, h_tokens):
        # 为BERT输入中的'<CLS>'、'<SEP>'和'<SEP>'词元保留位置
        while len(p_tokens) + len(h_tokens) > self.max_len - 3:
            if len(p_tokens) > len(h_tokens):
                p_tokens.pop()
            else:
                h_tokens.pop()

    def _tokenize(self, c_tokens, r_tokens):
        self._truncate_pair_of_tokens(c_tokens, r_tokens)
        tokens, segments = d2l.get_tokens_and_segments(c_tokens, r_tokens)
        token_ids = self.vocab[tokens] + [self.vocab['<pad>']] * (self.max_len - len(tokens))
        segments = segments + [0] * (self.max_len - len(segments))
        valid_len = len(tokens)
        return token_ids, segments, valid_len


class BERTClassifier(nn.Module):
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        self.encoder = bert.encoder
        self.hidden = bert.hidden

    def forward(self, inputs):
        tokens_X, segments_X, valid_lens_x = inputs
        encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)
        return self.hidden(encoded_X)


def load_pretrained_model(pretrained_model, num_hiddens, ffn_num_hiddens,
                          num_heads, num_layers, dropout, max_len, devices):
    data_dir = '../data/bert.small.torch'
    vocab = d2l.Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(data_dir,
                                                     'vocab.json')))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(
        vocab.idx_to_token)}
    bert = d2l.BERTModel(len(vocab), num_hiddens, norm_shape=[256],
                         ffn_num_input=256, ffn_num_hiddens=ffn_num_hiddens,
                         num_heads=4, num_layers=2, dropout=0.2,
                         max_len=max_len, key_size=256, query_size=256,
                         value_size=256, hid_in_features=256,
                         mlm_in_features=256, nsp_in_features=256)
    bert.load_state_dict(torch.load(os.path.join(data_dir,
                                                 'pretrained.params')))
    return bert, vocab


if __name__ == '__main__':
    text_path = './dataset/text/HAdobe5k.txt'
    caption_dic = read_data(text_path)
    # print(len(caption_dic))

    devices = [d2l.try_gpu()]
    bert, vocab = load_pretrained_model(
        'bert.small', num_hiddens=256, ffn_num_hiddens=512, num_heads=4,
        num_layers=2, dropout=0.1, max_len=512, devices=devices)

    batch_size = 32
    dataset = CaptionDataset(caption_dic=caption_dic, reg_dic=caption_dic, max_len=128, vocab=vocab)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    it = next(iter(train_iter))

    net = BERTClassifier(bert)
    output = net(it[0])
    print(output.shape)
