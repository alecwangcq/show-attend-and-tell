import torch
from torch.autograd import Variable
import torch.nn as nn


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

class EncoderVGG(nn.Module):
    def __init__(self, model_path=None):
        super(EncoderVGG, self).__init__()
        if model_path is None:
            vgg = models.vgg16(pretrained=True)
            self._vgg_extractor = nn.Sequential(*(vgg.features[i] for i in xrange(29)))
        else:
            self._vgg_extractor = torch.load(model_path)
            
    def forward(self, x):
        return self._vgg_extractor(x)

class Decoder(nn.Module):

    def __init__(self, vis_dim, vis_num, embed_dim, hidden_dim, vocab_size, num_layers=1, dropout_ratio=0.5):
        super(Decoder, self).__init__()

        self.embed_dim = embed_dim
        self.vis_dim = vis_dim
        self.vis_num = vis_num
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dropout_ratio = dropout_ratio

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout_ratio) if dropout_ratio < 1 else None
        self.lstm_cell = nn.LSTMCell(embed_dim + vis_dim, hidden_dim, num_layers)
        self.fc_dropout = nn.Dropout(dropout_ratio) if dropout_ratio < 1 else None
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

        # attention
        self.att_vw = nn.Linear(self.vis_dim, self.vis_dim, bias=False)
        self.att_hw = nn.Linear(self.hidden_dim, self.vis_dim, bias=False)
        self.att_bias = nn.Parameter(torch.zeros(vis_num))
        self.att_w = nn.Linear(self.vis_dim, 1, bias=False)

    def _attention_layer(self, features, hiddens):
        """
        :param features:  batch_size  * 196 * 512
        :param hiddens:  batch_size * hidden_dim
        :return:
        """
        att_fea = self.att_vw(features)
        # N-L-D
        att_h = self.att_hw(hiddens).unsqueeze(1)
        # N-1-D
        att_full = nn.ReLU()(att_fea + att_h + self.att_bias.view(1, -1, 1))
        att_out = self.att_w(att_full).squeeze(2)
        alpha = nn.Softmax()(att_out)
        # N-L
        context = torch.sum(features * alpha.unsqueeze(2), 1)
        return context, alpha

    def forward(self, features, captions, lengths):
        """
        :param features: batch_size * 196 * 512
        :param captions: batch_size * time_steps
        :param lengths:
        :return:
        """
        batch_size, time_step = captions.data.shape
        vocab_size = self.vocab_size
        embed = self.embed
        dropout = self.dropout
        attention_layer = self._attention_layer
        lstm_cell = self.lstm_cell
        fc_dropout = self.fc_dropout
        fc_out = self.fc_out

        word_embeddings = embed(captions)
        word_embeddings = dropout(word_embeddings) if dropout is not None else word_embeddings
        feas = torch.mean(features, 1)  # batch_size * 512
        h0, c0 = self.get_start_states(batch_size)

        predicts = to_var(torch.zeros(batch_size, time_step, vocab_size))

        for step in xrange(time_step):
            batch_size = sum(i >= step for i in lengths)
            if step != 0:
                feas, alpha = attention_layer(features[:batch_size, :], h0[:batch_size, :])
            words = (word_embeddings[:batch_size, step, :]).squeeze(1)
            inputs = torch.cat([feas, words], 1)
            h0, c0 = lstm_cell(inputs, (h0[:batch_size, :], c0[:batch_size, :]))
            outputs = fc_out(fc_dropout(h0)) if fc_dropout is not None else fc_out(h0)
            predicts[:batch_size, step, :] = outputs

        return predicts

    def sample(self, feature, max_len=20):
        # greedy sample
        embed = self.embed
        lstm_cell = self.lstm_cell
        fc_out = self.fc_out
        attend = self._attention_layer
        batch_size = feature.size(0)

        sampled_ids = []
        alphas = [0]

        words = embed(to_var(torch.ones(batch_size, 1).long())).squeeze(1)
        h0, c0 = self.get_start_states(batch_size)
        feas = torch.mean(feature, 1) # convert to batch_size*512

        for step in xrange(max_len):
            if step != 0:
                feas, alpha = attend(feature, h0)
                alphas.append(alpha)
            inputs = torch.cat([feas, words], 1)
            h0, c0 = lstm_cell(inputs, (h0, c0))
            outputs = fc_out(h0)
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted.unsqueeze(1))
            words = embed(predicted)

        sampled_ids = torch.cat(sampled_ids, 1)
        return sampled_ids.squeeze(), alphas

    def get_start_states(self, batch_size):
        hidden_dim = self.hidden_dim
        h0 = to_var(torch.zeros(batch_size, hidden_dim))
        c0 = to_var(torch.zeros(batch_size, hidden_dim))
        return h0, c0


if __name__ == '__main__':
    # for test
    vis_dim = 512
    vis_num = 196
    embed_dim = 512
    hidden_dim = 512
    vocab_size = 1000
    num_layers = 1
    dropout_ratio = 0.5
    model = Decoder(vis_dim, vis_num, embed_dim, hidden_dim, vocab_size, num_layers, dropout_ratio)

    n_sample = 10
    features = to_var(torch.randn(n_sample, vis_num, vis_dim))
    caption = to_var(torch.zeros(n_sample, 20).long())
    lengths = [1, 2, 3, 2, 3, 2, 3, 20, 6, 4]

    model.train()
    model.cuda()

    for name, param in model.named_parameters():
        print name, param.size()

    ss = model(features, caption, lengths)
