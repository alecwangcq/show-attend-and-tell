import numpy as np
import argparse
from torch.nn.utils.rnn import  pack_padded_sequence
from model.models import *
from utils.data_loader_coco import *
import torch.optim as optim


def train(dataloader, model, optimizer, criterion, epoch, total_epoch):
    total_step = len(dataloader)
    # print 'Total step:', total_step
    for i, (features, targets, lengths) in enumerate(dataloader):
        optimizer.zero_grad()
        features = to_var(features)
        targets = to_var(targets)
        predicts = model(features, targets[:, :-1], [l - 1 for l in lengths])
        predicts = pack_padded_sequence(predicts, [l-1 for l in lengths], batch_first=True)[0]
        targets = pack_padded_sequence(targets[:, 1:], [l-1 for l in lengths], batch_first=True)[0]
        loss = criterion(predicts, targets)
        loss.backward()
        optimizer.step()
        if (i+1)%100 == 0:
            print 'Epoch [%d/%d]: [%d/%d], loss: %5.4f, perplexity: %5.4f.'%(epoch, total_epoch,i,
                                                                             total_step,loss.data[0],
                                                                             np.exp(loss.data[0]))

def test():
    pass


def main(args):
    # dataset setting
    image_root = args.image_root
    feature_path = args.feature_path
    ann_path = args.ann_path
    vocab_path = args.vocab_path
    batch_size = args.batch_size
    shuffle = args.shuffle
    num_workers = args.num_workers
    
    dataloader = get_loader(image_root=image_root, 
                            feature_path=feature_path,
                            ann_path=ann_path, 
                            vocab_path=vocab_path, 
                            batch_size=batch_size,
                            shuffle=shuffle, 
                            num_workers=num_workers)

    # model setting
    vis_dim = args.vis_dim
    vis_num = args.vis_num
    embed_dim = args.embed_dim
    hidden_dim = args.hidden_dim
    vocab_size =args.vocab_size
    num_layers = args.num_layers
    dropout_ratio = args.dropout_ratio
    
    model = Decoder(vis_dim=vis_dim,
                    vis_num=vis_num, 
                    embed_dim=embed_dim,
                    hidden_dim=hidden_dim, 
                    vocab_size=vocab_size, 
                    num_layers=num_layers,
                    dropout_ratio=dropout_ratio)

    # optimizer setting
    lr = args.lr
    num_epochs = args.num_epochs
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # criterion
    criterion = nn.CrossEntropyLoss()
    model.cuda()
    model.train()
    
    print 'Number of epochs:', num_epochs
    for epoch in xrange(num_epochs):
        train(dataloader=dataloader, model=model, optimizer=optimizer, criterion=criterion,
              epoch=epoch, total_epoch=num_epochs)
        torch.save(model, './checkpoints/model_%d.pth'%(epoch))
        test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data loader
    parser.add_argument('--image_root', type=str,
                        default='./data/mscoco/images/train2014')
    parser.add_argument('--feature_path', type=str,
                        default='./data/name_feature_train_t.pth')
    parser.add_argument('--ann_path', type=str,
                        default='./data/mscoco/annotations/captions_train2014.json')
    parser.add_argument('--vocab_path', type=str,
                        default='./data/vocab.pkl')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=2)

    # model setting
    parser.add_argument('--vis_dim', type=int, default=512)
    parser.add_argument('--vis_num', type=int, default=196)
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--dropout_ratio', type=float, default=0.5)

    # optimizer setting
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--num_epochs', type=int, default=120)

    args = parser.parse_args()
    print args
    main(args)