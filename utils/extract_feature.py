import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from data_loader import get_loader
import argparse
from torch.autograd import Variable


def get_model(model_path=None):
    if model_path is None:
        vgg = models.vgg16_bn(pretrained=True)
        model = nn.Sequential(*(vgg.features[i] for i in xrange(29)))
        torch.save(model, './utils/vgg_model.pth')
    else:
        model = torch.load(model_path)
    return torch.nn.DataParallel(model, device_ids=[0, 1, 2])


def get_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    return transform


def extract_features(root, files, transform, batch_size, shuffle, num_workers, model):

    dataloader = get_loader(root, files, transform, batch_size, shuffle, num_workers)
    model = model.cuda()
    model.eval()

    features = []
    imnames = []
    n_iters = len(dataloader)
    for i, (images, names) in enumerate(dataloader):
        images = Variable(images).cuda()
        feas = model(images).cpu()
        features.append(feas.data)
        imnames.extend(names)

        if (i+1)%100 == 0:
            print 'iter [%d/%d] finsihed.'%(i, n_iters)

    return torch.cat(features, 0), imnames


def main(args):

    root = args.root
    files = args.files
    transform = get_transform()
    batch_size = args.batch_size
    shuffle = args.shuffle
    num_workers = args.num_workers
    model = get_model(args.model_path).cuda()
    model.eval()
    features, names = extract_features(root, files, transform, batch_size, shuffle, num_workers, model)
    torch.save((names, features), args.save_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str,
                        default='./data/mscoco/images/train2014',
                        help='the directory that contains images.')
    parser.add_argument('--files', type=str, default=None, help='file lists')
    parser.add_argument('--batch_size', type=int, default=196, help='batch size')
    parser.add_argument('--shuffle', type=bool, default=False, help='whether to shuffle the dataset')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of threads for data loader')
    parser.add_argument('--model_path', type=str, default='./model/vgg_fea.pth',
                        help='The path to the feature extraction model')
    parser.add_argument('--save_name', type=str, default='./data/fea_mscoco_val.pth', help='Where to save the files.')
    args = parser.parse_args()
    print args
    main(args)
    print 'Done.'




