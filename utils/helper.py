import matplotlib.pyplot as plt
from PIL import Image
import skimage.transform
import os

__all__=['decode_captions',
        'attention_visualization']

def decode_captions(captions, idx_to_word):
    N, D = captions.shape
    decoded = []
    for idx in xrange(N):
        words = []
        for wid in xrange(D):
            word = idx_to_word[captions[idx, wid]]
            if word == '<end>' or word == '<start>' or word == '<unk>':
                words.append('.')
            else:
                words.append(word)
        decoded.append(words)
    return decoded

def attention_visualization(root, image_name, caption, alphas):
    image = Image.open(os.path.join(root, image_name))
    image = image.resize([224, 224], Image.LANCZOS)
    plt.subplot(4,5,1)
    plt.imshow(image)
    plt.axis('off')
    
    words = caption[1:]
    for t in range(len(words)):
        if t > 18:
            break
        plt.subplot(4, 5, t+2)
        plt.text(0, 1, '%s'%(words[t]) , color='black', backgroundcolor='white', fontsize=8)
        plt.imshow(image)
        # print alphas
        alp_curr = alphas[t, :].view(14, 14)
        alp_img = skimage.transform.pyramid_expand(alp_curr.numpy(), upscale=16, sigma=20)
        plt.imshow(alp_img, alpha=0.85)
        plt.axis('off')
    plt.show()
    

