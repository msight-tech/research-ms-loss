
cub_root = 'resource/datasets/CUB_200_2011/'
images_file = cub_root + 'images.txt'
train_file = cub_root + 'train.txt'
test_file = cub_root + 'test.txt'


def main():
    train = []
    test = []
    with open(images_file) as f_img:
        for l_img in f_img:
            i, fname = l_img.split()
            label = int(fname.split('.', 1)[0])
            if label <= 100:
                train.append((fname, label - 1)) # labels 0 ... 99 (0-based labels for margin_loss)
            else:
                test.append((fname, label - 1))  # labels 100 ... 199

    for f, v in [(train_file, train), (test_file, test)]:
        with open(f, 'w') as tf:
            for fname, label in v:
                print("images/{},{}".format(fname, label), file=tf)


if __name__ == '__main__':
    main()
