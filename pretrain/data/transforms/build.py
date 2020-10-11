from . import transforms as T


def build_transforms(cfg, mode='train'):
    assert mode in ['train', 'test', 'val']
    min_size = cfg.SCALES[0]
    max_size = cfg.SCALES[1]
    assert min_size <= max_size

    if mode == 'train':
        flip_prob = cfg.TRAIN.FLIP_PROB
    elif mode == 'test':
        flip_prob = cfg.TEST.FLIP_PROB
    else:
        flip_prob = cfg.VAL.FLIP_PROB

    to_bgr255 = True

    #normalize_transform = T.Normalize(
    #    mean=cfg.NETWORK.PIXEL_MEANS, std=cfg.NETWORK.PIXEL_STDS, to_bgr255=to_bgr255
    #)

    normalize_transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # transform = T.Compose(
    #     [
    #         T.Resize(min_size, max_size),
    #         T.RandomHorizontalFlip(flip_prob),
    #         T.ToTensor(),
    #         normalize_transform,
    #         T.FixPadding(min_size, max_size, pad=0)
    #     ]
    # )
    #
    '''
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800] 
    if mode == 'train':
        transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                    ])
                ),
            normalize_transform,
            ])
    else:
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize_transform,
            ])

    '''
    transform = T.Compose(
        [
            T.Resize(min_size, max_size),
            T.RandomHorizontalFlip(flip_prob),
            #T.ToTensor(),
            normalize_transform,
        ]
    )
    

    return transform
