import numpy as np


def preprocess(data):
    nbins = 64
    eta_range = (-2, 2)
    phi_range = (-2, 2)

    images = []
    for jet in data:
        etas = jet[:, 0]
        phis = jet[:, 1]
        pts = np.exp(jet[:, 2])

        image, _, _ = np.histogram2d(
            etas, phis, bins=nbins, range=[eta_range, phi_range], weights=pts
        )

        total = image.sum()
        if total > 0:
            image /= total

        image = np.log1p(100 * image)

        images.append(image)

    images = np.array(images)

    return images
