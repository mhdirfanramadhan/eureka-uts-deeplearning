import keras

GAMMA = 2.0
ALPHA = 0.25


def focal_loss(gamma=GAMMA, alpha=ALPHA):
    return keras.losses.BinaryFocalCrossentropy(
        apply_class_balancing=True, alpha=alpha, gamma=gamma
    )
