# ======================================
# Loss + Optimizer
# ======================================

class_weights = torch.tensor(
    [
        0.0078,   # Class 0: Background
        0.1653,   # Class 1: No Damage
        1.4010,   # Class 2: Minor Damage
        1.0366,   # Class 3: Major Damage
        2.3893,   # Class 4: Destroyed
    ],
    dtype=torch.float32,
).to(DEVICE)


# Weighted CrossEntropy Loss
ce_loss = nn.CrossEntropyLoss(
    weight=class_weights
)


# Dice Loss
def dice_loss(outputs, targets, smooth=1e-6):

    probabilities = torch.softmax(outputs, dim=1)

    targets_one_hot = torch.nn.functional.one_hot(
        targets,
        num_classes=5,
    )

    targets_one_hot = targets_one_hot.permute(
        0, 3, 1, 2
    ).float()

    intersection = (
        probabilities * targets_one_hot
    ).sum(dim=(0, 2, 3))

    denominator = (
        probabilities.sum(dim=(0, 2, 3))
        + targets_one_hot.sum(dim=(0, 2, 3))
    )

    dice = (
        2.0 * intersection + smooth
    ) / (
        denominator + smooth
    )

    # Ignore background when calculating Dice loss
    dice = dice[1:]

    return 1.0 - dice.mean()


# Combined Loss
def combined_loss(outputs, targets):

    ce = ce_loss(outputs, targets)

    dice = dice_loss(
        outputs,
        targets,
    )

    return ce + dice


criterion = combined_loss


# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)