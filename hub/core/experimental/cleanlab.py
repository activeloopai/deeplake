from hub.integrations import pytorch_module_to_skorch as to_skorch
from hub.util.dataset import map_tensor_keys

from cleanlab.filter import find_label_issues
from cleanlab.rank import get_label_quality_scores

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.base import clone

import numpy as np

def get_dataset_tensors(dataset, tensors, dataloader_train_params, overwrite, verbose):
    """
    This function returns the tensors of a dataset. If a list of tensors is not provided,
    it will try to find them in the dataloader_train_params in the transform. If none of
    these are provided, it will iterate over the dataset tensors and return any tensors
    that match htype 'image' for images and htype 'class_label' for labels. Additionally,
    this function will also check if the dataset already has a label_issues group.
    """

    tensors_list = list(dataset.tensors)

    if (
        "label_issues/is_label_issue" in tensors_list
        or "label_issues/label_quality_scores" in tensors_list
    ):
        if overwrite:
            if verbose:
                print("Found existing label_issues tensor. Deleting the tensor...")
            dataset.delete_group("label_issues")
            dataset.commit("Removed label issues", allow_empty=True)

        else:
            raise ValueError(
                "The group of tensors label_issues already exist. Use overwrite = True, to overwrite the label_issues tensors."
            )

    if tensors is not None:
        tensors = map_tensor_keys(dataset, tensors)

    # Try to get the tensors from the dataloader parameters.
    elif (
        dataloader_train_params
        and "transform" in dataloader_train_params
        and isinstance(dataloader_train_params["transform"], dict)
    ):
        tensors = map_tensor_keys(
            dataset,
            [k for k in dataloader_train_params["transform"].keys() if k != "index"],
        )

    # Map the images and labels tensors to the corresponding tensors in the dataset.
    images_tensor, labels_tensor = None, None

    if tensors:
        for tensor in tensors:
            if dataset[tensor].htype == "image":
                images_tensor = tensor
            elif dataset[tensor].htype == "class_label":
                labels_tensor = tensor
    else:
        for tensor in tensors_list:
            if dataset[tensor].htype == "image":
                images_tensor = tensor
            elif dataset[tensor].htype == "class_label":
                labels_tensor = tensor

    if images_tensor and labels_tensor:
        tensors = [images_tensor, labels_tensor]

    else:
        raise ValueError(
            "Could not find the images and labels tensors. Please provide the images and labels tensors."
        )

    return tensors


def estimate_cv_predicted_probabilities(
    dataset, labels, model, folds, num_classes, verbose
):
    """
    This function computes an out-of-sample predicted
    probability for every example in a dataset using cross
    validation. Output is an np.array of shape (N, K) where N is
    the number of training examples and K is the number of classes.
    """

    # Initialize pred_probs array
    pred_probs = np.zeros(shape=(len(dataset), num_classes))

    # Create cross-validation object for out-of-sample predicted probabilities.
    kfold = StratifiedKFold(n_splits=folds, shuffle=True)

    if verbose:
        print(
            "Computing out-of-sample predicted probabilities with "
            f"{folds}-fold cross validation..."
        )

    # Loop through each fold and compute out-of-sample predicted probabilities.
    for fold, (train_idx, holdout_idx) in enumerate(kfold.split(X=dataset, y=labels)):

        if verbose:
            print(f"Fold {fold + 1} of {folds}...")

        # Initialize a fresh untrained model.
        model_copy = clone(model)

        # Select the training and holdout cross-validated sets.
        ds_train, ds_holdout = (
            dataset[train_idx.tolist()],
            dataset[holdout_idx.tolist()],
        )
        # Fit model to training set, predict on holdout set, and update pred_probs.
        model_copy.fit(X=ds_train)

        pred_probs_cross_val = model_copy.predict_proba(X=ds_holdout)
        pred_probs[holdout_idx] = pred_probs_cross_val

    return pred_probs

def append_label_issues_tensors(dataset, label_issues, label_quality_scores):
    """
    This function creates a group of tensors label_issues.
    After creating tensors, automatically commits the changes.
    """
    with dataset:

        dataset.create_group("label_issues")
        dataset.label_issues.create_tensor(
            "is_label_issue", htype="generic", dtype="bool"
        )
        dataset.label_issues.create_tensor(
            "label_quality_scores", htype="generic", dtype="float64"
        )

        for label_issue, label_quality_score in zip(label_issues, label_quality_scores):
            dataset.label_issues.is_label_issue.append(label_issue)
            dataset.label_issues.label_quality_scores.append(label_quality_score)

    commit_id = dataset.commit("Added label issues")

    return commit_id

def clean_labels(
    dataset,
    module,
    criterion,
    device,
    epochs,
    folds,
    verbose,
    tensors,
    dataloader_train_params,
    dataloader_valid_params,
    optimizer,
    optimizer_lr,
    overwrite
    # skorch_kwargs,
):
    """
    This function cleans the labels of a dataset. It wraps a PyTorch instance in a sklearn classifier.
    Next, it runs cross-validation to get out-of-sample predicted probabilities for each example.
    Then, it calls filter.find_label_issues to find label issues and rank.get_label_quality_scores
    to find label quality scores for each sample in the dataset. At the end, it creates tensors
    with label issues.
    """

    images_tensor, labels_tensor = get_dataset_tensors(
        dataset=dataset,
        tensors=tensors,
        dataloader_train_params=dataloader_train_params,
        overwrite=overwrite,
        verbose=verbose
    )

    # Get labels of a dataset
    labels = dataset[labels_tensor].numpy().flatten()

    # Get the number of unique classes.
    num_classes = len(np.unique(labels))

    # Wrap the PyTorch Module in scikit-learn interface.
    model = to_skorch(
        dataset=dataset,
        module=module,
        criterion=criterion,
        device=device,
        epochs=epochs,
        optimizer=optimizer,
        optimizer_lr=optimizer_lr,
        tensors=[images_tensor, labels_tensor],
        dataloader_train_params=dataloader_train_params,
        dataloader_valid_params=dataloader_valid_params,
        num_classes=num_classes
        # skorch_kwargs=skorch_kwargs
    )

    # Compute out-of-sample predicted probabilities.
    pred_probs = estimate_cv_predicted_probabilities(
        dataset=dataset,
        labels=labels,
        model=model,
        folds=folds,
        num_classes=num_classes,
        verbose=verbose,
    )

    if verbose:
        predicted_labels = pred_probs.argmax(axis=1)
        acc = accuracy_score(y_true=labels, y_pred=predicted_labels)
        print(f"Cross-validated estimate of accuracy on held-out data: {acc}")

    if verbose:
        print("Using predicted probabilities to identify label issues ...")

    label_issues = find_label_issues(labels=labels, pred_probs=pred_probs)

    label_quality_scores = get_label_quality_scores(
        labels=labels, pred_probs=pred_probs
    )

    if verbose:
        print(f"Identified {np.sum(label_issues)} examples with label issues.")

    if verbose:
        print("Creating tensors with label issues...")
    append_label_issues_tensors(dataset=dataset, label_issues=label_issues, label_quality_scores=label_quality_scores)

    return label_issues, label_quality_scores
