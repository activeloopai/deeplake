def assert_label_issues_tensors(dataset, overwrite, verbose):
    """
    This function checks if a dataset already has a label_issues group.
    If overwrite = True, it will delete the existing label_issues group,
    else it will raise an error.
    """
    tensors_list = list(dataset.tensors)

    if (
        "label_issues/is_label_issue" in tensors_list
        or "label_issues/label_quality" in tensors_list
        or "label_issues/predicted_label" in tensors_list
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


def create_label_issues_tensors(dataset, label_issues, overwrite, verbose):
    """
    This function creates a group of tensors label_issues.
    After creating tensors, automatically commits the changes.
    """
    from hub.integrations.cleanlab.utils import process_label_issues

    # Check if label_issues tensor already exists.
    assert_label_issues_tensors(dataset=dataset, overwrite=overwrite, verbose=verbose)

    # Process label_issues dataframe to numpy ndarrays.
    label_issues, label_quality_scores, predicted_labels = process_label_issues(
        dataset=dataset, label_issues=label_issues
    )

    if verbose:
        print("Creating tensors with label issues...")

    with dataset:

        dataset.create_group("label_issues")

        dataset.label_issues.create_tensor(
            "is_label_issue", htype="generic", dtype="bool"
        )

        dataset.label_issues.create_tensor(
            "label_quality", htype="generic", dtype=label_quality_scores.dtype
        )

        dataset.label_issues.create_tensor(
            "predicted_label", htype="class_label", dtype="uint32"
        )

        for label_issue, label_quality_score, predicted_label in zip(
            label_issues, label_quality_scores, predicted_labels
        ):
            dataset.label_issues.is_label_issue.append(label_issue)
            dataset.label_issues.label_quality.append(label_quality_score)
            dataset.label_issues.predicted_label.append(predicted_label)

    commit_id = dataset.commit("Added label issues")

    if verbose:
        print(
            'Examine labels with issues by running a query: select * where "label_issues/is_label_issue" == true'
        )
        print(
            'View the labels with the lowest label quality scores by sorting by "label_issues/label_quality_scores"'
        )
        print(
            'Explore the predicted (guessed) labels by viewing the "label_issues/predicted_label" tensor'
        )

    return commit_id
