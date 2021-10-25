def terms_of_access_prompt(dataset_org: str, dataset_name: str, terms: str):
    print()
    print()
    print(
        "The owner of the dataset you are trying to access requires that you agree to the following terms first:"
    )
    print()
    print("-" * 16)
    print(f"Dataset Owner: {dataset_org}")
    print(f"Dataset Name: {dataset_name}")
    print("-" * 16)

    print("Terms:")
    print(terms)
    print()
    print()

    print("-" * 16)
    x = input(
        f"In order to accept, please type the dataset's name ({dataset_name}) and press enter: "
    )

    return x == dataset_name
