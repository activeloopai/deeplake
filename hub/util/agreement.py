from typing import List
from hub.util.exceptions import AgreementNotAcceptedError


def agreement_prompt(agreements: List[str], org_id: str, ds_name: str):
    print(
        "\n\nThe owner of the dataset you are trying to access requires that you agree to the following terms first:\n"
    )
    print("-" * 16)
    print(f"Dataset Owner: {org_id}")
    print(f"Dataset Name: {ds_name}")
    print("-" * 16)

    print("Terms:")

    for agreement in agreements:
        print(agreement + "\n")

    print("-" * 16)
    user_input = input(
        f"In order to accept, please type the dataset's name ({ds_name}) and press enter: "
    )
    return user_input == ds_name


def handle_dataset_agreements(client, agreements: List[str], org_id: str, ds_name: str):
    accepted = agreement_prompt(agreements, org_id, ds_name)
    if not accepted:
        raise AgreementNotAcceptedError()
    client.accept_agreements(org_id, ds_name)
    print("Accepted agreement!")
