import pickle
from typing import Dict, Optional, Set
from hub.client.utils import get_user_name
from hub.client.config import ALL_AGREEMENTS_PATH
from hub.util.exceptions import AgreementNotAcceptedError, NotLoggedInError


def update_local_agreements(all_local_agreements: Dict[str, Set[str]]):
    with open(ALL_AGREEMENTS_PATH, "wb") as f:
        pickle.dump(all_local_agreements, f)


def get_all_local_agreements() -> Dict[str, Set[str]]:
    try:
        with open(ALL_AGREEMENTS_PATH, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}


def agreement_prompt(org_id: str, ds_name: str, agreement: str):
    print()
    print()
    print(
        "The owner of the dataset you are trying to access requires that you agree to the following terms first:"
    )
    print()
    print("-" * 16)
    print(f"Dataset Owner: {org_id}")
    print(f"Dataset Name: {ds_name}")
    print("-" * 16)

    print("Terms:")
    print(agreement)
    print()
    print()

    print("-" * 16)
    user_input = input(
        f"In order to accept, please type the dataset's name ({ds_name}) and press enter: "
    )
    return user_input == ds_name


def handle_dataset_agreement(
    agreement: Optional[str], path: str, ds_name: str, org_id: str
):
    if agreement is None:
        return
    user_name = get_user_name()
    if user_name == "public":
        raise NotLoggedInError()
    if user_name == "org_id":
        return
    all_local_agreements = get_all_local_agreements()
    agreement_set = all_local_agreements.get(user_name) or set()
    if path not in agreement_set:
        accepted = agreement_prompt(org_id, ds_name, agreement)
        if accepted:
            print("Accepted agreement!")
            agreement_set.add(path)
            all_local_agreements[user_name] = agreement_set
            update_local_agreements(all_local_agreements)
        else:
            raise AgreementNotAcceptedError()
