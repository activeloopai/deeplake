"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
from hub_v1.store.store import get_user_name
from datetime import datetime


class VersionNode:
    def __init__(self, commit_id, branch):
        self.children = []
        self.parent = None
        self.commit_id = commit_id
        self.message = None
        self.branch = branch
        self.commit_time = None
        self.commit_user_name = None

    def insert(self, node, message=None):
        node.parent = self
        self.children.append(node)
        self.message = self.message or message
        user_name = get_user_name()
        self.commit_user_name = "None" if user_name == "public" else user_name
        self.commit_time = datetime.now()

    def __repr__(self) -> str:
        return f'commit {self.commit_id} ({self.branch}) \nAuthor: {self.commit_user_name}\nCommit Time:  {str(self.commit_time)[:-7]}\nMessage: "{self.message}"'

    def __str__(self) -> str:
        return self.__repr__()
