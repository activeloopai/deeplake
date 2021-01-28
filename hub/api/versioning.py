class VersionNode:
    def __init__(self, commit_id, branch):
        self.children = []
        self.parent = None
        self.commit_id = commit_id
        self.message = None
        self.branch = branch

    def insert(self, node, message=None):
        node.parent = self
        self.children.append(node)
        self.message = self.message or message

    def __repr__(self) -> str:
        return f'{self.commit_id}({self.branch}): "{self.message}"'

    def __str__(self) -> str:
        return self.__repr__()

    # def delete(self):
    #     child