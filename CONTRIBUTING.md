Contributing to Hub
===================

## Formatting and Linting
Hub uses Black and Flake8 to ensure a consistent code format throughout the project.
if you are using vscode then Replace `.vscode/settings.json` content with the following:
```
{
    "[py]": {
        "editor.formatOnSave": true
    },
    "python.formatting.provider": "black",
    "python.linting.flake8Enabled": true,
    "python.linting.flake8Path": "flake8",
    "python.linting.flake8Args": [
        "--max-line-length=80",
        "--select=B,C,E,F,W,B950",
        "--ignore=E203,E501,W503"
    ],
    "python.linting.pylintEnabled": false,
    "python.linting.enabled": true,
}
```

To contribute a feature/fix:
----------------------

1. Submit the contribution as a GitHub pull request against the master branch.
2. Make sure that your code passes the unit tests `pytest .`
3. Make sure that your code passes the linter.
4. Add new unit tests for your code.

`GitHub Issues`: https://github.com/activeloopai/Hub/issues
