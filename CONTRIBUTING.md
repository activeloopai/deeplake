Contributing to Hub
===================

To contribute a feature/fix:
----------------------

1. Submit the contribution as a GitHub pull request against the master branch.
2. Make sure that your code passes the unit tests `pytest .`
3. Make sure that your code passes the linter.
4. Add new unit tests for your code.

`GitHub Issues`: https://github.com/activeloopai/Hub/issues


## How you can help

* Adding new datasets following [this example](https://docs.activeloop.ai/en/latest/concepts/dataset.html#how-to-upload-a-dataset)
* Fix an issue from GitHub Issues
* Add a feature. For an extended feature please create an issue to discuss.


## Get Started

Ready to contribute? Here's how to set up `Hub` for local development.

1. Fork the `Hub` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/hub.git

3. Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed,
   this is how you set up your fork for local development::

    $ mkvirtualenv Hub
    $ cd Hub/

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. While hacking your changes, make sure to cover all your developments with the required
   unit tests, and that none of the old tests fail as a consequence of your changes.
   For this, make sure to run the tests suite and check the code coverage::

    $ pytest .      # Run the tests

6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website.


## Formatting and Linting
Hub uses Black and Flake8 to ensure a consistent code format throughout the project.
if you are using vscode then Replace `.vscode/settings.json` content with the following:
```json
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
