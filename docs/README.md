# Deep Lake Documentation

This directory contains the documentation for the Deep Lake project. The documentation is written in markdown and built using <https://squidfunk.github.io/mkdocs-material/>.

## Building the Documentation

To build the documentation, you will need to have Docker installed.

To build the documentation, run the following command:

```bash
docker run --rm -it -p 8000:8000 -v ${PWD}:/docs -v "${PWD}"/../:/indra -v "${PWD}"/../node_modules:/source/typescript/node_modules  -v ${PWD}/../python:/source/python $(docker build --build-arg=GITHUB_TOKEN=$GITHUB_TOKEN -q .)
```

This will build the documentation and start a local server at `http://localhost:8000`.

NOTE: the command references an optional GITHUB_TOKEN env variable. That allows you to use the insiders build of some of the build libraries, but the permission setup is currently not quite right. So ignore those for now.

NOTE: the build process of the container is quiet due to the `-q` flag. To see the build output, you can run `docker build .` on its own.

## Version management

Versions are managed using ["mike"](https://github.com/jimporter/mike) which keeps copies of the built docs in the `docs_site` branch.

The mike cli can be used through the docker container by running the following command:

```bash
docker run --rm -v ${PWD}:/docs -v ${PWD}/../python:/source/python --entrypoint mike $(docker build --build-arg=GITHUB_TOKEN=$GITHUB_TOKEN -q .) [mike commands here]
````

## Writing Documentation

The TLDR section for the MKDocs page writing is <https://squidfunk.github.io/mkdocs-material/reference/> which
starts with the general page setup and then has links on the left for the main content blocks you can use.

Custom sorting of links in the navigation is done in the (optional) `.pages` file per directory.
For more information on what can be in these files, see <https://github.com/lukasgeiter/mkdocs-awesome-pages-plugin/>
