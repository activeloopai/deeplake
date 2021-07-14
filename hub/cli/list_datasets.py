import click
from hub.util import list_datasets


@click.command()
@click.option("--workspace", "-w", default=None, help="List datasets in the workspace")
def list_public_datasets(workspace: str):
    """Get a list of public datasets in Platform."""
    try:
        res = list_datasets.list_public_datasets(workspace)
        click.echo("\n".join(res))
    except Exception as e:
        raise SystemExit(f"Unable to list public datasets: {e}")


@click.command()
@click.option(
    "--workspace", "-w", default=None, help="List your datasets in the workspace"
)
def list_my_datasets(workspace: str):
    """Get a list of your datasets in the workspace from Platform."""
    try:
        res = list_datasets.list_my_datasets(workspace)
        if res:
            click.echo("\n".join(res))
    except Exception as e:
        raise SystemExit(f"Unable to list datasets: {e}")
