import click

from hub import list as hub_list


@click.command()
@click.option(
    "--workspace", "-w", default=None, help="List your datasets in the workspace"
)
def list_datasets(workspace: str):
    """Get a list of datasets in the workspace from Platform."""
    try:
        res = hub_list(workspace)
        if res:
            click.echo("\n".join(res))
    except Exception as e:
        raise SystemExit(f"Unable to list datasets: {e}")
