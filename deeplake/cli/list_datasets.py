import click
import deeplake as dl


@click.command()
@click.option(
    "--workspace", "-w", default=None, help="List your datasets in the workspace"
)
def list_datasets(workspace: str):
    """Get a list of datasets in the workspace from Platform."""
    try:
        res = dl.list(workspace)
        if res:
            click.echo("\n".join(res))
    except Exception as e:
        raise SystemExit(f"Unable to list datasets: {e}")
