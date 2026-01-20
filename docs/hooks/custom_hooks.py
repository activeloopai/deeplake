import logging
import shutil
import subprocess
import mkdocs.plugins as plugins
import yaml
import os

log = logging.getLogger("mkdocs")


def on_startup(command, dirty, **kwargs):
    if True:  # JS is disabled for now
        return
    log.info("Generating JS API...")
    shutil.rmtree("docs/api-js", ignore_errors=True)
    res = subprocess.run(
        [
            "typedoc",
            "--tsconfig",
            "/docs/deeplake.tsconfig.json",
            "--treatWarningsAsErrors",
        ]
    )
    if res.returncode != 0:
        log.error("Failed to generate JS API")
        exit(1)

    log.info("Generating JS API...DONE")

def extract_first_title(markdown):
   # Find first # heading
   lines = markdown.split('\n')
   for line in lines:
       if line.strip().startswith('# '):
           return line.strip('# ').strip()
   return None

@plugins.event_priority(100)  # This hook should run before the social plugin
def on_page_markdown(markdown, page, config, files):
    meta_file = page.file.src_path.rsplit('.', 1)[0] + '.meta.yaml'

    # For files like notebooks which don't support metadata in markdown, look for a .meta.yaml file to define it
    if meta_file in files:
        log.info(f"Using metadata from {meta_file} for {page.file.src_path}")
        with open(page.file.src_dir + "/" + meta_file, 'r') as f:
            meta_data = yaml.safe_load(f)
            if not meta_data:
                raise Exception(f"Invalid metadata file {meta_file}")

        if "description" in meta_data:
            page.meta['description'] = meta_data.get('description')
        if "title" in meta_data:
            page.meta['title'] = meta_data.get('title')
    elif 'title' not in page.meta or page.meta['title'] == None:
        page.meta['title'] = extract_first_title(markdown)
    return markdown

def on_post_build(config):
    """Copy llms.txt and llms directory to the site after build"""
    # Copy main llms.txt to root
    src_path = os.path.join(config['docs_dir'], 'llms.txt')
    dest_path = os.path.join(config['site_dir'], 'llms.txt')
    
    if os.path.exists(src_path):
        log.info(f"Copying llms.txt from {src_path} to {dest_path}")
        shutil.copy2(src_path, dest_path)
    else:
        log.warning(f"llms.txt not found at {src_path}")
    
    # Copy llms directory to site
    src_llms_dir = os.path.join(config['docs_dir'], 'llms')
    dest_llms_dir = os.path.join(config['site_dir'], 'llms')
    
    if os.path.exists(src_llms_dir):
        if os.path.exists(dest_llms_dir):
            shutil.rmtree(dest_llms_dir)
        shutil.copytree(src_llms_dir, dest_llms_dir)
        log.info(f"Copied llms directory from {src_llms_dir} to {dest_llms_dir}")
    else:
        log.warning(f"llms directory not found at {src_llms_dir}")