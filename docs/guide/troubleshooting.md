---
title: Troubleshooting
---

# Troubleshooting
## CLI Installation

#### If you get a `Permission denied` message:
```bash
  sudo pip3 install snark
```
#### If you don't have `sudo` access:
```bash
  pip3 install snark --user
```

**AND** add the following to your `~/.bashrc` file:
```bash
  export PY_USER_BIN=$(python3 -c 'import site; print(site.USER_BASE + "/bin")')
  export PATH=$PY_USER_BIN:$PATH
```
**AND** reload your `~/.bashrc`:
```bash
  source ~/.bashrc
```
#### Snark Not Found
If you tried all above and still get `snark not found` error message, try:
1) Updating your pip3 through `pip3 install --upgrade pip3`
2) Try installing specific version through tarball:
```bash
  pip3 uninstall snark
  pip3 install https://files.pythonhosted.org/packages/6b/c4/1112f032a3d90686d757e5b0b325564a047488fc74fa43a138148dc2b8a5/snark-0.3.2.0.tar.gz
```

In case of questions or issues please contact us at support@snark.ai
