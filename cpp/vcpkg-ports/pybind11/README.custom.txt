## Custom Port Notes

This is identical to the original port, but with the following changes:
- Removed "python" dependency from vcpkg.json
  - This makes vcpkg use the system python, which keeps the version built against in sync with the version ran
  - This also avoids having to wait for new python versions to be ported to vcpkg
- Commented out `file(INSTALL "usage" ...` from portfile.cmake
  - It was not copying correctly, and is not needed