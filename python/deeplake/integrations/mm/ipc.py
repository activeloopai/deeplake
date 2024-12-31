import socketserver


def _get_free_port() -> int:
    with socketserver.TCPServer(("localhost", 0), None) as s:  # type: ignore
        return s.server_address[1]
