#!/usr/bin/env python3
"""Simple HTTP server for the data directory."""
import os
import http.server
import functools

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory='.')
server = http.server.HTTPServer(('', 8742), handler)
print(f"Serving data/ at http://localhost:8742")
server.serve_forever()
