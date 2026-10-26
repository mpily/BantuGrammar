"""Public REST API for the Kitaita translator Space.

Wraps the research pipeline; never changes its logic. Research modules must
not import this package (it pulls in FastAPI). Contract: api/CONTRACT.md.
"""
API_VERSION = "1"
