"""mfluxible-mcp -- the MCP tool that generates images through a running mfluxible.

A second distribution rather than an extra on the server's, and the separation is the
point: this talks to mfluxible over HTTP and has no use for mflux, MLX or PyTorch, so
installing it must not drag several gigabytes of them along. That is the same rule the
repo's separate dependency sets have always kept (see CLAUDE.md); publishing it as
`mfluxible[mcp]` would have quietly broken it.

Released in lockstep with the server from the same repository -- one tag builds both --
and tests/test_mcp_server.py fails if the two versions drift apart.
"""

__version__ = "0.9.0"
