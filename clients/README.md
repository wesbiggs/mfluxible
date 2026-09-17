# mfluxible-mcp

An [MCP](https://modelcontextprotocol.io) tool that generates images by calling a
running [mfluxible](https://github.com/wesbiggs/mfluxible) server.

This is the client half. It makes HTTP calls and installs nothing heavy — no mflux, no
MLX, no PyTorch — so it can live on a machine that isn't the one with the GPU. The
server it talks to is the `mfluxible` package, which does need Apple Silicon.

```bash
uvx mfluxible-mcp
```

Three tools: `generate_image` (text-to-image, image-to-image, and inpainting with
rectangular mask regions), `check_image` (collect a generation that outlived the host's
tool-call timeout), and `preview_mask` (render a mask selection over an image without
touching the GPU).

Point it at a server with `MFLUXIBLE_URL`; it defaults to
`http://127.0.0.1:8420/mfluxible/v1/images/generations`.

**[Full documentation, including how to register it with each tested MCP host.](https://github.com/wesbiggs/mfluxible/blob/main/docs/mcp.md)**

Apache-2.0, same as the rest of the project.
