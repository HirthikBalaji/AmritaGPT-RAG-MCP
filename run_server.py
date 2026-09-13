"""
AmritaGPT Unified Server Entry Point.
Runs the institutional intelligence server in either MCP mode (stdio/sse)
or OpenAPI REST Gateway mode (FastAPI + Swagger UI + mounted MCP).
"""
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    parser = argparse.ArgumentParser(description="Run AmritaGPT Server (MCP & OpenAPI)")
    parser.add_argument(
        "--mode",
        choices=["mcp", "api", "dual"],
        default="mcp",
        help="Server mode: 'mcp' (Model Context Protocol), 'api' / 'dual' (FastAPI OpenAPI Gateway with Swagger UI & mounted MCP)"
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="Transport protocol for MCP mode (default: stdio)"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address for HTTP/REST/SSE server")
    parser.add_argument("--port", type=int, default=8000, help="Port for HTTP/REST/SSE server")
    args = parser.parse_args()

    if args.mode in ["api", "dual"]:
        import uvicorn
        from amritagpt.api.app import app

        print("=" * 70, file=sys.stderr)
        print("  AMRITAGPT OPENAPI REST GATEWAY & MCP SERVER", file=sys.stderr)
        print(f"  Interactive Dashboard:  http://localhost:{args.port}/dashboard", file=sys.stderr)
        print(f"  Interactive Swagger UI: http://localhost:{args.port}/docs", file=sys.stderr)
        print(f"  ReDoc Documentation:    http://localhost:{args.port}/redoc", file=sys.stderr)
        print(f"  OpenAPI 3.1 Spec:       http://localhost:{args.port}/openapi.json", file=sys.stderr)
        print(f"  Mounted MCP SSE:        http://localhost:{args.port}/mcp/sse", file=sys.stderr)
        print("=" * 70, file=sys.stderr)

        uvicorn.run(app, host=args.host, port=args.port)
    else:
        from amritagpt.mcp.server import create_amrita_mcp_server
        server = create_amrita_mcp_server()
        print(f"Starting AmritaGPT MCP Server on transport '{args.transport}'...", file=sys.stderr)
        if args.transport == "stdio":
            server.run(transport="stdio")
        else:
            server.run(transport=args.transport, port=args.port, host=args.host)


if __name__ == "__main__":
    main()
