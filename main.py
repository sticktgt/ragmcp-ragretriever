import os
import uvicorn
from .config import CONFIG

def main():
    server_cfg = CONFIG.get("server")
    host = server_cfg.get("host", "0.0.0.0")
    port = int(server_cfg.get("port", "8080"))
    # Serve the MCP HTTP app defined in mcp_server.py
    uvicorn.run("ragretriever.mcp_server:app", host=host, port=port, reload=False)

if __name__ == "__main__":
    main()
