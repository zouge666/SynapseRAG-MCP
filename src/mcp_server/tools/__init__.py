from mcp_server.protocol_handler import ToolDefinition
from mcp_server.tools.list_collections import list_collections, list_collections_tool_definition
from mcp_server.tools.query_knowledge_hub import query_knowledge_hub, query_knowledge_hub_tool_definition


def default_tools() -> dict[str, ToolDefinition]:
    tools = [list_collections_tool_definition(), query_knowledge_hub_tool_definition()]
    return {tool.name: tool for tool in tools}


__all__ = [
    "default_tools",
    "list_collections",
    "list_collections_tool_definition",
    "query_knowledge_hub",
    "query_knowledge_hub_tool_definition",
]
