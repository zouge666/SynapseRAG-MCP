from mcp_server.protocol_handler import ToolDefinition
from mcp_server.tools.query_knowledge_hub import query_knowledge_hub, query_knowledge_hub_tool_definition


def default_tools() -> dict[str, ToolDefinition]:
    tool = query_knowledge_hub_tool_definition()
    return {tool.name: tool}


__all__ = ["default_tools", "query_knowledge_hub", "query_knowledge_hub_tool_definition"]
