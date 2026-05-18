from mcp_server.protocol_handler import ProtocolHandler, ToolDefinition


def tool_handler(arguments: dict[str, object]) -> dict[str, object]:
    return {"content": [{"type": "text", "text": f"hello {arguments['name']}"}]}


def failing_tool_handler(arguments: dict[str, object]) -> dict[str, object]:
    raise RuntimeError("secret stack details")


def handler() -> ProtocolHandler:
    return ProtocolHandler(
        tools={
            "hello": ToolDefinition(
                name="hello",
                description="Say hello",
                input_schema={"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
                handler=tool_handler,
            )
        }
    )


def test_initialize_returns_server_info_and_capabilities() -> None:
    response = handler().handle_request(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18"},
        }
    )

    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 1
    assert response["result"]["protocolVersion"] == "2025-06-18"
    assert response["result"]["serverInfo"]["name"] == "synapserag-mcp"
    assert response["result"]["capabilities"] == {"tools": {}}


def test_tools_list_returns_registered_tool_schemas() -> None:
    response = handler().handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})

    assert response["result"] == {
        "tools": [
            {
                "name": "hello",
                "description": "Say hello",
                "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
            }
        ]
    }


def test_tools_call_routes_to_registered_tool() -> None:
    response = handler().handle_request(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "hello", "arguments": {"name": "Ada"}},
        }
    )

    assert response["result"] == {"content": [{"type": "text", "text": "hello Ada"}]}


def test_initialized_notification_returns_no_response() -> None:
    assert handler().handle_request({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}) is None


def test_invalid_method_returns_method_not_found() -> None:
    response = handler().handle_request({"jsonrpc": "2.0", "id": 4, "method": "missing", "params": {}})

    assert response["error"] == {"code": -32601, "message": "Method not found"}


def test_invalid_request_returns_invalid_request() -> None:
    response = handler().handle_request({"jsonrpc": "1.0", "id": 5, "method": "initialize"})

    assert response["error"] == {"code": -32600, "message": "Invalid Request"}


def test_invalid_params_returns_invalid_params() -> None:
    response = handler().handle_request({"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {"name": "missing"}})

    assert response["error"] == {"code": -32602, "message": "Invalid params"}


def test_internal_tool_exception_returns_internal_error_without_stack() -> None:
    active_handler = ProtocolHandler(
        tools={
            "boom": ToolDefinition(
                name="boom",
                description="Boom",
                input_schema={"type": "object"},
                handler=failing_tool_handler,
            )
        }
    )

    response = active_handler.handle_request(
        {"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {"name": "boom", "arguments": {}}}
    )

    assert response["error"] == {"code": -32603, "message": "Internal error"}
    assert "secret" not in str(response)
