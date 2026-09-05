"""JSON-RPC dispatch rules for the MCP server (see issue #403).

#384 stopped unknown *requests* from being answered with silence, which was
the cause of the client respawn loops. `ping` is not an unknown method though:
it is a required MCP utility, and the spec says a ping request is answered with
an empty result. Answering it with -32601 tells a client its liveness probe is
unsupported, which is a different wrong answer from the original silence.
"""

from leann.mcp import handle_request


def test_ping_returns_an_empty_result():
    response = handle_request({"jsonrpc": "2.0", "id": 3, "method": "ping"})

    assert response is not None, "silence is what caused the respawn loops"
    assert response["id"] == 3
    assert response["result"] == {}
    assert "error" not in response


def test_ping_with_params_is_still_a_ping():
    response = handle_request({"jsonrpc": "2.0", "id": 4, "method": "ping", "params": {}})

    assert response["result"] == {}


def test_unknown_request_still_gets_method_not_found():
    """#384's behaviour must survive: an unknown request is an error, not silence."""
    response = handle_request({"jsonrpc": "2.0", "id": 9, "method": "server/discover"})

    assert response["error"]["code"] == -32601


def test_notifications_get_no_response():
    """JSON-RPC 2.0: a notification has no id and must never be answered."""
    assert handle_request({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert handle_request({"jsonrpc": "2.0", "method": "some/unknown/notification"}) is None
