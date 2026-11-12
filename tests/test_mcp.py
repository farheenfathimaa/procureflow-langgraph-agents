"""MCP supplier tool tests (in-process, no transport or network)."""

from __future__ import annotations

import pytest

from mcp_server.server import get_supplier, search_suppliers


def test_search_suppliers_returns_citable_payloads():
    results = search_suppliers(
        "business laptops with three year warranty", category="laptops", max_results=3
    )

    assert results
    assert len(results) <= 3
    first = results[0]
    assert first["supplier_id"].startswith("SUP-LAP-")
    assert first["citation"]["doc_id"] == first["supplier_id"]
    assert first["unit_price_usd"] > 0
    assert first["summary"]


def test_search_suppliers_filters_by_category():
    results = search_suppliers("desk or chair", category="furniture", max_results=3)

    assert results
    assert all(item["category"] == "furniture" for item in results)


def test_search_suppliers_bounds_max_results():
    results = search_suppliers("supplier", max_results=1000)
    assert 1 <= len(results) <= 10


def test_get_supplier_by_id():
    supplier = get_supplier("sup-lap-001")

    assert supplier["name"] == "NovaTech Devices"
    assert supplier["country"] == "United States"
    assert supplier["unit_price_usd"] == 1150.0
    assert supplier["citation"]["doc_id"] == "SUP-LAP-001"


def test_unknown_supplier_raises():
    with pytest.raises(ValueError):
        get_supplier("SUP-DOES-NOT-EXIST")


async def test_server_registers_tools():
    from mcp_server.server import server

    names = {tool.name for tool in await server.list_tools()}
    assert {"search_suppliers", "get_supplier"}.issubset(names)
