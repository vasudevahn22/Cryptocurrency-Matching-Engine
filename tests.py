import pytest
from fastapi.testclient import TestClient
from main import app, OrderType, OrderSide
from decimal import Decimal

client = TestClient(app)

SYMBOL = "BTC-USDT"

# === Helper ===
def place_order(order_type, side, quantity, price=None, trigger_price=None):
    payload = {
        "symbol": SYMBOL,
        "order_type": order_type,
        "side": side,
        "quantity": str(quantity)
    }
    if price is not None:
        payload["price"] = str(price)
    if trigger_price is not None:
        payload["trigger_price"] = str(trigger_price)
    
    response = client.post("/orders", json=payload)

    if response.status_code != 200:
        print("ERROR RESPONSE:", response.status_code, response.text)

    return response

# === Basic Functionality ===
def test_limit_order_full_match():
    place_order("limit", "sell", Decimal("1"), Decimal("30000"))
    res = place_order("limit", "buy", Decimal("1"), Decimal("31000"))
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "filled"
    assert data["filled_quantity"] == "1"

def test_market_order_with_liquidity():
    place_order("limit", "sell", Decimal("0.5"), Decimal("29000"))
    place_order("limit", "sell", Decimal("0.5"), Decimal("29500"))
    res = place_order("market", "buy", Decimal("1"))
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "filled"
    assert data["filled_quantity"] == "1.0"

def test_ioc_order():
    place_order("limit", "sell", Decimal("0.5"), Decimal("28000"))
    res = place_order("ioc", "buy", Decimal("1"), Decimal("28000"))
    data = res.json()
    assert data["status"] in ["partial_filled", "cancelled"]

def test_fok_order_success():
    place_order("limit", "sell", Decimal("1"), Decimal("27000"))
    res = place_order("fok", "buy", Decimal("1"), Decimal("27000"))
    assert res.status_code == 200
    assert res.json()["status"] == "filled"
 
def test_fok_order_failure():
    res = place_order("fok", "buy", Decimal("2"), Decimal("27000"))
    assert res.status_code == 200
    assert res.json()["status"] == "cancelled"

# === Advanced Order Type Tests ===
def test_stop_loss_order():
    place_order("limit", "sell", Decimal("1"), Decimal("25000"))
    res = place_order("stop_loss", "buy", Decimal("1"), trigger_price=Decimal("25000"))
    assert res.status_code == 200
    assert res.json()["status"] in ["filled", "pending", "cancelled"]

def test_take_profit_order():
    place_order("limit", "buy", Decimal("1"), Decimal("26000"))
    res = place_order("take_profit", "sell", Decimal("1"), trigger_price=Decimal("26000"))
    assert res.status_code == 200
    assert res.json()["status"] in ["filled", "pending", "cancelled"]

def test_stop_limit_order():
    place_order("stop_limit", "buy", Decimal("1"), price=Decimal("27000"), trigger_price=Decimal("26500"))
    place_order("limit", "sell", Decimal("1"), Decimal("26500"))
    res = client.get(f"/orderbook/{SYMBOL}")
    assert res.status_code == 200

def test_stats():
    res = client.get("/stats")
    assert res.status_code == 200
    data = res.json()
    assert "orders_processed" in data
    assert "trades_executed" in data

def test_bbo():
    res = client.get(f"/bbo/{SYMBOL}")
    assert res.status_code == 200
    assert "best_bid" in res.json()
    assert "best_ask" in res.json()
