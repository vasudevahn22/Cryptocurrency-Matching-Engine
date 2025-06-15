from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from decimal import Decimal
from typing import Optional, List, Dict, Set, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timezone
from pathlib import Path
import asyncio
import json
import logging
import pickle
import uuid
import bisect
from collections import deque
import time

# === Configuration ===
FEE_TAKER = Decimal("0.001")
FEE_MAKER = Decimal("0.0005")
PERSISTENCE_FILE = Path("engine_state.pkl")

# === Enums ===
class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    IOC = "ioc"
    FOK = "fok"
    STOP_LOSS = "stop_loss"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT = "take_profit"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    PENDING = "pending"
    PARTIAL_FILLED = "partial_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

# === Order ===
@dataclass
class Order:
    order_id: str
    symbol: str
    order_type: OrderType
    side: OrderSide
    quantity: Decimal
    price: Optional[Decimal]
    trigger_price: Optional[Decimal] = None
    timestamp: datetime = datetime.now(timezone.utc)
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: Decimal = Decimal("0")
    remaining_quantity: Optional[Decimal] = None
    activated: bool = True

    def __post_init__(self):
        self.activated = True
        if self.remaining_quantity is None:
            self.remaining_quantity = self.quantity
        if self.quantity <= 0:
            raise ValueError("Quantity must be positive")
        if self.order_type in [OrderType.LIMIT, OrderType.IOC, OrderType.FOK, OrderType.STOP_LIMIT]:
            if not self.price or self.price <= 0:
                raise ValueError("Price must be positive for limit/IOC/FOK orders")
        if self.order_type in [OrderType.STOP_LOSS, OrderType.STOP_LIMIT, OrderType.TAKE_PROFIT]:
            self.activated = False

    def fill(self, qty: Decimal):
        self.filled_quantity += qty
        self.remaining_quantity -= qty
        self.status = OrderStatus.FILLED if self.remaining_quantity == 0 else OrderStatus.PARTIAL_FILLED

    def cancel(self):
        self.status = OrderStatus.CANCELLED

    def reject(self):
        self.status = OrderStatus.REJECTED

    def activate(self):
        self.activated = True
    
    def is_active(self) -> bool:
        return self.order_type not in {
        OrderType.STOP_LOSS,
        OrderType.STOP_LIMIT,
        OrderType.TAKE_PROFIT
    } or self.activated

    def is_triggered(self, price: Decimal) -> bool:
        if self.order_type == OrderType.STOP_LOSS:
            return (self.side == OrderSide.BUY and price >= self.trigger_price) or (self.side == OrderSide.SELL and price <= self.trigger_price)
        if self.order_type == OrderType.TAKE_PROFIT:
            return (self.side == OrderSide.BUY and price <= self.trigger_price) or (self.side == OrderSide.SELL and price >= self.trigger_price)
        if self.order_type == OrderType.STOP_LIMIT:
            return (self.side == OrderSide.BUY and price >= self.trigger_price) or (self.side == OrderSide.SELL and price <= self.trigger_price)
        return False

# === Trade ===
@dataclass
class Trade:
    trade_id: str
    symbol: str
    price: Decimal
    quantity: Decimal
    timestamp: datetime
    aggressor_side: OrderSide
    maker_order_id: str
    taker_order_id: str
    maker_fee: Decimal
    taker_fee: Decimal

    @classmethod
    def create(cls, symbol, price, quantity, aggressor, maker, taker):
        return cls(
            trade_id=str(uuid.uuid4()),
            symbol=symbol,
            price=price,
            quantity=quantity,
            timestamp=datetime.now(timezone.utc),
            aggressor_side=aggressor,
            maker_order_id=maker.order_id,
            taker_order_id=taker.order_id,
            maker_fee=quantity * price * FEE_MAKER,
            taker_fee=quantity * price * FEE_TAKER
        )

    def to_dict(self):
        return {
            "timestamp": self.timestamp.isoformat() + "Z",
            "symbol": self.symbol,
            "trade_id": self.trade_id,
            "price": str(self.price),
            "quantity": str(self.quantity),
            "aggressor_side": self.aggressor_side.value,
            "maker_order_id": self.maker_order_id,
            "taker_order_id": self.taker_order_id,
            "maker_fee": str(self.maker_fee),
            "taker_fee": str(self.taker_fee)
        }

class PriceLevel:
    def __init__(self, price: Decimal):
        self.price = price
        self.orders: deque[Order] = deque()
        self.total_quantity = Decimal("0")

    def add(self, order: Order):
        self.orders.append(order)
        self.total_quantity += order.remaining_quantity

    def remove(self, order: Order):
        self.orders.remove(order)
        self.total_quantity -= order.remaining_quantity

    def update_quantity(self, quantity: Decimal):
        self.total_quantity -= quantity

    def is_empty(self):
        return not self.orders

class OrderBookSide:
    def __init__(self, is_bid: bool):
        self.is_bid = is_bid
        self.levels: Dict[Decimal, PriceLevel] = {}
        self.prices: List[Decimal] = []

    def add_order(self, order: Order):
        price = order.price
        if price not in self.levels:
            self.levels[price] = PriceLevel(price)
            self.prices.append(price)
            self.prices.sort(reverse=self.is_bid)
        self.levels[price].add(order)

    def remove_order(self, order: Order):
        price = order.price
        level = self.levels.get(price)
        if level:
            level.remove(order)
            if level.is_empty():
                del self.levels[price]
                self.prices.remove(price)

    def best_price(self) -> Optional[Decimal]:
        return self.prices[0] if self.prices else None

    def get_next_order(self) -> Optional[Order]:
        if not self.prices:
            return None
        best = self.levels[self.prices[0]]
        return best.orders[0] if best.orders else None

    def update_after_fill(self, price: Decimal, qty: Decimal):
        if price in self.levels:
            self.levels[price].update_quantity(qty)

    def get_depth(self, levels=10):
        return [[str(price), str(self.levels[price].total_quantity)] for price in self.prices[:levels]]

class OrderBook:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.bids = OrderBookSide(is_bid=True)
        self.asks = OrderBookSide(is_bid=False)
        self.orders: Dict[str, Order] = {}
        self.trades: List[Trade] = []
        self.lock = asyncio.Lock()

    def add_order(self, order: Order):
        self.orders[order.order_id] = order
        side = self.bids if order.side == OrderSide.BUY else self.asks
        side.add_order(order)

    def remove_order(self, order_id: str):
        order = self.orders.get(order_id)
        if order:
            side = self.bids if order.side == OrderSide.BUY else self.asks
            side.remove_order(order)
            del self.orders[order_id]

    def get_order(self, order_id: str) -> Optional[Order]:
        return self.orders.get(order_id)

    def get_depth_snapshot(self, levels=10):
        return {
            "symbol": self.symbol,
            "bids": self.bids.get_depth(levels),
            "asks": self.asks.get_depth(levels),
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z"
        }

    def get_bbo(self) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        best_bid = self.bids.best_price()
        best_ask = self.asks.best_price()
        return best_bid, best_ask

    def get_market_data(self) -> dict:
        bid, ask = self.get_bbo()
        return {
            "symbol": self.symbol,
            "best_bid": str(bid) if bid else None,
            "best_ask": str(ask) if ask else None,
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z"
        }

class MatchingEngine:
    def __init__(self):
        self.order_books: Dict[str, OrderBook] = {}
        self.trades: List[Trade] = []
        self.start_time = datetime.now(timezone.utc)
        self.orders_processed = 0
        self.trades_executed = 0
        self.lock = asyncio.Lock()

    def get_or_create_order_book(self, symbol: str) -> OrderBook:
        if symbol not in self.order_books:
            self.order_books[symbol] = OrderBook(symbol)
        return self.order_books[symbol]

    def get_bbo_price(self, book: OrderBook, side: OrderSide) -> Optional[Decimal]:
        return book.asks.best_price() if side == OrderSide.BUY else book.bids.best_price()

    def match_order(self, order: Order) -> List[Trade]:
        trades = []
        book = self.get_or_create_order_book(order.symbol)
        ob_side = book.asks if order.side == OrderSide.BUY else book.bids

        while order.remaining_quantity > 0:
            best_order = ob_side.get_next_order()
            if not best_order:
                break
            price_check = (order.side == OrderSide.BUY and (order.order_type == OrderType.MARKET or order.price >= best_order.price)) or \
                          (order.side == OrderSide.SELL and (order.order_type == OrderType.MARKET or order.price <= best_order.price))
            if not price_check:
                break
            qty = min(order.remaining_quantity, best_order.remaining_quantity)
            order.fill(qty)
            best_order.fill(qty)
            ob_side.update_after_fill(best_order.price, qty)
            trade = Trade.create(order.symbol, best_order.price, qty, order.side, best_order, order)
            trades.append(trade)
            book.trades.append(trade)
            self.trades.append(trade)
            self.trades_executed += 1
            if best_order.status == OrderStatus.FILLED:
                book.remove_order(best_order.order_id)
        if order.remaining_quantity > 0 and order.order_type in [OrderType.LIMIT]:
            book.add_order(order)
        elif order.remaining_quantity > 0 and order.order_type in [OrderType.MARKET, OrderType.IOC, OrderType.FOK]:
            order.cancel()
        if order.status == OrderStatus.FILLED:
            if order.order_id in book.orders:
                book.remove_order(order.order_id)
        return trades

    async def submit_order(self, order: Order) -> dict:
        async with self.lock:
            start = time.perf_counter()
            book = self.get_or_create_order_book(order.symbol)
            self.orders_processed += 1
            trades = []

            # Trigger activation
            if not order.is_active():
                bbo_price = self.get_bbo_price(book, order.side)
                if bbo_price and order.is_triggered(bbo_price):
                    order.activate()
                    if order.order_type == OrderType.STOP_LIMIT:
                        order.order_type = OrderType.LIMIT
                    else:
                        order.order_type = OrderType.MARKET

            if order.is_active():
                trades = self.match_order(order)

            end = time.perf_counter()
            return {
                "order_id": order.order_id,
                "status": order.status.value,
                "original_quantity": str(order.quantity),
                "filled_quantity": str(order.filled_quantity),
                "remaining_quantity": str(order.remaining_quantity),
                "fills": [t.to_dict() for t in trades],
                "latency_ms": round((end - start) * 1000, 4)
            }

    def get_performance_stats(self):
        uptime = datetime.now(timezone.utc) - self.start_time
        return {
            "uptime_seconds": int(uptime.total_seconds()),
            "orders_processed": self.orders_processed,
            "trades_executed": self.trades_executed,
            "orders_per_second": round(self.orders_processed / max(uptime.total_seconds(), 1), 2)
        }

    def persist(self):
        with open(PERSISTENCE_FILE, "wb") as f:
            pickle.dump(self.order_books, f)

    def restore(self):
        if PERSISTENCE_FILE.exists():
            with open(PERSISTENCE_FILE, "rb") as f:
                self.order_books = pickle.load(f)

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    engine.persist()

app = FastAPI(lifespan=lifespan)
engine = MatchingEngine()
engine.restore()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class OrderRequest(BaseModel):
    symbol: str
    order_type: OrderType
    side: OrderSide
    quantity: Decimal
    price: Optional[Decimal] = None
    trigger_price: Optional[Decimal] = None

@app.post("/orders")
async def submit_order(req: OrderRequest):
    order = Order(
        order_id=str(uuid.uuid4()),
        symbol=req.symbol,
        order_type=req.order_type,
        side=req.side,
        quantity=req.quantity,
        price=req.price,
        trigger_price=req.trigger_price
    )
    order.__post_init__()
    return await engine.submit_order(order)

@app.get("/stats")
def performance():
    return engine.get_performance_stats()

@app.get("/orderbook/{symbol}")
def get_orderbook(symbol: str):
    book = engine.get_or_create_order_book(symbol)
    return book.get_depth_snapshot()

@app.get("/bbo/{symbol}")
def get_bbo(symbol: str):
    return engine.get_or_create_order_book(symbol).get_market_data()

#WebSocket Clients
trade_clients: Set[WebSocket] = set()
orderbook_clients: Set[WebSocket] = set()

@app.websocket("/ws/trades")
async def ws_trades(websocket: WebSocket):
    await websocket.accept()
    trade_clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        trade_clients.remove(websocket)

@app.websocket("/ws/orderbook/{symbol}")
async def ws_orderbook(websocket: WebSocket, symbol: str):
    await websocket.accept()
    orderbook_clients.add(websocket)
    try:
        while True:
            snapshot = engine.get_or_create_order_book(symbol).get_depth_snapshot()
            await websocket.send_text(json.dumps(snapshot))
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        orderbook_clients.remove(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)