# Cryptocurrency-Matching-Engine
# ⚡ Crypto Matching Engine

A high-performance, real-time cryptocurrency matching engine built with **FastAPI**, featuring advanced order types, WebSocket trade broadcasting, fee logic, and persistent state recovery.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-green)
![Status](https://img.shields.io/badge/status-production--ready-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

---

## 🚀 Features

- ✅ **Real-Time Matching Engine**
- ✅ **Order Types**: Market, Limit, IOC, FOK, Stop-Loss, Stop-Limit, Take-Profit
- ✅ **Maker-Taker Fee Model**
- ✅ **RESTful APIs** for BBO, Order Book, and Order Submission
- ✅ **WebSocket Broadcasts** for live trade updates
- ✅ **Persistence**: Automatic recovery using JSON-based state file
- ✅ **Test Coverage**: PyTest-based unit and integration tests

---

## 🏗️ Architecture

```text
FastAPI
│
├── REST Endpoints
│   ├── /submit-order/
│   ├── /orderbook/{symbol}
│   └── /best-bid-offer/{symbol}
│
├── WebSocket Endpoint
│   └── /ws
│
├── OrderBook
│   ├── Add, Remove, Match orders
│   └── Match logic for each order type
│
├── Persistence Layer
│   └── Saves/Loads order book to state.json
│
└── Test Suite
    └── tests.py (PyTest + httpx)
```
| Type          | Description                               |
| ------------- | ----------------------------------------- |
| `market`      | Executes immediately at best price        |
| `limit`       | Matches or adds to order book             |
| `ioc`         | Immediate-or-cancel, partial fill allowed |
| `fok`         | Fill-or-kill, requires full fill          |
| `stop_loss`   | Triggers when price crosses stop          |
| `stop_limit`  | Triggers limit order when stop is hit     |
| `take_profit` | Triggers when price exceeds target        |
