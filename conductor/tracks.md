# Project Tracks

This file tracks all major tracks for the project. Each track has its own detailed plan in its respective folder.

---

- [ ] **Track: Implement initial data acquisition and preprocessing for Bitcoin time series analysis.**
  *Link: [./tracks/data_acquisition_preprocessing_20260217/](./tracks/data_acquisition_preprocessing_20260217/)*

---

- [x] **Track: Your task is to create a trading bot in the src/app directory (create it). It should trade on the spot market using the binance api (configure it for using the test api). It shoul be able to use several trading strategies. Propose a strategy interface. Start by implementing a very basic MACD strategy.
The requirements are
### **1. Functional Requirements**
* **1.1. Autonomous Execution:** The core trading loop must execute asynchronously and autonomously, requiring no human intervention once the initialization phase is complete.
* **1.2. Exchange Authentication:** The system must securely ingest and manage user credentials (e.g., via `.env` file or secure secret manager) to authenticate with the target cryptocurrency exchange API.
* **1.3. Graceful Shutdown:** The system must implement signal handling (e.g., trapping `SIGINT`/`SIGTERM`) to halt operations securely. It must close open network connections, resolve pending states, and prevent orphaned or partial transactions before exiting.
* **1.4. Configurable Risk Management:** The system must expose adjustable parameters (e.g., via a `config.json` or YAML file) to define specific risk tolerance thresholds (e.g., maximum drawdown, stop-loss percentages, volatility limits).
* **1.5. Capital Allocation Limits:** The system must enforce a strict, user-defined maximum capital allocation limit (fiat or crypto base asset) to restrict total exposure during trading operations.

### **2. Documentation Requirements**
* **2.1. Lifecycle Documentation:** A `README.md` must detail the exact CLI commands, prerequisites, and operational flows required to initialize, start, and safely terminate the system.
* **2.2. Configuration Schema:** Provide comprehensive documentation for all configurable parameters. This must include acceptable data types, default values, and a detailed explanation of how each parameter alters the system's execution and risk assessment logic.

### **3. Testing Requirements**
* **3.1. API Integration Testing:** Develop automated integration tests against the target exchange's API (utilizing a testnet or sandbox environment where applicable) to verify reliable authentication, data ingestion, and order execution.
* **3.2. Business Logic Unit/Component Testing:** Implement isolated component tests to validate core domain logic. Coverage must explicitly verify that the system strictly adheres to the defined maximum capital allocation and risk limit constraints.
* **3.3. Model Backtesting & Evaluation:** Build an evaluation pipeline to assess the predictive model's efficacy. The pipeline must ingest historical market data to backtest the model and output performance metrics.
* **3.4. Telemetry & Monitoring:** Implement a logging and monitoring framework to track system health, operational performance, execution latency, and error rates in real-time.

### **4. Interface Requirements**
* **4.1. CLI Implementation:** Develop a Command Line Interface (CLI) application exposing robust commands (e.g., `start`, `stop`, `status`) to manage the system's operational lifecycle.
* **4.2. Pre-flight Validation & Error Handling:** The CLI must perform pre-flight checks upon initialization. If required configurations, credentials, or parameters are missing or malformed, the system must abort startup and return clear, actionable, and strictly typed error messages to `stderr`.**
  *Link: [./tracks/trading_bot_20260331/](./tracks/trading_bot_20260331/)*