# Security and trading safety

This repository is a research and portfolio project. It does not enable live trading.

- No exchange API key or secret is required for the included demo.
- Credentials, environment files, databases, logs, and model artifacts are excluded by `.gitignore`.
- The forward observer uses public market data and records `orders_sent: 0`.
- A research result never enables deployment automatically.
- Do not add real credentials to source code, Docker Compose files, screenshots, or committed reports.

If this project is extended to an exchange testnet, use least-privilege keys with withdrawal disabled and keep all secrets outside the repository.
