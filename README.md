# Implied Volatility Surface Generator using Black-Scholes Model and Brent’s Method

[Visit the Application](https://theivsurface.streamlit.app/)

[Full Report PDF](./docs/Implied%20Volatility%20Surface%20Report.pdf)

---

**Author**: Micah Yihun Aldrich  
**University**: University of Illinois Urbana-Champaign

---

## Abstract

Breaking into quantitative finance has become increasingly competitive. With distributed computing systems and the growing number of technically skilled applicants, financial institutions like JPMorgan and trading firms like Jane Street now accept a very small percentage of intern applications.

This project focuses on developing a solution to demonstrate expertise in **options theory**, using the **Black-Scholes model** and **Brent’s method** to generate an **implied volatility (IV) surface** for stock options. The application is deployed on **Streamlit** and showcases proficiency in **full-stack development** and **data visualization**.

---

## Introduction

Options are derivatives that derive value from an underlying asset. **Implied Volatility (IV)** refers to the forecasted volatility of the asset based on the market price of its options. 

The **IV surface** is a 3D plot showing how volatility varies across strike prices and expiration dates for options on the same underlying asset. This project visualizes the **IV surface**, allowing traders to gain insights and uncover potential arbitrage opportunities.

---

## Methodology

**Ticker Data**:  
- Retrieved using the **yfinance** library, which fetches data from the Yahoo Finance API.

**Tools & Libraries**:
- **Streamlit**: For building the web interface.
- **Plotly**: For 3D plotting of the IV surface.
- **SciPy**: For performing the numerical analysis using **Brent’s Method**.

**Black-Scholes Model & Brent’s Method**:
- The backend calculates the theoretical price of an option using the Black-Scholes model.
- **Brent’s Method** (root-finding algorithm) is used to compute implied volatility by solving the difference between the market price and the theoretical price.

**IV Surface Plot**:
- The 3D surface plot visualizes **Implied Volatility** across strike prices and time to expiration.
- The x-axis = Time to Expiration  
- The y-axis = Strike Price / Moneyness  
- The z-axis = Implied Volatility  

---

## Implementation

### Web Application Design:
- A **Streamlit** interface allows users to input parameters like ticker symbol, risk-free rate, dividend yield, etc.
- The application dynamically updates the IV surface in real-time.

### Backend Architecture:
- The backend retrieves live option data from **Yahoo Finance**.
- It calculates the implied volatility using **Black-Scholes** and **Brent’s Method** and updates the surface plot using **Plotly**.

---

## Results

As of **October 24, 2024**, the project results in a web application where users can input parameters and visualize an interactive 3D Implied Volatility surface. 

Key features:
- Intuitive parameter input via sidebar.
- Dynamic, real-time updates of the IV surface.

**Example: SPY Ticker Display**  
**Example: NVDA Ticker Display**

<img width="1440" alt="Screenshot 2024-10-24 at 3 25 45 PM" src="https://github.com/user-attachments/assets/f073f0f1-255e-4891-9e35-821ef8246bdb">


---

## Future Work

Future iterations will explore:
- **Arbitrage Strategies**: Using the IV surface to identify trading opportunities.
- **Snapshots**: Capture the IV surface at regular intervals and store them in a database for historical analysis.
- **Replay Feature**: Allow users to replay historical IV surfaces to observe how volatility evolves over time.

---
