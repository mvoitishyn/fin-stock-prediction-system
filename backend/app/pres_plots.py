import numpy as np
import matplotlib.pyplot as plt
from model import load_data, featurize_data
import pandas as pd
import matplotlib.dates as mdates


# random-walk simulation

n_steps = 1000
n_walks = 5

FONT_SIZE = 14  # or whatever size you prefer


def simulate_random_walk(n_steps):
    steps = np.random.choice([-1, 1], size=n_steps)
    return np.cumsum(steps)



plt.figure()
for i in range(n_walks):
    walk = simulate_random_walk(n_steps)
    plt.plot(walk)
plt.title("5 Simulated 1D Random Walks", color="black", fontsize = FONT_SIZE)
plt.xlabel("Step", color = "black",fontsize = FONT_SIZE)
plt.ylabel("Position", color = "black",fontsize = FONT_SIZE)
plt.tick_params(axis='x', colors='black',labelsize = FONT_SIZE)
plt.tick_params(axis='y', colors='black',labelsize = FONT_SIZE)
plt.savefig("../presentationplots/random_walk_5_sim.png", transparent=True)
plt.close()

final_positions = [simulate_random_walk(n_steps)[-1] for _ in range(10000)]
plt.figure()
plt.hist(final_positions, bins=50)
plt.title(f"Distribution of Final Positions after {n_steps} Steps", color='black', fontsize = FONT_SIZE)
plt.xlabel("Position", color = "black", fontsize=FONT_SIZE)
plt.ylabel("Frequency", color = "black", fontsize = FONT_SIZE)
plt.tick_params(axis='x', colors='black', labelsize = FONT_SIZE)
plt.tick_params(axis='y', colors='black', labelsize = FONT_SIZE)
plt.savefig("../presentationplots/random_walk_distribution.png", transparent=True)
plt.close()


# distribution of closepricereturn
rawdf = load_data("INTC")
X_train, X_test, y_train, y_test = featurize_data(
    df=rawdf,
    use_baseline_data=True, 
    use_macro_data=True, 
    use_extra_data=True
)

processeddf = pd.read_csv("../data/processed/stocksdata/INTC_processed.csv")

result = np.vstack((y_train, y_test))

plt.hist(result, bins=100)
plt.title(f"Distribution of Close Price Return for Intel Corporation(INTC)", color='black')
plt.xlabel("Return", color = "black", fontsize = FONT_SIZE)
plt.ylabel("Frequency", color = "black", fontsize = FONT_SIZE)
plt.tick_params(axis='x', colors='black', labelsize = FONT_SIZE)
plt.tick_params(axis='y', colors='black', labelsize = FONT_SIZE)
plt.savefig("../presentationplots/distribution_close_price_return.png", transparent=True)
plt.close()

dates = pd.to_datetime(rawdf["Date"])
plt.plot(dates, rawdf["Close"])
plt.title(f"Raw Adjusted Close Price for Intel Corporation(INTC)", color='black')
plt.xlabel("Date", color = "black", fontsize = FONT_SIZE)
plt.ylabel("Adj Close Price", color = "black", fontsize = FONT_SIZE)
plt.tick_params(axis='x', colors='black', labelsize = 12)
plt.tick_params(axis='y', colors='black', labelsize = FONT_SIZE)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=24))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.tight_layout()
plt.savefig("../presentationplots/close_price_raw.png", transparent=True)
plt.close()


# volatility next-day simulation

mu = 0                  
sigmas = [0.01, 0.05, 0.1]  
N = 10000
base_price = 200

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for ax, sigma in zip(axes, sigmas):
    returns = np.random.normal(mu, sigma, size=N)
    prices = base_price * (1 + returns)
    
    p_low, p_high = np.percentile(prices, [0.1, 99.9])
    
    x = np.linspace(p_low, p_high, 1000)
    r = (x / base_price - 1)
    pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-r**2 / (2 * sigma**2)) / base_price
    
    ax.hist(prices, bins=50, density=True, alpha=0.6)
    ax.plot(x, pdf, lw=2)
    ax.set_xlim(p_low, p_high)
    ax.set_title(f"σ = {sigma*100:.0f}%", color='black', fontsize = FONT_SIZE)
    ax.set_xlabel('Price ($)', color='black', fontsize = FONT_SIZE)
    ax.tick_params(colors='black', labelsize = FONT_SIZE)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')

axes[0].set_ylabel('Density', color='black',fontsize=FONT_SIZE)
axes[0].tick_params(colors='black', labelsize=FONT_SIZE)

fig.suptitle('Next-Day Price Distributions for Varying Volatilities', color='black', fontsize=16)

fig.patch.set_facecolor('black')
for ax in axes:
    ax.set_facecolor('black')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

plt.savefig("../presentationplots/volatility_sim.png", transparent=True)

from math import erf, sqrt

def norm_cdf(x):
    """Standard normal cumulative distribution function."""
    return 0.5 * (1 + erf(x / sqrt(2)))

sigmas = [0.01, 0.05, 0.1]  # 1%, 5%, 10%
threshold_return = 0.05    # 5% gain to reach $210 from $200

# compute probabilities
for sigma in sigmas:
    z = threshold_return / sigma
    probability = 1 - norm_cdf(z)
    print(f"σ = {sigma*100:.0f}%: z = {z:.2f}, P(price ≥ $210) = {probability:.6f}")

