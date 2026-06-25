import math

def norm_cdf(x):
    """Cumulative distribution function for the standard normal distribution."""
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def norm_pdf(x):
    """Probability density function for the standard normal distribution."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def d1(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return 0.0
    return (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

def d2(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return 0.0
    return d1(S, K, T, r, sigma) - sigma * math.sqrt(T)

def bs_price(S, K, T, r, sigma, option_type="CE"):
    """Calculate Black-Scholes Option Price."""
    if T <= 0:
        return max(0, S - K) if option_type == "CE" else max(0, K - S)
    
    d_1 = d1(S, K, T, r, sigma)
    d_2 = d2(S, K, T, r, sigma)
    
    if option_type == "CE":
        return S * norm_cdf(d_1) - K * math.exp(-r * T) * norm_cdf(d_2)
    else:
        return K * math.exp(-r * T) * norm_cdf(-d_2) - S * norm_cdf(-d_1)

def bs_vega(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return 0.0
    d_1 = d1(S, K, T, r, sigma)
    return S * norm_pdf(d_1) * math.sqrt(T)

def calculate_iv(target_price, S, K, T, r, option_type="CE", max_iter=100, tol=1e-5):
    """
    Calculate Implied Volatility using Newton-Raphson method.
    Returns annualized implied volatility as a decimal.
    """
    if target_price <= 0 or T <= 0: return 0.001
    
    sigma = 0.2 # Initial guess (20% IV)
    
    for i in range(max_iter):
        price = bs_price(S, K, T, r, sigma, option_type)
        diff = price - target_price
        
        if abs(diff) < tol:
            return sigma
            
        vega = bs_vega(S, K, T, r, sigma)
        if vega == 0:
            break
            
        sigma = sigma - diff / vega
        
        # Keep sigma within bounds
        if sigma <= 0: sigma = 0.001
        if sigma > 3.0: sigma = 3.0
        
    return sigma

def calculate_delta(S, K, T, r, sigma, option_type="CE"):
    """Calculate Delta for the option."""
    if T <= 0 or sigma <= 0:
        if option_type == "CE": return 1.0 if S > K else 0.0
        else: return -1.0 if S < K else 0.0
        
    d_1 = d1(S, K, T, r, sigma)
    if option_type == "CE":
        return norm_cdf(d_1)
    else:
        return norm_cdf(d_1) - 1.0
