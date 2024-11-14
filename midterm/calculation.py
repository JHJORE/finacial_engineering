from scipy.optimize import fsolve

def bond_price_equation(r):
    price = 1028.50
    cash_flows = [80, 80, 80, 80, 80, 1080]
    times = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    
    # Calculate the sum of the discounted cash flows
    discounted_cash_flows = sum(cash_flow / (1 + r) ** time for cash_flow, time in zip(cash_flows, times))
    return discounted_cash_flows - price

# Use fsolve to find the value of r (the internal yield of return)
initial_guess = 0.05 
ytm_solution = fsolve(bond_price_equation, initial_guess)

print(ytm_solution[0] * 100 ) # Return YTM as a percentage