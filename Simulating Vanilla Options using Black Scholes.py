
import math
from scipy.stats import norm
import random

def calc_discounted_call_intrinsic_value(S, K, r, T)  :
    
    discounted_call_intrinsic_value = max(0,(S-K)*math.exp(-r*T))

    return discounted_call_intrinsic_value



def calc_discounted_put_intrinsic_value(S, K, r, T) :
    
    discounted_put_intrinsic_value = max(0,(K-S)*math.exp(-r*T))  

    return discounted_put_intrinsic_value



def calc_call_bs_value(S, K, r, T, sigma)  :
    
    d1 = (math.log(S/K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = (math.log(S/K) + (r - 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))

    bs_call = (S * norm.cdf(d1, 0.0, 1.0) - K * math.exp(-r * T) * norm.cdf(d2, 0.0, 1.0))

    return bs_call



def calc_put_bs_value(S, K, r, T, sigma)  :
    
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = (math.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))

    bs_put = (K * math.exp(-r * T) * norm.cdf(-d2, 0.0, 1.0) - S * norm.cdf(-d1, 0.0, 1.0))

    return bs_put



def simulate_returns(mu, sigma, dt, num_sims) -> list :

    List_returns = [0] * num_sims 
    drift = (mu - 0.5 * sigma**2)*dt
    vol = sigma * math.sqrt(dt)

        
    for i in range(num_sims):
        r = drift + vol * random.gauss(0.0 , 1.0)
        List_returns[i] = r
 
    return List_returns



def evaluate_options(S, K, r, T, sigma, num_sims) :
    
    List_returns = simulate_returns(r, sigma, T, num_sims)
    List_S_end = [0]*num_sims

    for i in range(len(List_returns)):
        ret = List_returns[i]
        S_end = S * math.exp(ret)
        List_S_end[i] = S_end 

    sum_call_discount_intrinsic = 0
    sum_put_discount_intrinsic = 0

    for i in range(len(List_S_end)):
        sum_call_discount_intrinsic += calc_discounted_call_intrinsic_value(List_S_end[i], K, r, T)

    for i in range(len(List_S_end)):
        sum_put_discount_intrinsic += calc_discounted_put_intrinsic_value(List_S_end[i], K, r, T)

    
    average_sum_call_discount_intrinsic = sum_call_discount_intrinsic / num_sims         
    average_sum_put_discount_intrinsic = sum_put_discount_intrinsic / num_sims
    
    bs_call = calc_call_bs_value(S, K, r, T, sigma)
    bs_put = calc_put_bs_value(S, K, r, T, sigma)
    
    return {
        'call_bs' : bs_call,
        'call_estimate' : average_sum_call_discount_intrinsic,
        'put_bs' : bs_put,
        'put_estimate' : average_sum_put_discount_intrinsic,
        'num_sims' : num_sims
      }
    
    

def main() :
    
    random.seed(1234)
    r = 0.05
    T = 0.5
    sigma = 0.25
    S = 100
    K = 100

    try:
        with open('assignment3_results.csv', 'w') as f :
            f.write("NUM_SIMS,BS_CALL,EST_CALL,DIFF_CALL,BS_PUT,EST_PUT,DIFF_PUT\n")
            for num_sims in [100, 1000, 10000, 100000, 1000000] :
                print("Running", num_sims)
                test_eval = evaluate_options(S, K, r, T, sigma, num_sims)
                f.write(f"{num_sims},{test_eval['call_bs']:.2f},{test_eval['call_estimate']:.2f},{(test_eval['call_bs']-test_eval['call_estimate']):.2f}")
                f.write(f",{test_eval['put_bs']:.2f},{test_eval['put_estimate']:.2f},{(test_eval['put_bs']-test_eval['put_estimate']):.2f}\n")

    except Exception as E:
        print("We ran into trouble")
        print(E)

if __name__ == "__main__" :
    main()
