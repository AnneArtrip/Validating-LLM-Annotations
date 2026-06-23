import pandas as pd
from datetime import datetime
from statistics import mean
from utils.llm_api_calls import llm_embedding
from utils.cosine_similarity import cosine_similarity

def test_validity(
        filename,
        simulation_func,
        label_hat,
        emb_original,
        simulated_text = None,
        num_sim_tau = 100,
        num_sim_boot = 100,
        alpha = 0.05):
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H%M%S")

    # Simulate text and embedding given provided label
    if simulated_text == None:
        simulated_text = simulation_func(label_hat,num_sim_tau)
    emb_simulated = [llm_embedding(simulated_text[nn]) for nn in range(num_sim_tau)]

    # Test statistic
    tau = mean([cosine_similarity(emb_original, emb_simulated[nn]) for nn in range(num_sim_tau)])

    # Inference: #
    bootstrap_text = simulation_func(label_hat,num_sim_boot)

    results = []
    for nn in range(0,num_sim_boot):
        
        # Simulate boostrap text statistic
        emb_bootstrap = llm_embedding(bootstrap_text[nn])
        tau_bootstrap = mean([cosine_similarity(emb_simulated[ii], emb_bootstrap) for ii in range(num_sim_tau)])

        # Append to results
        results.append({
            "iteration": nn,
            "label": label_hat,
            "simulated_text": simulated_text,
            "bootstrap_text": bootstrap_text[nn],
            "tau_bootstrap": tau_bootstrap})

    df = pd.DataFrame(results)
    crit = df.tau_bootstrap.quantile(alpha)
    reject = True if crit > tau else False

    out = pd.DataFrame({
        'crit': [crit],
        'tau': [tau],
        'reject': [reject]}, index=[0])
    
    return out, df
