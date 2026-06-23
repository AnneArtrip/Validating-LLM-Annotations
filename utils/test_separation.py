import pandas as pd
from datetime import datetime
from itertools import product
from statistics import mean
from utils.llm_api_calls import llm_embedding
from utils.cosine_similarity import cosine_similarity

def test_separation(
        filename,
        simulation_func,
        label,
        counterfactual_labels, # Other eligible labels not equal to label_hat
        simulated_text = None,
        num_sim_tau = 100,
        num_sim_boot = 100,
        alpha = 0.05):

    timestamp = datetime.now().strftime("%Y-%m-%d %H%M%S")

    # Simulate text and embedding given provided label
    if simulated_text == None:
        simulated_text = simulation_func(label,num_sim_tau)

    num_sim_tau1 = len(simulated_text)
    emb_simulated = [llm_embedding(simulated_text[nn]) for nn in range(num_sim_tau1)]

    # Loop over each counterfactual label
    out = []
    out_sim = []
    for label_ in counterfactual_labels:

        # Counterfactual text and embedding
        counterfactual_text = simulation_func(label_,num_sim_tau)
        emb_counterfactual = [llm_embedding(counterfactual_text[nn]) for nn in range(num_sim_tau)]

        # Compute test statistic
        tau = mean([cosine_similarity(emb_counterfactual[n2], emb_simulated[n1])
                    for n2, n1 in product(range(num_sim_tau), range(num_sim_tau1))])

        # Inference: # 
        bootstrap_text = simulation_func(label_,num_sim_boot)
        results = []
        for nn in range(0,num_sim_boot):

            # Resimulate iteration 
            num_tries = 0
            max_num_tries = 10
            while bootstrap_text[nn]==None and num_tries<max_num_tries:
                bootstrap_text[nn] = simulation_func(label_,1)[0]
                num_tries = num_tries+1

            # Simulate boostrap text statistic
            emb_bootstrap = llm_embedding(bootstrap_text[nn])
            tau_bootstrap = mean([cosine_similarity(emb_bootstrap, emb_simulated[n1]) 
                    for n1 in range(num_sim_tau1)])

            # Append to results
            results.append({
                "iteration": nn,
                "label": label,
                "simulated_text": simulated_text,
                "label_": label_,
                "bootstrap_text": bootstrap_text[nn],
                "tau_bootstrap": tau_bootstrap})

        df = pd.DataFrame(results)
        crit = df.tau_bootstrap.quantile(1-alpha)
        reject = True if tau > crit else False

        out.append({
            "label": label,
            "label_": label_,
            "crit": crit,
            "tau": tau,
            "reject": reject})

        out_sim.append(df)

    return pd.DataFrame(out), out_sim
