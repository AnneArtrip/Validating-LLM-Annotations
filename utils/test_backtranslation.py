import pandas as pd

def test_backtranslation(
        annotation_func, 
        simulation_func,
        labels,
        tested_labels = None,
        num_sim = 100):

    # Preallocation
    results = []

    if tested_labels==None:
        tested_labels=labels

    if not isinstance(tested_labels,list):
        tested_labels = [tested_labels]
        
    # Simulation (for each label)
    for label in tested_labels:
        
        # Simulate all texts
        simulated_texts = simulation_func(label,num_sim)

        for nn in range(0,num_sim):

            # Step 1: Simulate text
            try:
                
                simulated_text = simulated_texts[nn]
                
                # Step 2: Generate label
                label_hat = annotation_func(simulated_text)
                
                # Append to results
                results.append({"iteration": nn,
                                "label": label,
                                "failed": False,
                                "simulated_text": simulated_text,
                                "label_hat": label_hat
                                })

            except Exception:
                results.append({"iteration": nn,
                                "label": label,
                                "failed": True,
                                "simulated_text": None,
                                "label_hat": None
                                })

    df = pd.DataFrame(results)
    #misclassified = df[df["label"] != df["label_hat"]]

    return df
    