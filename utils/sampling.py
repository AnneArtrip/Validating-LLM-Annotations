
import pandas as pd

def random_sampling(data,
                    N,
                    method='random', # {'random', 'stratified'}
                    seed=1
                    ):

    if N>=len(data):
        raise ValueError('N must be smaller than the dimension of the data set!')

    # Random sampling
    if method=='random':
        data_subsample = data.sample(n=N, random_state=seed)  
    
    # Stratified sampling
    elif method=='stratified':

        # Get the distribution of labels in the original dataframe
        original_distribution = data['human_label'].value_counts(normalize=True)

        # Calculate how many samples to take from each category to maintain distribution
        sample_sizes = (original_distribution * 100).round().astype(int)
                
        # Adjust if the sum doesn't exactly equal 100
        diff = 100 - sample_sizes.sum()
        if diff != 0:
            # Add/subtract the difference to/from the largest category
            largest_category = original_distribution.index[0]
            sample_sizes[largest_category] += diff

        # Create an empty dataframe for the stratified sample
        data_subsample = pd.DataFrame(columns=data.columns)

        # Sample from each category according to calculated sizes
        for label, size in sample_sizes.items():
            if size > 0:  # Only sample if we need at least one row
                label_df = data[data['human_label'] == label]
                # If we need more samples than available, sample with replacement
                replace = size > len(label_df)
                sampled = label_df.sample(n=size, random_state=seed, replace=replace)
                data_subsample = pd.concat([data_subsample, sampled])

        # Shuffle the final dataframe to randomize the order
        data_subsample = data_subsample.sample(frac=1, random_state=seed).reset_index(drop=True)

        # Verify the distribution
        #verification = data_subsample['human_label'].value_counts(normalize=True)

    else:
        raise ValueError('Provided method is not valid!')

    return(data_subsample)

