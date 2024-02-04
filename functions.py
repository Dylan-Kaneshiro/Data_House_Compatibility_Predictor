import numpy as np

# Used to test if all the listed attributes of all team members and applicants are the same
def all_equal(attribute_list):
    return len(set(attribute_list)) == 1

# I decided to treat all attributes as independent from eachother, since large negative covariances between attributes made the cdf go to 0 very quickly
def make_independent(cov_matrix):
    dim = len(cov_matrix)
    for i in range(dim):
        for j in range(dim):
            if not i==j:
                cov_matrix[i,j] = 0
    return cov_matrix

# Define compatibility as the geometric mean of the probabilities for each attribute that a person on the team will have a lower value
def compatibility(model, attributes, candidate_dict):
    num_attributes = len(attributes)
    candidate_values = [candidate_dict[attribute] for attribute in attributes]
    return model.cdf(np.array(candidate_values))**(1/num_attributes)