import json
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
import numpy as np
import gradio as gr

from functions import *

def main(file):

    # Read input file
    with open(file.name) as json_file:
        data = json.load(json_file)

    # Validate input
    team_stats = [person['attributes'] for person in data['team']]
    applicant_stats = [person['attributes'] for person in data['applicants']]
    if len(team_stats) < 2:
        print('Input file must have at least 2 team members to score applicants')
        return None
    if len(team_stats[0]) == 0:
        print('Team members must have at least 1 attribute')
        return None
    if not all_equal([tuple(person.keys()) for person in team_stats + applicant_stats]):
        print('Input file has incomplete data. Please input all attributes for all people')
        return None
    
    
    # Build model
    # Model each attribute as a normal distribution, with each attribute independent from eachother
    attributes = tuple(team_stats[0].keys())
    dist = [[person[attribute] for person in team_stats] for attribute in attributes]
    multivariate_normal_model = {
        "attributes": attributes,
        "distribution": dist,
        "means":[np.mean(feature) for feature in dist],
        "covariance_matrix": make_independent(np.cov(dist))
    }
    model = mvn(mean=multivariate_normal_model['means'],
            cov=multivariate_normal_model['covariance_matrix'], allow_singular=True)
    
    # Calculate scores
    scores = [{'name': applicant['name'], 
               'score': compatibility(model, multivariate_normal_model['attributes'], applicant['attributes'])} 
               for applicant in data['applicants']]


    # Write results to scores.json
    json_object = json.dumps({"scoredApplicants": scores}, indent=4)
    with open("scores.json", "w") as outfile:
        outfile.write(json_object)

    return "scores.json"

demo = gr.Interface(main, "file", "file")

demo.launch()