"""
Defines the default scoring function to guide the genchem RL agent.
"""

import os
from reinvent_scoring.scoring import ScoringFunctionFactory
from reinvent_scoring.scoring.scoring_function_parameters import ScoringFunctionParameters


class WrapperScoringClass:

    def __init__(self, scoring_class):
        self.scoring_class = scoring_class

    def get_final_score(self, smile):

        output = {}

        try:

            if isinstance(smile, str):
                score = self.scoring_class.get_final_score([smile])
            else:
                raise ValueError("Scoring error due to wrong dtype")

            output.update({
                "valid_smile": True,
                "score": float(score.total_score[0]),
                "regression_model": float(score.profile[0].score[0]),
                "matching_substructure": float(score.profile[1].score[0]),
                "custom_alerts": float(score.profile[2].score[0]),
                "QED_score": float(score.profile[3].score[0]),
                "raw_regression_model": float(score.profile[4].score[0]),
            })

        except TypeError:

            output.update({
                "valid_smile": False,
                "score": 0.0,
                "regression_model": 0.0,
                "matching_substructure": 0.0,
                "custom_alerts": 0.0,
                "QED_score": 0.0,
                "raw_regression_model": 0.0,
            })

        return output


scoring_function_parameters = {
    "name": "custom_product",  # this is our default one (alternative: "custom_sum")
    "parallel": False,  # sets whether components are to be executed
    # in parallel; note, that python uses "False" / "True"
    # but the JSON "false" / "true"

    # the "parameters" list holds the individual components
    "parameters": [

        # add component: an activity model
        {
            "component_type": "predictive_property",  # this is a scikit-learn model, returning
            # activity values
            "name": "Regression model",  # arbitrary name for the component
            "weight": 2,  # the weight ("importance") of the component (default: 1)
            "specific_parameters": {
                "model_path": os.path.join(os.path.dirname(__file__),
                                           '../../../../pytorchrl/envs/generative_chemistry/reinvent/models/Aurora_model.pkl'),
                # absolute model path
                "scikit": "regression",  # model can be "regression" or "classification"
                "descriptor_type": "ecfp_counts",  # sets the input descriptor for this model
                "size": 2048,  # parameter of descriptor type
                "radius": 3,  # parameter of descriptor type
                "use_counts": True,  # parameter of descriptor type
                "use_features": True,  # parameter of descriptor type
                "transformation": {
                    "transformation_type": "sigmoid",  # see description above
                    "high": 9,  # parameter for sigmoid transformation
                    "low": 4,  # parameter for sigmoid transformation
                    "k": 0.25  # parameter for sigmoid transformation
                }
            }
        },

        # add component: enforce the match to a given substructure
        {
            "component_type": "matching_substructure",
            "name": "Matching substructure",  # arbitrary name for the component
            "weight": 1,  # the weight of the component (default: 1)
            "specific_parameters": {
                "smiles": ["c1ccccc1CC"]  # a match with this substructure is required
            }
        },

        # add component: enforce to NOT match a given substructure
        {
            "component_type": "custom_alerts",
            "name": "Custom alerts",  # arbitrary name for the component
            "weight": 1,  # the weight of the component (default: 1)
            "specific_parameters": {
                "smiles": [  # specify the substructures (as list) to penalize
                    "[*;r8]",
                    "[*;r9]",
                    "[*;r10]",
                    "[*;r11]",
                    "[*;r12]",
                    "[*;r13]",
                    "[*;r14]",
                    "[*;r15]",
                    "[*;r16]",
                    "[*;r17]",
                    "[#8][#8]",
                    "[#6;+]",
                    "[#16][#16]",
                    "[#7;!n][S;!$(S(=O)=O)]",
                    "[#7;!n][#7;!n]",
                    "C#C",
                    "C(=[O,S])[O,S]",
                    "[#7;!n][C;!$(C(=[O,N])[N,O])][#16;!s]",
                    "[#7;!n][C;!$(C(=[O,N])[N,O])][#7;!n]",
                    "[#7;!n][C;!$(C(=[O,N])[N,O])][#8;!o]",
                    "[#8;!o][C;!$(C(=[O,N])[N,O])][#16;!s]",
                    "[#8;!o][C;!$(C(=[O,N])[N,O])][#8;!o]",
                    "[#16;!s][C;!$(C(=[O,N])[N,O])][#16;!s]"
                ]
            }
        },

        # add component: calculate the QED drug-likeness score (using RDkit)
        {
            "component_type": "qed_score",
            "name": "QED Score",  # arbitrary name for the component
            "weight": 1,  # the weight of the component (default: 1)
        }]
}

scoring_params = ScoringFunctionParameters(
    scoring_function_parameters["name"],
    scoring_function_parameters["parameters"],
    scoring_function_parameters["parallel"],
)

scoring_class = ScoringFunctionFactory(scoring_params)
wrapper_scoring_class = WrapperScoringClass(scoring_class)
scoring_function = wrapper_scoring_class.get_final_score
