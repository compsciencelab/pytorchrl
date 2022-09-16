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
            elif smile is None:
                raise TypeError
            else:
                raise ValueError("Scoring error due to wrong dtype")

            output.update({
                "valid_smile": True,
                "score": float(score.total_score[0]),
                "reward": float(score.total_score[0]),
                "DRD2": float(score.profile[0].score[0]),
                "custom_alerts": float(score.profile[1].score[0]),
                "raw_DRD2": float(score.profile[2].score[0]),
            })

        except TypeError:

            output.update({
                "valid_smile": False,
                "score": 0.0,
                "reward": 0.0,
                "DRD2": 0.0,
                "custom_alerts": 0.0,
                "raw_DRD2": 0.0,
            })

        return output


scoring_function_parameters = {
    "name": "custom_sum",
    "parallel": False,  # Do not change

    "parameters": [
        {
            "component_type": "predictive_property",
            "name": "DRD2",
            "weight": 1,
            "specific_parameters": {
                "model_path": os.path.join(os.path.dirname(__file__),
                                           '../../../../pytorchrl/envs/generative_chemistry/libinvent/models/drd2.pkl'),
                "scikit": "classification",
                "descriptor_type": "ecfp",
                "size": 2048,
                "radius": 3,
                "transformation": {
                    "transformation_type": "no_transformation"
                }
            }
        },
        {
            "component_type": "custom_alerts",
            "name": "Custom alerts",
            "weight": 1,
            "specific_parameters": {
                "smiles": [
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
        }]
}

scoring_params = ScoringFunctionParameters(
    scoring_function_parameters["name"],
    scoring_function_parameters["parameters"],
    scoring_function_parameters["parallel"])

scoring_class = ScoringFunctionFactory(scoring_params)
wrapper_scoring_class = WrapperScoringClass(scoring_class)
scoring_function = wrapper_scoring_class.get_final_score
