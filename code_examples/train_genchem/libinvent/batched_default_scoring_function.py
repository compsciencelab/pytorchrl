"""
Defines the default scoring function to guide the genchem RL agent.
"""

import os
import numpy as np
from reinvent_scoring.scoring import ScoringFunctionFactory
from reinvent_scoring.scoring.scoring_function_parameters import ScoringFunctionParameters


class WrapperScoringClass:

    def __init__(self, scoring_class):
        self.scoring_class = scoring_class

    def get_final_score(self, smiles):

        output = {}
        scores = self.scoring_class.get_final_score(smiles)
        valid_smiles = np.zeros_like(scores.total_score, dtype=bool)
        valid_smiles[scores.valid_idxs] = True

        output.update({
            "valid_smile": valid_smiles,
            "score": scores.total_score,
            "reward": scores.total_score,
            "DRD2": scores.profile[0].score,
            "custom_alerts": scores.profile[1].score,
            "raw_DRD2": scores.profile[2].score,
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
                "model_path": os.path.join(os.path.dirname(__file__), "models/drd2.pkl"),
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

if __name__ == "__main__":

    # TEST
    smiles_batch = [
        "Cc1ccc2c(c1)sc1c(=O)[nH]c3ccc(C(=O)NCCCN(C)C)cc3c12",
        "O=C(NCCN1CCOCC1)c1cc(C(F)(F)F)cc(C(F)(F)F)c1",
        "COc1cc(CN2CCCC(C(=O)Nc3ccccc3Oc3cccnc3)C2)ccc1F",
        "LOLOLOLOOLOLOLLLLOLLOLO",
        "COc1cccc(NC(=O)c2oc3ccccc3c2NC(=O)c2ccc3c(c2)OCO3)c1",
        "Cc1ncc(COP(=O)(O)O)c(CNC(Cc2ccc(O)c(O)c2)C(=O)O)c1O",
        "Cc1ccccc1N1C(=O)NC(=O)C(=Cc2cc(Br)c(N3CCOCC3)o2)C1=O",
        "COc1nc(=Nc2ccc(Cl)cc2)sn1C",
        "CCOC(=O)C(C)NP(=O)(COc1ccc(C)c2c1-c1ncsc1C2)NC(C)C(=O)OCC",
        "Cc1ccc(SCC(=O)OCC(=O)NC2(C#N)CCCCC2)cc1",
        "CC(C)(C)OC(=O)NC(Cc1ccc2cc(O)ccc2c1)C(=O)O",
    ]
    scores = scoring_function(smiles_batch)

    for k, v in scores.items():
        print(f"{k}: \n    type {type(v)} shape {v.shape}")
