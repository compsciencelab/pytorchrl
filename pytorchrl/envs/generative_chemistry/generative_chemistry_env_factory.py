import os
from pytorchrl.envs.generative_chemistry.vocabulary import SMILESTokenizer, create_vocabulary
from pytorchrl.envs.generative_chemistry.environment import GenChemEnv
from reinvent_scoring.scoring import ScoringFunctionFactory

if __name__ == "__main__":

    smiles_list = ["[*:0]N1CCN(CC1)CCCCN[*:1]"]
    tokenizer = SMILESTokenizer()
    voc = create_vocabulary(smiles_list, tokenizer)

    scoring_function = {
        "name": "test_scoring_func",
        "parallel": False,  # Do not change
        "parameters": [
            {
                "component_type": "predictive_property",
                "name": "DRD2",
                "weight": 1,
                "specific_parameters": {
                    # "model_path": os.path.join(ipynb_path, "models/drd2.pkl"), # TODO: I guess here is loafing the pretrained model
                    # "scikit": "classification",
                    # "descriptor_type": "ecfp",
                    # "size": 2048,
                    # "radius": 3,
                    # "transformation": {
                    #     "transformation_type": "no_transformation"
                    # }
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
    scoring_function = ScoringFunctionFactory(scoring_function)

    env = GenChemEnv(
        scoring_function=scoring_function,
        base_molecule=smiles_list[0],
        vocabulary=voc)

    import ipdb; ipdb.set_trace()
