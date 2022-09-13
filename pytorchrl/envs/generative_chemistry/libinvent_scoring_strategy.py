from typing import List, Any

from typing import List

from reinvent_chemistry import Conversions
from reinvent_chemistry.library_design import BondMaker, AttachmentPoints
from reinvent_scoring import FinalSummary, ScoringFunctionFactory
from reinvent_scoring.scoring.diversity_filters.reinvent_core.base_diversity_filter import BaseDiversityFilter
from running_modes.reinforcement_learning.dto.sampled_sequences_dto import SampledSequencesDTO

import numpy as np
from reinvent_chemistry.library_design.reaction_filters.reaction_filter import ReactionFilter
from reinvent_scoring import FinalSummary, ScoringFunctionComponentNameEnum, LoggableComponent, ComponentParameters, ComponentSummary
from reinvent_scoring.scoring.diversity_filters.lib_invent.base_diversity_filter import BaseDiversityFilter


class LibinventScoringStrategy:
    def __init__(self, scoring_function):

        self._bond_maker = BondMaker()
        self._conversion = Conversions()
        self._attachment_points = AttachmentPoints()
        self.scoring_function = scoring_function
        # self.reaction_filter = ReactionFilter(strategy_configuration.reaction_filter)

    def join_scaffold_and_decorations(self, scaffold, decorations):
        scaffold = self._attachment_points.add_attachment_point_numbers(scaffold, canonicalize=False)
        molecule = self._bond_maker.join_scaffolds_and_decorations(scaffold, decorations)
        smile = self._conversion.mol_to_smiles(molecule)
        return smile
