from __future__ import annotations

from synthetic_privacy_audit.attacks.calibrated_bayesian_mia import CalibratedBayesianMembershipInferenceAttack
from synthetic_privacy_audit.attacks.cluster_persistence import ClusterPersistenceAttack
from synthetic_privacy_audit.attacks.conditional_inference import ConditionalInferenceAttack
from synthetic_privacy_audit.attacks.distance_density_mia import DistanceDensityMembershipInferenceAttack
from synthetic_privacy_audit.attacks.ensemble_mia import EnsembleMembershipInferenceAttack
from synthetic_privacy_audit.attacks.fuzzy_outlier_linkage import FuzzyOutlierLinkageAttack
from synthetic_privacy_audit.attacks.gen_lra_mia import GenLikelihoodRatioMembershipInferenceAttack
from synthetic_privacy_audit.attacks.groundhog_linkability import GroundhogLinkabilityAttack
from synthetic_privacy_audit.attacks.intersection_linkability import IntersectionLinkabilityAttack
from synthetic_privacy_audit.attacks.joint_attribute_inference import JointAttributeInferenceAttack
from synthetic_privacy_audit.attacks.knn_stability import KNNStabilityAttack
from synthetic_privacy_audit.attacks.likelihood_ratio_mia import ShadowLikelihoodRatioMembershipInferenceAttack
from synthetic_privacy_audit.attacks.linear_reconstruction_attribute_inference import LinearReconstructionAttributeInferenceAttack
from synthetic_privacy_audit.attacks.logan_mia import LOGANMembershipInferenceAttack
from synthetic_privacy_audit.attacks.nearest_neighbor_attribute_inference import NearestNeighborAttributeInferenceAttack
from synthetic_privacy_audit.attacks.outlier_conditioned_inference import OutlierConditionedInferenceAttack
from synthetic_privacy_audit.attacks.outlier_stratified_mia import OutlierStratifiedMembershipInferenceAttack
from synthetic_privacy_audit.attacks.property_meta_classifier import PropertyMetaClassifierAttack
from synthetic_privacy_audit.attacks.rap_reconstruction import RAPReconstructionAttack
from synthetic_privacy_audit.attacks.record_influence import RecordInfluenceAttack
from synthetic_privacy_audit.attacks.shadow_attribute_inference import ShadowAttributeInferenceAttack
from synthetic_privacy_audit.attacks.shadow_model_mia import ShadowModelMembershipInferenceAttack
from synthetic_privacy_audit.attacks.shadow_property_inference import ShadowPropertyInferenceAttack
from synthetic_privacy_audit.attacks.singling_out import SinglingOutAttack
from synthetic_privacy_audit.attacks.statistical_property_inference import StatisticalPropertyInferenceAttack
from synthetic_privacy_audit.attacks.temporal_linkability import TemporalLinkabilityAttack
from synthetic_privacy_audit.attacks.topk_attribute_inference import TopKAttributeInferenceAttack


def all_attacks():
    """The complete Excel-derived attack suite executed by the main runner."""
    return [
        JointAttributeInferenceAttack(),
        RAPReconstructionAttack(),
        NearestNeighborAttributeInferenceAttack(),
        LinearReconstructionAttributeInferenceAttack(),
        ShadowAttributeInferenceAttack(),
        ConditionalInferenceAttack(),
        TopKAttributeInferenceAttack(),
        OutlierConditionedInferenceAttack(),
        FuzzyOutlierLinkageAttack(),
        OutlierStratifiedMembershipInferenceAttack(),
        KNNStabilityAttack(),
        RecordInfluenceAttack(),
        SinglingOutAttack(),
        IntersectionLinkabilityAttack(),
        ClusterPersistenceAttack(),
        TemporalLinkabilityAttack(),
        StatisticalPropertyInferenceAttack(),
        ShadowPropertyInferenceAttack(),
        PropertyMetaClassifierAttack(),
        ShadowModelMembershipInferenceAttack(),
        DistanceDensityMembershipInferenceAttack(),
        EnsembleMembershipInferenceAttack(),
        LOGANMembershipInferenceAttack(),
        ShadowLikelihoodRatioMembershipInferenceAttack(),
        GroundhogLinkabilityAttack(),
        GenLikelihoodRatioMembershipInferenceAttack(),
        CalibratedBayesianMembershipInferenceAttack(),
    ]

