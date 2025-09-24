import os

PHASE_EARLY = "EarlySprouting"
PHASE_VG    = "VegetativeGrowth"
PHASE_BP    = "BulkingPhase"
PHASE_RM    = "RhizomeMaturation"

def phase_ckpt_map(env=os.environ):
    return {
        PHASE_VG: env.get("MAML_CKPT_VG", "models/VG_maml_4shots.pth"),
        PHASE_BP: env.get("MAML_CKPT_BP", "models/BP_maml_3shots.pth"),
        PHASE_RM: env.get("MAML_CKPT_RM", "models/RM_maml_3shots.pth"),
    }
