PHASE_EARLY = "EarlySprouting"
PHASE_VG    = "VegetativeGrowth"
PHASE_BP    = "BulkingPhase"
PHASE_RM    = "RhizomeMaturation"

WEEK_TO_PHASE = {
    **{w: PHASE_EARLY for w in range(1, 5)},      # 1..4
    **{w: PHASE_VG    for w in range(5, 10)},     # 5..9
    **{w: PHASE_BP    for w in range(10, 15)},    # 10..14
    **{w: PHASE_RM    for w in range(15, 21)},    # 15..20
}

def phase_from_week(week: int) -> str:
    if week not in WEEK_TO_PHASE:
        raise ValueError(f"Week {week} out of supported range (1..20)")
    return WEEK_TO_PHASE[week]
