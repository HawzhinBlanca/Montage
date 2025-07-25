from montage.providers.audio_normalizer import NormalizationTarget

def test_enum_values():
    assert NormalizationTarget.EBU_R128.value == "ebu"
    assert list(NormalizationTarget) == [
        NormalizationTarget.EBU_R128,
        NormalizationTarget.FILM,
        NormalizationTarget.SPEECH,
    ]