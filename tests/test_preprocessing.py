from log_psplines.preprocessing import estimate_psd_with_lines
from log_psplines.example_datasets.lvk_data import  LVKData

def test_estimate_psd_with_lines():
    """
    Test the estimate_psd_with_lines function with a sample LVKData instance.
    """
    # Load sample LVK data
    lvk_data = LVKData.load()

    # Extract a segment of the strain data for testing
    strain_segment = lvk_data.strain[:4096]  # Use first 1024 samples for testing



    # Estimate PSD with lines
    psd_estimate = estimate_psd_with_lines(strain_segment,)

    # Check if the output is a valid PSD
    assert psd_estimate is not None, "PSD estimate should not be None"
    assert len(psd_estimate) > 0, "PSD estimate should have positive length"

    print("Test passed: PSD estimated successfully.")





