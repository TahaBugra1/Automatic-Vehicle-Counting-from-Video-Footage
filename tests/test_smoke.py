def test_package_importable():
    import visionflow

    assert visionflow.__version__ == "0.1.0"
