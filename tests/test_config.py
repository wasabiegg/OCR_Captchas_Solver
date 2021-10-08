from config import BaseConfig


# Test
def test_load_config():
    base_config_1: BaseConfig = BaseConfig.get_config()
    assert base_config_1 is not None

    base_config_1.characters = ["2", "2", "2"]
    base_config_2: BaseConfig = BaseConfig.get_config()
    assert base_config_2 is not None
    assert base_config_2.characters == ["2", "2", "2"]

    assert base_config_2.data_dir.is_dir()

    base_config_2.characters = [str(i) for i in range(10)]

    assert base_config_1.characters == [str(i) for i in range(10)]
