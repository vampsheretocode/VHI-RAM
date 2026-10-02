import yaml
import sys

def test_manifest_schema():
    print("Testing data manifest schema...")
    try:
        with open('data/manifest_schema.yml', 'r') as f:
            schema = yaml.safe_load(f)
        assert 'schema_version' in schema
        assert 'datasets' in schema
        print("Manifest schema valid.")
    except Exception as e:
        print(f"Manifest schema test failed: {e}")
        sys.exit(1)

def test_experiment_schema():
    print("Testing experiment config schema...")
    try:
        with open('configs/experiment_schema.yml', 'r') as f:
            schema = yaml.safe_load(f)
        assert 'experiment_id' in schema
        assert 'data' in schema
        assert 'models' in schema
        print("Experiment schema valid.")
    except Exception as e:
        print(f"Experiment schema test failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    test_manifest_schema()
    test_experiment_schema()
    print("[SUCCESS] All schema tests passed.")
